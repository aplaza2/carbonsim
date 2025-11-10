import os
import threading
import numpy as np
import pandas as pd
import functools
from datetime import datetime, timedelta
from codecarbon import EmissionsTracker

from .logging_utils import print_final_horizon, print_no_data, print_error
from .config import load_config
from .regressor_models import PolyRegressor
from .drift import PageHinkley

def _mae(e): return np.nanmean(np.abs(e))
def _rmse(e): return np.sqrt(np.nanmean(e**2))
def _mape(e, y): return np.nanmean(np.abs(e) / np.maximum(np.abs(y), 1e-12)) * 100.0


class CarbonSimulator:
    """
    CarbonSimulator monitors and records the carbon footprint of an experiment.

    This class uses an EmissionsTracker to measure energy consumption and carbon emissions
    during the execution of an experiment. Each experiment has a unique `run_id` to
    distinguish measurements in the CSV log. Additionally, it makes future emission
    projections and can detect possible changes or drifts in emission patterns.

    Main parameters:
    - experiment_id: Identifier for the experiment.
    - interval_sec: Time interval (in seconds) between measurements.
    - monitor_csv: CSV file where measurements and projections are recorded.
    - horizon_sec: Time horizon for final projections.
    """
    def __init__(self, **kwargs):
    
        cfg = load_config(config_path=".carbonsim.config", **kwargs)
        self.cfg = cfg
        self.carbon_csv = cfg.carbon_csv
        self.monitor_csv = cfg.monitor_csv
        self.projections_file = cfg.projections_file
        self.interval_sec = cfg.interval_sec
        self.horizon_sec = cfg.horizon_sec
        self.log_level = cfg.log_level
        self.automatic_projections = cfg.automatic_projections

        self.tracker = EmissionsTracker(
            output_file=self.carbon_csv,
            project_name=self.cfg.project_name,
            experiment_id=self.cfg.experiment_id,
            measure_power_secs=self.cfg.measure_power_secs,
            save_to_file=self.cfg.save_to_file,
            log_level=self.cfg.codecarbon_log_level,
            tracking_mode=self.cfg.tracking_mode,
            on_csv_write=self.cfg.on_csv_write,
            csv_write_interval=self.cfg.csv_write_interval

        )
   
        measure_power_secs = getattr(self.tracker._conf, "measure_power_secs", self.cfg.measure_power_secs)
        csv_write_interval = getattr(self.tracker._conf, "csv_write_interval", self.cfg.csv_write_interval)
        if self.interval_sec <= measure_power_secs * csv_write_interval:
            raise ValueError(
                f"interval_sec={self.interval_sec} debe ser mayor que "
                f"measure_power_secs*csv_write_interval={measure_power_secs * csv_write_interval}"
            )

        self.start_time = None
        self.timer = None
        self._lock = threading.Lock()
        if self.automatic_projections:
            self._init_monitor_csv()

        self._ph = PageHinkley(delta=self.cfg.ph_delta, lamb=self.cfg.ph_lambda) \
            if self.cfg.drift_detector == "ph" else None

    # ---------- IO ----------
    def _init_monitor_csv(self):
        if not os.path.exists(self.monitor_csv):
            pd.DataFrame(columns=[
                "project_name","run_id","experiment_id",
                "elapsed_sec","real_emissions_kg",
                "proj_linear","proj_poly2","proj_poly3",
                "ci_low_linear","ci_up_linear","ci_low_poly2","ci_up_poly2","ci_low_poly3","ci_up_poly3",
                "error_linear","error_poly2","error_poly3",
                "sim_next_linear","sim_next_poly2","sim_next_poly3",
                "sim_ci_low_linear","sim_ci_up_linear","sim_ci_low_poly2","sim_ci_up_poly2","sim_ci_low_poly3","sim_ci_up_poly3",
                "best_model_eval","metric_value_eval","drift_flag"
            ]).to_csv(self.monitor_csv, index=False)

    def _read_carbon_csv(self):
        df = pd.read_csv(
            self.carbon_csv,
            parse_dates=["timestamp"],
            dtype={
                "ram_power": "float64",
                "cpu_energy": "float64",
                "gpu_energy": "float64",
                "cloud_provider": "string",
                "cpu_model": "string",
            },
            low_memory=False
        )

        if df.empty:
            return df
        
        current_run_id = str(getattr(self.tracker, "run_id", None))
        df = df[df["run_id"] == current_run_id].copy()

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df["emissions"] = pd.to_numeric(df["emissions"], errors="coerce")
        df = df.dropna(subset=["timestamp","emissions"])
        return df

    # ---------- utils ----------
    @staticmethod
    def _elapsed_seconds(ts: pd.Series) -> np.ndarray:
        t0 = ts.iloc[0]
        return (ts - t0).dt.total_seconds().values

    def _fit_models(self, t: np.ndarray, y: np.ndarray):
        models = {}
        for deg in range(1, self.cfg.degree_max + 1):
            models[deg] = PolyRegressor(
                degree=deg,
                regularization=self.cfg.regularization,
                alpha=self.cfg.alpha
            ).fit(t, y)
        return models

    def _choose_best(self, errs: dict, y_real: float | None):
        values = {}
        for deg, e in errs.items():
            if e is None or not np.isfinite(e):
                values[deg] = np.inf
                continue
            if self.cfg.metric == "mae":
                values[deg] = _mae(np.array([e]))
            elif self.cfg.metric == "rmse":
                values[deg] = _rmse(np.array([e]))
            else:
                values[deg] = _mape(np.array([e]), np.array([y_real if y_real is not None else 0.0]))
        best_deg = min(values, key=lambda k: (values[k], k))  # desempate por simplicidad
        best_name = {1:"linear", 2:"poly2", 3:"poly3"}[best_deg]
        return best_name, values[best_deg]

    # ---------- ciclo ----------
    def start(self):
        """
        Starts tracking the experiment's carbon emissions.

        - Launches the EmissionsTracker to begin recording energy and emissions.
        - Schedules the first call to `_monitor_step` to compute projections and update the monitor CSV.
        - Records the experiment start time.
        """
        self.start_time = datetime.now()
        self.tracker.start()

    def stop(self):
        """
        Stop tracking and, **once stopped**, generate the full monitor CSV (by reading
        emissions_realtime.csv) and then the final projections file.
        """
        # Stop tracker first so carbon_csv is flushed & complete
        try:
            self.tracker.stop()
        except Exception:
            # best-effort stop; still try to process files
            pass

        if not self.automatic_projections:
            return

        # Build the monitor CSV from the realtime carbon file
        try:
            self._build_monitor_from_realtime()
        except Exception as e:
            print_error(e, self.log_level)
            return

        # Now generate final projections based on the monitor CSV (same logic you had)
        try:
            # Read the monitor we just built
            mon = pd.read_csv(self.monitor_csv)
            # if empty nothing to do
            if mon.empty:
                print_no_data(self.log_level)
                return

            # Determine best_final across whole monitor (same code as before)
            if self.cfg.metric == "mape":
                if "real_emissions_kg" in mon:
                    vals = []
                    for col in ["error_linear","error_poly2","error_poly3"]:
                        if col in mon:
                            e = mon[col].values
                            yv = mon["real_emissions_kg"].values
                            vals.append(_mape(e, yv))
                        else:
                            vals.append(np.inf)
                    best_final = ["linear","poly2","poly3"][int(np.nanargmin(vals))]
                else:
                    best_final = "linear"
            else:
                vals = []
                for col in ["error_linear","error_poly2","error_poly3"]:
                    if col in mon:
                        e = mon[col].values
                        vals.append(_mae(e) if self.cfg.metric == "mae" else _rmse(e))
                    else:
                        vals.append(np.inf)
                best_final = ["linear","poly2","poly3"][int(np.nanargmin(vals))]

            # Build final projections using last available models fitted on full realtime
            # Reuse the logic you already had in _monitor_step final block:
            df = self._read_carbon_csv()
            if df.empty:
                print_no_data(self.log_level)
                return

            times = self._elapsed_seconds(df["timestamp"])
            emissions = df["emissions"].values.astype(float)
            models_fit = self._fit_models(times, emissions)

            t0 = df["timestamp"].iloc[0]
            horizon_abs = self.start_time + timedelta(seconds=self.horizon_sec)
            target_h = max((horizon_abs - t0).total_seconds(), float(times[-1]))

            preds_h, ci_h = {}, {}
            for deg, model in models_fit.items():
                pred, lo, up = model.predict(np.array([target_h]), ci_alpha=self.cfg.ci_alpha, add_pred_var=True)
                preds_h[deg] = float(pred[0])
                ci_h[deg] = (
                    float(lo[0]) if lo is not None else None,
                    float(up[0]) if up is not None else None
                )

            best_deg = {"linear":1,"poly2":2,"poly3":3}[best_final]

            header = not os.path.exists(self.projections_file)
            final_row = {
                "project_name": mon["project_name"].iloc[-1] if "project_name" in mon.columns else self.cfg.project_name,
                "run_id": mon["run_id"].iloc[-1] if "run_id" in mon.columns else None,
                "experiment_id": mon["experiment_id"].iloc[-1] if "experiment_id" in mon.columns else self.cfg.experiment_id,
                "timestamp_last": df["timestamp"].iloc[-1].isoformat(),
                "horizon_sec": self.horizon_sec,
                "target_elapsed_sec": target_h,
                "proj_linear": preds_h.get(1, np.nan),
                "proj_poly2": preds_h.get(2, np.nan),
                "proj_poly3": preds_h.get(3, np.nan),
                "ci_low_linear": ci_h.get(1,(None,None))[0],
                "ci_up_linear":  ci_h.get(1,(None,None))[1],
                "ci_low_poly2":  ci_h.get(2,(None,None))[0],
                "ci_up_poly2":   ci_h.get(2,(None,None))[1],
                "ci_low_poly3":  ci_h.get(3,(None,None))[0],
                "ci_up_poly3":   ci_h.get(3,(None,None))[1],
                "best_model": best_final,
                "best_projection": preds_h.get(best_deg, np.nan)
            }
            pd.DataFrame([final_row]).to_csv(self.projections_file, mode="a", header=header, index=False)
            print_final_horizon(preds_h, ci_h, self.horizon_sec, best_deg, best_final, self.log_level)

        except Exception as e:
            print_error(e, self.log_level)

    def _build_monitor_from_realtime(self):
        """
        Construye emissions_monitor.csv leyendo todo emissions_realtime.csv y
        creando una fila por intervalo de tamaño self.interval_sec.

        Para cada intervalo k (t = k * interval_sec):
        - usa todas las filas de realtime con timestamp <= t para obtener el último valor real
        - si k == 1: coloca proyecciones de evaluación y sim_next en NaN/valor por defecto
        - si k > 1: ajusta modelos con los datos hasta t y calcula:
            * proj_*  = evaluación en t (out-of-sample usando hasta t)
            * sim_next_* = predicción en t + interval_sec  (simulación del próximo intervalo)
            * intervalos de confianza si el regresor lo provee
        - guarda error = proj - real
        Finalmente escribe todo el DataFrame a self.monitor_csv (sobrescribe).
        """
        df = self._read_carbon_csv()
        if df is None or df.empty:
            print_no_data(self.log_level)
            return

        # Datos básicos
        df = df.sort_values("timestamp").reset_index(drop=True)
        t0 = df["timestamp"].iloc[0]
        total_secs = (df["timestamp"].iloc[-1] - t0).total_seconds()
        n_intervals = int(np.floor(total_secs / self.interval_sec))
        # Si al menos una medición y total_secs < interval_sec, igual creamos 1 fila:
        if n_intervals == 0 and len(df) > 0:
            n_intervals = 1

        rows = []
        for k in range(1, n_intervals + 1):
            cutoff = t0 + timedelta(seconds=k * self.interval_sec)
            df_k = df[df["timestamp"] <= cutoff]
            if df_k.empty:
                continue

            last_row = df_k.iloc[-1]
            project_name = last_row.get("project_name", None)
            run_id = last_row.get("run_id", None)
            experiment_id = last_row.get("experiment_id", None)
            elapsed = float((cutoff - t0).total_seconds())
            real = float(last_row["emissions"])

            if k == 1:
                proj_eval = {1: real, 2: real, 3: real}
                ci_eval = {1: (None, None), 2: (None, None), 3: (None, None)}
                errs = {1: np.nan, 2: np.nan, 3: np.nan}
                sim_next = {1: np.nan, 2: np.nan, 3: np.nan}
                sim_ci = {1: (None, None), 2: (None, None), 3: (None, None)}
                best_name = "linear"
                metric_value = np.nan
                drift_flag = False
            else:
                times = (df_k["timestamp"] - t0).dt.total_seconds().values
                emissions = df_k["emissions"].values.astype(float)
                models = self._fit_models(times, emissions)

                proj_eval, ci_eval, errs = {}, {}, {}
                sim_next, sim_ci = {}, {}
                for deg, model in models.items():
                   
                    pred_eval, lo_eval, up_eval = model.predict(np.array([elapsed]), ci_alpha=self.cfg.ci_alpha, add_pred_var=False)
                    proj_eval[deg] = float(pred_eval[0])
                    ci_eval[deg] = (
                        float(lo_eval[0]) if lo_eval is not None else None,
                        float(up_eval[0]) if up_eval is not None else None
                    )
                    
                    pred_next, lo_n, up_n = model.predict(np.array([elapsed + self.interval_sec]), ci_alpha=self.cfg.ci_alpha, add_pred_var=True)
                    sim_next[deg] = float(pred_next[0])
                    sim_ci[deg] = (
                        float(lo_n[0]) if lo_n is not None else None,
                        float(up_n[0]) if up_n is not None else None
                    )
                    errs[deg] = proj_eval[deg] - real

                best_name, metric_value = self._choose_best(errs, real if self.cfg.metric == "mape" else None)
                drift_flag = False  # Mantengo simple: si quieres aplicar Page-Hinkley por fila, podemos acumular errores y usar self._ph.update()

            ci1 = ci_eval.get(1,(None,None)); ci2 = ci_eval.get(2,(None,None)); ci3 = ci_eval.get(3,(None,None))
            sci1 = sim_ci.get(1,(None,None)); sci2 = sim_ci.get(2,(None,None)); sci3 = sim_ci.get(3,(None,None))

            row = {
                "project_name": project_name,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "elapsed_sec": elapsed,
                "real_emissions_kg": real,
                "proj_linear": proj_eval.get(1, np.nan), "proj_poly2": proj_eval.get(2, np.nan), "proj_poly3": proj_eval.get(3, np.nan),
                "ci_low_linear": ci1[0], "ci_up_linear": ci1[1],
                "ci_low_poly2": ci2[0], "ci_up_poly2": ci2[1],
                "ci_low_poly3": ci3[0], "ci_up_poly3": ci3[1],
                "error_linear": errs.get(1, np.nan), "error_poly2": errs.get(2, np.nan), "error_poly3": errs.get(3, np.nan),
                "sim_next_linear": sim_next.get(1, np.nan), "sim_next_poly2": sim_next.get(2, np.nan), "sim_next_poly3": sim_next.get(3, np.nan),
                "sim_ci_low_linear": sci1[0], "sim_ci_up_linear": sci1[1],
                "sim_ci_low_poly2": sci2[0], "sim_ci_up_poly2": sci2[1],
                "sim_ci_low_poly3": sci3[0], "sim_ci_up_poly3": sci3[1],
                "best_model_eval": best_name,
                "metric_value_eval": metric_value,
                "drift_flag": bool(drift_flag),
            }
            rows.append(row)

        # Escribir monitor
        if rows:
            df_rows = pd.DataFrame(rows)
            file_exists = os.path.exists(self.monitor_csv)
            df_rows.to_csv(self.monitor_csv, mode="a", header=not file_exists, index=False)
        
        else:
            print_no_data(self.log_level)


def carbon_simulation(*dargs, **decorator_kwargs):
    """
    Decorador para ejecutar bajo CarbonSimulator.
    Permite usarlo con @carbon_simulation o @carbon_simulation(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sim = CarbonSimulator(**decorator_kwargs)
            sim.start()
            try:
                result = func(*args, **kwargs)
            finally:
                sim.stop()
            return result
        return wrapper

    # Caso sin paréntesis: @carbon_simulation
    if len(dargs) == 1 and callable(dargs[0]):
        return decorator(dargs[0])
    # Caso con paréntesis: @carbon_simulation(...)
    return decorator

