import os
import threading
import numpy as np
import pandas as pd
import functools
from datetime import datetime, timedelta
from codecarbon import EmissionsTracker

from .logging_utils import print_emission_log, print_final_horizon, print_no_data, print_error
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
        df = pd.read_csv(self.carbon_csv, parse_dates=["timestamp"])
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
        self._schedule_next()

    def stop(self):
        """
        Stops tracking the experiment's carbon emissions.

        - Cancels the timer that schedules `_monitor_step`.
        - Waits for the last ongoing measurement to finish.
        - Calls `_monitor_step(final=True)` to record the final horizon projection.
        - Stops the EmissionsTracker.
        """
        if self.timer:
            self.timer.cancel()
        self._monitor_step(final=True)
        self.tracker.stop()

    def _schedule_next(self):
        self.timer = threading.Timer(self.interval_sec, self._monitor_step)
        self.timer.daemon = True
        self.timer.start()

    def _apply_window_and_drift(self, t: np.ndarray, y: np.ndarray, last_error_abs: float | None):
        drift_flag = False
        if self._ph is not None and last_error_abs is not None and np.isfinite(last_error_abs):
            if self._ph.update(last_error_abs):
                drift_flag = True
                self._ph.reset()
        if self.cfg.window_size > 0:
            t = t[-self.cfg.window_size:]
            y = y[-self.cfg.window_size:]
        if drift_flag:
            k = min(5, len(t))
            t = t[-k:]
            y = y[-k:]
        return t, y, drift_flag

    def _monitor_step(self, final=False):
        with self._lock:
            try:
                df = self._read_carbon_csv()
                if df is None or df.empty:
                    print_no_data(self.log_level)
                    if not final:
                        self._schedule_next()
                    return

                times = self._elapsed_seconds(df["timestamp"])
                emissions = df["emissions"].values.astype(float)

                last_row = df.iloc[-1]
                project_name = last_row.get("project_name", None)
                run_id = last_row.get("run_id", None)
                experiment_id = last_row.get("experiment_id", None)
                last_elapsed = float(times[-1])
                y_real = float(emissions[-1])

                # 1) Evaluación out-of-sample
                if len(times) <= 1:
                    proj_eval = {1:y_real, 2:y_real, 3:y_real}
                    ci_eval = {1:(None,None), 2:(None,None), 3:(None,None)}
                    errs = {1:np.nan, 2:np.nan, 3:np.nan}
                    best_name, best_metric = "linear", np.nan
                    last_error_abs = None
                    t_fit, y_fit, drift_flag = self._apply_window_and_drift(times, emissions, None)
                else:
                    t_tr = times[:-1]
                    y_tr = emissions[:-1]
                    t_tr, y_tr, _ = self._apply_window_and_drift(t_tr, y_tr, None)
                    models = self._fit_models(t_tr, y_tr)

                    proj_eval, ci_eval, errs = {}, {}, {}
                    for deg, model in models.items():
                        pred, lo, up = model.predict(np.array([times[-1]]), ci_alpha=self.cfg.ci_alpha, add_pred_var=False)
                        proj_eval[deg] = float(pred[0])
                        ci_eval[deg] = (
                            float(lo[0]) if lo is not None else None,
                            float(up[0]) if up is not None else None
                        )
                        errs[deg] = proj_eval[deg] - y_real

                    best_name, best_metric = self._choose_best(errs, y_real if self.cfg.metric == "mape" else None)
                    best_deg = {"linear":1,"poly2":2,"poly3":3}[best_name]
                    last_error_abs = abs(errs.get(best_deg, np.nan))
                    t_fit, y_fit, drift_flag = self._apply_window_and_drift(times, emissions, last_error_abs)

                # 2) Simulación próximo intervalo
                models_fit = self._fit_models(t_fit, y_fit)
                t_next = last_elapsed + self.interval_sec
                sim_next, sim_ci = {}, {}
                for deg, model in models_fit.items():
                    pred, lo, up = model.predict(np.array([t_next]), ci_alpha=self.cfg.ci_alpha, add_pred_var=True)
                    sim_next[deg] = float(pred[0])
                    sim_ci[deg] = (
                        float(lo[0]) if lo is not None else None,
                        float(up[0]) if up is not None else None
                    )

                # 3) Guardar monitor.csv
                if not final:
                    def get_eval(deg): return proj_eval.get(deg, np.nan)
                    def get_eval_ci(deg): return ci_eval.get(deg, (None, None))
                    def get_sim(deg): return sim_next.get(deg, np.nan)
                    def get_sim_ci(deg): return sim_ci.get(deg, (None, None))

                    ci1 = get_eval_ci(1); ci2 = get_eval_ci(2); ci3 = get_eval_ci(3)
                    sci1 = get_sim_ci(1); sci2 = get_sim_ci(2); sci3 = get_sim_ci(3)

                    row = {
                        "project_name": project_name,
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "elapsed_sec": last_elapsed,
                        "real_emissions_kg": y_real,
                        "proj_linear": get_eval(1), "proj_poly2": get_eval(2), "proj_poly3": get_eval(3),
                        "ci_low_linear": ci1[0], "ci_up_linear": ci1[1],
                        "ci_low_poly2": ci2[0], "ci_up_poly2": ci2[1],
                        "ci_low_poly3": ci3[0], "ci_up_poly3": ci3[1],
                        "error_linear": (proj_eval.get(1, np.nan) - y_real) if len(times) > 1 else np.nan,
                        "error_poly2": (proj_eval.get(2, np.nan) - y_real) if len(times) > 1 else np.nan,
                        "error_poly3": (proj_eval.get(3, np.nan) - y_real) if len(times) > 1 else np.nan,
                        "sim_next_linear": get_sim(1), "sim_next_poly2": get_sim(2), "sim_next_poly3": get_sim(3),
                        "sim_ci_low_linear": sci1[0], "sim_ci_up_linear": sci1[1],
                        "sim_ci_low_poly2": sci2[0], "sim_ci_up_poly2": sci2[1],
                        "sim_ci_low_poly3": sci3[0], "sim_ci_up_poly3": sci3[1],
                        "best_model_eval": best_name,
                        "metric_value_eval": best_metric,
                        "drift_flag": bool(drift_flag),
                    }
                    pd.DataFrame([row]).to_csv(self.monitor_csv, mode="a", header=False, index=False)

                    # 4) Logs
                    print_emission_log(
                        log_level=self.log_level,
                        last_elapsed=last_elapsed,
                        y_real=y_real,
                        proj_eval=proj_eval,
                        ci_eval=ci_eval,
                        sim_next=sim_next,
                        best_name=best_name,
                        best_metric=best_metric,
                        interval_sec=self.interval_sec,
                        metric_name=self.cfg.metric
                    )

                # 5) Proyección final al horizonte
                if final:
                    t0 = df["timestamp"].iloc[0]
                    horizon_abs = self.start_time + timedelta(seconds=self.horizon_sec)
                    target_h = max((horizon_abs - t0).total_seconds(), last_elapsed)

                    preds_h, ci_h = {}, {}
                    for deg, model in models_fit.items():
                        pred, lo, up = model.predict(np.array([target_h]), ci_alpha=self.cfg.ci_alpha, add_pred_var=True)
                        preds_h[deg] = float(pred[0])
                        ci_h[deg] = (
                            float(lo[0]) if lo is not None else None,
                            float(up[0]) if up is not None else None
                        )

                    # Elegir mejor modelo sobre histórico monitor
                    try:
                        mon = pd.read_csv(self.monitor_csv)
                    except Exception:
                        mon = pd.DataFrame()
                    if not mon.empty:
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
                    else:
                        best_final = "linear"

                    best_deg = {"linear":1,"poly2":2,"poly3":3}[best_final]

                    header = not os.path.exists(self.projections_file)
                    final_row = {
                        "project_name": project_name,
                        "run_id": run_id,
                        "experiment_id": experiment_id,
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
            finally:
                if not final:
                    self._schedule_next()



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

