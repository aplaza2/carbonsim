import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from .utils import _read_exclude_ids, _prepare_df, _require_columns, _apply_filters, _save_plot, _prepare_runs_info, _plot_time_series


def plot_projections(
    monitor_file: str = "emissions_monitor.csv",
    output_dir: str = "plots",
    run_id: Optional[str] = None,
    project_name: Optional[str] = None,
    exclude_runs: Optional[Iterable[str]] = None,
    exclude_file: Optional[str] = None,
    show_confidence: bool = True,
    label_by: str = "experiment_id"
):
    """Grafica proyecciones acumuladas de emisiones (solo proyecciones)."""
    excludes = _read_exclude_ids(exclude_runs, exclude_file)
    df = _prepare_df(monitor_file)
    
    required_cols = [
        "project_name", "run_id", "experiment_id", "elapsed_sec",
        "proj_linear", "proj_poly2", "proj_poly3",
        "ci_low_linear","ci_up_linear","ci_low_poly2","ci_up_poly2","ci_low_poly3","ci_up_poly3",
        "best_model_eval"
    ]
    _require_columns(df, required_cols, monitor_file)
    df = _apply_filters(df, run_id, project_name, excludes)

    for rid, run_df in df.groupby("run_id"):
        proj_name = run_df["project_name"].iloc[0]
        folder = os.path.join(output_dir, proj_name, "projections")

        plt.figure(figsize=(11,6))
        proj_cols = ["proj_linear", "proj_poly2", "proj_poly3"]
        best_model = run_df["best_model_eval"].iloc[-1] if "best_model_eval" in run_df.columns else None
        x_proj = pd.to_numeric(run_df["elapsed_sec"], errors="coerce")

        for label, col in zip(["linear","poly2","poly3"], proj_cols):
            y_vals = pd.to_numeric(run_df[col], errors="coerce")
            lbl = f"{label} projection"
            if best_model and label.startswith(best_model):
                lbl += " (best)"
            plt.plot(x_proj, y_vals, linestyle="--", label=lbl)

            if show_confidence:
                lo_col = f"ci_low_{label}"
                hi_col = f"ci_up_{label}"
                if lo_col in run_df.columns and hi_col in run_df.columns:
                    y_lo = pd.to_numeric(run_df[lo_col], errors="coerce")
                    y_hi = pd.to_numeric(run_df[hi_col], errors="coerce")
                    if not (y_lo.isna().all() or y_hi.isna().all()):
                        plt.fill_between(x_proj, y_lo, y_hi, alpha=0.15)

        label_value = str(run_df[label_by].iloc[0])
        plt.title(f"Projections for {label_value}")
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Cumulative emissions (kg CO2eq)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        filename = f"projections-{label_value}.png"
        _save_plot(plt.gcf(), folder, filename)


def plot_emissions(
    emissions_file: str = "emissions_realtime.csv",
    output_dir: str = "plots",
    project_name: Optional[str] = None,
    run_id: Optional[str] = None,
    exclude_runs: Optional[Iterable[str]] = None,
    exclude_file: Optional[str] = None,
    label_by: str = "experiment_id",
    mark_endpoints: bool = True
):
    """Grafica todas las emisiones reales por proyecto (agrupando por run_id o experiment_id)."""
    excludes = _read_exclude_ids(exclude_runs, exclude_file)
    df = _prepare_df(emissions_file)
    _require_columns(df, ["run_id", "project_name", "timestamp", "emissions"], emissions_file)
    df = _apply_filters(df, run_id, project_name, excludes)

    if df.empty:
        print("[carbonsim INFO] No hay datos para graficar luego de filtros/excludes.")
        return

    for proj, df_proj in df.groupby("project_name"):
        folder = os.path.join(output_dir, proj)
        runs_info = _prepare_runs_info(df_proj, label_by, "emissions")
        _plot_time_series(runs_info, title=f"Emissions for project {proj}",
                          xlabel="Elapsed time (s)", ylabel="Cumulative emissions (kg CO2eq)",
                          output_folder=folder, filename="emissions.png",
                          mark_endpoints=mark_endpoints)


def plot_emissions_rates(
    emissions_file: str = "emissions_realtime.csv",
    output_dir: str = "plots",
    project_name: Optional[str] = None,
    run_id: Optional[str] = None,
    excludes: Optional[Iterable[str]] = None,
    mark_endpoints: bool = True,
    label_by: str = "experiment_id"
):
    """Grafica emissions_rate por proyecto (agrupando por run_id o experiment_id)."""
    if excludes is None:
        excludes = []
    df = _prepare_df(emissions_file)
    _require_columns(df, ["project_name", "run_id", "timestamp", "emissions_rate"], emissions_file)
    df = _apply_filters(df, run_id, project_name, excludes)

    if df.empty:
        print("[carbonsim INFO] No hay datos para graficar con los filtros dados.")
        return

    for proj, df_proj in df.groupby("project_name"):
        folder = os.path.join(output_dir, proj)
        runs_info = _prepare_runs_info(df_proj, label_by, "emissions_rate")
        _plot_time_series(runs_info, title=f"Emission Rate for project {proj}",
                          xlabel="Elapsed time (s)", ylabel="Emission rate (kg CO2eq/s)",
                          output_folder=folder, filename="emissions_rates.png",
                          mark_endpoints=mark_endpoints)
