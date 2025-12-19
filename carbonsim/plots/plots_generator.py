import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from .plots_utils import _read_ids, _prepare_df, _require_columns, _apply_filters, _save_plot, _prepare_runs_info, _plot_time_series, _to_float


def plot_emissions(
    emissions_file: str = "emissions_realtime.csv",
    output_dir: str = "plots",
    include_runs: Optional[Iterable[str]] = None,
    include_file: Optional[str] = None,
    exclude_runs: Optional[Iterable[str]] = None,
    exclude_file: Optional[str] = None,
    label_by: str = "experiment_id",
    mark_endpoints: bool = True,
    xlim: Optional[float] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    title: Optional[str] = None,
    filename: str = "emissions.png"
):
    """Grafica todas las emisiones reales por proyecto (agrupando por run_id o experiment_id)."""
    includes = _read_ids(include_runs, include_file)
    excludes = _read_ids(exclude_runs, exclude_file)
    df = _prepare_df(emissions_file)
    _require_columns(df, ["run_id", "project_name", "timestamp", "emissions"], emissions_file)
    df = _apply_filters(df, includes, excludes)

    if df.empty:
        print("[carbonsim INFO] No hay datos para graficar luego de filtros/excludes.")
        return

    for proj, df_proj in df.groupby("project_name"):
        folder = os.path.join(output_dir, proj)
        runs_info = _prepare_runs_info(df_proj, label_by, "emissions")
        if title is None:
            title = f"Emissions for project {proj}"
        _plot_time_series(
            runs_info,
            title=title,
            xlabel="Elapsed time (s)",
            ylabel="Cumulative emissions (kg CO2eq)",
            output_folder=folder,
            filename=filename,
            mark_endpoints=mark_endpoints,
            xlim=xlim,
            xscale=xscale,
            yscale=yscale
        )


def plot_emissions_rates(
    emissions_file: str = "emissions_realtime.csv",
    output_dir: str = "plots",
    include_runs: Optional[Iterable[str]] = None,
    include_file: Optional[str] = None,
    exclude_runs: Optional[Iterable[str]] = None,
    exclude_file: Optional[str] = None,
    mark_endpoints: bool = True,
    label_by: str = "experiment_id",
    xlim: Optional[float] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    title: Optional[str] = None,
    filename: str = "emissions_rates.png"
):
    """Grafica emissions_rate por proyecto (agrupando por run_id o experiment_id)."""
    includes = _read_ids(include_runs, include_file)
    excludes = _read_ids(exclude_runs, exclude_file)
    df = _prepare_df(emissions_file)
    _require_columns(df, ["run_id", "project_name", "timestamp", "emissions"], emissions_file)
    df = _apply_filters(df, includes, excludes)

    if df.empty:
        print("[carbonsim INFO] No hay datos para graficar con los filtros dados.")
        return

    for proj, df_proj in df.groupby("project_name"):
        folder = os.path.join(output_dir, proj)
        runs_info = _prepare_runs_info(df_proj, label_by, "emissions_rate")
        if title is None:
            title = f"Emission Rate for project {proj}"
        _plot_time_series(
            runs_info, 
            title=title,
            xlabel="Elapsed time (s)", 
            ylabel="Emission rate (kg CO2eq/s)",
            output_folder=folder, 
            filename=filename,
            mark_endpoints=mark_endpoints,
            xlim=xlim,
            xscale=xscale,
            yscale=yscale
        )


def plot_projections(
    monitor_file: str = "emissions_monitor.csv",
    output_dir: str = "plots",
    include_runs: Optional[Iterable[str]] = None,
    include_file: Optional[str] = None,
    exclude_runs: Optional[Iterable[str]] = None,
    exclude_file: Optional[str] = None,
    show_confidence: bool = True,
    label_by: str = "experiment_id"
):
    """Grafica proyecciones acumuladas de emisiones en el horizonte de tiempo. Se realiza un gráfico por experimento"""
    includes = _read_ids(include_runs, include_file)
    excludes = _read_ids(exclude_runs, exclude_file)
    df = _prepare_df(monitor_file)
    
    required_cols = [
        "project_name", "run_id", "experiment_id", "elapsed_sec",
        "proj_linear", "proj_poly2", "proj_poly3",
        "ci_low_linear","ci_up_linear","ci_low_poly2","ci_up_poly2","ci_low_poly3","ci_up_poly3",
        "best_model_eval"
    ]
    _require_columns(df, required_cols, monitor_file)
    df = _apply_filters(df, includes, excludes)

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


def plot_multiple_projections(
    monitor_file: str = "emissions_monitor.csv",
    projections_file: str = "emissions_projections.csv",
    output_dir: str = "plots",
    include_runs: Optional[Iterable[str]] = None,
    include_file: Optional[str] = None,
    exclude_runs: Optional[Iterable[str]] = None,
    exclude_file: Optional[str] = None,
    show_confidence: bool = True,
    label_by: str = "experiment_id",
):
    """
    Una recta por experimento:
      - tramo real (monitor_csv)
      - tramo proyectado (projections_csv, best_model)
    """

    # -------- filtros include / exclude --------
    includes = _read_ids(include_runs, include_file)
    excludes = _read_ids(exclude_runs, exclude_file)

    mon = _prepare_df(monitor_file)
    proj = _prepare_df(projections_file)
    proj = proj.drop_duplicates(subset=["run_id"], keep="last")


    _require_columns(
        mon,
        ["project_name", "run_id", "experiment_id",
         "elapsed_sec", "real_emissions_kg"],
        monitor_file,
    )
    _require_columns(
        proj,
        ["project_name", "run_id", "experiment_id",
         "target_elapsed_sec", "best_model",
         "proj_linear", "proj_poly2", "proj_poly3"],
        projections_file,
    )

    mon = _apply_filters(mon, includes, excludes)
    proj = _apply_filters(proj, includes, excludes)

    # Indexar proyecciones por run_id (1 fila por run)
    proj_idx = proj.set_index("run_id")

    for project_name, mon_proj in mon.groupby("project_name"):
        folder = os.path.join(output_dir, project_name, "projections")
        plt.figure(figsize=(12, 7))

        for run_id, run_df in mon_proj.groupby("run_id"):
            if run_id not in proj_idx.index:
                continue

            run_df = run_df.sort_values("elapsed_sec")
            p = proj_idx.loc[run_id]

            label_value = str(run_df[label_by].iloc[0])
            best_model = p["best_model"]
            proj_col = f"proj_{best_model}"

            # -------- tramo real --------
            x_real = run_df["elapsed_sec"].values
            y_real = run_df["real_emissions_kg"].values

            line_real, = plt.plot(
                x_real,
                y_real,
                linewidth=2,
                label=f"{label_value} ({best_model})",
            )

            color = line_real.get_color()


            # -------- tramo proyectado --------
            t_last = float(x_real[-1])
            t_target = _to_float(p["target_elapsed_sec"])
            y_target = _to_float(p[proj_col])

            if t_target > t_last:
                plt.plot(
                    [t_last, t_target],
                    [y_real[-1], y_target],
                    linestyle="--",
                    linewidth=2,
                    color=color
                )

                # -------- intervalo de confianza --------
                if show_confidence:
                    lo = p.get(f"ci_low_{best_model}")
                    hi = p.get(f"ci_up_{best_model}")
                    if pd.notna(lo) and pd.notna(hi):
                        plt.fill_between(
                            [t_last, t_target],
                            [y_real[-1], lo],
                            [y_real[-1], hi],
                            alpha=0.15,
                            color=color
                        )

        plt.title(f"Best Emissions Projections - {project_name}")
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Cumulative emissions (kg CO₂eq)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title=None)
        filename = f"best-projections-{project_name}.png"
        _save_plot(plt.gcf(), folder, filename)
