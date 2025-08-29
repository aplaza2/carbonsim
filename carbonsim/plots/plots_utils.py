import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Set, List, Tuple

def _read_exclude_ids(exclude_runs: Optional[Iterable[str]] = None,
                      exclude_file: Optional[str] = None) -> Set[str]:
    """Construye el conjunto de run_id a excluir combinando lista y archivo."""
    excludes: Set[str] = set()
    if exclude_runs:
        excludes.update(map(str, exclude_runs))
    if exclude_file:
        try:
            with open(exclude_file, "r", encoding="utf-8") as f:
                file_ids = [line.strip() for line in f if line.strip()]
            excludes.update(file_ids)
        except Exception as e:
            print(f"[carbonsim WARNING] No se pudo leer exclude_file='{exclude_file}': {e}")
    if excludes:
        print(f"[carbonsim EXCLUDE] run_id excluidos: {sorted(excludes)}")
    return excludes


def _prepare_df(file_path: str,
                parse_ts_cols: Iterable[str] = ("timestamp", "timestamp_last")) -> pd.DataFrame:
    """Lee CSV y convierte a datetime las columnas de timestamp si existen."""
    df = pd.read_csv(file_path)
    for col in parse_ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {name}: {missing}")


def _apply_filters(df: pd.DataFrame,
                   run_id: Optional[str] = None,
                   project_name: Optional[str] = None,
                   excludes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Aplica filtros por run_id, project_name y exclusions."""
    if run_id:
        df = df[df["run_id"] == run_id]
    if project_name:
        df = df[df["project_name"] == project_name]
    if excludes:
        df = df[~df["run_id"].astype(str).isin(excludes)]
    return df


def _save_plot(fig, folder: str, filename: str):
    """Guarda figura en disco, creando directorio si no existe."""
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[carbonsim PLOT SAVED] {out_path}")


def _prepare_runs_info(df: pd.DataFrame, label_by: str, value_col: str) -> List[Tuple[str, pd.Series, pd.Series, float]]:
    """
    Prepara información de cada run para graficar.
    Retorna lista de tuples: (label_value, x, y, final_value), ordenadas por final_value descendente.
    """
    runs_info = []
    for rid, df_run in df.groupby("run_id"):
        df_run_sorted = df_run.sort_values("timestamp")
        x = (df_run_sorted["timestamp"] - df_run_sorted["timestamp"].iloc[0]).dt.total_seconds()
        y = pd.to_numeric(df_run_sorted[value_col], errors="coerce")
        label_value = str(df_run_sorted[label_by].iloc[0]) if label_by in df_run_sorted.columns else rid
        final_value = y.iloc[-1]
        runs_info.append((label_value, x, y, final_value))
    runs_info.sort(key=lambda t: t[3], reverse=True)
    return runs_info


def _plot_time_series(runs_info, title: str, xlabel: str, ylabel: str,
                      output_folder: str, filename: str, mark_endpoints: bool = True):
    """Función genérica para graficar curvas de tiempo de múltiples runs."""
    plt.figure(figsize=(11, 6))
    for label_value, x, y, final_value in runs_info:
        plt.plot(x, y, label=label_value)
        if mark_endpoints:
            plt.scatter(x.iloc[-1], y.iloc[-1], color=plt.gca().lines[-1].get_color(), s=30, zorder=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    _save_plot(plt.gcf(), output_folder, filename)