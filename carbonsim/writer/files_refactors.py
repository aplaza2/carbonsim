from ..config import load_config
import pandas as pd
from typing import Any


def _check_valid_column(column: str) -> None:
    """Valida que la column sea permitida"""
    if column not in ["project_name", "experiment_id", "run_id"]:
        raise ValueError(f"[carbonsim ERROR] Columna inválida: {column}")


def rename_data(column: str, old_value: str, new_value: str, **kwargs: Any) -> None:
    """
    Modifica valores en column dada de los archivos indicados en config.
    kwargs se pasan como overrides a load_config.
    """
    _check_valid_column(column)

    config = load_config(**kwargs)
    archivos = [config.carbon_csv, config.monitor_csv, config.projections_file]

    for archivo in archivos:
        df = pd.read_csv(archivo)
        df.loc[df[column] == old_value, column] = new_value
        df.to_csv(archivo, index=False)
        print(f"[carbonsim FILES] Modificado '{old_value}' → '{new_value}' en '{column}' de {archivo}")


def rename_data_by_run_id(run_id: str, column: str, new_value: str, **kwargs: Any) -> None:
    """
    Modifica project_name o experiment_id solo para filas con un run_id dado.
    kwargs se pasan como overrides a load_config.
    """
    if column not in ["project_name", "experiment_id"]:
        raise ValueError("[carbonsim ERROR]Solo se puede modificar 'project_name' o 'experiment_id'")

    config = load_config(**kwargs)
    archivos = [config.carbon_csv, config.monitor_csv, config.projections_file]

    for archivo in archivos:
        df = pd.read_csv(archivo)
        df.loc[df["run_id"] == run_id, column] = new_value
        df.to_csv(archivo, index=False)
        print(f"[carbonsim FILES] Modificado '{column}' → '{new_value}' para run_id '{run_id}' en {archivo}")

def delete_run(run_id: str, **kwargs: Any) -> None:
    """
    Elimina todas las filas con un run_id dado en los archivos indicados en config.
    kwargs se pasan como overrides a load_config.
    """
    config = load_config(**kwargs)
    archivos = [config.carbon_csv, config.monitor_csv, config.projections_file]

    for archivo in archivos:
        df = pd.read_csv(archivo)
        filas_iniciales = len(df)
        df = df[df["run_id"] != run_id]
        filas_finales = len(df)
        df.to_csv(archivo, index=False)
        print(f"[carbonsim FILES] Eliminadas {filas_iniciales - filas_finales} filas con run_id '{run_id}' en {archivo}")