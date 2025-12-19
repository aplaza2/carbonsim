# CarbonSim — Simulación y proyección de emisiones de CodeCarbon

**CarbonSim** utiliza la variación personalisada de CodeCaron para monitorear las **emisiones acumuladas** registradas por CodeCarbon y realizar proyecciones. En particular:

- Evalúa, en cada intervalo, la **capacidad predictiva** de tres modelos (lineal, polinómico grado 2 y 3) de forma **out-of-sample**.  
- **Simula** la acumulación del **próximo intervalo**.  
- Al finalizar, **elige el mejor modelo** según una métrica configurable (MAE/RMSE/MAPE) y **proyecta** al **horizonte** deseado.  
- Incluye **intervalos de confianza**, **regularización (Ridge)**, **ventana móvil**, y **detección de drift** (Page–Hinkley).  

> **Entrada**: `emissions_realtime.csv` (CodeCarbon; **emisiones acumuladas**)  
> **Salidas**:  
> - `emissions_monitor.csv`: evaluación por intervalo + simulación del próximo  
> - `projections.csv` (o el archivo que configures): proyección final al horizonte  

---

## Instalación

```bash
pip install git+https://github.com/aplaza2/custom_codecarbon
pip install git+https://github.com/aplaza2/carbonsim.git
```

## Uso Básico

```python
from carbonsim import CarbonSimulator, carbon_simulation

# ---------------------------
# 1. Usando kwargs directamente
# ---------------------------
sim = CarbonSimulator(
    carbon_csv="custom.csv",
    horizon_sec=120,
    interval_sec=10,
    log_level="NEXT"
)
sim.start()
# ... tu código aquí ...
sim.stop()

# ---------------------------
# 2. Usando decorador
# ---------------------------
@carbon_simulation(horizon_sec=120, interval_sec=10, log_level="FULL")
def run_experiment():
    # Código de tu experimento aquí
    pass

run_experiment()

```

## Configuración vía archivo `.carbonsim.config`

CarbonSim permite cargar parámetros desde un archivo de configuración. Ejemplo:

```ini
[carbonsim]
project_name = Tests
experiment_id = Exp-1
interval_sec = 60
horizon_sec = 3600
metric = mae
degree_max = 3
regularization = ridge
alpha = 0.01
window_size = 100
drift_detector = ph
ph_delta = 0.005
ph_lambda = 0.05
ci_alpha = 0.05
log_level = FULL
automatic_projections = False
carbon_csv = emissions_realtime.csv
monitor_csv = emissions_monitor.csv
projections_file = emissions_projections.csv
codecarbon_log_level = info

```

## Parámetros principales de `CarbonSimConfig`

| Parámetro              | Default                    | Descripción                                                                   |
| ---------------------- | -------------------------- | ----------------------------------------------------------------------------- |
| `monitor_csv`          | `"emissions_monitor.csv"`  | Archivo de evaluación por intervalo                                           |
| `projections_file`     | `"emissions_projections.csv"`        | Archivo de proyección final                                                   |
| `interval_sec`         | `60`                       | Intervalo de medición en segundos                                             |
| `horizon_sec`          | `3600`                     | Horizonte de proyección en segundos                                           |
| `metric`               | `"mae"`                    | Métrica de evaluación: `mae`, `rmse`, `mape`                                  |
| `degree_max`           | `3`                        | Grado máximo del modelo polinómico (1..3)                                     |
| `regularization`       | `"none"`                   | Regularización: `none` o `ridge`                                              |
| `alpha`                | `0.0`                      | Lambda de Ridge                                                               |
| `window_size`          | `0`                        | Ventana móvil: 0=usar todo, >0 usar ventana de ese tamaño                     |
| `drift_detector`       | `"ph"`                     | Detector de drift: `none` o `ph` (Page–Hinkley)                               |
| `ph_delta`             | `0.005`                    | Delta para Page–Hinkley                                                       |
| `ph_lambda`            | `0.05`                     | Lambda para Page–Hinkley                                                      |
| `ci_alpha`             | `0.05`                     | Nivel de confianza para intervalos                                            |
| `log_level`            | `FULL`                     | Nivel de logs: `NONE`, `NEXT`, `FULL`, `ERROR`                                         |
| `automatic_projections`| `False`                    | Generación de proyecciones automática |
| `project_name`         | `"Proyections"`            | Nombre de proyecto para CodeCarbon                                            |
| `experiment_id`        | `"DefaultExperiment"`      | ID de experimento para CodeCarbon                                             |
| `carbon_csv`           | `"emissions_realtime.csv"` | CSV de emisiones acumuladas de CodeCarbon                                     |
| `measure_power_secs`   | `5`                        | Frecuencia de medición de energía                                             |
| `csv_write_interval`   | `1`                        | Frecuencia de escritura de CSV                                                |
| `tracking_mode`        | `"process"`                | Modo de tracking de CodeCarbon                                                |
| `save_to_file`         | `True`                     | Guardar resultados en archivo CSV                                             |
| `on_csv_write`         | `"append"`                 | Comportamiento al escribir en CSV                                             |
| `codecarbon_log_level` | `"error"`               | Nivel de logs de CodeCarbon (`critical`, `error`, `warning`, `info`, `debug`) |

## Niveles de Log (`log_level`)

- `NONE`: no muestra información en consola
- `NEXT`: solo muestra la estimación del siguiente intervalo
- `FULL`: muestra toda la información, incluyendo métricas de evaluación y estimaciones por modelo
- `ERROR`: solo muestra los errores

### Ejemplos de salida de logs:
```text
[carbonsim t=60.0s] Emissions=0.123456 | Estimation for next check (at 120s)=0.125678
[carbonsim t=120.0s] Emissions=0.246789 | Estimation for next check (at 180s)=0.250123
[carbonsim FINAL] Estimation for run program 3600s -> 5.678901 Kg CO2eq
```

## Archivos de salida

- `emissions_monitor.csv`: evaluación de cada intervalo con errores y proyecciones
- `projections.csv`: proyección final hacia el horizonte
- `emissions_realtime.csv`: CSV de CodeCarbon con las emisiones acumuladas

## Generación manual de proyecciones
```
generate_carbon_projections(**kwargs)
```

## Otros módulos

### carbonsim.plots
Este módulo permite graficar emisiones y proyecciones acumuladas de distintos experimentos.

#### `plot_emissions(**kwargs)`

Grafica emisiones reales acumuladas a lo largo del tiempo, agrupando por run_id o experiment_id.

Parámetros:

- `emissions_file` (str): CSV de emisiones (default: emissions_realtime.csv).
- `output_dir` (str): Carpeta de salida.
- `project_name` (str): Filtrar por proyecto.
- `include_runs` / `include_file`: Filtros de inclusión.
- `exclude_runs` / `exclude_file`: Filtros de exclusión.
- `label_by` (str): Agrupamiento para etiquetas (experiment_id por defecto).
- `mark_endpoints` (bool): Marca los valores finales de cada curva.
- `xlim` (int): Límite de tiempo a graficar.
- `xscale` (str): Tipo de escala para el eje x.
- `yscale` (str): Tipo de escala para el eje y.

Salida:
Gráficas `.png` en `plots/<project_name>/emissions.png`.

#### `plot_emissions_rates(**kwargs)`

Grafica la tasa instantánea de emisiones (emissions_rate) a lo largo del tiempo.

Parámetros:

- `emissions_file` (str): CSV de emisiones (default: emissions_realtime.csv).
- `output_dir` (str): Carpeta de salida.
- `include_runs` / `include_file`: Filtros de inclusión.
- `exclude_runs` / `exclude_file`: Filtros de exclusión.
- `label_by` (str): Agrupamiento (experiment_id por defecto).
- `mark_endpoints` (bool): Marca los valores finales.
- `xlim` (int): Límite de tiempo a graficar.
- `xscale` (str): Tipo de escala para el eje x.
- `yscale` (str): Tipo de escala para el eje y.

Salida:
Gráficas `.png` en `plots/<project_name>/emissions_rates.png`.

#### `plot_projections(**kwargs)`

Grafica **proyecciones acumuladas de emisiones** individualmente.

Parámetros:

- `monitor_file` (str): CSV de monitoreo (default: emissions_monitor.csv).
- `output_dir` (str): Carpeta donde guardar las gráficas (default: plots).
- `include_runs` / `include_file`: Filtros de inclusión.
- `exclude_runs` / `exclude_file`: Filtros de exclusión.
- `show_confidence` (bool): Si True, dibuja bandas de confianza (default: True).
- `label_by` (str): Etiqueta usada en el título de la gráfica (experiment_id por defecto).

Salida:
Gráficas `.png` en `plots/<project_name>/projections/`.

#### `plot_multiple_projections(**kwargs)`

Grafica **proyecciones acumuladas de emisiones** agrupados por conjunto de experimento.

Parámetros:

- `monitor_file` (str): CSV de monitoreo (default: emissions_monitor.csv).
- `projections_file` (str): CSV de proyecciones (default: emissions_projections.csv).
- `output_dir` (str): Carpeta donde guardar las gráficas (default: plots).
- `include_runs` / `include_file`: Filtros de inclusión.
- `exclude_runs` / `exclude_file`: Filtros de exclusión.
- `show_confidence` (bool): Si True, dibuja bandas de confianza (default: True).
- `label_by` (str): Etiqueta usada en el título de la gráfica (experiment_id por defecto).

Salida:
Gráficas `.png` en `plots/<project_name>/projections/`.

### carbonsim.writer

Este módulo permite editar, renombrar o eliminar datos en los CSV definidos en la configuración (CarbonSimConfig).

#### `rename_data(column, old_value, new_value, **kwargs)`

Reemplaza valores en una columna de todos los archivos de resultados.

Parámetros:

- `column` (str): Nombre de la columna a modificar.
- `old_value` (str): Valor a reemplazar.
- `new_value` (str): Nuevo valor.
- `**kwargs`: Sobrescribe parámetros de configuración (config_path=..., etc).

#### `rename_data_by_run_id(run_id, column, new_value, **kwargs)`

Modifica project_name o experiment_id únicamente en filas con un run_id específico.

Parámetros:

- `run_id` (str): Identificador de la corrida.
- `column` (str): Solo puede ser "project_name" o "experiment_id".
- `new_value` (str): Nuevo valor.
- `**kwargs`: Sobrescribe configuración.

#### `delete_run(run_id, **kwargs)`

Elimina todas las filas asociadas a un run_id de los archivos de resultados.

Parámetros:

- `run_id` (str): Identificador de la corrida a eliminar.
- `**kwargs`: Sobrescribe configuración.