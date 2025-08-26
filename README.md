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
from carbonsim import CarbonSimulator, CarbonSimConfig, carbon_simulation

# ---------------------------
# 1. Usando configuración explícita
# ---------------------------
cfg = CarbonSimConfig(
    carbon_csv="custom.csv",
    monitor_csv="monitor.csv",
    horizon_sec=120,
    interval_sec=10,
    metric="rmse",
    log_level="FULL"
)
sim = CarbonSimulator(cfg)
sim.start()
# ... tu código aquí ...
sim.stop()

# ---------------------------
# 2. Usando kwargs directamente
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
# 3. Usando decorador
# ---------------------------
@carbon_simulation(horizon_sec=120, interval_sec=10, log_level="FULL")
def run_experiment():
    # Código de tu experimento aquí
    for i in range(5):
        print("Ejecutando iteración", i)

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
carbon_csv = emissions_realtime.csv
monitor_csv = emissions_monitor.csv
projections_file = projections.csv
codecarbon_log_level = info

```

## Parámetros principales de `CarbonSimConfig`

| Parámetro              | Default                    | Descripción                                                                   |
| ---------------------- | -------------------------- | ----------------------------------------------------------------------------- |
| `monitor_csv`          | `"emissions_monitor.csv"`  | Archivo de evaluación por intervalo                                           |
| `projections_file`     | `"projections.csv"`        | Archivo de proyección final                                                   |
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
