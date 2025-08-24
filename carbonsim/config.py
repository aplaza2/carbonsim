import os
import configparser
from dataclasses import dataclass
from .logging_utils import LogLevel

@dataclass
class CarbonSimConfig:
    monitor_csv: str = "emissions_monitor.csv"
    projections_file: str = "projections.csv"
    interval_sec: int = 60
    horizon_sec: int = 3600          # 1 hora
    metric: str = "mae"              # mae | rmse | mape
    degree_max: int = 3              # 1..3
    regularization: str = "none"     # none | ridge
    alpha: float = 0.0               # ridge lambda
    window_size: int = 0             # 0=usar todo, >0 usar ventana móvil
    drift_detector: str = "ph"       # none | ph
    ph_delta: float = 0.005
    ph_lambda: float = 0.05
    ci_alpha: float = 0.05
    log_level: LogLevel = LogLevel.FULL 

    # Parámetros CodeCarbon
    project_name: str = "Proyections"
    experiment_id: str = "DefaultExperiment"
    carbon_csv: str = "emissions_realtime.csv"
    measure_power_secs: int = 5
    csv_write_interval: int = 1
    tracking_mode: str = "process"
    save_to_file: bool = True
    on_csv_write: str = "append"
    codecarbon_log_level: str = "error"  # critical | error | warning | info | debug

    @staticmethod
    def from_file(path: str | None = None):
        cfg = CarbonSimConfig()
        if path and os.path.exists(path):
            parser = configparser.ConfigParser()
            parser.read(path, encoding="utf-8")
            section = parser["carbonsimulator"] if "carbonsimulator" in parser else parser["DEFAULT"]
            for field_name, fdef in cfg.__dataclass_fields__.items():
                if field_name in section:
                    value = section[field_name]
                    ftype = type(getattr(cfg, field_name))
                    if ftype is bool:
                        value = section.getboolean(field_name)
                    elif ftype is int:
                        value = section.getint(field_name)
                    elif ftype is float:
                        value = section.getfloat(field_name)
                    else:
                        value = str(value)
                    setattr(cfg, field_name, value)
        return cfg

    def validate(self):
        self.metric = self.metric.lower()
        assert self.metric in {"mae", "rmse", "mape"}
        self.degree_max = int(max(1, min(3, self.degree_max)))
        self.regularization = self.regularization.lower()
        assert self.regularization in {"none", "ridge"}
        assert self.interval_sec > self.measure_power_secs * self.csv_write_interval, \
            f"interval_sec debe ser > measure_power_secs*csv_write_interval = {self.measure_power_secs*self.csv_write_interval}"
        return self
