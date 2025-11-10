import os
import configparser
from dataclasses import dataclass, fields
from .logging_utils import LogLevel
from typing import Any

@dataclass
class CarbonSimConfig:
    monitor_csv: str = "emissions_monitor.csv"
    projections_file: str = "emissions_projections.csv"
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
    automatic_projections: bool = False

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
            section = parser["carbonsim"] if "carbonsim" in parser else parser["DEFAULT"]
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
            f"interval_sec debe ser > measure_power_secs * csv_write_interval = {self.measure_power_secs*self.csv_write_interval}"
        
        lv = getattr(self, "log_level", None)
        if lv is None:
            self.log_level = LogLevel.NONE
        elif isinstance(lv, LogLevel):
            pass
        elif isinstance(lv, str):
            try:
                self.log_level = LogLevel[lv.strip().upper()]
            except KeyError:
                raise ValueError(
                    f"log_level inválido: {lv}. Opciones: {list(LogLevel.__members__.keys())}"
                )
        else:
            raise TypeError(f"log_level debe ser str o LogLevel, no {type(lv)}")


        return self
    
def _cast_type(old_value, new_value):
        """Helper: castea según el tipo del valor en cfg."""
        if isinstance(old_value, bool):
            return new_value.lower() in ("true", "1", "yes", "y")
        if isinstance(old_value, int):
            return int(new_value)
        if isinstance(old_value, float):
            return float(new_value)
        return new_value


def _find_config_file() -> str | None:
    candidates = [
        os.path.join(os.getcwd(), ".carbonsim.config"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".carbonsim.config"),
        os.path.expanduser("~/.carbonsim.config"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_config(**overrides: Any) -> CarbonSimConfig:
    # 1 Usar valores por defecto
    cfg = CarbonSimConfig()

    # 2 Detectar archivo de config
    config_file = overrides.pop("config_path", _find_config_file())
    if config_file and os.path.exists(config_file):
        parser = configparser.ConfigParser()
        parser.read(config_file, encoding="utf-8")
        section = parser["carbonsim"] if "carbonsim" in parser else parser["DEFAULT"]

        for field in fields(cfg):
            if field.name in section:
                old_value = getattr(cfg, field.name)
                new_value = section[field.name]
                setattr(cfg, field.name, _cast_type(old_value, new_value))

    # 3 Aplicar overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f"[carbonsim WARNING] '{k}' no es un parámetro válido de CarbonSimConfig")

    return cfg.validate()

