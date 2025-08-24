from enum import Enum
import numpy as np

class LogLevel(Enum):
    NONE = 0          # No imprime nada
    BEST = 1          # Solo la mejor estimación
    FULL = 2          # Toda la información como tu print actual


def print_emission_log(log_level, last_elapsed, y_real, proj_eval, ci_eval, sim_next, best_name, best_metric, interval_sec, metric_name):
    """
    Imprime los logs de emisiones según log_level.
    - FULL: imprime toda la info
    - BEST: imprime solo la mejor estimación para el siguiente intervalo
    - NONE: no imprime nada
    """

    if log_level == LogLevel.NONE:
        return

    def fmt_ci(ci):
        return "NA" if (ci is None or ci[0] is None) else f"[{ci[0]:.6g},{ci[1]:.6g}]"

    if log_level == LogLevel.FULL:
        ci_eval_linear = ci_eval.get(1, (None, None))
        ci_eval_poly2  = ci_eval.get(2, (None, None))
        ci_eval_poly3  = ci_eval.get(3, (None, None))

        print(
            f"[carbonsim t={last_elapsed:.1f}s] Real={y_real:.6g} | "
            f"Eval@t Lin={proj_eval.get(1, y_real):.6g} {fmt_ci(ci_eval_linear)}; "
            f"P2={proj_eval.get(2, y_real):.6g} {fmt_ci(ci_eval_poly2)}; "
            f"P3={proj_eval.get(3, y_real):.6g} {fmt_ci(ci_eval_poly3)} | "
            f"Next(at {interval_sec + last_elapsed:.1f}s) Lin={sim_next.get(1, y_real):.6g}, "
            f"P2={sim_next.get(2, y_real):.6g}, P3={sim_next.get(3, y_real):.6g} | "
            f"Best={best_name} ({metric_name}={best_metric:.6g})"
        )

    elif log_level == LogLevel.BEST:
        # Solo la mejor estimación numérica para el siguiente intervalo
        best_next_value = sim_next.get(best_name, None) if isinstance(best_name, int) else sim_next.get(1, None)
        print(
            f"[carbonsim | t={last_elapsed:.1f}s] "
            f"Real Emissions = {y_real:.6g} kg | "
            f"Next Interval (at {interval_sec + last_elapsed:.1f}s) Estimated = {best_next_value:.6g} kg"
        )


def print_final_horizon(preds_h: dict[int, float],
                        ci_h: dict[int, tuple[float, float]],
                        horizon_sec: float,
                        best_deg: int,
                        best_final: str,
                        log_level: LogLevel):
    """
    Imprime la estimación final al horizonte según el log_level.
    """
    if log_level == LogLevel.NONE:
        return
    def ci_str(c):
        return "NA" if (c is None or c[0] is None) else f"[{c[0]:.6g},{c[1]:.6g}]"

    if log_level == LogLevel.FULL:
        print(
            f"[carbonsim FINAL] Horizon={horizon_sec}s -> "
            f"Lin={preds_h.get(1, np.nan):.6g} {ci_str(ci_h.get(1,(None,None)))} | "
            f"P2={preds_h.get(2, np.nan):.6g} {ci_str(ci_h.get(2,(None,None)))} | "
            f"P3={preds_h.get(3, np.nan):.6g} {ci_str(ci_h.get(3,(None,None)))} | "
            f"Best={best_final} => {preds_h.get(best_deg, np.nan):.6g}"
        )
    elif log_level == LogLevel.BEST:
        print(
            f"[CarbonSim | FINAL] Horizon = {horizon_sec}s | "
            f"Estimated Emissions for Run = {preds_h.get(best_deg, np.nan):.6g} kg CO2eq"
        )


def print_error(message: str, log_level: LogLevel):
    """
    Imprime un mensaje de error según el log_level.
    """
    if log_level == LogLevel.NONE:
        return
    print(f"[carbonsim ERROR] {message}")


def print_no_data(log_level: LogLevel):
    """
    Imprime un mensaje de no hay datos según el log_level.
    """
    if log_level == LogLevel.NONE:
        return
    print("[carbonsim WARNING] No hay datos suficientes para realizar la estimación.")