from dataclasses import dataclass

@dataclass
class PageHinkley:
    delta: float = 0.005   # sensibilidad a cambios pequeños
    lamb: float = 0.05     # umbral de alarma
    min_instances: int = 5

    mean_: float = 0.0
    cum_sum_: float = 0.0
    min_cum_sum_: float = 0.0
    n_: int = 0
    drift_: bool = False

    def update(self, x: float) -> bool:
        """
        x: nuevo valor (p.ej. residuo absoluto o relativo).
        Retorna True si detecta drift.
        """
        self.n_ += 1
        if self.n_ == 1:
            self.mean_ = x
            self.cum_sum_ = 0.0
            self.min_cum_sum_ = 0.0
            self.drift_ = False
            return False

        self.mean_ = self.mean_ + (x - self.mean_) / self.n_
        self.cum_sum_ += x - self.mean_ - self.delta
        self.min_cum_sum_ = min(self.min_cum_sum_, self.cum_sum_)

        self.drift_ = (self.n_ >= self.min_instances) and ((self.cum_sum_ - self.min_cum_sum_) > self.lamb)
        return self.drift_

    def reset(self):
        self.mean_ = 0.0
        self.cum_sum_ = 0.0
        self.min_cum_sum_ = 0.0
        self.n_ = 0
        self.drift_ = False
