from dataclasses import dataclass
import numpy as np
from typing import Literal, Tuple, Optional


Metric = Literal["mae", "rmse", "mape"]


@dataclass
class PolyRegressor:
    degree: int = 1
    regularization: Literal["none", "ridge"] = "none"
    alpha: float = 0.0  # lambda de ridge
    fit_intercept: bool = True

    # almacenados tras fit
    beta_: Optional[np.ndarray] = None
    XtX_inv_: Optional[np.ndarray] = None
    sigma2_: Optional[float] = None
    x_mean_: Optional[float] = None
    x_scale_: Optional[float] = None
    n_: Optional[int] = None
    p_: Optional[int] = None
    last_obs_: Optional[float] = None  # último valor real (para monotonicidad)

    # parámetros internos de estabilidad numérica
    _cond_threshold: float = 1e10      # umbral para activar fallback
    _ridge_fallback_alpha: float = 1e-3

    def _poly_features(self, x: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(x)] if self.fit_intercept else []
        for d in range(1, self.degree + 1):
            cols.append(x ** d)
        return np.vstack(cols).T

    def _standardize(self, t: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None:
            self.x_mean_ = float(np.mean(t))
        if self.x_scale_ is None:
            s = float(np.std(t))
            self.x_scale_ = s if s > 0 else 1.0
        return (t - self.x_mean_) / self.x_scale_

    def fit(self, t_sec: np.ndarray, y: np.ndarray):
        """
        t_sec: tiempos en segundos (1D)
        y: emisiones acumuladas (1D)
        """
        t = self._standardize(np.asarray(t_sec, dtype=float))
        X = self._poly_features(t)
        y = np.asarray(y, dtype=float)
        self.n_, self.p_ = X.shape

        # Guardar último observado (monotonicidad)
        self.last_obs_ = float(y[-1])

        XtX = X.T @ X
        use_ridge = (self.regularization == "ridge" and self.alpha > 0.0)

        # Activar fallback si la condición es mala
        if not use_ridge:
            try:
                cond = np.linalg.cond(XtX)
            except Exception:
                cond = np.inf
            if not np.isfinite(cond) or cond > self._cond_threshold:
                use_ridge = True
                if self.alpha <= 0.0:
                    self.alpha = self._ridge_fallback_alpha

        if use_ridge:
            lamI = self.alpha * np.eye(X.shape[1])
            A = XtX + lamI
            try:
                beta = np.linalg.solve(A, X.T @ y)
                XtX_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                # Pseudoinversa como “último recurso”
                XtX_inv = np.linalg.pinv(A, rcond=1e-12)
                beta = XtX_inv @ X.T @ y
        else:
            try:
                XtX_inv = np.linalg.inv(XtX)
                beta = XtX_inv @ X.T @ y
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX, rcond=1e-12)
                beta = XtX_inv @ X.T @ y

        self.beta_ = beta
        # sigma^2 (residuos de entrenamiento)
        try:
            resid = y - X @ beta
            dof = max(1, self.n_ - self.p_)
            sigma2 = float(np.sum(resid ** 2) / dof)
            self.sigma2_ = sigma2 if np.isfinite(sigma2) else np.nan
        except Exception:
            self.sigma2_ = np.nan

        self.XtX_inv_ = XtX_inv
        return self

    def predict(
        self,
        t_sec: np.ndarray,
        ci_alpha: Optional[float] = None,
        add_pred_var: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Devuelve y_hat y opcionalmente intervalos (infer, upper).
        add_pred_var=True produce intervalo de predicción (media + var resid).
        """
        assert self.beta_ is not None, "Modelo no ajustado"

        t_sec = np.asarray(t_sec, dtype=float)
        t = (t_sec - self.x_mean_) / self.x_scale_
        X = self._poly_features(t)
        y_hat = X @ self.beta_

        # Monotonicidad: no bajar del último real
        if self.last_obs_ is not None:
            y_hat = np.maximum(y_hat, self.last_obs_)

        if ci_alpha is None or self.sigma2_ is None or self.XtX_inv_ is None:
            return y_hat, None, None

        # z-score aproximado (lookup común)
        def approx_z(alpha: float) -> float:
            table = {0.1: 1.6449, 0.05: 1.96, 0.01: 2.5758}
            key = round(alpha, 2)
            return float(table.get(key, 1.96))

        z = approx_z(ci_alpha)

        try:
            var_mean = np.einsum("bi,ij,bj->b", X, self.XtX_inv_, X) * self.sigma2_
            var_mean = np.maximum(var_mean, 0.0)
            if add_pred_var:
                var_pred = var_mean + (0.0 if not np.isfinite(self.sigma2_) else self.sigma2_)
                var_pred = np.maximum(var_pred, 0.0)
                se = np.sqrt(var_pred)
            else:
                se = np.sqrt(var_mean)

            if not np.all(np.isfinite(se)):
                return y_hat, None, None

            lower = y_hat - z * se
            upper = y_hat + z * se
            # Monotonicidad también en el límite inferior
            if self.last_obs_ is not None:
                lower = np.maximum(lower, self.last_obs_)

            # Si IC no son finitos, devolver None
            if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
                return y_hat, None, None

            return y_hat, lower, upper

        except Exception:
            # Si algo falla en IC, devolvemos sólo la predicción
            return y_hat, None, None
