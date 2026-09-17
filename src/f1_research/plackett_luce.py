"""Regularized Plackett-Luce maximum-likelihood ranker for complete F1 orders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, FEATURES, NUMERIC


class PlackettLuceRanker:
    """Linear utility model trained on whole finishing permutations.

    ``predict`` returns a lower-is-better score to stay compatible with the existing
    probability/distribution functions. Internally, higher utility is better.
    """

    def __init__(self, l2: float = 5.0, max_iter: int = 500, tol: float = 1e-7):
        if l2 < 0 or max_iter < 1 or tol <= 0:
            raise ValueError("Invalid Plackett-Luce optimizer settings")
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.preprocess = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("category", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
        ])
        self.coef_: np.ndarray | None = None
        self.optimization_: dict | None = None

    @staticmethod
    def _event_indices(frame: pd.DataFrame) -> list[np.ndarray]:
        events = []
        for _, event in frame.groupby("event_id", sort=False):
            if event.finish_position.isna().any() or event.finish_position.duplicated().any():
                raise ValueError("PL training requires one complete unique order per event")
            ordered = event.sort_values("finish_position", kind="stable")
            events.append(ordered.index.to_numpy())
        if not events:
            raise ValueError("No events supplied to Plackett-Luce ranker")
        return events

    def fit(self, frame: pd.DataFrame) -> "PlackettLuceRanker":
        required = set(FEATURES) | {"event_id", "finish_position"}
        if required - set(frame):
            raise ValueError(f"Missing PL columns: {sorted(required - set(frame))}")
        local = frame.reset_index(drop=True).copy()
        x = np.asarray(self.preprocess.fit_transform(local[FEATURES]), dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("PL preprocessing produced non-finite features")
        events = self._event_indices(local)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            utilities = x @ beta
            nll = 0.5 * self.l2 * float(beta @ beta)
            grad = self.l2 * beta.copy()
            for idx in events:
                # idx is already in finishing order after _event_indices reset semantics.
                order = local.loc[idx].sort_values("finish_position", kind="stable").index.to_numpy()
                u = utilities[order]
                xe = x[order]
                # Last remaining entrant has probability one and contributes zero.
                for k in range(len(order) - 1):
                    remaining = u[k:]
                    lse = logsumexp(remaining)
                    nll -= u[k] - lse
                    weights = np.exp(remaining - lse)
                    grad -= xe[k] - weights @ xe[k:]
            return float(nll), grad

        initial = np.zeros(x.shape[1], dtype=float)
        result = minimize(lambda beta: objective(beta), initial, jac=True,
                          method="L-BFGS-B", options={"maxiter": self.max_iter, "ftol": self.tol})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("Plackett-Luce optimization produced non-finite parameters")
        self.coef_ = np.asarray(result.x, dtype=float)
        self.optimization_ = {
            "success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "negative_log_likelihood": float(result.fun),
            "features": int(len(result.x)), "events": len(events), "l2": self.l2,
        }
        return self

    def utility(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("PlackettLuceRanker is not fitted")
        x = np.asarray(self.preprocess.transform(frame[FEATURES]), dtype=float)
        result = x @ self.coef_
        if not np.isfinite(result).all():
            raise ValueError("PL prediction produced non-finite utility")
        return result

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return -self.utility(frame)

    def feature_names(self) -> list[str]:
        if self.coef_ is None:
            raise RuntimeError("PlackettLuceRanker is not fitted")
        return self.preprocess.get_feature_names_out().tolist()

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_names(), "coefficient": self.coef_}).sort_values(
            "coefficient", key=lambda series: series.abs(), ascending=False)
