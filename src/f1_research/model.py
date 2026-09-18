"""Train-only preprocessing and past-only temperature-calibrated race distributions.

The score learner is normalized-rank regression, not a maximum-likelihood PL fit.
Its scores parameterize a Plackett-Luce distribution evaluated on held-out races.
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, FEATURES, NUMERIC

MODELS = ("uniform", "qualifying_order", "recent_form", "ridge_rank", "gradient_boosting")
METRICS = ("position_mae", "winner_log_loss", "winner_brier", "winner_accuracy", "podium_recall", "spearman_rank", "kendall_rank", "ndcg")


def estimator(kind="gradient_boosting"):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    learner = Ridge(alpha=10.0) if kind == "ridge_rank" else GradientBoostingRegressor(
        n_estimators=80, max_depth=2, min_samples_leaf=12,
        learning_rate=0.04, loss="huber", random_state=42,
    )
    return Pipeline([("preprocess", preprocessing), ("model", learner)])


def target(frame):
    size = frame.groupby("event_id")["driver"].transform("size")
    return (frame["finish_position"] - 1) / (size - 1).clip(lower=1)


def probability(scores, temperature):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("A race needs at least two finite scores")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = -(scores - scores.min()) / temperature
    weights = np.exp(np.maximum(logits, -700))
    return weights / weights.sum()


def distribution(scores, temperature, samples=4096, seed=42):
    """Exact winner and expected rank; deterministic Monte Carlo rank marginals.

    Gumbel top-k samples the PL permutation. Every sample contains each entrant once.
    Monte Carlo podium probabilities sum to min(3,N); no independent binary heads.
    """
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    p = probability(scores, temperature)
    rng = np.random.default_rng(seed)
    utility = np.log(p)
    orders = np.argsort(-(utility + rng.gumbel(size=(samples, len(p)))), axis=1)
    ranks = np.empty_like(orders)
    np.put_along_axis(ranks, orders, np.arange(1, len(p) + 1)[None, :], axis=1)
    # P(j ahead of i) = p_j / (p_i + p_j), including self contribution .5 removed.
    expected = 0.5 + (p[None, :] / (p[:, None] + p[None, :])).sum(axis=1)
    return {
        "win_probability": p,
        "podium_probability": (ranks <= min(3, len(p))).mean(axis=0),
        "top10_probability": (ranks <= min(10, len(p))).mean(axis=0),
        "expected_position": expected,
        "position_p10": np.quantile(ranks, 0.1, axis=0, method="inverted_cdf"),
        "position_p90": np.quantile(ranks, 0.9, axis=0, method="inverted_cdf"),
    }


def scores_for(name, event, learner=None):
    if name == "uniform":
        return np.zeros(len(event))
    if name == "qualifying_order":
        return event["quali_position"].fillna(len(event) + 1).to_numpy() / len(event)
    if name == "recent_form":
        return event["driver_form"].to_numpy()
    return learner.predict(event[FEATURES])


def score_event(event, scores, temperature):
    p = probability(scores, temperature)
    truth = event["finish_position"].to_numpy()
    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    winner = truth == 1
    # Higher relevance means a better actual finish. NDCG therefore rewards getting the
    # whole ordering right rather than only the winner/podium.
    relevance = (len(event) + 1 - truth).astype(float)
    ndcg = float(ndcg_score(relevance.reshape(1, -1), (-scores).reshape(1, -1)))
    spearman = float(spearmanr(ranks, truth).statistic)
    kendall = float(kendalltau(ranks, truth).statistic)
    return {
        "position_mae": float(np.mean(np.abs(ranks - truth))),
        "winner_log_loss": float(-np.log(max(p[winner][0], 1e-15))),
        "winner_brier": float(np.sum((p - winner) ** 2)),
        "winner_accuracy": float(truth[np.argmin(scores)] == 1),
        "podium_recall": float(np.sum((ranks <= 3) & (truth <= 3)) / min(3, len(event))),
        "spearman_rank": spearman,
        "kendall_rank": kendall,
        "ndcg": ndcg,
    }, ranks


def fit_models(history, calibration_events=4):
    events = history[["event_id", "date"]].drop_duplicates().sort_values(["date", "event_id"])
    if calibration_events < 1 or len(events) < calibration_events + 3:
        raise ValueError("Need calibration_events >= 1 and at least three earlier fit events")
    cutoff = events.iloc[-calibration_events]["date"]
    train, calibration = history[history["date"] < cutoff], history[history["date"] >= cutoff]
    if train["event_id"].nunique() < 3:
        raise ValueError("Insufficient strictly earlier fit events")
    fitted = {}
    for name in MODELS:
        learner = estimator(name).fit(train[FEATURES], target(train)) if name in (
            "ridge_rank", "gradient_boosting") else None
        pairs = [(event, scores_for(name, event, learner))
                 for _, event in calibration.groupby("event_id", sort=True)]
        candidates = [1.0] if name == "uniform" else np.geomspace(0.02, 2.0, 30)
        losses = [np.mean([score_event(e, s, t)[0]["winner_log_loss"] for e, s in pairs])
                  for t in candidates]
        fitted[name] = (learner, float(candidates[int(np.argmin(losses))]))
    # Do not refit: doing so would change the score scale used for calibration.
    return fitted, {
        "train_events": int(train["event_id"].nunique()),
        "calibration_events": int(calibration["event_id"].nunique()),
        "train_end": str(train["date"].max()),
        "calibration_end": str(calibration["date"].max()),
    }
