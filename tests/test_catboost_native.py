import sys
import types

import numpy as np
import pandas as pd

from f1_research.features import CATEGORICAL, FEATURES, NUMERIC
from f1_research.modern_models import CandidateSpec, build_estimator


class FakeCatBoostRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_frame = None
        self.predict_frame = None

    def fit(self, frame, target):
        self.fit_frame = frame.copy()
        self.target = np.asarray(target, dtype=float)
        return self

    def predict(self, frame):
        self.predict_frame = frame.copy()
        return np.linspace(0.1, 0.9, len(frame))


def _frame():
    rows = 4
    data = {
        "team": ["Ferrari", None, "McLaren", "Mercedes"],
        "circuit": ["Monza", "Monza", None, "Spa"],
        "regulation_era": ["2026_plus", "2026_plus", "2026_plus", None],
    }
    for index, name in enumerate(NUMERIC):
        values = np.arange(rows, dtype=float) + index
        values[1] = np.nan
        data[name] = values
    return pd.DataFrame(data, columns=FEATURES)


def test_catboost_path_preserves_named_native_categories_and_imputes_numeric(monkeypatch):
    module = types.ModuleType("catboost")
    module.CatBoostRegressor = FakeCatBoostRegressor
    monkeypatch.setitem(sys.modules, "catboost", module)

    train = _frame()
    model = build_estimator(CandidateSpec("catboost", {"depth": 4}), random_state=11)
    fitted = model.fit(train, np.asarray([0.0, 0.3, 0.6, 1.0]))

    assert fitted is model
    assert model.model_.kwargs["cat_features"] == CATEGORICAL
    assert model.model_.kwargs["depth"] == 4
    assert model.model_.kwargs["random_seed"] == 11
    assert model.model_.kwargs["allow_writing_files"] is False
    assert list(model.model_.fit_frame.columns) == FEATURES
    assert not model.model_.fit_frame[NUMERIC].isna().any().any()
    assert model.model_.fit_frame.loc[1, "team"] == "__MISSING__"
    assert all(model.model_.fit_frame[name].map(type).eq(str).all() for name in CATEGORICAL)

    future = _frame()
    future.loc[0, "team"] = "New 2026 Team"
    predictions = model.predict(future)
    assert np.isfinite(predictions).all()
    assert model.model_.predict_frame.loc[0, "team"] == "New 2026 Team"
    assert not model.model_.predict_frame[NUMERIC].isna().any().any()
