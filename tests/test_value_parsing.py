import numpy as np
import pytest

from f1_research.value_parsing import strict_optional_bool


def test_strict_optional_bool_accepts_explicit_encodings():
    assert strict_optional_bool(True, field="x") is True
    assert strict_optional_bool(False, field="x") is False
    assert strict_optional_bool(np.bool_(True), field="x") is True
    assert strict_optional_bool(1, field="x") is True
    assert strict_optional_bool(0.0, field="x") is False
    assert strict_optional_bool("TRUE", field="x") is True
    assert strict_optional_bool("false", field="x") is False
    assert strict_optional_bool(None, field="x") is None


def test_strict_optional_bool_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="Unsupported explicit boolean"):
        strict_optional_bool("maybe", field="x")
    with pytest.raises(ValueError, match="Unsupported explicit boolean"):
        strict_optional_bool(2, field="x")
