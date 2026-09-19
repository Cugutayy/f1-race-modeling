import numpy as np
import pytest

from f1_research.demo import synthetic_history
from f1_research.features import build_features
from f1_research.group_rankers import NativeGroupRanker, RankingSpec, group_arrays


def test_group_arrays_preserve_whole_events_and_higher_relevance_for_better_finish():
    frame = build_features(synthetic_history(events=3, drivers=5))
    ordered, target, groups, group_ids = group_arrays(frame)

    assert groups == [5, 5, 5]
    assert len(ordered) == len(target) == len(group_ids) == 15
    assert np.all(np.diff(group_ids) >= 0)
    first = target[:5]
    assert first.tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]


def test_group_arrays_reject_incomplete_classification():
    frame = build_features(synthetic_history(events=2, drivers=5))
    broken = frame[~((frame.event_id == frame.event_id.iloc[0]) & (frame.finish_position == 5))].copy()
    with pytest.raises(ValueError, match="consecutive"):
        group_arrays(broken)


def test_unknown_native_ranker_fails_before_optional_dependency_use():
    frame = build_features(synthetic_history(events=2, drivers=5))
    model = NativeGroupRanker(RankingSpec("not-a-ranker", {}))
    with pytest.raises(ValueError, match="Unknown native ranking model"):
        model.fit(frame)
