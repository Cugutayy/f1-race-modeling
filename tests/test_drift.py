import numpy as np

from f1_research.drift import drift_report, population_stability_index


def test_psi_is_near_zero_for_identical_samples():
    sample = np.linspace(0, 1, 100)
    assert population_stability_index(sample, sample) < 1e-12


def test_drift_report_flags_shift():
    reference = {"pace": list(np.linspace(0, 1, 100))}
    current = {"pace": list(np.linspace(2, 3, 100))}
    report = drift_report(reference, current)
    assert report["warning"] is True
