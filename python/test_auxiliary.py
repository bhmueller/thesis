"""Test functions from auxiliary.py."""
import numpy as np
from numpy.testing import assert_allclose
from auxiliary import compute_confidence_intervals


def test_compute_confidence_intervals():
    # Sample data and get var and mean.
    n_vals = 10 ** 5
    rng = np.random.default_rng()
    vals = rng.standard_normal(n_vals)
    param_estimate = np.mean(vals)
    std_dev = np.std(vals, ddof=1)
    # std_error = std_dev / np.sqrt(n_vals)
    critical_value = 1.96
    confidence_intervals = compute_confidence_intervals(
        param_estimate, std_dev, critical_value
    )

    assert_allclose(confidence_intervals["lower_bound"], -1.96, atol=1e-02)
    assert_allclose(confidence_intervals["upper_bound"], 1.96, atol=1e-02)
