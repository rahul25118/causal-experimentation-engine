import numpy as np
import pytest
from src.eda_diagnostics import run_srm_check
from src.variance_reduction import apply_univariate_cuped, compute_welch_ttest
from src.sequential_testing import compute_msprt_pvalue


def test_srm_detection():
    healthy = run_srm_check(500_100, 499_900)
    assert healthy["srm_detected"] is False

    broken = run_srm_check(520_000, 480_000)
    assert broken["srm_detected"] is True


def test_cuped_unbiasedness_and_variance_reduction():
    rng = np.random.default_rng(123)
    n = 50_000
    x_pre = rng.normal(50.0, 10.0, size=n)
    variant = rng.integers(0, 2, size=n)
    true_ate = 2.0
    y_post = 0.8 * x_pre + variant * true_ate + rng.normal(0.0, 5.0, size=n)

    y_cuped, theta = apply_univariate_cuped(y_post, x_pre)
    assert np.isclose(theta, 0.8, atol=0.03)
    assert np.var(y_cuped) < np.var(y_post) * 0.5

    res = compute_welch_ttest(y_cuped[variant == 0], y_cuped[variant == 1])
    assert np.isclose(res["ate"], true_ate, atol=0.1)


def test_msprt_always_valid_null_control():
    # Under H0 (delta = 0), p-value must remain >= 0.05
    _, p_null = compute_msprt_pvalue(delta_hat=0.0, var_delta=0.01, tau_sq=0.5)
    assert p_null == 1.0

    # Under strong signal, p-value must drop below 0.05
    _, p_sig = compute_msprt_pvalue(delta_hat=1.5, var_delta=0.001, tau_sq=0.5)
    assert p_sig < 0.05
