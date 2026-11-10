"""test_framework.py - simple checks for framework.py (Appendix B).

Run:  python -m pytest test_framework.py
"""

import numpy as np
import framework as fw


def test_psd_check():
    """Equal correlations are valid; the 3-asset example in 2.5.1 is not."""
    assert fw.is_psd(fw.correlation_matrix(30, 0.6))
    bad = np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
    assert not fw.is_psd(bad)


def test_repair_gives_valid_correlation_matrix():
    """After repair the matrix is PSD, symmetric and has 1 on the diagonal."""
    C = np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
    R = fw.fix_correlation(C)
    assert fw.is_psd(R)
    assert np.allclose(R, R.T)
    assert np.allclose(np.diag(R), 1)


def test_incremental_update_matches_full_formula():
    """Equation (8) gives the same variance as computing x'Sigma x again."""
    inst = fw.make_instance(10, 2, 0.5, seed=1)
    cov = inst["cov"]
    x = np.random.default_rng(0).uniform(0, 0.1, 10)
    i, delta = 4, 0.03
    x_new = x.copy()
    x_new[i] += delta
    fast = fw.variance_after_add(x @ cov @ x, cov @ x, cov, i, delta)
    assert np.isclose(fast, x_new @ cov @ x_new)


def test_risk_floor_equation_7():
    """Equal weights with equal correlation give sigma^2 (rho + (1-rho)/N)."""
    N, rho, s = 30, 0.6, 0.2
    cov = s**2 * fw.correlation_matrix(N, rho)
    x = np.full(N, 1 / N)
    assert np.isclose(x @ cov @ x, s**2 * (rho + (1 - rho) / N))


def test_greedy_respects_limits():
    """Budget, risk limit and pool limit are never broken."""
    for rho in [0.0, 0.9]:
        for aware in [False, True]:
            inst = fw.make_instance(30, 5, rho, seed=2)
            lots = fw.greedy(inst, aware)
            spent = inst["price"] @ lots
            assert np.all(spent <= inst["budget"] + 1e-9)
            for k in range(5):
                assert fw.account_risk(inst, lots, k) <= inst["risk_limit"][k] + 1e-9
            assert fw.check(inst, lots)["pool_ok"]


def test_empty_account_is_not_feasible():
    """An account with nothing invested must be flagged as not feasible."""
    inst = fw.make_instance(10, 2, 0.3, seed=0)
    lots = np.zeros((10, 2), dtype=int)
    assert not any(fw.check(inst, lots)["feasible"])


def test_same_seed_gives_same_answer():
    """The same seed gives the same problem and the same allocation."""
    a = fw.greedy(fw.make_instance(20, 4, 0.5, seed=7))
    b = fw.greedy(fw.make_instance(20, 4, 0.5, seed=7))
    assert np.array_equal(a, b)
