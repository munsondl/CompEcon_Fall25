# tests/test_shocks.py

from src.shocks import tauchen
import numpy as np

def test_tauchen_basic():
    n = 5
    mu = 200.0       # thousand dollars per truck per year (your mean)
    rho = 0.9
    sigma = 10.0     # example volatility

    z, P = tauchen(n, mu, rho, sigma)

    # Basic shape tests
    assert len(z) == n
    assert P.shape == (n, n)

    # Rows of transition matrix must sum to 1
    for i in range(n):
        assert abs(P[i].sum() - 1.0) < 1e-8

def test_tauchen_monotonic_grid():
    z, _ = tauchen(7, 200.0, 0.9, 10.0)
    assert all(z[i] < z[i+1] for i in range(len(z)-1))
