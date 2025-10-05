# Cal_risk_functions_iso_const.py
import numpy as np

# ---------- Problem primitives ----------
def P_func(x, xtilde, Sigma, sigma2):
    return 0.5 * (x - xtilde).T @ Sigma @ (x - xtilde) + 0.5 * sigma2

def _assert_isotropic(Sigma, atol=1e-12):
    diag = np.diag(Sigma)
    if not np.allclose(Sigma, np.diag(diag), atol=atol):
        raise ValueError("This fast path assumes Σ is diagonal.")
    if not np.allclose(diag, diag[0], atol=atol):
        raise ValueError("This fast path assumes Σ = cc · I.")
    return float(diag[0])

# ---------- Closed-form GF (mean) under Σ=cc·I and constant γ ----------
def gf_mean_iso(ts, x0, xtilde, cc, delta_reg, gamma, d):
    lam = (cc + delta_reg) * gamma * d
    coef = np.exp(-lam * ts)[:, None]                     # (N,1)
    drift = (cc / (cc + delta_reg)) * (1.0 - np.exp(-lam * ts))[:, None]
    return coef * x0[None, :] + drift * xtilde[None, :]

def evaluate_P_over_gf_iso(ts, x0, xtilde, Sigma, sigma2, gamma, delta_reg, d):
    cc = _assert_isotropic(Sigma)
    m_t = gf_mean_iso(ts, x0, xtilde, cc, delta_reg, gamma, d)   # (N,d)
    P_vals = np.array([P_func(m, xtilde, Sigma, sigma2) for m in m_t], dtype=float)
    return ts, P_vals

# ---------- Fast Volterra solver (exponential kernel, O(N)) ----------
def solve_volterra_P_iso(ts, P_vals_gf, Sigma, gamma, delta_reg, sigma_sgd, d):

    cc = _assert_isotropic(Sigma)
    lam = (cc + delta_reg) * gamma * d

    cK = (gamma**2) * (d**2) * (cc**2)                 
    cH = (sigma_sgd**2) * (gamma**2) * (d**2) * cc / 2 

    N = len(ts)
    P_vals = np.zeros(N, dtype=float)
    P_vals[0] = P_vals_gf[0]

    S = 0.0
    J = 0.0

    for i in range(1, N):
        dt = ts[i] - ts[i-1]
        rho = np.exp(-2.0 * lam * dt)

        S = rho * (S + P_vals[i-1] * dt)
        J = rho * (J + dt)

        P_vals[i] = P_vals_gf[i] + cK * S + cH * J

    return ts, P_vals
