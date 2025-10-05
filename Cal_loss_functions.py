import numpy as np
from numpy.linalg import norm
from scipy.integrate import trapezoid

# --- User-provided setup & risk curve interpolator ---
from parameter_setup import *             # expects: Sigma = cc * I_d, delta_reg, gamma, x0, xtilde, sigma_sgd_list, jj, T, d, etc.
from Read_risk import interp_P            # callable: interp_P(t) -> P_t
sigma_sgd = sigma_sgd_list[jj]

# --------- Check isotropy & constants ---------
def _assert_isotropic_Sigma(Sigma, atol=1e-12):
    diag = np.diag(Sigma)
    if not np.allclose(Sigma, np.diag(diag), atol=atol):
        raise ValueError("This specialized module requires Sigma to be diagonal (cc * I).")
    if not np.allclose(diag, diag[0], atol=atol):
        raise ValueError("This specialized module requires Sigma = cc * I.")
    return float(diag[0])

cc = _assert_isotropic_Sigma(Sigma)
delta = float(delta_reg)
g = float(gamma)  # constant learning rate

# Drift rate lambda = (cc + delta) * gamma
lam = (cc + delta) * g

# q(t) = (1/d) * (2 cc P_t + sigma^2)
def q_of(t: float) -> float:
    return (2.0 * cc * float(interp_P(t)) + sigma_sgd**2) / d

# --------- Closed-form mean and (isotropic) covariance before any jump ---------
def mean_at(t: float) -> np.ndarray:
    """m(t) = e^{-lam t} x0 + (cc/(cc+delta)) * (1 - e^{-lam t}) * xtilde"""
    coef = np.exp(-lam * t)
    return coef * x0 + (cc / (cc + delta)) * (1.0 - coef) * xtilde

def scalar_v_at(t: float, Nint: int = 400) -> float:
    """
    V(t) = v(t) * I_d with
      v(t) = g^2 * ∫_0^t e^{-2 lam (t-u)} q(u) du
           = g^2 * [ (sigma^2/d) * (1 - e^{-2 lam t}) / (2 lam) + (2 cc / d) ∫_0^t e^{-2 lam (t-u)} P_u du ]
    """
    if t <= 0.0:
        return 0.0
    # analytic constant-noise term
    const_term = (sigma_sgd**2 / d) * (1.0 - np.exp(-2.0 * lam * t)) / (2.0 * lam)
    # numeric P_u term
    us = np.linspace(0.0, t, Nint)
    Pu = np.array([float(interp_P(u)) for u in us], dtype=float)
    kern = np.exp(-2.0 * lam * (t - us))
    Pu_term = (2.0 * cc / d) * trapezoid(Pu * kern, us)
    return (g**2) * (const_term + Pu_term)

def cov_at(t: float, Nint: int = 400) -> np.ndarray:
    """Return V(t) = v(t) * I_d."""
    v = scalar_v_at(t, Nint=Nint)
    return v * np.eye(d)

# --------- Jump (surgical differentiation) at time s ---------
def eta_at(s: float) -> float:
    """η = γ/d with constant schedule."""
    return g / d

def jump_rank1_coeffs(v_s: float, a: np.ndarray, s: float):
    """
    For C = I - η (a a^T + δ I) = (1 - η δ) I - η a a^T,
    V(s^+) = C (v_s I) C^T + (η^2 σ^2) I
           = [ v_s (1 - η δ)^2 + η^2 σ^2 ] I + v_s [ -2 η (1 - η δ) + η^2 ||a||^2 ] (a a^T).
    Return (beta, k) where V(s^+) = beta * I + k * (a a^T).
    """
    eta = eta_at(s)
    beta = v_s * (1.0 - eta * delta)**2 + (eta**2) * (sigma_sgd**2)
    k = v_s * ( -2.0 * eta * (1.0 - eta * delta) + (eta**2) * (norm(a)**2) )
    return beta, k

def post_jump_mean(m_s: np.ndarray, a: np.ndarray, b: float, s: float) -> np.ndarray:
    """m(s^+) = C m_s + c with C, c as above."""
    eta = eta_at(s)
    return (1.0 - eta * delta) * m_s - eta * np.outer(a, a) @ m_s + eta * b * a

# --------- Propagation from s to t (t >= s) ---------
def drift_tail_mean(s: float, t: float) -> np.ndarray:
    """∫_s^t e^{-lam (t-u)} * g * (cc I) xtilde du = (cc/(cc+delta)) * (1 - e^{-lam (t-s)}) xtilde"""
    return (cc / (cc + delta)) * (1.0 - np.exp(-lam * (t - s))) * xtilde

def scalar_tail_cov(s: float, t: float, Nint: int = 200) -> float:
    """
    Tail covariance from s to t (isotropic):
      g^2 ∫_s^t e^{-2 lam (t-u)} q(u) du
    """
    if t <= s:
        return 0.0
    const_term = (sigma_sgd**2 / d) * (1.0 - np.exp(-2.0 * lam * (t - s))) / (2.0 * lam)
    us = np.linspace(s, t, Nint)
    Pu = np.array([float(interp_P(u)) for u in us], dtype=float)
    kern = np.exp(-2.0 * lam * (t - us))
    Pu_term = (2.0 * cc / d) * trapezoid(Pu * kern, us)
    return (g**2) * (const_term + Pu_term)

def propagate_post_jump(beta: float, k: float, a: np.ndarray, s: float, t: float, Nint: int = 200):
    """
    If V(s^+) = beta I + k (a a^T), then for t>=s:
      V(t;s) = e^{-2 lam (t-s)} [beta I + k (a a^T)] + tail_scalar(s,t) I
             = [ e^{-2 lam Δ} beta + tail ] I + [ e^{-2 lam Δ} k ] (a a^T).
    """
    decay = np.exp(-2.0 * lam * (t - s))
    tail = scalar_tail_cov(s, t, Nint=Nint)
    beta_t = decay * beta + tail
    k_t = decay * k
    return beta_t, k_t

def mean_post_jump_to_t(m_sp: np.ndarray, s: float, t: float) -> np.ndarray:
    """m(t;s) = e^{-lam (t-s)} m(s^+) + drift_tail_mean(s,t)."""
    return np.exp(-lam * (t - s)) * m_sp + drift_tail_mean(s, t)

# --------- Rényi integrand and epsilon ---------
def _solve_Minv_b(beta, r1, u, r2, v, b, eps=1e-12):
    """
    Solve (beta I + r1 u u^T + r2 v v^T) x = b for possibly mixed-sign r1,r2.
    Uses Woodbury with U=[u,v], C=diag(r1,r2) (no square-roots).
    Handles the cases r1=0 and/or r2=0.
    """
    beta = float(beta)
    invb = 1.0 / max(beta, eps)

    # Build active columns
    U_cols, r_list = [], []
    if abs(r1) > eps:
        U_cols.append(u.reshape(-1))
        r_list.append(r1)
    if abs(r2) > eps:
        U_cols.append(v.reshape(-1))
        r_list.append(r2)

    if not U_cols:  # purely isotropic
        return invb * b

    U = np.column_stack(U_cols)            # d x k (k=1 or 2)
    Rinv = np.diag([1.0/ri for ri in r_list])  # k x k (signed)
    G = U.T @ U                            # k x k

    # Woodbury: (B + U C U^T)^{-1} = B^{-1} - B^{-1} U (C^{-1} + U^T B^{-1} U)^{-1} U^T B^{-1}
    S = Rinv + invb * G                    # k x k
    y = np.linalg.solve(S, invb * (U.T @ b))
    return invb * (b - U @ y)


def _logdet_M(beta, r1, u, r2, v, eps=1e-12):
    """
    log det(beta I + r1 u u^T + r2 v v^T) with mixed-sign r1,r2.
    Uses det(I + U C U^T / beta) = det(I + C (U^T U) / beta).
    """
    d_dim = u.shape[0]
    beta = max(float(beta), eps)

    U_cols, r_list = [], []
    if abs(r1) > eps:
        U_cols.append(u.reshape(-1))
        r_list.append(r1)
    if abs(r2) > eps:
        U_cols.append(v.reshape(-1))
        r_list.append(r2)

    if not U_cols:
        return d_dim * np.log(beta)

    U = np.column_stack(U_cols)            # d x k
    R = np.diag(r_list)                    # k x k
    G = U.T @ U                            # k x k
    small = np.eye(len(r_list)) + (1.0/beta) * (R @ G)
    sign, logdet_small = np.linalg.slogdet(small)
    if sign <= 0:
        # add tiny jitter to keep things numeric if extremely close to singular
        small = small + 1e-12 * np.eye(small.shape[0])
        sign, logdet_small = np.linalg.slogdet(small)
    return d_dim * np.log(beta) + logdet_small


def renyi_integrand_special(mu1, beta1, k1, a, mu2, beta2, k2, a2, alpha, eps=1e-12):
    """
    Return exp((alpha-1) * D_alpha) exactly for
      Σ1 = beta1 I + k1 a a^T,  Σ2 = beta2 I + k2 a2 a2^T,
    allowing mixed-sign k1,k2 and α>1.
    """
    mu1 = np.asarray(mu1).reshape(-1)
    mu2 = np.asarray(mu2).reshape(-1)
    a   = np.asarray(a).reshape(-1)
    a2  = np.asarray(a2).reshape(-1)
    diff = mu1 - mu2

    # Mixture covariance M_alpha
    betaM = alpha * beta1 + (1.0 - alpha) * beta2
    r1M   = alpha * k1
    r2M   = (1.0 - alpha) * k2

    # Quadratic term: diff^T M_alpha^{-1} diff
    Minv_diff = _solve_Minv_b(betaM, r1M, a, r2M, a2, diff, eps=eps)
    qf = float(diff @ Minv_diff)
    term1 = 0.5 * alpha * (alpha - 1.0) * qf

    # Log-dets: det Σ1, det Σ2 (rank-1), and det M_alpha (mixed rank-2)
    a2norm = float(a @ a)
    a22norm = float(a2 @ a2)

    beta1 = max(float(beta1), eps)
    beta2 = max(float(beta2), eps)

    det1 = d * np.log(beta1) + np.log(max(1.0 + (k1 / beta1) * a2norm, eps))
    det2 = d * np.log(beta2) + np.log(max(1.0 + (k2 / beta2) * a22norm, eps))
    detM = _logdet_M(betaM, r1M, a, r2M, a2, eps=eps)

    log_det_ratio_half = 0.5 * (alpha * det1 + (1.0 - alpha) * det2 - detM)
    return float(np.exp(term1 + log_det_ratio_half))


def epsilon_alpha(a: np.ndarray, b: float, a_prime: np.ndarray, b_prime: float,
                  timet: float, T: float, alpha: float, Ns: int = 120) -> float:
    """
    Specialized ε_α(t) for isotropic Sigma and constant γ.
    """
    if timet < 0 or T <= 0 or timet > T:
        raise ValueError("Require 0 <= t <= T and T > 0.")
    if timet == 0.0:
        return 0.0

    ss = np.linspace(0.0, timet, Ns)
    integrand_vals = np.zeros_like(ss)

    for i, s in enumerate(ss):
        # m(s), v(s)
        m_s = mean_at(float(s))
        v_s = scalar_v_at(float(s))

        # Post-jump mean/cov at s^+ for the two neighboring updates
        m1_sp = post_jump_mean(m_s, a, b, s)
        m2_sp = post_jump_mean(m_s, a_prime, b_prime, s)

        beta1_sp, k1_sp = jump_rank1_coeffs(v_s, a, s)
        beta2_sp, k2_sp = jump_rank1_coeffs(v_s, a_prime, s)

        # Propagate to t
        beta1_t, k1_t = propagate_post_jump(beta1_sp, k1_sp, a, s, timet)
        beta2_t, k2_t = propagate_post_jump(beta2_sp, k2_sp, a_prime, s, timet)

        m1_t = mean_post_jump_to_t(m1_sp, s, timet)
        m2_t = mean_post_jump_to_t(m2_sp, s, timet)

        # exp((α-1) D_α)
        integrand_vals[i] = renyi_integrand_special(m1_t, beta1_t, k1_t, a,
                                                   m2_t, beta2_t, k2_t, a_prime,
                                                   alpha)

    integral_val = trapezoid(integrand_vals, ss)
    eps_t = (1.0 / (alpha - 1.0)) * np.log((1.0 / T) * integral_val + (T - timet) / T)
    return float(eps_t)
