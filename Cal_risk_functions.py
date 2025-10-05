import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

warnings.filterwarnings("ignore")

def P_func(x, xtilde, Sigma, sigma2):
    """
    Population risk: P(x) = [(x - xtilde)^\top Sigma_x (x - xtilde) + E(xi^2)]/2 by taking expectations over a and b
    """
    return 0.5 * (x - xtilde).T @ Sigma @ (x - xtilde) + 0.5 * sigma2

import numpy as np
from scipy.linalg import expm
from scipy.integrate import cumulative_trapezoid

def gamma_func(s, gamma, d):
    return gamma*d


def Gamma(ts, gamma, d):
    # Gamma = gamma*d*ts # contant learning rate
    gammas = np.array([gamma_func(t, gamma, d) for t in ts])
    Gamma = cumulative_trapezoid(gammas, ts, initial=0.0)
    return Gamma

    

def G(Sigma,t, s, M, gamma,d,Gamma_t, Gamma_s, delta_reg):
    A = Sigma + delta_reg * np.eye(Sigma.shape[0])
    exp_term = expm(-2 * A * (Gamma_t - Gamma_s))
    return (gamma_func(s, gamma, d)**2 / Sigma.shape[0]) * np.trace(Sigma @ M @ exp_term)

def G_prime(Sigma,t, s, M, gamma,d,Gamma_t, Gamma_s, delta_reg,sigma_sgd):
    A = Sigma + delta_reg * np.eye(Sigma.shape[0])
    exp_term = expm(-2 * A * (Gamma_t - Gamma_s))
    return (sigma_sgd**2 * gamma_func(s, gamma, d)**2 / (2 * Sigma.shape[0])) * np.trace(M @ exp_term)

from tqdm import tqdm
import numpy as np

def solve_volterra_P(ts, P_vals_gf, HessianP, Nabla2P, Sigma,gamma, delta_reg, sigma_sgd):
    d = Sigma.shape[0]
    Gamma_vals = Gamma(ts, gamma, d)
    P_vals = np.zeros(len(ts))

    for i, t in enumerate(tqdm(ts, desc="Solving Volterra P")):
        G_integral = 0
        G_prime_integral = 0
        for j in range(i):
            s = ts[j]
            dt = ts[j+1] - ts[j] if j+1 < len(ts) else ts[j] - ts[j-1]
            Gamma_t = Gamma_vals[i]
            Gamma_s = Gamma_vals[j]
            G_val = G(Sigma,t, s, HessianP, gamma,d,Gamma_t, Gamma_s, delta_reg)
            Gp_val = G_prime(Sigma,t, s, Nabla2P, gamma,d,Gamma_t, Gamma_s, delta_reg,sigma_sgd)
            G_integral += G_val * P_vals[j] * dt
            G_prime_integral += Gp_val * dt
        P_vals[i] = P_vals_gf[i] + G_integral + G_prime_integral

    return ts, P_vals


def compute_X_gf(ts, x0, xtilde, Sigma, gamma, delta):
    d = x0.shape[0]
    A = Sigma + delta * np.eye(d)
    Gamma_vals = Gamma(ts, gamma, d)
    X_gf = np.zeros((len(ts), d))

    for i, t in enumerate(ts):
        Gamma_t = Gamma_vals[i]

        # First term: exp(-A * Gamma_t) x0
        term1 = expm(-A * Gamma_t) @ x0

        # Second term: Σ xtilde ∫₀ᵗ exp(-A (Γ(t) - Γ(s))) γ(s) ds
        integral = np.zeros((d,))

        for j in range(i):
            dt = ts[j+1] - ts[j] if j+1 < len(ts) else ts[j] - ts[j-1]
            Gamma_s = Gamma_vals[j]
            decay = expm(-A * (Gamma_t - Gamma_s))
            integral += (gamma_func(ts[j], gamma, d) * decay @ xtilde) * dt

        term2 = Sigma @ integral

        X_gf[i] = term1 + term2

    return Gamma_vals, X_gf


def evaluate_P_over_gf(ts, x0, xtilde, Sigma, sigma2,gamma, delta, P_func):
    _, X_gf_vals = compute_X_gf(ts, x0, xtilde, Sigma, gamma, delta)
    P_vals = np.array([P_func(x, xtilde, Sigma, sigma2) for x in X_gf_vals])
    return ts, P_vals





