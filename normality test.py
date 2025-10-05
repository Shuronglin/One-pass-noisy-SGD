import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numpy.linalg import inv, det
from numpy.random import default_rng
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import chi2, norm
from parameter_setup import *
from Read_risk import interp_P
from matplotlib.ticker import MaxNLocator, ScalarFormatter

warnings.filterwarnings("ignore")
sigma_sgd = sigma_sgd_list[jj]

def run_SGD_get_one_xm(m, A_matrix, b_vec, gamma, delta_reg, sigma_sgd, x0, decay_rate=0.0, rng=None):
    if rng is None:
        rng = default_rng()
    A = np.asarray(A_matrix, dtype=np.float64, order="C")
    b = np.asarray(b_vec, dtype=np.float64, order="C")
    x = np.asarray(x0, dtype=np.float64).copy()
    for k in range(m):
        a_k = A[k]
        r = a_k.dot(x) - b[k]
        grad = r * a_k + delta_reg * x
        gamma_k = gamma / (1.0 + decay_rate * k)
        noise = sigma_sgd * rng.standard_normal(x.shape)
        x -= gamma_k * (grad + noise)
    return x


def A_fn(t):
    return np.exp(-gamma_sde * (cc + delta_reg) * t)


def B_fn(t):
    return (1.0 - A_fn(t)) * (cc / (cc + delta_reg)) * xtilde


def v_fn(t, d):
    dense_u = np.linspace(0.0, t, 200)
    P_dense = interp_P(dense_u)
    integrand = np.exp(2.0 * gamma_sde * (cc + delta_reg) * dense_u) * P_dense
    integral_val = trapezoid(integrand, dense_u)
    term1 = gamma_sde * sigma_sgd**2 * (1.0 - A_fn(t)**2) / (2.0 * d * (cc + delta_reg))
    term2 = (2.0 * cc * gamma_sde**2 * (A_fn(t)**2) / d) * integral_val
    return term1 + term2


def qqplot_mahalanobis(X, mu, v_scalar):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    N, d = X.shape
    Z = X - mu[None, :]

    if not np.isfinite(v_scalar) or v_scalar <= 0:
        v_scalar = np.mean(np.var(Z, axis=0))

    m2_sorted = np.sort((Z * Z).sum(axis=1) / v_scalar)
    probs = (np.arange(1, N + 1) - 0.5) / N
    theo = chi2.ppf(probs, df=d)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(theo, m2_sorted, s=8, alpha=0.6)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lo = min(x0, y0)
    hi = max(x1, y1)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0)

    ax.set_xlabel(r"HSGD quantiles")
    ax.set_ylabel(r"SGD quantiles")
    # ax.set_title("Mahalanobis QQ Plot")
    tick_values=(800,1000,1200)
    ax.set(xticks=tick_values, yticks=tick_values,)  
    
    fig.tight_layout()
    filename = os.path.join(folder_name, f"mvn_qq_{sigma_sgd}_iter_{miter}.pdf")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)



from multiprocessing import get_context
import os
import numpy as np
from numpy.random import default_rng


def init_worker():
    seed = (os.getpid() ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFF
    global _rng
    _rng = default_rng(seed)

from numpy.random import default_rng, SeedSequence

def _one_sim_with_seed(child_ss):
    rng = default_rng(child_ss)  # deterministic per task
    return run_SGD_get_one_xm(
        miter, A_matrix, b_vec, gamma, delta_reg, sigma_sgd, x0,
        decay_rate=decay_rate, rng=rng
    )

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    miter = 1500
    N_sims = 10000

    n, d = A_matrix.shape
    t = miter / d
    A_t = A_fn(t)
    B_t = B_fn(t)
    mu_t = A_t * np.asarray(x0, dtype=np.float64) + B_t
    v_t = v_fn(t, d)

    master_seed = 1234
    ss = SeedSequence(master_seed)
    child_seeds = ss.spawn(N_sims)  # one per simulation, deterministic

    ctx = get_context("fork")
    with ctx.Pool(processes=os.cpu_count()-3) as pool:
        parts = pool.map(_one_sim_with_seed, child_seeds)

    X = np.vstack(parts)
    qqplot_mahalanobis(X, mu_t, v_t)