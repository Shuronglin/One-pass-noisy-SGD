
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Import user parameters and data ----------
from parameter_setup import (
    n, d, T, sigma_sgd_list, jj, folder_name, clipping_list,
    Sigma, sigma2, decay_rate, xtilde, x0,
    gamma, delta_reg, alpha, A_matrix, b_vec, cc, timepoints
)


sigma = sigma_sgd_list[jj]
file_map = {
    sigma: os.path.join(folder_name, f"loss_curves_{sigma}.csv")
}
timepoints = np.logspace(
    np.log10(1/d),
    np.log10(n/d),
    num=30
)
x_vals = timepoints * d  # X-grid used by CSV outputs (should span up to n)

# ---------------- Privacy bound (Theorem 3.3, shuffled/Jensen) --------
def eps_step_k(alpha, eta, Sg, sigma, mu, k):
    """
    Position-dependent one-pass cost for release after k steps since the
    differing-record step, specialized to b=1. This is ε^{(0)}_k(α).

    ε^{(0)}_k = A * r2^(k-1) / sum_{s=0}^{k-1} r2^s,
    where A = α η S_g^2 / (4 σ^2), r = 1 - η μ, r2 = r^2.
    Handles r2≈1 with the k-limit.
    """
    if k < 1:
        return 0.0
    r = 1.0 - eta * mu
    r2 = r * r
    A = (alpha * eta * (Sg ** 2)) / (4.0 * (sigma ** 2))  # b=1 ⇒ /b^2 drops out

    if abs(r2 - 1.0) < 1e-12:
        denom = float(k)  # geometric sum limit
    else:
        denom = (1.0 - r2 ** k) / (1.0 - r2)

    return A * (r2 ** (k - 1)) / denom


def eps_shuffle_at_iter(j, alpha, eta, Sg, sigma, mu, n):
    """
    Shuffle-averaged (uniform position) RDP at iteration j (1..n) for one pass.
    Uses Jensen/log-avg-exp over positions:
      ε_j = (1/(α-1)) * log( ((n-j) + sum_{k=1..j} exp((α-1) ε^{(0)}_k)) / n )
    where k = (# steps since differing-record step) ranges 1..j.
    """
    if j <= 0:
        return 0.0

    # Build the list of per-position costs ε^{(0)}_k for k=1..j
    eps_k = np.array([eps_step_k(alpha, eta, Sg, sigma, mu, k) for k in range(1, j + 1)])

    # log-avg-exp with a 0-cost mass for positions rpos > j
    a = alpha - 1.0
    # Contributions: (n-j) positions contribute exp(0)=1; j positions contribute exp(a*eps_k)
    # Use log-sum-exp for stability.
    max_term = np.maximum(0.0, np.max(a * eps_k) if eps_k.size > 0 else 0.0)
    sum_exp = (n - j) * np.exp(-max_term) + np.sum(np.exp(a * eps_k - max_term))
    return (max_term + np.log(sum_exp) - np.log(n)) / a


def compute_bound_curves(n, sigma, alpha, eta, mu, Sg_list):
    """
    Returns per-Sg curves of shuffled ε_α(j) for j = 1..n.
    """
    iters = np.arange(1, n + 1)
    curves = {}
    for Sg in Sg_list:
        eps_curve = np.array([
            eps_shuffle_at_iter(j=int(j), alpha=alpha, eta=eta,
                                Sg=Sg, sigma=sigma, mu=mu, n=n)
            for j in iters
        ])
        curves[Sg] = (eps_curve, iters)
    return curves

# ---------- Helpers to load CSV (vector t0..tK) ----------
def load_y_values(csv_path, num_pts_expected):
    # Try wide (one row, columns t0..tK)
    try:
        df = pd.read_csv(csv_path)
        wanted = [f"t{i}" for i in range(num_pts_expected)]
        if all(col in df.columns for col in wanted):
            return df[wanted].iloc[0].to_numpy(dtype=float)
    except Exception:
        pass

    # Try tall (index t0..tK, single column)
    try:
        df2 = pd.read_csv(csv_path, index_col=0)
        wanted = [f"t{i}" for i in range(num_pts_expected)]
        if len(df2.columns) >= 1 and all(k in df2.index for k in wanted):
            return df2.loc[wanted, df2.columns[0]].to_numpy(dtype=float)
    except Exception:
        pass

    # Try raw row of numbers
    try:
        df3 = pd.read_csv(csv_path, header=None)
        arr = df3.values.flatten()
        if arr.size >= num_pts_expected:
            return arr[:num_pts_expected].astype(float)
    except Exception:
        pass

    raise ValueError(
        f"Could not parse y-values from {csv_path}. "
        f"Expected columns t0..t{num_pts_expected-1} or a single {num_pts_expected}-number row."
    )

# ---------------- Main ----------------
if __name__ == "__main__":
    np.random.seed(123)

    # list of sensitivity / clipping thresholds
    Sg_list = clipping_list

    # Build theorem (shuffle-averaged) curves
    curves = compute_bound_curves(
        n=n, sigma=sigma, alpha=alpha, eta=gamma, mu=delta_reg, Sg_list=Sg_list
    )

    # Epsilon floor so log-y is happy
    positives = []
    for Sg, (eps_curve, iters) in curves.items():
        positives.extend(list(eps_curve[eps_curve > 0]))
    eps_floor = max(1e-12, 0.1 * np.min(positives) if len(positives) > 0 else 1e-12)


    # --- Plot ---
    plt.figure(figsize=(4, 3.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 1) Plot theorem curves for each Sg (dashed)
    #     Add (j=0, ε≈eps_floor) anchor so the log plot starts near (0,0)
    j0, y0 = 0, eps_floor
    for i, Sg in enumerate(sorted(curves.keys())):
        c = colors[i % len(colors)]
        eps_curve, iters = curves[Sg]
        x_plot = np.concatenate([[j0], iters])
        y_plot = np.concatenate([[y0], np.clip(eps_curve, eps_floor, None)])
        plt.plot(x_plot, y_plot, '-', linewidth=1.5, color=c, label=f"Ye & Shokri(2022), Clip={Sg}")

    # Axes formatting
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1, None)
    plt.ylim(8e-5, 2.5*1e-3)
    plt.xlabel("Iteration")
    plt.ylabel(r"Rényi privacy loss $\varepsilon_\alpha$")
    # plt.title(f"Privacy curves: HSGD vs Shuffle-avg (α = {alpha}, σ={sigma})")

    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(
        folder_name, f"privacy_loss_sigma{sigma}_comparison.pdf"
    )
    plt.savefig(out_path, dpi=300, format='pdf', bbox_inches='tight')
    print(f"Saved: {out_path}")
