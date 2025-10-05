import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import get_context

from parameter_setup import *                      

from Cal_loss_functions import (
    epsilon_alpha,          
    lam, delta, cc, d     
)

# ----------------- helpers -----------------
def mean_at(t: float) -> np.ndarray:
    """
    Closed form for baseline mean m(t) under constant γ and isotropic Σ=cc I:
    m(t) = e^{-λ t} x0 + (cc/(cc+delta)) * (1 - e^{-λ t}) * xtilde.
    """

    return np.exp(-lam * t) * x0 + (cc / (cc + delta)) * (1.0 - np.exp(-lam * t)) * xtilde


def load_pairs(n_pairs=3000, seed=4):
    """
    Randomly choose n_pairs distinct (i,j) pairs with i<j.
    """
    rng = np.random.default_rng(seed)
    n = len(b_vec)

    pairs = set()
    # sample with replacement then deduplicate
    while len(pairs) < n_pairs:
        ii = rng.integers(0, n, size=10000)
        jj = rng.integers(0, n, size=10000)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        lo = np.minimum(ii, jj)
        hi = np.maximum(ii, jj)
        for x, y in zip(lo, hi):
            pairs.add((int(x), int(y)))
            if len(pairs) >= n_pairs:
                break
    pairs = np.array(sorted(pairs), dtype=int)
    print(f"pairs chosen: {len(pairs)}")
    return pairs


_G_t = None
_G_alpha = None
_G_m_s = None

def _init_worker(t_cur, alpha_in, s_mid):
    global _G_t, _G_alpha, _G_m_s
    _G_t = float(t_cur)
    _G_alpha = float(alpha_in)
    _G_m_s = mean_at(float(s_mid))


def _score_pair(pair):
    """
    Compute (ba_diff, g_norm_mid, eps_t) for one (i,j) using globals set by _init_worker.
    """
    i, j = int(pair[0]), int(pair[1])
    a,  b  = A_matrix[i, :], float(b_vec[i])
    ap, bp = A_matrix[j, :], float(b_vec[j])

    ba_diff = float(np.linalg.norm(b * a - bp * ap))
    eps_val = float(epsilon_alpha(a, b, ap, bp, timet=T/2, T=T, alpha=_G_alpha))
    aa, aap = np.outer(a, a), np.outer(ap, ap)
    gtheta  = (b * a - bp * ap) - (aa - aap + delta * np.eye(d)) @ _G_m_s
    g_norm  = float(np.linalg.norm(gtheta))

    return ba_diff, g_norm, eps_val

# ----------------- main experiment -----------------
def run_scatter(t=None, alpha_in=None, max_pairs=None, n_pairs=500, s_mid_frac=0, n_workers=None, seed=3):
    os.makedirs(folder_name, exist_ok=True)
    if t is None:
        t = T/2  
    if alpha_in is None:
        alpha_in = alpha
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)

    pairs = load_pairs()
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    s_mid = float(s_mid_frac) * float(t)


    try:
        ctx = get_context('fork')
    except ValueError:
        ctx = get_context('spawn')

    print(f"Scoring {len(pairs)} pairs at t={t:.4f} (s_mid={s_mid:.4f}) using {n_workers} workers...")
    t0 = time.time()
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(t, alpha_in, s_mid)) as pool:
        results = pool.map(_score_pair, [tuple(p) for p in pairs])
    print(f"Done in {time.time()-t0:.1f}s")

    badiff_arr = np.array([r[0] for r in results], dtype=float)
    gnorm_arr  = np.array([r[1] for r in results], dtype=float)
    eps_arr    = np.array([r[2] for r in results], dtype=float)


    df = pd.DataFrame({
        "i": pairs[:,0].astype(int),
        "j": pairs[:,1].astype(int),
        "ba_diff": badiff_arr,
        "g_norm_mid": gnorm_arr,
        "eps_t": eps_arr
    })
    out_csv = os.path.join(folder_name, f"pairs_gnorm_eps_t{t:.3f}_alpha{alpha_in:.2f}.csv")
    df.to_csv(out_csv, index=False)


    
    
    # -------- Scatter: ε(t) vs ||g(theta)|| --------
    plt.figure(figsize=(4,3.5))



    bins = np.linspace(gnorm_arr.min(), gnorm_arr.max(), 500)
    digitized = np.digitize(gnorm_arr, bins)
    
    sample_idx = [np.random.choice(np.where(digitized == b)[0]) 
                  for b in np.unique(digitized) if np.any(digitized == b)]
    
    plt.scatter(gnorm_arr[sample_idx], eps_arr[sample_idx], alpha=0.6, edgecolor=None, s=18)

    plt.yscale("log")
    plt.xlabel(r"$\|g(\theta)\|_2$")
    plt.ylabel(r"Rényi privacy loss")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, f"scatter_eps_vs_gnorm_t{t:.3f}.pdf"))
    plt.close()


    def safe_corr(x, y):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if np.all(x == x[0]) or np.all(y == y[0]):
            return np.nan
        return np.corrcoef(np.log10(np.maximum(x, 1e-300)),
                           np.log10(np.maximum(y, 1e-300)))[0,1]

    corr_badiff = safe_corr(badiff_arr, eps_arr)
    corr_gnorm  = safe_corr(gnorm_arr,  eps_arr)

    print(f"Saved CSV to: {out_csv}")
    print(f"log–log corr( ||g(θ)||(s={s_mid_frac:.2f}·t) , ε ): {corr_gnorm:.3f}")

if __name__ == "__main__":
    run_scatter()
