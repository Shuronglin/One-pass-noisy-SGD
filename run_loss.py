# run_privacy_curves.py
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import get_context

# --- Your setup & P_t interpolator ---
from parameter_setup import *          
from Read_risk import interp_P
from Cal_loss_functions import epsilon_alpha

top_pairs_specified=None
# top_pairs_specified = [[903,1344]]


os.makedirs(folder_name, exist_ok=True)


# --------- Pair selection ---------
def generate_unique_pairs(top_k=30, bottom_k=30):

    sorted_idx = np.argsort(b_vec, kind="stable")
    bottom_idx = sorted_idx[:bottom_k]       
    top_idx    = sorted_idx[-top_k:]         

    ii, jj = np.meshgrid(top_idx, bottom_idx, indexing='ij')
    pairs = np.column_stack((ii.ravel(), jj.ravel()))
    pairs = np.sort(pairs, axis=1)           
    pairs = np.unique(pairs, axis=0)         
    return pairs

# --------- Core computations ---------
def compute_eps_curve(pair):
    """Compute ε_α(t) over all timepoints for one (i,j) pair."""
    i, j = int(pair[0]), int(pair[1])
    a,  b    = A_matrix[i], b_vec[i]
    a_p, b_p = A_matrix[j], b_vec[j]
    return [epsilon_alpha(a, b, a_p, b_p, timet=t, T=T, alpha=alpha) for t in timepoints]

def init_worker():
    seed = (os.getpid() ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFF
    np.random.seed(seed)

def compute_eps_t0(pair, t0=0.2):
    i, j = int(pair[0]), int(pair[1])
    eps = epsilon_alpha(A_matrix[i], b_vec[i], A_matrix[j], b_vec[j], timet=t0, T=T, alpha=alpha)
    ba_diff = np.linalg.norm(b_vec[i] * A_matrix[i] - b_vec[j] * A_matrix[j])
    aaT_diff = np.linalg.norm(np.outer(A_matrix[i], A_matrix[i]) - np.outer(A_matrix[j], A_matrix[j]))
    return eps, ba_diff, aaT_diff

def get_top_pairs(pairs, n_top_pairs=10, t0_frac=0.2, n_workers=None):
    if top_pairs_specified: return top_pairs_specified
        
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
    t0 = max(1e-6, float(t0_frac) * T)

    try:
        ctx = get_context('fork')
    except ValueError:
        ctx = get_context('spawn')
    start = time.time()
    with ctx.Pool(processes=n_workers, initializer=init_worker) as pool:
        scores = pool.starmap(compute_eps_t0, [(tuple(p), t0) for p in pairs])
    took = time.time() - start

    eps_arr    = np.array([s[0] for s in scores], dtype=float)
    ba_arr     = np.array([s[1] for s in scores], dtype=float)
    aaT_arr    = np.array([s[2] for s in scores], dtype=float)

    sorted_idx = np.argsort(eps_arr)
    top_idx    = sorted_idx[-n_top_pairs:][::-1]
    top_pairs  = pairs[top_idx]


    df_all = pd.DataFrame(pairs, columns=["idx_i", "idx_j"])
    df_all["eps_t0"] = eps_arr
    df_all["||b a - b' a'||"] = ba_arr
    df_all["||aa^T - a'a'^T||"] = aaT_arr
    return top_pairs

def get_curves_for_pairs(pairs, n_workers=None):
    """Compute ε-curves for a list of pairs in parallel; returns array shape (num_pairs, num_timepoints)."""
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)

    try:
        ctx = get_context('fork')
    except ValueError:
        ctx = get_context('spawn')

    start = time.time()
    with ctx.Pool(processes=n_workers, initializer=init_worker) as pool:
        curves = pool.map(compute_eps_curve, [tuple(p) for p in pairs])
    took = time.time() - start
    return np.array(curves, dtype=float)

# --------- Main ---------
if __name__ == "__main__":

    pairs = generate_unique_pairs()
    top_pairs = get_top_pairs(pairs, n_top_pairs=10, t0_frac=0.2)

    eps_curve_arr = get_curves_for_pairs(top_pairs)

    df_curves = pd.DataFrame(
        eps_curve_arr,
        index=[f"pair_{int(i)}_{int(j)}" for i, j in top_pairs],
        columns=[f"t{idx}" for idx in range(eps_curve_arr.shape[1])]
    )
    curves_path = f"{folder_name}/loss_curves_{sigma_sgd_list[jj]}.csv"
    df_curves.to_csv(curves_path, index=True)
    print(f"Saved curves CSV: {curves_path}")


    # iterations = timepoints * d
    # max_curve = eps_curve_arr.max(axis=0)

    # plt.figure()
    # for curve in eps_curve_arr:
    #     plt.plot(iterations, curve, alpha=0.7)
    # plt.plot(iterations, max_curve, '--', linewidth=2.0, label="max curve")
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('iteration'); plt.ylabel(r'$\varepsilon_\alpha$')
    # plt.title(f'Privacy loss curves (σ={sigma_sgd_list[jj]})')
    # plt.legend(); plt.tight_layout()
    # fig_path = f"{folder_name}/loss_curves_{sigma_sgd_list[jj]}.pdf"
    # plt.savefig(fig_path)
    # print(f"Saved: {fig_path}")

    # plt.figure()
    # plt.plot(iterations, max_curve)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('iteration'); plt.ylabel(r'$\varepsilon_\alpha$')
    # plt.title(f'Max privacy loss curve (σ={sigma_sgd_list[jj]})')
    # plt.tight_layout()
    # max_fig_path = f"{folder_name}/loss_max_curve_{sigma_sgd_list[jj]}.pdf"
    # plt.savefig(max_fig_path)
    # print(f"Saved: {max_fig_path}")
