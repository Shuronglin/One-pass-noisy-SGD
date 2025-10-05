#!/usr/bin/env python3
# risk_clipping_one_sigma_shared_noise.py
import os
import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from parameter_setup import *
from Cal_risk_functions_iso_const import P_func  


def run_SGD_get_risk(A_matrix, b_vec, gamma, delta_reg, sigma_sgd, x0,
                     clip_threshold=None, noise_seq=None):
    n, d = A_matrix.shape
    if noise_seq is None:
        
        rng = np.random.default_rng()
        noise_seq = rng.normal(0.0, 1.0, size=(n, d))

    x = x0.copy()
    risks = [P_func(x, xtilde, Sigma, sigma2)]
    max_grad_norm = 0.0

    for k in range(n):
        a = A_matrix[k]
        b = b_vec[k]
        grad = (a @ x - b) * a + delta_reg * x
        gnorm = np.linalg.norm(grad)
        if gnorm > max_grad_norm:
            max_grad_norm = gnorm

        
        if clip_threshold is not None and gnorm > clip_threshold and gnorm > 0:
            grad = grad * (clip_threshold / gnorm)

       
        noise = sigma_sgd * noise_seq[k]

       
        x = x - gamma * (grad + noise)
        risks.append(P_func(x, xtilde, Sigma, sigma2))

    tag = f"(clip={clip_threshold})" if clip_threshold is not None else "(no clip)"
    print(f"[σ={sigma_sgd}] Max grad norm (pre-clip): {max_grad_norm:.4f} {tag}")
    return np.array(risks)


# --------- main ---------
if __name__ == "__main__":
    
    sigma = float(sigma_sgd_list[jj])
    print(f"\n=== Running SGD for σ = {sigma} with shared per-step noise ===")
    caps = list(clipping_list) + [None]        
    rng_master = np.random.default_rng(12345)
    shared_noise_seq = rng_master.normal(0.0, 1.0, size=(n, d))

    # Run SGD 
    risks_dict = {}
    for cap in caps:
        risks_dict[cap] = run_SGD_get_risk(
            A_matrix, b_vec, gamma, delta_reg, sigma, x0,
            clip_threshold=cap, noise_seq=shared_noise_seq
        )

    out_dir = folder_name
    os.makedirs(out_dir, exist_ok=True)
    save_path_npz = os.path.join(out_dir, f"risk_data_sigma{sigma}_clipping_sharednoise.npz")

    caps_encoded = np.array([-1.0 if c is None else float(c) for c in caps], dtype=float)
    risks_arr = np.vstack([risks_dict[c] for c in caps])  

    np.savez_compressed(
        save_path_npz,
        risks=risks_arr,             
        sigma=np.array([sigma], float),
        caps=caps_encoded,           
        clipping_list=np.array(clipping_list, float),
        n=np.array([n], int),
        d=np.array([d], int),
        seed=np.array([12345], int)
    )
    print(f"Saved results to '{save_path_npz}'")

    plt.figure(figsize=(4, 3.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    
    clip_styles = {None: '-'}
    linestyles_pool = ['-', '-', '-', (0, (3, 1, 1, 1)), (0, (5, 1))]
    for i, clip in enumerate(clipping_list):
        clip_styles[clip] = linestyles_pool[i % len(linestyles_pool)]
    

    max_iter = len(next(iter(risks_dict.values()))) - 1
    plot_idx = np.unique(np.logspace(0, np.log10(max_iter), num=200, dtype=int))
    plot_idx = np.unique(np.concatenate([[0], plot_idx]))

    clips =  list(clipping_list)
    for i, clip in enumerate(clips):
        ls = clip_styles[clip]
        lbl = f"(no clip)" if clip is None else f"SGD (clip={clip})"
        plt.plot(
            plot_idx,
            risks_dict[clip][plot_idx],
            linestyle=ls,
            color=colors[i % len(colors)],  
            linewidth=1.2,
            alpha=1,
            label=lbl
        )
    eps = 0.05
    plt.plot(
        plot_idx * (1 - eps),  # shift for visibility 
        risks_dict[None][plot_idx],
        linestyle=clip_styles[None],
        color="black",
        linewidth=1,
        alpha=0.9,
        label=f"SGD (no clip)"
    )


    plt.xscale("log")
    plt.yscale("log")
    

    plt.gca().yaxis.set_major_locator(mticker.FixedLocator([1e-1]))
    plt.gca().yaxis.set_minor_locator(mticker.NullLocator())
    plt.gca().set_yticks([1e-1])        
    plt.gca().set_yticklabels(["$10^{-1}$"]) 
    
    plt.xlabel("Iteration")
    plt.ylabel("Population Risk")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    save_path_pdf = os.path.join(out_dir, f"risk_curves_sigma{sigma}_clipping.pdf")
    plt.savefig(save_path_pdf, dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {save_path_pdf}")
