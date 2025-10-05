# run_risk_iso_const.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import get_context

from parameter_setup import *          
from Cal_risk_functions_iso_const import (
    P_func, evaluate_P_over_gf_iso, solve_volterra_P_iso
)


def build_ts(n, d, n1=300, n2=300):
    arr1 = np.linspace(0.0, 1.0, n1)
    arr2 = np.linspace(1.0, n / d, n2)
    ts = np.concatenate((arr1, arr2[1:]))
    return ts

def run_SGD_get_risk(A_matrix, b_vec, gamma, delta_reg, sigma_sgd, x0, decay_rate=0.0):
    n, d = A_matrix.shape
    x = x0.copy()
    risks = [P_func(x, xtilde, Sigma, sigma2)]
    max_grad_norm = 0.0  

    for k in range(n):
        a = A_matrix[k]
        b = b_vec[k]
        grad = (a @ x - b) * a + delta_reg * x
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_grad_norm:
            max_grad_norm = grad_norm

        noise = sigma_sgd * np.random.normal(0, 1, size=d)
        gamma_k = gamma / (1 + decay_rate * k)  
        x = x - gamma_k * (grad + noise)
        risks.append(P_func(x, xtilde, Sigma, sigma2))
    return risks

def compute_for_sigma(sigma_sgd):
    """
    Worker: compute SGD risk (for reference) and Volterra risk using the fast solver.
    """
    ts = build_ts(n, d)
    # GF risk (fast) for isotropic Σ + constant γ
    ts, P_gf = evaluate_P_over_gf_iso(ts, x0, xtilde, Sigma, sigma2, gamma, delta_reg, d)
    # Solve Volterra (fast)
    ts, P_vals = solve_volterra_P_iso(ts, P_gf, Sigma, gamma, delta_reg, sigma_sgd, d)
    # Optional: simulate SGD for comparison (can be slow; comment out if you only need theory)
    pop_risk = run_SGD_get_risk(A_matrix, b_vec, gamma, delta_reg, sigma_sgd, x0, decay_rate=0.0)
    return pop_risk, ts, P_vals

def save_results_npz(ts_list, pop_risk_list, P_vals_list, sigma_sgd_list, file_name):
    ts_arr       = np.vstack(ts_list)
    pop_risk_arr = np.vstack(pop_risk_list)
    P_vals_arr   = np.vstack(P_vals_list)
    sigma_arr    = np.array(sigma_sgd_list)
    np.savez_compressed(file_name, ts=ts_arr, pop_risk=pop_risk_arr, P_vals=P_vals_arr, sigma=sigma_arr)
    print(f"Saved compressed data to '{file_name}'")

def plot_risk_Volterra(pop_risk_list, ts_list, P_vals_list, sigma_sgd_list):
    iteration = [x * d for x in ts_list]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(4.5, 4))

    for i, sigma in enumerate(sigma_sgd_list):
        c = colors[i % len(colors)]
        # SGD (solid)
        plt.plot(range(len(pop_risk_list[i])), pop_risk_list[i], label=f"SGD σ={sigma}", linewidth=1.5, color=c)
        # Volterra (dashed)
        plt.plot(iteration[i], P_vals_list[i], '--', label=f"HSGD σ={sigma}", linewidth=2, color=c)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, None)
    # plt.ylim(0, 0.4) 
    plt.xlabel("Iteration")
    plt.ylabel("Population Risk")
    # plt.xticks( [, 1000])
    
    plt.legend()
    # plt.grid(False)
    plt.tight_layout()
    
    save_path = os.path.join(folder_name, "risk_curves.pdf")
    plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        ctx = get_context('fork')
    except ValueError:
        ctx = get_context('spawn')
    with ctx.Pool(processes=min(6, (os.cpu_count() or 4)), initializer=lambda: np.random.seed(os.getpid())) as pool:
        results = pool.map(compute_for_sigma, sigma_sgd_list)

    pop_risk_list, ts_list, P_vals_list = zip(*results)
    pop_risk_list = list(pop_risk_list)
    ts_list       = list(ts_list)
    P_vals_list   = list(P_vals_list)

    # Save data
    out_dir = folder_name
    os.makedirs(out_dir, exist_ok=True)
    save_data_path = os.path.join(out_dir, "risk_data.npz")
    save_results_npz(ts_list, pop_risk_list, P_vals_list, sigma_sgd_list, file_name=save_data_path)

    # Plot
    plot_risk_Volterra(pop_risk_list, ts_list, P_vals_list, sigma_sgd_list)
