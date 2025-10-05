import numpy as np
from scipy.interpolate import interp1d

def load_results_npz(file_path='risk_data.npz'):
    with np.load(file_path) as data:
        ts_arr        = data['ts']
        pop_risk_arr  = data['pop_risk']
        P_vals_arr    = data['P_vals']
        sigma_arr     = data['sigma']
    return ts_arr, pop_risk_arr, P_vals_arr, sigma_arr

import os
from parameter_setup import *
file_path = os.path.join(folder_name, 'risk_data.npz')
ts_list, pop_risk_list, P_vals_list, sigma_sgd_list = load_results_npz(file_path)


def _safe_interp1d(ts, ys, kind="cubic", tol=1e-9):
    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)

    m = np.isfinite(ts) & np.isfinite(ys)
    ts, ys = ts[m], ys[m]
    if ts.size == 0:
        raise ValueError("No finite points to interpolate.")

    order = np.argsort(ts)
    ts, ys = ts[order], ys[order]

    t_u = [ts[0]]
    y_sum = [ys[0]]
    cnt = [1]
    for t, y in zip(ts[1:], ys[1:]):
        if abs(t - t_u[-1]) <= tol:
            y_sum[-1] += y
            cnt[-1] += 1
        else:
            t_u.append(t)
            y_sum.append(y)
            cnt.append(1)
    t_u = np.array(t_u, dtype=float)
    y_u = np.array(y_sum, dtype=float) / np.array(cnt, dtype=float)


    k_needed = {"linear": 1, "quadratic": 2, "cubic": 3}.get(kind, 3)
    if t_u.size <= k_needed:
        kind = "linear"

    return interp1d(t_u, y_u, kind=kind, fill_value="extrapolate", assume_sorted=True)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

iteration = [x*d for x in ts_list]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(4, 3.5))



for i, sigma in enumerate(sigma_sgd_list):
    c = colors[i % len(colors)]
    index0=0
    plt.plot(range(index0, len(pop_risk_list[i])),
             pop_risk_list[i][index0:],
             '--',
             label=f"SGD σ={sigma}",
             linewidth=1,
             color=c)
    
    
    mask = iteration[i] >= index0
    plt.plot(iteration[i][mask],
             P_vals_list[i][mask],
             '-',
             label=f"HSGD σ={sigma}",
             linewidth=1.5,
             color=c)


plt.xscale("log")
plt.yscale("log")
plt.xlim(1, None)
# plt.ylim(0, 0.4) 
plt.xlabel("Iteration")
plt.ylabel("Population Risk")


plt.legend()
plt.tight_layout()

save_path = os.path.join(folder_name, "risk_curves.pdf")
plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
plt.show()


##########
ts, P_vals = ts_list[jj], P_vals_list[jj]
interp_P = _safe_interp1d(ts, P_vals, kind="cubic", tol=1e-9)
