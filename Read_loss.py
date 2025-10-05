import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------- Config ----------
from parameter_setup import *
file_map = {
    sigma: os.path.join(folder_name, f"loss_curves_{sigma}.csv")
    for sigma in sigma_sgd_list
}

T = n/d
num_pts = len(timepoints)
x_vals = timepoints * d  # "iterations"


def load_y_values(csv_path):
    
    df = pd.read_csv(csv_path)
    wanted = [f"t{i}" for i in range(num_pts)]
    if all(col in df.columns for col in wanted):
        y = df[wanted].iloc[0].to_numpy(dtype=float)
        return y
    
    df2 = pd.read_csv(csv_path, index_col=0)
    first_value_col = df2.columns[0]
    if all(k in df2.index for k in wanted):
        y = df2.loc[wanted, first_value_col].to_numpy(dtype=float)
        return y
   
    df3 = pd.read_csv(csv_path, header=None)
    arr = df3.values.flatten()
    if arr.size >= num_pts:
        return arr[:num_pts].astype(float)
    raise ValueError(f"Could not parse y-values from {csv_path}. Expected columns t0..t14 or a single 15-number row.")

# ---------- Load all series ----------
series = {}
for sigma, path in file_map.items():
    y = load_y_values(path)
    series[sigma] = y

# ---------- Plot ----------
ax=plt.figure(figsize=(4, 3.5))


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, sigma in enumerate(sorted(series.keys(), reverse=False)):
    c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
    plt.plot(x_vals, series[sigma], '-', linewidth=1.5, label=f"HSGD σ={sigma}", color=c)

plt.yscale("log")
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Rényi-DP loss")

ax = plt.gca() 


ax.yaxis.set_major_locator(ticker.FixedLocator([1e-3]))
ax.yaxis.set_minor_locator(ticker.NullLocator())  # kill minor ticks
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())  # shows 10^{-3}

plt.xlim(1, None)
plt.legend()
plt.tight_layout()

save_path = os.path.join(folder_name, "loss_curves.pdf")
plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
plt.show()

