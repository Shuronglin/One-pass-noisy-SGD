import numpy as np
import os

n = 1500   # training samples
d = 1000  # feature dimension
T = n/d
sigma_sgd_list = [1, 1.25, 1.5] #DP noise
jj = 2 # index of sigma_sgd_list for running privacy loss
clipping_list = [3,5,7]

    
Sigma = np.eye(d)/(12*d)                              
sigma2 = 0.01/d
decay_rate = 0.000
np.random.seed(2)
xtilde = np.random.uniform(0, 1/np.sqrt(d), size=d)
# Initialize SGD
x0 = np.random.normal(0, 2, size=d)

gamma = 0.05  # constant learning rate
delta_reg = 0.10  # regularization strength
gamma_sde = gamma*d
alpha = 2.0  # Renyi order

A_matrix = np.random.uniform(0, np.sqrt(1/d), size=(n, d))
xi = np.random.normal(0, np.sqrt(sigma2), size=n)
xi = np.clip(xi, -3*np.sqrt(sigma2), 3*np.sqrt(sigma2))
b_vec = A_matrix @ xtilde + xi
cc = 1/(12*d)



# Create folder and save PDF
sigma_str = "-".join(str(s) for s in sigma_sgd_list)
folder_name = f"n{n}_d{d}_sigma{sigma_str}_eta{gamma}"
os.makedirs(folder_name, exist_ok=True)

## loss
n_top_pairs = 1
timepoints = np.logspace(
    np.log10(0.001),
    np.log10(T),
    num=30
)


