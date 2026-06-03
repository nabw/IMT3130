"""
Computes empirical convergence rates for finite difference schemes and outputs results to a CSV.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

Ns = [round(10**exp) for exp in np.linspace(1,8.5,15)]

fig, ax = plt.subplots(figsize=(3,3), dpi=150)

fderr = []
bderr = []
cderr = []
hs = []

for N in Ns:
    xs = np.linspace(0, 1, N)
    h = xs[1] - xs[0]
    hs.append(h)
    ys = np.sin(2*np.pi*xs)
    true_deriv = 2*np.pi*np.cos(2*np.pi*xs)

    # forward difference

    fd = []
    bd = []
    cd = []

    t0 = time.time()

    # Forward difference (excluding last point)
    fd = (ys[1:] - ys[:-1]) / h

    # Backward difference (excluding first point)
    bd = (ys[1:] - ys[:-1]) / h

    # Central difference (excluding first and last points)
    cd = (ys[2:] - ys[:-2]) / (2 * h)
    
    fderr.append(np.max(np.abs(fd - true_deriv[:-1])))
    bderr.append(np.max(np.abs(bd - true_deriv[1:])))
    cderr.append(np.max(np.abs(cd - true_deriv[1:-1])))
    print(f"done with N = {N} in {time.time() - t0} s.")

ax.loglog(hs, fderr,  label='forward')
ax.loglog(hs, bderr, '--', label='backward')
ax.loglog(hs, cderr, label='centered')
ax.set_ylabel("error sup norm")
ax.set_xlabel("h")

plt.legend()
plt.tight_layout()
plt.show()

results = {
    'h': hs,
    'fd': fderr,
    'bd': bderr,
    'cd': cderr
}

pd.DataFrame(results).to_csv('convergence_rates_FD.csv', index=False)
