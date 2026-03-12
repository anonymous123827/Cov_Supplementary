import re
import numpy as np
import pandas as pd

def crps_from_quantiles(y, taus, qvals, full=True, grid_n=2001):
    taus = np.asarray(taus, float)
    qvals = np.asarray(qvals, float)

    if full:
        taus_ext = np.concatenate([[0.0], taus, [1.0]])
        q_ext = np.concatenate([[qvals[0]], qvals, [qvals[-1]]])
        tau_grid = np.linspace(0.0, 1.0, grid_n)
        q_tau = np.interp(tau_grid, taus_ext, q_ext)
    else:
        tau_grid = np.linspace(taus.min(), taus.max(), grid_n)
        q_tau = np.interp(tau_grid, taus, qvals)

    u = y - q_tau
    rho = u * (tau_grid - (u < 0).astype(float))
    return 2.0 * np.trapz(rho, tau_grid)

df = pd.read_csv("actual_vs_forecast_chronos-bolt-tiny_9_no_co.csv")

qcols = [c for c in df.columns if re.fullmatch(r"q\d+", c)]
taus = np.array([int(c[1:]) / 100 for c in qcols], float)

order = np.argsort(taus)
taus = taus[order]
qcols = [qcols[i] for i in order]

y = df["y_true"].to_numpy(float)
Q = df[qcols].to_numpy(float)

crps_full = np.array([crps_from_quantiles(y[i], taus, Q[i], full=True) for i in range(len(y))])
crps_central = np.array([crps_from_quantiles(y[i], taus, Q[i], full=False) for i in range(len(y))])

print("Quantiles used:", list(zip(qcols, taus)))
print("Mean CRPS (full 0–1, flat tails):", crps_full.mean())
print("Mean CRPS (central only):", crps_central.mean())
