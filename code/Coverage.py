from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np

Z_90 = 1.2815515655446004
Z_975 = 1.959963984540054
SCALE_95_FROM_80 = Z_975 / Z_90

Q025_ALIASES = ["q2.5", "q02_5", "q025", "q2_5", "p2.5", "p025"]
Q975_ALIASES = ["q97.5", "q97_5", "q975", "q97_5", "p97.5", "p975"]

def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)

    mask = np.isfinite(y) & np.isfinite(lo2) & np.isfinite(hi2)
    if mask.sum() == 0:
        return float("nan")
    inside = (y[mask] >= lo2[mask]) & (y[mask] <= hi2[mask])
    return float(inside.mean())

def compute_coverage(path: str) -> dict:
    df = pd.read_csv(path)

    if "y_true" not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: missing required column 'y_true'")

    if "q10" not in df.columns or "q90" not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: need 'q10' and 'q90' for 80% coverage")

    y = df["y_true"].to_numpy(dtype=float)
    q10 = df["q10"].to_numpy(dtype=float)
    q90 = df["q90"].to_numpy(dtype=float)

    cov80 = _coverage(y, q10, q90)

    q025_col = _first_existing(df, Q025_ALIASES)
    q975_col = _first_existing(df, Q975_ALIASES)

    if q025_col and q975_col:
        lo95 = df[q025_col].to_numpy(dtype=float)
        hi95 = df[q975_col].to_numpy(dtype=float)
        cov95 = _coverage(y, lo95, hi95)
        cov95_note = f"true 95% from [{q025_col}, {q975_col}]"
    else:
        mu = (q10 + q90) / 2.0
        half80 = (q90 - q10) / 2.0
        half95 = half80 * SCALE_95_FROM_80
        lo95 = mu - half95
        hi95 = mu + half95
        cov95 = _coverage(y, lo95, hi95)
        cov95_note = "approx 95% from [q10, q90] (Normal/symmetric assumption)"

    return {
        "file": os.path.basename(path),
        "n_rows": int(len(df)),
        "coverage_80": cov80,
        "coverage_95": cov95,
        "coverage_95_note": cov95_note,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more CSV paths")
    ap.add_argument("--out", default="coverage_summary.csv", help="Output summary CSV")
    args = ap.parse_args()

    rows = []
    for p in args.inputs:
        rows.append(compute_coverage(p))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    with pd.option_context("display.max_colwidth", 120):
        print(out_df)

    print(f"\nSaved: {args.out}")
