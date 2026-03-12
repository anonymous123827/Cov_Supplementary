
from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_QCOL_RE = re.compile(r"^(?:q|p)(\d+(?:\.\d+)?)$", re.IGNORECASE)

def detect_y_col(df: pd.DataFrame) -> str:
    if "y_true" in df.columns:
        return "y_true"
    # fallback options
    for cand in ["y", "actual", "truth", "target"]:
        if cand in df.columns:
            return cand
    # case-insensitive search
    lower_map = {c.lower(): c for c in df.columns}
    for cand in ["y_true", "y", "actual", "truth", "target"]:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError("Could not find ground-truth column. Expected 'y_true' or similar.")


def detect_quantiles(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    q_cols: List[str] = []
    taus: List[float] = []

    for c in df.columns:
        m = _QCOL_RE.match(str(c).strip())
        if not m:
            continue
        v = float(m.group(1))
        tau = v / 100.0 if v > 1.0 else v  # allow q10 (->0.10) or q0.1 (->0.1)
        if 0.0 <= tau <= 1.0:
            q_cols.append(c)
            taus.append(tau)

    if len(q_cols) < 2:
        raise ValueError("Need at least 2 quantile columns like q10 and q90.")

    order = np.argsort(taus)
    q_cols = [q_cols[i] for i in order]
    taus = np.array([taus[i] for i in order], dtype=float)
    return q_cols, taus


def compute_pit_from_quantiles(
    y: np.ndarray,
    Q: np.ndarray,
    taus: np.ndarray,
    seed: int = 0,
    enforce_monotone: bool = True,
    tail_randomize: bool = True,
) -> np.ndarray:
    """
    PIT via piecewise-linear inversion of the quantile function.

    - If y is below the minimum available quantile: PIT in [0, min_tau]
    - If y is above the maximum available quantile: PIT in [max_tau, 1]

    If tail_randomize=True, PIT is uniform in those tail ranges (randomized PIT tails).
    Otherwise, PIT is clamped to the boundary taus.
    """
    y = np.asarray(y, dtype=float)
    Q = np.asarray(Q, dtype=float)
    taus = np.asarray(taus, dtype=float)

    if enforce_monotone:
        # fix quantile crossing artifacts
        Q = np.maximum.accumulate(Q, axis=1)

    rng = np.random.default_rng(seed)
    pit = np.empty(len(y), dtype=float)

    min_tau = float(taus[0])
    max_tau = float(taus[-1])

    for i in range(len(y)):
        yi = y[i]
        qi = Q[i]

        # lower tail
        if yi <= qi[0]:
            pit[i] = rng.uniform(0.0, min_tau) if tail_randomize else min_tau
            continue

        # upper tail
        if yi >= qi[-1]:
            pit[i] = rng.uniform(max_tau, 1.0) if tail_randomize else max_tau
            continue

        # interior: find interval qi[j] <= yi <= qi[j+1]
        j = np.searchsorted(qi, yi, side="right") - 1
        j = int(np.clip(j, 0, len(qi) - 2))

        ql, qu = qi[j], qi[j + 1]
        tl, tu = taus[j], taus[j + 1]

        if qu == ql:
            pit[i] = 0.5 * (tl + tu)
        else:
            w = (yi - ql) / (qu - ql)
            pit[i] = tl + w * (tu - tl)

    return pit


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    y = np.arange(1, n + 1, dtype=float) / n
    return x, y

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more CSV files.")
    ap.add_argument("--rename-pressure-to-IOCI", action="store_true",
                    help="Create '_IOCI' copies of files that contain '_pressure' in the filename.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for PIT tail randomization.")
    ap.add_argument("--outfig", type=str, default="pit_reliability.png", help="Output figure filename (PNG/PDF).")
    ap.add_argument("--title", type=str, default="PIT Reliability Diagram (ECDF of PIT vs Uniform)",
                    help="Figure title.")
    ap.add_argument("--labels-from-filename", action="store_true",
                    help="Use filename stem as legend label (default).")
    ap.add_argument("--bins", type=int, default=10,
                    help="If --also-hist is set, number of bins for histogram.")
    ap.add_argument("--also-hist", action="store_true",
                    help="Additionally save an overlaid PIT histogram figure.")
    ap.add_argument("--outhist", type=str, default="pit_hist.png",
                    help="Output filename for overlaid PIT histogram (only if --also-hist).")
    ap.add_argument("--no-tail-randomize", action="store_true",
                    help="Disable randomized PIT tails; clamp to boundary taus instead.")

    args = ap.parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.inputs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    pits: Dict[str, np.ndarray] = {}
    for p in paths:
        df = pd.read_csv(p)
        y_col = detect_y_col(df)
        q_cols, taus = detect_quantiles(df)

        y = df[y_col].to_numpy(dtype=float)
        Q = df[q_cols].to_numpy(dtype=float)

        pit = compute_pit_from_quantiles(
            y=y,
            Q=Q,
            taus=taus,
            seed=args.seed,
            enforce_monotone=True,
            tail_randomize=(not args.no_tail_randomize),
        )

        label = p.stem  # default
        pits[label] = pit

    # ---- Reliability diagram (one combined PIT diagram) ----
    plt.figure(figsize=(7.5, 5.5))
    xx = np.linspace(0, 1, 200)
    plt.plot(xx, xx, linestyle="--", label="Ideal")

    for label, pit in pits.items():
        x_ecdf, y_ecdf = ecdf(pit)
        plt.step(x_ecdf, y_ecdf, where="post", label=label)

    plt.title(args.title)
    plt.xlabel("PIT")
    plt.ylabel("CDF")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(args.outfig, dpi=300)
    plt.close()

    # ---- Optional: overlaid PIT histogram ----
    # if args.also_hist:
    #     plt.figure(figsize=(7.5, 5.0))
    #     # Overlaid histograms (density)
    #     for label, pit in pits.items():
    #         plt.hist(pit, bins=args.bins, density=True, histtype="step", linewidth=2, label=label)
    #     plt.axhline(1.0, linestyle="--", label="Ideal")
    #     plt.title("PIT Histogram (Overlaid)")
    #     plt.xlabel("PIT value")
    #     plt.ylabel("Density")
    #     plt.xlim(0, 1)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(args.outhist, dpi=300)
    #     plt.close()

    # Minimal console output
    print("Wrote:", args.outfig)
    if args.also_hist:
        print("Wrote:", args.outhist)

main()