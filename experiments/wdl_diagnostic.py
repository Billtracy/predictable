"""Where does the W/D/L (1X2) model win and lose?

The outcome side is the only one with real signal (RPS beats uniform), so it's worth
understanding before improving. This reports:
  - the 3x3 confusion matrix (predicted argmax vs actual) and how often the model
    ever picks DRAW (Poisson/xG models are notoriously draw-blind);
  - per-class Murphy decomposition (reliability=bias, resolution=signal) and mean
    predicted prob vs actual frequency (calibration bias per class);
  - overall accuracy, multiclass log loss, and RPS.

Uses the fitted GOAL_DISPERSION and the cached point-in-time xG from fit_dispersion.

Usage:
  python experiments/wdl_diagnostic.py epl_2024
"""

import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.poisson import GOAL_DISPERSION
from fit_dispersion import simulate_markets, load_or_build_cache

SEED = 42
CLASSES = ("home", "draw", "away")


def actual_idx(hs, as_):
    if hs > as_:
        return 0
    if as_ > hs:
        return 2
    return 1


def murphy(preds, outcomes, n_bins=10):
    preds = np.asarray(preds)
    outcomes = np.asarray(outcomes)
    o_bar = outcomes.mean()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(preds, edges[1:-1]), 0, n_bins - 1)
    rel = res = 0.0
    for k in range(n_bins):
        mask = idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        rel += nk * (preds[mask].mean() - outcomes[mask].mean()) ** 2
        res += nk * (outcomes[mask].mean() - o_bar) ** 2
    n = len(preds)
    return rel / n, res / n, o_bar * (1 - o_bar)


def main(competition_id):
    rows = load_or_build_cache(competition_id, refresh=False)
    if not rows:
        print("No cache; run fit_dispersion first.")
        return

    rng = np.random.default_rng(SEED)
    P = np.zeros((len(rows), 3))   # predicted probs
    A = np.zeros(len(rows), dtype=int)  # actual class idx
    for i, r in enumerate(rows):
        p_home, p_draw, p_away, _, _ = simulate_markets(r["home_xg"], r["away_xg"], GOAL_DISPERSION, rng)
        s = p_home + p_draw + p_away or 1.0
        P[i] = [p_home / s, p_draw / s, p_away / s]
        A[i] = actual_idx(r["home_score"], r["away_score"])

    pred = P.argmax(axis=1)
    n = len(rows)

    print(f"\nW/D/L diagnostic — {competition_id} ({n} matches, d={GOAL_DISPERSION})\n")

    # Confusion matrix
    print("Confusion (rows=actual, cols=predicted):")
    print(f"{'':>8} " + " ".join(f"{c:>6}" for c in CLASSES) + f"  {'total':>6}")
    conf = np.zeros((3, 3), dtype=int)
    for a, p in zip(A, pred):
        conf[a, p] += 1
    for a in range(3):
        print(f"{CLASSES[a]:>8} " + " ".join(f"{conf[a, p]:>6}" for p in range(3)) + f"  {conf[a].sum():>6}")
    print(f"{'pred tot':>8} " + " ".join(f"{conf[:, p].sum():>6}" for p in range(3)))
    acc = (pred == A).mean()
    print(f"\nAccuracy: {acc*100:.1f}%   |   times DRAW was picked: {int((pred==1).sum())}/{n} "
          f"({(pred==1).mean()*100:.1f}%)   |   actual draws: {int((A==1).sum())} ({(A==1).mean()*100:.1f}%)")

    # Per-class calibration + decomposition
    print("\nPer-class (one-vs-rest):")
    print(f"{'class':>6} | {'base':>6} | {'mean p':>6} | {'bias':>7} | {'reliab':>7} | {'resol':>7}")
    print("-" * 52)
    for c in range(3):
        y = (A == c).astype(float)
        p = P[:, c]
        rel, res, _ = murphy(p, y)
        base = y.mean()
        print(f"{CLASSES[c]:>6} | {base:>6.3f} | {p.mean():>6.3f} | {p.mean()-base:>+7.3f} | {rel:>7.4f} | {res:>7.4f}")

    # Overall proper scores
    eps = 1e-15
    logloss = -np.mean([math.log(max(P[i, A[i]], eps)) for i in range(n)])
    rps = 0.0
    for i in range(n):
        cp = ca = 0.0
        for k in range(2):
            cp += P[i, k]
            ca += 1.0 if A[i] == k else 0.0
            rps += (cp - ca) ** 2
    rps /= 2 * n
    print(f"\nLog loss: {logloss:.4f} (uniform 1.099)   RPS: {rps:.4f} (uniform 0.222)")

    print("\nReading it: a 'bias' far from 0 means that class is systematically over/under-")
    print("predicted (fix with calibration). 'resol' near 0 for a class means the model")
    print("can't identify those games. Draw resolution is usually ~0 — the classic problem.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/wdl_diagnostic.py <competition_id>")
        sys.exit(1)
    main(sys.argv[1])
