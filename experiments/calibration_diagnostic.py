"""Calibration / resolution diagnostic for the goal totals markets.

The dispersion sweep left every goal market marginally ABOVE its base-rate Brier.
That can mean two very different things:
  - the model has real match-to-match signal (resolution) but is mis-calibrated
    (systematic bias) -> a cheap calibration layer unlocks it; or
  - the model has ~no resolution -> it cannot beat the league average no matter how
    it's calibrated, and only better FEATURES (pace/tempo) can help.

This script tells them apart. For each market it reports the Murphy decomposition of
the Brier score (Brier = reliability - resolution + uncertainty):
  - uncertainty = base-rate Brier (the bar to beat; intrinsic to the market)
  - reliability = miscalibration (lower is better; calibration removes this)
  - resolution  = discrimination / usable signal (higher is better; features add this)
and the out-of-sample Brier after 5-fold isotonic and Platt calibration, so we can see
whether calibration alone crosses the base-rate bar.

Uses the fitted GOAL_DISPERSION and the cached point-in-time xG from fit_dispersion.

Usage:
  python experiments/calibration_diagnostic.py epl_2024
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from models.poisson import GOAL_DISPERSION
from fit_dispersion import simulate_markets, load_or_build_cache, GOAL_LINES

SEED = 42


def build_predictions(rows):
    """Predicted prob + realized outcome for each market, at the shipped dispersion."""
    rng = np.random.default_rng(SEED)
    markets = ["BTTS"] + [f"Over {l}" for l in GOAL_LINES]
    preds = {m: [] for m in markets}
    ys = {m: [] for m in markets}

    for r in rows:
        _, _, _, p_btts, overs = simulate_markets(r["home_xg"], r["away_xg"], GOAL_DISPERSION, rng)
        hs, as_ = r["home_score"], r["away_score"]

        preds["BTTS"].append(p_btts)
        ys["BTTS"].append(1.0 if (hs > 0 and as_ > 0) else 0.0)
        for l in GOAL_LINES:
            preds[f"Over {l}"].append(overs[l])
            ys[f"Over {l}"].append(1.0 if (hs + as_ > l) else 0.0)

    return {m: np.array(preds[m]) for m in markets}, {m: np.array(ys[m]) for m in markets}


def murphy_decomposition(preds, outcomes, n_bins=10):
    o_bar = outcomes.mean()
    uncertainty = o_bar * (1 - o_bar)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(preds, edges[1:-1]), 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        mask = idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        p_bar_k = preds[mask].mean()
        o_bar_k = outcomes[mask].mean()
        reliability += nk * (p_bar_k - o_bar_k) ** 2
        resolution += nk * (o_bar_k - o_bar) ** 2
    n = len(preds)
    return reliability / n, resolution / n, uncertainty


def cv_calibrated_brier(preds, outcomes, method, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    oof = np.zeros_like(preds, dtype=float)
    for tr, te in kf.split(preds):
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(preds[tr], outcomes[tr])
            oof[te] = model.predict(preds[te])
        else:  # platt
            model = LogisticRegression()
            model.fit(preds[tr].reshape(-1, 1), outcomes[tr])
            oof[te] = model.predict_proba(preds[te].reshape(-1, 1))[:, 1]
    return np.mean((oof - outcomes) ** 2)


def main(competition_id):
    rows = load_or_build_cache(competition_id, refresh=False)
    if not rows:
        print("No cached matches; run fit_dispersion first.")
        return

    preds, ys = build_predictions(rows)
    markets = ["BTTS"] + [f"Over {l}" for l in GOAL_LINES]

    print(f"\nCalibration / resolution diagnostic — {competition_id} ({len(rows)} matches, d={GOAL_DISPERSION})")
    print("Brier = reliability - resolution + uncertainty.  Goal: cross uncertainty (base-rate) bar.\n")

    header = (
        f"{'market':<9} | {'raw':>7} | {'base':>7} | {'reliab':>7} | {'resol':>7} "
        f"| {'iso-cv':>7} | {'platt-cv':>8} | verdict"
    )
    print(header)
    print("-" * len(header))

    for m in markets:
        p, y = preds[m], ys[m]
        raw = np.mean((p - y) ** 2)
        rel, res, unc = murphy_decomposition(p, y)
        iso = cv_calibrated_brier(p, y, "isotonic")
        platt = cv_calibrated_brier(p, y, "platt")
        best_cal = min(iso, platt)

        if res < 0.0010:
            verdict = "NO signal - needs features"
        elif best_cal < unc - 0.0005:
            verdict = "calibration UNLOCKS signal"
        else:
            verdict = "marginal - features likely needed"

        print(
            f"{m:<9} | {raw:>7.4f} | {unc:>7.4f} | {rel:>7.4f} | {res:>7.4f} "
            f"| {iso:>7.4f} | {platt:>8.4f} | {verdict}"
        )

    print("-" * len(header))
    print("\nReading it:")
    print("  * resol (resolution) is the usable signal. Near 0 => the model can't tell")
    print("    high-scoring matches from low-scoring ones; no calibration fixes that.")
    print("  * iso-cv / platt-cv below the 'base' column => calibration alone beats the")
    print("    base rate (out-of-sample) and is worth shipping.")
    print("  * If resolution is tiny AND calibrated Brier ~= base, the gain must come")
    print("    from pace/tempo FEATURES, not post-processing.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/calibration_diagnostic.py <competition_id>")
        sys.exit(1)
    main(sys.argv[1])
