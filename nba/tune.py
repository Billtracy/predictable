"""Grid search over model parameters on a tuning window, scored on a holdout.

Score = RMSE of the predicted final total (and margin) against the actual result.
That is a smooth, proper target; tip hit rate is far too noisy to tune on directly
(±1.4% at n=1200). The holdout seasons are only reported, never searched.

Usage:
    python -m nba.tune                      # tune on 2023+2024, holdout 2025+2026
"""
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nba.backtest import walk_forward
from nba.db import load_games
from nba.model import Design, rest_context

TUNE = (2023, 2024)
HOLDOUT = (2025, 2026)
GRID = {
    "half_life": (30.0, 60.0, 120.0, 240.0),
    "prev_season_weight": (0.25, 0.5, 1.0),
    "eff_ridge": (5.0, 10.0, 20.0, 40.0),
    "pace_ridge": (5.0, 20.0),
}

_G = {}


def _init():
    games = load_games()
    ctx = rest_context(games)
    _G.update(games=games, ctx=ctx, design=Design(games, ctx))


def _rmse(recs, key, actual):
    return float(np.sqrt(np.mean([(r[actual] - r[key]) ** 2 for r in recs])))


def score(args):
    seasons, params = args
    recs = walk_forward(_G["games"], seasons, design=_G["design"], context=_G["ctx"], **params)
    return params, _rmse(recs, "m_total", "total"), _rmse(recs, "m_margin", "margin"), len(recs)


def main(argv):
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(combos)} combinations on {TUNE}")
    with ProcessPoolExecutor(initializer=_init) as pool:
        results = list(pool.map(score, [(TUNE, c) for c in combos]))
        results.sort(key=lambda r: r[1])
        print(f"\n{'params':<78} total  margin")
        for p, t, m, n in results[:12]:
            print(f"{str(p):<78} {t:6.3f} {m:6.3f}")
        print("...")
        for p, t, m, n in results[-3:]:
            print(f"{str(p):<78} {t:6.3f} {m:6.3f}")

        best = results[0][0]
        best_margin = min(results, key=lambda r: r[2])[0]
        print(f"\nbest for totals: {best}\nbest for margin: {best_margin}")
        for label, p in (("best", best), ("default", {})):
            _, t, m, n = list(pool.map(score, [(HOLDOUT, p)]))[0]
            print(f"holdout {HOLDOUT} {label:<8} total RMSE {t:6.3f}  margin RMSE {m:6.3f}  (n={n})")


if __name__ == "__main__":
    main(sys.argv[1:])
