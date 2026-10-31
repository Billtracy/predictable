"""Grid-search the UCL model parameters on one season, then check the winner on a
holdout season. Ranked by mean Brier over the over/under lines of the target
(goals: 1.5/2.5/3.5; corners: 7.5-12.5).

Usage (from the repo root):
    python -m ucl.tune                          # goals: tune 25/26, hold out 26/27
    python -m ucl.tune --target corners
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per worker, set before numpy loads

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

from ucl import config as C
from ucl.backtest import CORNER_PARAMS, GOAL_PARAMS, brier, evaluate
from ucl.db import load_matches

HALF_LIVES = (180, 365, 730)
TEAM_RIDGES = (4.0, 8.0, 16.0, 32.0)
LEAGUE_RIDGES = (2.0, 5.0, 10.0)

_MATCHES = None


def _init():
    global _MATCHES
    _MATCHES = load_matches()


def score(args):
    season, target, params = args
    kw = {"goal_params": params} if target == "goals" else {"corner_params": params}
    r = evaluate(season, matches=_MATCHES, targets=(target,), **kw)
    lines = r["goal_lines"] if target == "goals" else r["corner_lines"]
    briers = {line: brier(pairs) for line, pairs in lines.items()}
    tips = r["goal_tips"] if target == "goals" else r["corner_tips"]
    return {"params": params, "mean": sum(briers.values()) / len(briers), "briers": briers,
            "tip_hit": sum(w for _, _, w, _ in tips) / len(tips), "n": len(tips)}


def base_rates(season, target, matches):
    cols = ("home_goals", "away_goals") if target == "goals" else ("home_corners", "away_corners")
    lines = (1.5, 2.5, 3.5) if target == "goals" else C.CORNER_LINES
    totals = [m[cols[0]] + m[cols[1]] for m in matches
              if m["competition"] == C.UCL and m["season"] == season and m["stage"] != "qualification"
              and m["status"] == "FT" and m[cols[0]] is not None]
    out = {}
    for line in lines:
        rate = sum(t > line for t in totals) / len(totals)
        out[line] = rate * (1 - rate)
    return out


def fmt(r):
    hl, tr, lr = r["params"]
    b = "  ".join(f"O{k} {v:.4f}" for k, v in r["briers"].items())
    return f"hl {hl:>5} team {tr:>4} league {lr:>4} | mean {r['mean']:.4f} [{b}] | tip hit {r['tip_hit'] * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("goals", "corners"), default="goals")
    ap.add_argument("--tune", type=int, default=2025)
    ap.add_argument("--holdout", type=int, default=C.CURRENT_SEASON)
    a = ap.parse_args()

    grid = [(a.tune, a.target, p) for p in itertools.product(HALF_LIVES, TEAM_RIDGES, LEAGUE_RIDGES)]
    with ProcessPoolExecutor(initializer=_init) as pool:
        results = sorted(pool.map(score, grid), key=lambda r: r["mean"])
        matches = load_matches()
        base = base_rates(a.tune, a.target, matches)
        print(f"{a.target}: tuning on {a.tune}/{(a.tune + 1) % 100:02d}, {results[0]['n']} matches, {len(grid)} configs")
        print("base-rate Briers: " + "  ".join(f"O{k} {v:.4f}" for k, v in base.items())
              + f" | mean {sum(base.values()) / len(base):.4f}\n")
        for r in results[:8]:
            print("  " + fmt(r))
        print("  ...\n  " + fmt(results[-1]) + "   (worst)")

        default = GOAL_PARAMS if a.target == "goals" else CORNER_PARAMS
        check = [results[0]["params"]] + ([default] if results[0]["params"] != default else [])
        hold = list(pool.map(score, [(a.holdout, a.target, p) for p in check]))

    base = base_rates(a.holdout, a.target, matches)
    print(f"\nHoldout {a.holdout}/{(a.holdout + 1) % 100:02d}: {hold[0]['n']} matches")
    print("base-rate Briers: " + "  ".join(f"O{k} {v:.4f}" for k, v in base.items())
          + f" | mean {sum(base.values()) / len(base):.4f}")
    for label, r in zip(("best", "current default"), hold):
        print(f"  {label:<16}" + fmt(r))


if __name__ == "__main__":
    main()
