"""Fit the goal-model dispersion parameter.

The live model (models/poisson.py) injects per-match noise whose std comes from
`calculate_unpredictability(fifa_rank)`. For club football FIFA rank is absent, so
that std is an unfit guess (~0.5) that the seeded backtest showed hurts the very
markets we care about (BTTS / over-under).

This experiment replaces that guess with a single dispersion value and grid-searches
it. Because home_xg/away_xg are deterministic (predictors/goals.py:50-51), we cache
them once per match (the expensive part) and then re-simulate cheaply for each
candidate dispersion. The re-simulation faithfully mirrors simulate_match's goal
path: adj = max(0.01, N(xg, d)) -> Poisson -> Dixon-Coles reweight of the four
low-score cells -> derive W/D/L, BTTS and over-lines from that distribution.

Usage:
  python experiments/fit_dispersion.py epl_2024            # cache (if needed) + sweep
  python experiments/fit_dispersion.py epl_2024 --refresh  # force rebuild of xG cache
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from models.team_ratings import build_context
from models.poisson import DIXON_COLES_RHO

GOAL_LINES = (1.5, 2.5, 3.5)
DISPERSION_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
N_SIMS = 20000
SEED = 42

CACHE_DIR = os.environ.get(
    "PREDICTABLE_SCRATCH",
    "/tmp/claude-1000/-home-elonjobs-Documents-development-python-Predictable/"
    "58e66ce7-099e-4c5e-b193-390cf1221073/scratchpad",
)

# Current live-model numbers from the full-data backtest, for side-by-side reference
# (FIFA-rank noise injection). Source: backtest.py epl_2024 run.
CURRENT_MODEL = {
    "BTTS": 0.2551,
    "Over 1.5": 0.1646,
    "Over 2.5": 0.2527,
    "Over 3.5": 0.2291,
    "rps": 0.2107,
    "acc": 0.5145,
}


def cache_path(competition_id):
    return os.path.join(CACHE_DIR, f"xg_cache_{competition_id}.json")


def build_xg_cache(competition_id):
    """Compute point-in-time xG for every completed match (leak-free) and cache it.

    Mirrors predictors/goals.predict_goals lines 33-51 but skips the Monte Carlo,
    so it is fast. Matches with no prior data (build_context is None) are skipped,
    exactly as the backtest skips them.
    """
    with get_db() as db:
        cur = db.cursor()
        cur.execute(
            """
            SELECT m.id, m.date, m.home_team_id, m.away_team_id,
                   m.home_score, m.away_score
            FROM matches m
            WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN')
            ORDER BY m.date ASC
            """,
            (competition_id,),
        )
        matches = cur.fetchall()

    rows = []
    skipped = 0
    for i, m in enumerate(matches):
        ctx = build_context(competition_id, m["home_team_id"], m["away_team_id"], m["date"])
        if ctx is None or not ctx["home_ratings"] or not ctx["away_ratings"]:
            skipped += 1
            continue

        g = ctx["avgs"]["avg_goals_per_team"]
        home_xg = ctx["home_ratings"]["attack_goals"] * ctx["away_ratings"]["defense_goals"] * g * ctx["home_advantage"]
        away_xg = ctx["away_ratings"]["attack_goals"] * ctx["home_ratings"]["defense_goals"] * g * ctx["away_advantage"]

        rows.append(
            {
                "home_xg": home_xg,
                "away_xg": away_xg,
                "home_score": m["home_score"],
                "away_score": m["away_score"],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  cached {i + 1}/{len(matches)} matches...")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path(competition_id), "w") as f:
        json.dump(rows, f)
    print(f"Cached xG for {len(rows)} matches (skipped {skipped} with no prior data).")
    return rows


def load_or_build_cache(competition_id, refresh):
    path = cache_path(competition_id)
    if refresh or not os.path.exists(path):
        print("Building point-in-time xG cache (one-time, ~30-60s)...")
        return build_xg_cache(competition_id)
    with open(path) as f:
        rows = json.load(f)
    print(f"Loaded xG cache: {len(rows)} matches ({path}).")
    return rows


def simulate_markets(home_xg, away_xg, dispersion, rng):
    """Faithful vectorized copy of simulate_match's goal path. Returns predicted
    probabilities: (p_home, p_draw, p_away, p_btts, {line: p_over})."""
    adj_home = np.maximum(0.01, rng.normal(home_xg, dispersion, N_SIMS))
    adj_away = np.maximum(0.01, rng.normal(away_xg, dispersion, N_SIMS))
    hg = rng.poisson(adj_home)
    ag = rng.poisson(adj_away)

    # Dixon-Coles reweighting of the four low-score cells (then implicit renormalize
    # via weighted means). tau matches models/poisson.dixon_coles_tau.
    rho = DIXON_COLES_RHO
    w = np.ones(N_SIMS)
    w[(hg == 0) & (ag == 0)] = 1.0 - home_xg * away_xg * rho
    w[(hg == 0) & (ag == 1)] = 1.0 + home_xg * rho
    w[(hg == 1) & (ag == 0)] = 1.0 + away_xg * rho
    w[(hg == 1) & (ag == 1)] = 1.0 - rho
    wsum = w.sum()

    p_home = w[hg > ag].sum() / wsum
    p_away = w[ag > hg].sum() / wsum
    p_draw = w[hg == ag].sum() / wsum
    p_btts = w[(hg > 0) & (ag > 0)].sum() / wsum

    total = hg + ag
    overs = {line: w[total > line].sum() / wsum for line in GOAL_LINES}
    return p_home, p_draw, p_away, p_btts, overs


def _rps(p_home, p_draw, p_away, actual_idx):
    """Ranked Probability Score for the ordered (home, draw, away) market."""
    cum_p = 0.0
    cum_a = 0.0
    total = 0.0
    probs = (p_home, p_draw, p_away)
    for i in range(2):
        cum_p += probs[i]
        cum_a += 1.0 if i == actual_idx else 0.0
        total += (cum_p - cum_a) ** 2
    return total / 2.0


def evaluate(rows, dispersion):
    rng = np.random.default_rng(SEED)
    n = len(rows)

    brier = {"BTTS": 0.0, "Over 1.5": 0.0, "Over 2.5": 0.0, "Over 3.5": 0.0}
    yes_count = {k: 0 for k in brier}
    rps_sum = 0.0
    correct = 0

    for r in rows:
        p_home, p_draw, p_away, p_btts, overs = simulate_markets(
            r["home_xg"], r["away_xg"], dispersion, rng
        )
        hs, as_ = r["home_score"], r["away_score"]

        # W/D/L
        if hs > as_:
            actual_idx = 0
        elif as_ > hs:
            actual_idx = 2
        else:
            actual_idx = 1
        rps_sum += _rps(p_home, p_draw, p_away, actual_idx)
        pred_idx = int(np.argmax([p_home, p_draw, p_away]))
        if pred_idx == actual_idx:
            correct += 1

        # Binary goal markets
        y_btts = 1.0 if (hs > 0 and as_ > 0) else 0.0
        brier["BTTS"] += (p_btts - y_btts) ** 2
        yes_count["BTTS"] += y_btts

        tot = hs + as_
        for line in GOAL_LINES:
            key = f"Over {line}"
            y = 1.0 if tot > line else 0.0
            brier[key] += (overs[line] - y) ** 2
            yes_count[key] += y

    result = {k: brier[k] / n for k in brier}
    result["rps"] = rps_sum / n
    result["acc"] = correct / n
    result["_yes_rate"] = {k: yes_count[k] / n for k in brier}
    result["goal_avg"] = np.mean([result["BTTS"], result["Over 1.5"], result["Over 2.5"], result["Over 3.5"]])
    return result


def base_rate_brier(rows):
    """Brier you'd get by emitting the constant marginal (yes-rate) for each market.
    This is the bar every market must beat to add value."""
    n = len(rows)
    yes = {"BTTS": 0, "Over 1.5": 0, "Over 2.5": 0, "Over 3.5": 0}
    for r in rows:
        hs, as_ = r["home_score"], r["away_score"]
        yes["BTTS"] += 1 if (hs > 0 and as_ > 0) else 0
        for line in GOAL_LINES:
            yes[f"Over {line}"] += 1 if (hs + as_ > line) else 0
    return {k: (v / n) * (1 - v / n) for k, v in yes.items()}


def main(competition_id, refresh):
    rows = load_or_build_cache(competition_id, refresh)
    if not rows:
        print("No cached matches; aborting.")
        return

    base = base_rate_brier(rows)
    markets = ["BTTS", "Over 1.5", "Over 2.5", "Over 3.5"]

    print(f"\nDispersion sweep on {competition_id}  ({len(rows)} matches, {N_SIMS} sims/match)")
    print("Lower Brier = better; a market only adds value if it beats its base-rate row.\n")

    header = f"{'dispersion':>10} | " + " | ".join(f"{m:>9}" for m in markets) + f" | {'goal-avg':>8} | {'W/D/L RPS':>9} | {'acc':>6}"
    print(header)
    print("-" * len(header))

    print(f"{'base-rate':>10} | " + " | ".join(f"{base[m]:>9.4f}" for m in markets) + f" | {np.mean(list(base.values())):>8.4f} | {'--':>9} | {'--':>6}")
    print(f"{'CURRENT':>10} | " + " | ".join(f"{CURRENT_MODEL[m]:>9.4f}" for m in markets) + f" | {np.mean([CURRENT_MODEL[m] for m in markets]):>8.4f} | {CURRENT_MODEL['rps']:>9.4f} | {CURRENT_MODEL['acc']:>6.3f}")
    print("-" * len(header))

    results = {}
    for d in DISPERSION_GRID:
        res = evaluate(rows, d)
        results[d] = res
        print(f"{d:>10.2f} | " + " | ".join(f"{res[m]:>9.4f}" for m in markets) + f" | {res['goal_avg']:>8.4f} | {res['rps']:>9.4f} | {res['acc']:>6.3f}")

    best = min(results, key=lambda d: results[d]["goal_avg"])
    print("-" * len(header))
    print(f"\nBest dispersion by goal-market average Brier: {best:.2f}  (goal-avg {results[best]['goal_avg']:.4f})")
    print("Per-market winners:")
    for m in markets:
        bd = min(results, key=lambda d: results[d][m])
        beats = "beats" if results[bd][m] < base[m] else "STILL ABOVE"
        print(f"  {m:<9} best d={bd:.2f}  Brier {results[bd][m]:.4f}  ({beats} base-rate {base[m]:.4f})")
    print(f"\nW/D/L RPS at best d: {results[best]['rps']:.4f}  (current {CURRENT_MODEL['rps']:.4f}, uniform 0.222)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/fit_dispersion.py <competition_id> [--refresh]")
        sys.exit(1)
    main(sys.argv[1], "--refresh" in sys.argv)
