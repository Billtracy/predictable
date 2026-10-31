"""Leak-free walk-forward backtest of the UCL goals and corners models.

For each UCL matchday in the target season, refit on every match strictly before
that day (all competitions), then predict that day's games. Scored on the markets
that matter: over/under goals and over/under corners, plus the EPL-style tips.

Usage (from the repo root):
    python -m ucl.backtest                    # 25/26 season, default params
    python -m ucl.backtest --season 2026      # current season so far
    python -m ucl.backtest --ucl-only         # ablation: ignore domestic leagues
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

from ucl import config as C
from ucl.db import load_matches, team_names
from ucl.model import UCLModel, team_leagues
from ucl.tips import corner_tip, goal_tip, tip_prob, tip_won

GOAL_LINES = (1.5, 2.5, 3.5)
GOAL_TIP_LINES = ("Over 1.5", "Over 2.5", "Under 3.5")
CORNER_TIP_LINES = ("Over 7.5", "Over 8.5", "Under 11.5", "Under 12.5")

GOAL_PARAMS = (C.HALF_LIFE_DAYS, C.TEAM_RIDGE, C.LEAGUE_RIDGE)
CORNER_PARAMS = (C.CORNER_HALF_LIFE_DAYS, C.CORNER_TEAM_RIDGE, C.CORNER_LEAGUE_RIDGE)


def rps(probs, actual_idx):
    cum_p = cum_a = total = 0.0
    for i in range(len(probs) - 1):
        cum_p += probs[i]
        cum_a += 1.0 if i == actual_idx else 0.0
        total += (cum_p - cum_a) ** 2
    return total / (len(probs) - 1)


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def binary_line(name, pairs):
    """pairs: [(p_yes, actual 0/1)]. Brier vs the base-rate ('climatology') Brier,
    the same bar the EPL backtest uses."""
    n = len(pairs)
    if not n:
        return f"  {name:<12} no data"
    rate = sum(y for _, y in pairs) / n
    acc = sum((p > 0.5) == (y == 1) for p, y in pairs) / n * 100
    majority = max(rate, 1 - rate) * 100
    return (f"  {name:<12} Brier {brier(pairs):.4f} (base-rate {rate * (1 - rate):.4f})   "
            f"acc {acc:5.1f}% (majority {majority:5.1f}%)   yes-rate {rate * 100:.1f}%")


def evaluate(season, ucl_only=False, goal_params=GOAL_PARAMS, corner_params=CORNER_PARAMS,
             matches=None, targets=("goals", "corners")):
    """Walk-forward over the season; returns the raw scored predictions."""
    comps = [C.UCL] if ucl_only else list(C.COMPETITIONS)
    matches = load_matches(comps) if matches is None else [m for m in matches if m["competition"] in comps]
    leagues = {} if ucl_only else team_leagues(matches)
    names = team_names()

    # Score the competition proper (league phase onward); qualifiers still train.
    games = [m for m in matches if m["competition"] == C.UCL and m["season"] == season
             and m["stage"] != "qualification" and m["status"] == "FT" and m["home_goals"] is not None]
    if not games:
        raise SystemExit(f"No completed UCL matches for season {season}. Run `python -m ucl.ingest` first.")

    by_day = defaultdict(list)
    for m in games:
        by_day[m["date"][:10]].append(m)

    # Ties: the deciding match says who went through; predict it before leg 1.
    legs = defaultdict(dict)
    for m in games:
        if m["tie_id"] and m["leg"]:
            legs[m["tie_id"]][m["leg"]] = m
    ties_by_day = defaultdict(list)
    for tie in legs.values():
        if 1 in tie and 2 in tie and tie[2]["tie_winner_id"]:
            ties_by_day[tie[1]["date"][:10]].append((tie[1], tie[2]["tie_winner_id"], False))
    for m in games:
        if m["round_name"] == "final" and m["tie_winner_id"]:
            ties_by_day[m["date"][:10]].append((m, m["tie_winner_id"], True))

    r = {"goal_lines": defaultdict(list), "corner_lines": defaultdict(list),
         "goal_tips": [], "corner_tips": [], "goal_totals": [], "corner_totals": [],
         "wdl_rps": [], "tie_pairs": [], "n_days": len(by_day)}
    gmodel = UCLModel(*goal_params, target="goals")
    cmodel = UCLModel(*corner_params, target="corners")

    for day in sorted(by_day):
        as_of = f"{day}T00:00:00+00:00"
        if "goals" in targets:
            gmodel.fit(matches, as_of, leagues)
        if "corners" in targets:
            cmodel.fit(matches, as_of, leagues)
        for m in by_day[day]:
            neutral = bool(m["neutral"])
            if "goals" in targets:
                p = gmodel.predict(m["home_id"], m["away_id"], C.UCL, neutral)
                hg, ag = m["home_goals"], m["away_goals"]
                total = hg + ag
                r["wdl_rps"].append(rps((p["home_win"], p["draw"], p["away_win"]),
                                        0 if hg > ag else 1 if hg == ag else 2))
                r["goal_totals"].append((p["home_xg"] + p["away_xg"], total))
                for line in GOAL_LINES:
                    r["goal_lines"][line].append((p["over"][line], int(total > line)))
                tip = goal_tip(p["home_xg"] + p["away_xg"])
                r["goal_tips"].append((tip, tip_prob(tip, p["over"]), tip_won(tip, total), total))
            if "corners" in targets and m["home_corners"] is not None:
                c = cmodel.predict_corners(m["home_id"], m["away_id"], C.UCL, neutral)
                total = m["home_corners"] + m["away_corners"]
                r["corner_totals"].append((c["total_corners"], total))
                for line in C.CORNER_LINES:
                    r["corner_lines"][line].append((c["over"][line], int(total > line)))
                tip = corner_tip(c["total_corners"])
                r["corner_tips"].append((tip, tip_prob(tip, c["over"]), tip_won(tip, total), total))
        if "goals" in targets:
            for first, winner, single in ties_by_day.get(day, []):
                pa = gmodel.tie_advance_prob(first["home_id"], first["away_id"], single_leg_neutral=single)
                r["tie_pairs"].append((pa, int(winner == first["home_id"]),
                                       names.get(first["home_id"], "?"), names.get(first["away_id"], "?")))
    r["goal_model"], r["corner_model"] = gmodel, cmodel
    return r


def tip_report(name, tips, fixed_lines):
    """Hit rate of the EPL-style tip vs always tipping one fixed line."""
    n = len(tips)
    if not n:
        return [f"  {name}: no data"]
    hits = sum(w for _, _, w, _ in tips)
    expected = sum(p for _, p, _, _ in tips)
    out = [f"  {name}: {hits}/{n} = {hits / n * 100:.1f}% hit   (model expected {expected / n * 100:.1f}%)"]
    counts = Counter(t for t, *_ in tips)
    for tip in sorted(counts, key=lambda t: -counts[t]):
        sub = [t for t in tips if t[0] == tip]
        out.append(f"      {tip:<11} tipped {len(sub):>3}x   hit {sum(w for _, _, w, _ in sub) / len(sub) * 100:5.1f}%"
                   f"   model said {sum(p for _, p, _, _ in sub) / len(sub) * 100:5.1f}%")
    actuals = [a for *_, a in tips]
    always = "   ".join(f"{line} {sum(tip_won(line, a) for a in actuals) / n * 100:.1f}%" for line in fixed_lines)
    out.append(f"      always the same line: {always}")
    return out


def run(season, ucl_only, goal_params, corner_params):
    r = evaluate(season, ucl_only, goal_params, corner_params)
    n = len(r["goal_tips"])
    label = "UCL-only (ablation)" if ucl_only else "domestic + UCL"
    print(f"\nUCL {season}/{(season + 1) % 100:02d} backtest: {n} matches, {r['n_days']} matchdays | {label}")
    print(f"goals params {goal_params} | corners params {corner_params}\n")

    print("Over/under goals (90-minute score)")
    for line in GOAL_LINES:
        print(binary_line(f"Over {line}", r["goal_lines"][line]))
    pred, act = zip(*r["goal_totals"])
    print(f"  total goals: predicted mean {np.mean(pred):.2f}, actual {np.mean(act):.2f}; "
          f"corr {np.corrcoef(pred, act)[0, 1]:.3f}")

    nc = len(r["corner_tips"])
    print(f"\nOver/under corners ({nc} matches with corner counts)")
    if nc:
        for line in C.CORNER_LINES:
            print(binary_line(f"Over {line}", r["corner_lines"][line]))
        pred, act = zip(*r["corner_totals"])
        print(f"  total corners: predicted mean {np.mean(pred):.2f}, actual {np.mean(act):.2f} "
              f"(var/mean {np.var(act) / np.mean(act):.2f}); corr {np.corrcoef(pred, act)[0, 1]:.3f}")

    print("\nTips (EPL rules)")
    for line in tip_report("Goals tip", r["goal_tips"], GOAL_TIP_LINES) + \
            tip_report("Corners tip", r["corner_tips"], CORNER_TIP_LINES):
        print(line)

    ties = r["tie_pairs"]
    print(f"\nAlso: W/D/L RPS {np.mean(r['wdl_rps']):.4f}"
          + (f" | ties Brier {brier([(p, y) for p, y, *_ in ties]):.4f} over {len(ties)} (coin flip 0.2500)" if ties else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--ucl-only", action="store_true")
    ap.add_argument("--half-life", type=float, default=C.HALF_LIFE_DAYS)
    ap.add_argument("--team-ridge", type=float, default=C.TEAM_RIDGE)
    ap.add_argument("--league-ridge", type=float, default=C.LEAGUE_RIDGE)
    ap.add_argument("--corner-half-life", type=float, default=C.CORNER_HALF_LIFE_DAYS)
    ap.add_argument("--corner-team-ridge", type=float, default=C.CORNER_TEAM_RIDGE)
    ap.add_argument("--corner-league-ridge", type=float, default=C.CORNER_LEAGUE_RIDGE)
    a = ap.parse_args()
    run(a.season, a.ucl_only, (a.half_life, a.team_ridge, a.league_ridge),
        (a.corner_half_life, a.corner_team_ridge, a.corner_league_ridge))


if __name__ == "__main__":
    main()
