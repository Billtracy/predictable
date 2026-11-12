"""Compare EPL match-card tip rules on completed fixtures.

Run with ``python -m experiments.tip_selection_backtest``. The optional JSON
report includes the picks and results by line and by displayed matchweek.
"""

import argparse
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import poisson

from experiments.corner_strategy_compare import bet_won, evaluate_lines
from experiments.epl_tip_replay import kickoff_day, load_epl_fixtures, load_epl_history
from predictors.corners import CORNER_STRATEGIES, DEFAULT_CORNER_STRATEGY
from ucl.model import TARGETS, UCLModel


HISTORY_SEASONS = ("epl_2023", "epl_2024", "epl_2025", "epl_2026")
REPLAY_SEASONS = HISTORY_SEASONS[1:]
GOAL_PICKS = ("Over 1.5", "Over 2.5", "Under 3.5")
CORNER_PICKS = {
    "tight": ("Over 7.5", "Over 8.5", "Under 11.5", "Under 12.5"),
    "classic": ("Over 7.5", "Over 8.5", "Under 11.5"),
}
POLICIES = ("current", "highest_probability", "largest_lift", "same_daily_mix", "team_model_same_mix")
BASELINE_WINDOW = 380
MIN_BASELINE_MATCHES = 50
CORNER_TEAM_RIDGE = 32.0


def fixture_day(fixture):
    return kickoff_day(fixture["date"])


def pick_probabilities(rates, picks):
    """Price each candidate line from the combined expected total."""
    thresholds = np.array([float(pick.split()[1]) for pick in picks])
    overs = np.array([pick.startswith("Over ") for pick in picks])
    expected_totals = np.sum(rates, axis=1)[:, None]
    return np.where(overs, poisson.sf(thresholds, expected_totals), poisson.cdf(thresholds, expected_totals))


def assign_picks(probabilities, picks, slots):
    # Each column is one slot for a pick the site would have issued that day.
    _, assigned = linear_sum_assignment(probabilities[:, [picks.index(pick) for pick in slots]], maximize=True)
    return [slots[i] for i in assigned]


def replay(fixtures, history, corner_strategy):
    picks_by_market = {"goals": GOAL_PICKS, "corners": CORNER_PICKS[corner_strategy]}
    dated_history = sorted(
        ((fixture_day(f), f) for f in history),
        key=lambda item: (item[0], item[1]["date"], item[1]["match_id"]),
    )
    leagues = {f[side]: "epl" for _, f in dated_history for side in ("home_club", "away_club")}
    history_by_market = {}
    for market in picks_by_market:
        days, totals, training = [], [], []
        home_count, away_count = TARGETS[market]
        for day, f in dated_history:
            observed = f[market]
            if observed["counts"] is None:
                continue
            days.append(day)
            totals.append(observed["actual"])
            training.append({
                "competition": "epl", "date": f["date"], "status": "FT", "neutral": False,
                "home_id": f["home_club"], "away_id": f["away_club"],
                home_count: observed["counts"][0], away_count: observed["counts"][1],
            })
        history_by_market[market] = (days, totals, training)

    results = defaultdict(list)
    for day, group in groupby(fixtures, key=fixture_day):
        slate = list(group)
        for market, picks in picks_by_market.items():
            priced = [f for f in slate if f[market]["pick"] is not None]
            if not priced:
                continue

            current = [f[market]["pick"] for f in priced]
            probabilities = pick_probabilities([f[market]["rates"] for f in priced], picks)
            reassigned = assign_picks(probabilities, picks, current)

            days, totals, training = history_by_market[market]
            end = bisect_left(days, day)
            past_totals = totals[max(0, end - BASELINE_WINDOW):end]
            prior_rates = {}
            if len(past_totals) >= MIN_BASELINE_MATCHES:
                prior_rates = {pick: np.mean([bet_won(pick, total) for total in past_totals]) for pick in picks}

            team_reassigned = current
            if end >= MIN_BASELINE_MATCHES and len(set(current)) > 1:
                options = {"team_ridge": CORNER_TEAM_RIDGE} if market == "corners" else {}
                model = UCLModel(target=market, **options).fit(training[:end], day, leagues)
                rates = [model.expected_goals(f["home_club"], f["away_club"], competition="epl") for f in priced]
                team_reassigned = assign_picks(pick_probabilities(rates, picks), picks, current)

            for i, fixture in enumerate(priced):
                if fixture[market]["actual"] is None:
                    continue
                probabilities_by_pick = dict(zip(picks, probabilities[i]))
                highest = max(picks, key=probabilities_by_pick.__getitem__)
                lift = current[i]
                if prior_rates:
                    lift = max(picks, key=lambda pick: (
                        probabilities_by_pick[pick] - prior_rates[pick], probabilities_by_pick[pick]
                    ))
                results[market].append({
                    "season": fixture["season"], "gw": fixture["round"],
                    "match_id": fixture["match_id"], "date": fixture["date"],
                    "actual": fixture[market]["actual"],
                    "picks": dict(zip(POLICIES, (current[i], highest, lift, reassigned[i], team_reassigned[i]))),
                    "prior_rates": prior_rates,
                })
    return results


def summarize(fixtures):
    def actual(fixture):
        return fixture["actual"]

    summaries = {}
    for policy in POLICIES:
        def pick(fixture):
            return fixture["picks"][policy]

        stats = evaluate_lines(fixtures, pick, actual)
        count = stats["bets"]
        if count:
            # This is the frontend's blind comparison, with its baseline weighted
            # by the mix of lines that this policy actually chose.
            blind = sum(
                line["bets"] * sum(bet_won(label, actual(f)) for f in fixtures)
                for label, line in stats["by_pick"].items()
            ) * 100 / (count * len(fixtures))
            stats["line_mix_baseline"] = round(blind, 1)
            stats["lift_pp"] = round(stats["wins"] * 100 / count - blind, 1)

        by_round = defaultdict(Counter)
        known_baselines = []
        for f in fixtures:
            won = bet_won(pick(f), actual(f))
            counts = by_round[f["gw"]]
            counts["picks"] += 1
            counts["won"] += won
            if f["prior_rates"]:
                known_baselines.append((won, f["prior_rates"][pick(f)]))
        stats["by_matchweek"] = dict(sorted(by_round.items()))
        if known_baselines:
            won, expected = np.mean(known_baselines, axis=0)
            stats["past_only_baseline"] = {
                "picks": len(known_baselines), "hit_rate": round(100 * won, 1),
                "expected_hit_rate": round(100 * expected, 1), "lift_pp": round(100 * (won - expected), 1),
            }
        summaries[policy] = stats
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs="+", choices=REPLAY_SEASONS, default=REPLAY_SEASONS)
    parser.add_argument("--corner-strategy", choices=CORNER_STRATEGIES, default=DEFAULT_CORNER_STRATEGY)
    parser.add_argument("--output", type=Path, help="Write the detailed JSON report to this path")
    args = parser.parse_args()
    seasons = sorted(set(args.seasons))
    history = list(load_epl_history(HISTORY_SEASONS))
    results = replay(load_epl_fixtures(seasons, args.corner_strategy), history, args.corner_strategy)
    report = {"corner_strategy": args.corner_strategy, "cutoff": "start of kickoff day (UTC)", "seasons": {}}
    for season in seasons:
        print(f"\n{season}")
        season_report = {}
        for market in ("goals", "corners"):
            fixtures = [f for f in results[market] if f["season"] == season]
            policies = summarize(fixtures)
            season_report[market] = {"policies": policies, "fixtures": fixtures}
            if not fixtures:
                print(f"  {market:7} No fixtures with both a forecast and a verified result")
            for policy, stats in policies.items():
                if stats["bets"]:
                    print(f"  {market:7} {policy:20} {stats['wins']:3}/{stats['bets']:<3} "
                          f"{stats['hit_rate']:5.1f}%  same-line baseline {stats['line_mix_baseline']:5.1f}% "
                          f"({stats['lift_pp']:+.1f} pp)")
        report["seasons"][season] = season_report

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
