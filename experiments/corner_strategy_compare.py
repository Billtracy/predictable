import os
import sys
import json
from collections import defaultdict
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from predictors.acca import build_accas, grade_leg
from predictors.corners import predict_corners, corner_tip, CORNER_STRATEGIES, DEFAULT_CORNER_STRATEGY
from predictors.goals import predict_goals, goal_tip

COMPETITIONS = ["epl_2024", "epl_2025", "epl_2026"]
MATCHES_PER_GW = 10
# Yes/no goal markets the model prices: key -> (label, P(yes) from the prediction, did it land)
GOAL_MARKETS = {
    "btts": ("Both teams to score", lambda f: f["goals"]["btts_prob"], lambda a: a["home_score"] > 0 and a["away_score"] > 0),
    "o15": ("Over 1.5 goals", lambda f: f["goals"]["o15_prob"], lambda a: a["home_score"] + a["away_score"] > 1.5),
    "o25": ("Over 2.5 goals", lambda f: f["goals"]["o25_prob"], lambda a: a["home_score"] + a["away_score"] > 2.5),
}
CALIBRATION_BINS = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 101)]
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corner_strategy_backtest.json")


def bet_won(pick_name, actual):
    direction, line = pick_name.split()
    line = float(line)
    return actual < line if direction == "Under" else actual > line


def load_fixtures():
    """Leak-free predictions for every finished match, shaped like the API's fixtures.

    Gameweeks are chunks of 10 over all of a season's matches by date (as the API
    does), so a postponed match doesn't shift later gameweeks.
    """
    fixtures = []
    for competition_id in COMPETITIONS:
        print(f"Predicting {competition_id}...")
        with get_db() as db:
            cur = db.cursor()
            cur.execute(
                """
                SELECT m.id, m.date, m.status, ht.name as home_name, at.name as away_name, m.home_score, m.away_score
                FROM matches m
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                WHERE m.competition_id = ?
                ORDER BY m.date ASC
                """,
                (competition_id,),
            )
            matches = cur.fetchall()

            cur.execute(
                """
                SELECT ms.match_id, SUM(ms.corners) as total_corners, COUNT(*) as rows
                FROM match_stats ms
                JOIN matches m ON ms.match_id = m.id
                WHERE m.competition_id = ? AND ms.corners IS NOT NULL
                GROUP BY ms.match_id
                HAVING rows = 2
                """,
                (competition_id,),
            )
            corner_totals = {r["match_id"]: r["total_corners"] for r in cur.fetchall()}

        for idx, match in enumerate(matches):
            actual_corners = corner_totals.get(match["id"])
            if match["status"] not in ("FT", "AET", "PEN") or actual_corners is None or match["home_score"] is None:
                continue
            try:
                cp = predict_corners(match["home_name"], match["away_name"], competition_id, as_of_date=match["date"], verbose=False)
                gp = predict_goals(match["home_name"], match["away_name"], competition_id, as_of_date=match["date"], verbose=False)
            except Exception:
                continue
            if not cp or not gp:
                continue
            res = gp["results"]
            outcome_total = (res["home_win_prob"] + res["draw_prob"] + res["away_win_prob"]) or 1.0
            fixtures.append({
                "season": competition_id,
                "gw": idx // MATCHES_PER_GW + 1,
                "match_id": match["id"],
                "home_team": match["home_name"],
                "away_team": match["away_name"],
                "goals": {
                    "home_xg": round(gp["home_xg"], 2),
                    "away_xg": round(gp["away_xg"], 2),
                    "btts_prob": res.get("btts_prob", 0),
                    "o15_prob": res.get("over_lines", {}).get(1.5, 0),
                    "o25_prob": res.get("over_lines", {}).get(2.5, 0),
                    "outcome_probs": [res["home_win_prob"] / outcome_total, res["draw_prob"] / outcome_total, res["away_win_prob"] / outcome_total],
                },
                "corners": {"total_expected": round(cp["results"]["total_expected"], 1)},
                "actual": {"home_score": match["home_score"], "away_score": match["away_score"], "corners": actual_corners},
            })
    return fixtures


def evaluate_tips(fixtures, strategy):
    return evaluate_lines(
        fixtures,
        lambda f: corner_tip(strategy, f["corners"]["total_expected"], f["home_team"], f["away_team"]),
        lambda f: f["actual"]["corners"],
    )


def evaluate_goal_tips(fixtures):
    return evaluate_lines(
        fixtures,
        lambda f: goal_tip(f["goals"]["home_xg"] + f["goals"]["away_xg"]),
        lambda f: f["actual"]["home_score"] + f["actual"]["away_score"],
    )


def evaluate_lines(fixtures, tip_for, actual_for):
    """Hit rate of an over/under tip rule, with each line's base rate: how often
    that line lands across every fixture, i.e. the score for backing it blind."""
    by_pick = defaultdict(lambda: [0, 0])
    by_gw = defaultdict(lambda: [0, 0])
    for f in fixtures:
        pick = tip_for(f)
        won = bet_won(pick, actual_for(f))
        by_pick[pick][0] += won
        by_pick[pick][1] += 1
        by_gw[(f["season"], f["gw"])][0] += won
        by_gw[(f["season"], f["gw"])][1] += 1

    wins = sum(w for w, _ in by_pick.values())
    bets = sum(n for _, n in by_pick.values())
    gw_rates = [w / n for w, n in by_gw.values() if n]
    return {
        "bets": bets,
        "wins": wins,
        "hit_rate": round(wins / bets * 100, 1) if bets else 0.0,
        "gameweeks": len(gw_rates),
        "perfect_gws": sum(1 for x in gw_rates if x == 1.0),
        "gws_80_plus": sum(1 for x in gw_rates if x >= 0.8),
        "worst_gw_rate": round(min(gw_rates) * 100, 1) if gw_rates else 0.0,
        "by_pick": {
            pick: {
                "bets": n, "wins": w, "hit_rate": round(w / n * 100, 1),
                "base_rate": round(sum(bet_won(pick, actual_for(f)) for f in fixtures) / len(fixtures) * 100, 1),
            }
            for pick, (w, n) in sorted(by_pick.items())
        },
    }


def evaluate_goal_markets(fixtures):
    """Yes/no calls (yes when P >= 50%) and probability quality for each goal market,
    against the naive bar of always calling the more common answer."""
    result = {}
    for key, (label, prob_for, landed_for) in GOAL_MARKETS.items():
        probs = [prob_for(f) / 100 for f in fixtures]
        landed = [landed_for(f["actual"]) for f in fixtures]
        n = len(fixtures)
        rate = sum(landed) / n if n else 0.0
        correct = sum((p >= 0.5) == y for p, y in zip(probs, landed))
        calibration = []
        for lo, hi in CALIBRATION_BINS:
            idx = [i for i, p in enumerate(probs) if lo <= p * 100 < hi]
            if idx:
                calibration.append({
                    "range": f"{lo}–{min(hi, 100)}%",
                    "matches": len(idx),
                    "predicted": round(sum(probs[i] for i in idx) / len(idx) * 100, 1),
                    "actual": round(sum(landed[i] for i in idx) / len(idx) * 100, 1),
                })
        result[key] = {
            "label": label,
            "matches": n,
            "accuracy": round(correct / n * 100, 1) if n else 0.0,
            "baseline_accuracy": round(max(rate, 1 - rate) * 100, 1),
            "actual_rate": round(rate * 100, 1),
            "avg_predicted": round(sum(probs) / n * 100, 1) if n else 0.0,
            "brier": round(sum((p - y) ** 2 for p, y in zip(probs, landed)) / n, 4) if n else 0.0,
            "baseline_brier": round(rate * (1 - rate), 4),
            "calibration": calibration,
        }

    # 1X2: does the model's favourite win?
    hits = 0
    for f in fixtures:
        a = f["actual"]
        outcome = 0 if a["home_score"] > a["away_score"] else 1 if a["home_score"] == a["away_score"] else 2
        probs = f["goals"]["outcome_probs"]
        hits += probs.index(max(probs)) == outcome
    result["outcome"] = {"label": "Match result (favourite)", "matches": len(fixtures),
                         "accuracy": round(hits / len(fixtures) * 100, 1) if fixtures else 0.0}
    return result


def gameweek_breakdown(fixtures):
    """One row per gameweek: tip records and whether each acca landed (None if no full acca)."""
    by_gw = defaultdict(list)
    for f in fixtures:
        by_gw[f["gw"]].append(f)
    rows = []
    for gw, gw_fixtures in sorted(by_gw.items()):
        goal = [bet_won(goal_tip(f["goals"]["home_xg"] + f["goals"]["away_xg"]), f["actual"]["home_score"] + f["actual"]["away_score"]) for f in gw_fixtures]
        corner = [bet_won(corner_tip(DEFAULT_CORNER_STRATEGY, f["corners"]["total_expected"], f["home_team"], f["away_team"]), f["actual"]["corners"]) for f in gw_fixtures]
        fixtures_by_id = {f["match_id"]: f for f in gw_fixtures}
        accas = [all(grade_leg(leg, fixtures_by_id) for leg in legs) if len(legs) == 3 else None for legs in build_accas(gw_fixtures)]
        rows.append({
            "gw": gw,
            "matches": len(gw_fixtures),
            "goal_tips": [sum(goal), len(goal)],
            "corner_tips": [sum(corner), len(corner)],
            "ultimate_acca": accas[0],
            "corner_acca": accas[1],
        })
    return rows


def evaluate_accas(fixtures):
    """Both gameweek accas, built exactly as the API builds them. Only full 3-leg accas count."""
    by_gw = defaultdict(list)
    for f in fixtures:
        by_gw[(f["season"], f["gw"])].append(f)

    # How often each leg's line lands across every fixture: the market's fair price for it, before margin
    all_by_id = {f["match_id"]: f for f in fixtures}
    base_rates = {}
    def base_rate(pick):
        if pick not in base_rates:
            graded = [grade_leg({"match_id": mid, "pick": pick}, all_by_id) for mid in all_by_id]
            graded = [g for g in graded if g is not None]
            base_rates[pick] = sum(graded) / len(graded) if graded else None
        return base_rates[pick]

    result = {name: {"accas": 0, "won": 0, "legs": 0, "legs_won": 0, "fair_odds_sum": 0.0} for name in ("ultimate", "corner")}
    for gw_fixtures in by_gw.values():
        fixtures_by_id = {f["match_id"]: f for f in gw_fixtures}
        for name, legs in zip(("ultimate", "corner"), build_accas(gw_fixtures)):
            if len(legs) < 3:
                continue
            graded = [grade_leg(leg, fixtures_by_id) for leg in legs]
            r = result[name]
            r["accas"] += 1
            r["won"] += all(graded)
            r["legs"] += len(graded)
            r["legs_won"] += sum(graded)
            fair = 1.0
            for leg in legs:
                fair /= base_rate(leg["pick"]) or 1.0
            r["fair_odds_sum"] += fair
    for r in result.values():
        r["win_rate"] = round(r["won"] / r["accas"] * 100, 1) if r["accas"] else 0.0
        r["leg_hit_rate"] = round(r["legs_won"] / r["legs"] * 100, 1) if r["legs"] else 0.0
        # Average acca price with no bookmaker margin, from how often each leg's line lands blind
        r["fair_odds"] = round(r.pop("fair_odds_sum") / r["accas"], 3) if r["accas"] else None
    return result


def summarise(fixtures):
    return {
        "matches": len(fixtures),
        "strategies": {key: evaluate_tips(fixtures, key) for key in CORNER_STRATEGIES},
        "accas": evaluate_accas(fixtures),
        "goal_tips": evaluate_goal_tips(fixtures),
        "goal_markets": evaluate_goal_markets(fixtures),
    }


def print_block(title, block):
    print(f"\n{'=' * 64}\n{title}  ({block['matches']} matches)\n{'=' * 64}")
    for key, res in block["strategies"].items():
        print(f"[{key}] {res['wins']}/{res['bets']} = {res['hit_rate']}%   perfect GWs {res['perfect_gws']}/{res['gameweeks']}   "
              f"GWs >= 80% {res['gws_80_plus']}   worst GW {res['worst_gw_rate']}%")
        for pick, p in res["by_pick"].items():
            print(f"    {pick:<11} {p['wins']:>4}/{p['bets']:<4} {p['hit_rate']:>5}%")
    g = block["goal_tips"]
    print(f"[goal tips] {g['wins']}/{g['bets']} = {g['hit_rate']}%")
    for pick, p in g["by_pick"].items():
        print(f"    {pick:<11} {p['wins']:>4}/{p['bets']:<4} {p['hit_rate']:>5}%   (base {p['base_rate']}%)")
    for key, m in block["goal_markets"].items():
        extra = f"   (always-majority {m['baseline_accuracy']}%, Brier {m['brier']} vs base {m['baseline_brier']})" if "brier" in m else ""
        print(f"[{key}] call accuracy {m['accuracy']}%{extra}")
    for name, a in block["accas"].items():
        print(f"[{name} acca] won {a['won']}/{a['accas']} = {a['win_rate']}%   legs {a['legs_won']}/{a['legs']} = {a['leg_hit_rate']}%   fair odds {a['fair_odds']}")


def main():
    fixtures = load_fixtures()
    seasons = {}
    for c in COMPETITIONS:
        season_fixtures = [f for f in fixtures if f["season"] == c]
        seasons[c] = {**summarise(season_fixtures), "gameweeks": gameweek_breakdown(season_fixtures)}
    historical = summarise([f for f in fixtures if f["season"] != COMPETITIONS[-1]])

    for c in COMPETITIONS:
        print_block(c, seasons[c])
    print_block(f"{' + '.join(COMPETITIONS[:-1])} combined", historical)

    # The page shows the historical (full-season) figures; this season's are reported alongside
    with open(OUTPUT_FILE, "w") as fh:
        json.dump({
            "meta": {
                "competitions": COMPETITIONS[:-1],
                "matches": historical["matches"],
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "accas": historical["accas"],
                "goal_tips": historical["goal_tips"],
                "goal_markets": historical["goal_markets"],
                "seasons": seasons,
            },
            "strategies": historical["strategies"],
        }, fh, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
