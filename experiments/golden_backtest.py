"""Backtest a 'golden game' per gameweek: the single goals tip and the single
corners tip the model rates most likely to land, across all ten fixtures.

Candidates are the site's own tip lines (goals: Over 1.5 / Under 3.5, corners:
Over 7.5 / Under 12.5). Every fixture is priced by the fitted model with data
from before its kickoff day only, and gameweeks are the site's date-ordered
chunks of ten.

    python experiments/golden_backtest.py [--seasons epl_2024 epl_2025 epl_2026] [--verbose]
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from experiments.epl_tip_replay import kickoff_day
from predictors.corners import CORNER_TIP_LINES, predict_corners
from predictors.goals import GOAL_TIP_LINES, predict_goals
from predictors.tips import most_likely_tip, tip_probability, tip_won

FINISHED = ("FT", "AET", "PEN")
PER_GW = 10
MIN_PLAYED = 8  # postponed matches leave holes in the date-ordered chunks
MARKETS = {"goals": GOAL_TIP_LINES, "corners": CORNER_TIP_LINES}


def _season(season):
    with get_db() as db:
        matches = db.execute(
            """
            SELECT m.id, m.date, m.status, m.home_score, m.away_score,
                   h.name AS home, a.name AS away
            FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            WHERE m.competition_id = ?
            ORDER BY m.date
            """,
            (season,),
        ).fetchall()
        corners = {
            r["match_id"]: r["c"]
            for r in db.execute(
                """
                SELECT s.match_id, SUM(s.corners) AS c, COUNT(*) AS n
                FROM match_stats s JOIN matches m ON m.id = s.match_id
                WHERE m.competition_id = ? AND s.corners IS NOT NULL
                GROUP BY s.match_id HAVING n = 2
                """,
                (season,),
            )
        }
    return matches, corners


def _candidates(match, corners):
    """[(market, tip, probability, actual total)] for one finished fixture."""
    as_of = kickoff_day(match["date"])
    gp = predict_goals(match["home"], match["away"], SEASON, as_of_date=as_of, verbose=False)
    cp = predict_corners(match["home"], match["away"], SEASON, as_of_date=as_of, verbose=False)
    if not gp or not cp:
        return []
    actual = {
        "goals": match["home_score"] + match["away_score"],
        "corners": corners.get(match["id"]),
    }
    lines = {"goals": gp["results"]["over_lines"], "corners": cp["results"]["over_lines"]}
    out = []
    for market, tips in MARKETS.items():
        if actual[market] is None:
            continue
        tip = most_likely_tip(tips, lines[market])
        out.append((market, tip, tip_probability(tip, lines[market]), actual[market]))
    return out


class Tally:
    def __init__(self):
        self.n = self.won = 0
        self.prob = 0.0

    def add(self, won, prob=0.0):
        self.n += 1
        self.won += bool(won)
        self.prob += prob

    def __str__(self):
        if not self.n:
            return "      -"
        s = f"{self.won:3d}/{self.n:<3d} {self.won / self.n * 100:5.1f}%"
        if self.prob:
            s += f"  (model said {self.prob / self.n:4.1f}%)"
        return s


def run(seasons, verbose):
    global SEASON
    rows = []  # (season, gw, golden goals, golden corners, every tip, double)
    totals = defaultdict(lambda: defaultdict(Tally))
    for SEASON in seasons:
        matches, corners = _season(SEASON)
        for gw in range(len(matches) // PER_GW):
            chunk = [m for m in matches[gw * PER_GW:(gw + 1) * PER_GW] if m["status"] in FINISHED]
            if len(chunk) < MIN_PLAYED:
                continue  # skip gameweeks gutted by postponements
            picks = {"goals": [], "corners": []}
            for m in chunk:
                for market, tip, prob, actual in _candidates(m, corners):
                    picks[market].append((prob, m, tip, tip_won(tip, actual)))
            if not picks["goals"] or not picks["corners"]:
                continue

            line = [f"{SEASON} GW{gw + 1:>2}"]
            golden = {}
            for market in MARKETS:
                ranked = sorted(picks[market], key=lambda p: p[0], reverse=True)
                golden[market] = ranked
                prob, m, tip, won = ranked[0]
                for key in (SEASON, "ALL"):
                    totals[key][f"golden {market}"].add(won, prob)
                    for p in ranked:
                        totals[key][f"every {market} tip"].add(p[3], p[0])
                line.append(f"{market[0].upper()}: {'✓' if won else '✗'} {m['home']} v {m['away']} {tip} ({prob:.0f}%)")

            # Both golden legs as a double; bookmakers need two different
            # fixtures, so if they collide take the next-best corner pick.
            g = golden["goals"][0]
            c = next(p for p in golden["corners"] if p[1]["id"] != g[1]["id"])
            for key in (SEASON, "ALL"):
                totals[key]["golden double"].add(g[3] and c[3], g[0] * c[0] / 100)
            if verbose:
                print("  ".join(line))

    print()
    for key in [*seasons, "ALL"]:
        if key not in totals:
            continue
        print(f"== {key} ==")
        for label in ("golden goals", "golden corners", "golden double", "every goals tip", "every corners tip"):
            print(f"  {label:<18} {totals[key][label]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["epl_2024", "epl_2025", "epl_2026"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args.seasons, args.verbose)
