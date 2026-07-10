import math
import os
import sys

import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.database import get_db
from predictors.corners import predict_corners
from predictors.goals import predict_goals

# Outcome order used for the probability vector in calibration metrics.
OUTCOMES = ("home_win", "draw", "away_win")

# Total-goals lines evaluated in the binary-markets section.
GOAL_LINES = (1.5, 2.5, 3.5)

# Total-corners lines evaluated (must be produced by simulate_match's corner path).
CORNER_LINES = (8.5, 9.5, 10.5)


def _ranked_probability_score(probs, actual_idx):
    """Ranked Probability Score for an ordered 3-outcome market (home/draw/away).

    RPS rewards probabilities that are close *in rank order* to the result, which
    is the standard scoring rule for football W/D/L. Lower is better (0 = perfect).
    """
    cum_p = 0.0
    cum_a = 0.0
    total = 0.0
    for i in range(len(probs) - 1):
        cum_p += probs[i]
        cum_a += 1.0 if i == actual_idx else 0.0
        total += (cum_p - cum_a) ** 2
    return total / (len(probs) - 1)


class BinaryMarket:
    """Accumulates (predicted probability, actual outcome) pairs for a yes/no
    market and reports accuracy, Brier score and the base rate.

    The base rate doubles as the naive baseline: always predicting the majority
    class scores max(base_rate, 1 - base_rate), so the model only adds value on
    a market if its accuracy beats that number.
    """

    def __init__(self, name):
        self.name = name
        self.probs = []   # model P(yes), in [0, 1]
        self.actuals = []  # 1 if yes happened, else 0

    def add(self, prob_yes, actual_yes):
        self.probs.append(prob_yes)
        self.actuals.append(1 if actual_yes else 0)

    @property
    def n(self):
        return len(self.probs)

    def report_line(self):
        n = self.n
        if n == 0:
            return f"  {self.name:<18} no data"
        correct = sum(
            1 for p, a in zip(self.probs, self.actuals) if (p > 0.5) == (a == 1)
        )
        accuracy = correct / n * 100
        brier = sum((p - a) ** 2 for p, a in zip(self.probs, self.actuals)) / n
        base_rate = sum(self.actuals) / n
        majority = max(base_rate, 1 - base_rate) * 100
        # Brier of always forecasting the base rate ("climatology"); the model
        # must beat this for its probabilities to carry real information.
        base_brier = base_rate * (1 - base_rate)
        return (
            f"  {self.name:<18} acc {accuracy:5.1f}%  (always-majority {majority:5.1f}%)"
            f"   Brier {brier:.4f}  (base-rate {base_brier:.4f})   yes-rate {base_rate * 100:.1f}%"
        )


def run_backtest(competition_id):
    print(f"Running backtest for {competition_id}...")

    # Fixed seed so metric changes between runs reflect model changes, not
    # Monte Carlo noise (~±0.003 on log loss / Brier at 10k sims per match).
    np.random.seed(42)

    with get_db() as db:
        cur = db.cursor()
        # Chronological order: each match is predicted using only prior matches.
        cur.execute(
            """
            SELECT m.id, m.date, ht.name as home_name, at.name as away_name,
                   m.home_score, m.away_score
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN')
            ORDER BY m.date ASC
            """,
            (competition_id,),
        )
        matches = cur.fetchall()

    if not matches:
        print("No completed matches found.")
        return

    scored = 0
    skipped = 0
    correct_outcomes = 0
    total_home_error = 0.0
    total_away_error = 0.0
    total_log_loss = 0.0
    total_brier = 0.0
    total_rps = 0.0

    btts_market = BinaryMarket("BTTS")
    over_markets = {line: BinaryMarket(f"Over {line} goals") for line in GOAL_LINES}

    EPS = 1e-15  # keep log-loss finite for zero-probability outcomes

    for match in matches:
        home_name = match["home_name"]
        away_name = match["away_name"]
        actual_home_score = match["home_score"]
        actual_away_score = match["away_score"]

        if actual_home_score > actual_away_score:
            actual_outcome = "home_win"
        elif actual_away_score > actual_home_score:
            actual_outcome = "away_win"
        else:
            actual_outcome = "draw"
        actual_idx = OUTCOMES.index(actual_outcome)

        # Predict using ONLY data available before this match's date (no leakage).
        try:
            pred = predict_goals(
                home_name, away_name, competition_id, as_of_date=match["date"], verbose=False
            )
        except Exception as e:
            print(f"Error predicting {home_name} vs {away_name}: {e}")
            skipped += 1
            continue

        if not pred:
            skipped += 1
            continue

        res = pred["results"]
        # Probability vector in OUTCOMES order, normalized to sum to 1.
        raw = [res["home_win_prob"], res["draw_prob"], res["away_win_prob"]]
        s = sum(raw) or 1.0
        probs = [p / s for p in raw]

        predicted_idx = max(range(3), key=lambda i: probs[i])
        if predicted_idx == actual_idx:
            correct_outcomes += 1

        # Calibration metrics
        total_log_loss += -math.log(max(probs[actual_idx], EPS))
        total_brier += sum((probs[i] - (1.0 if i == actual_idx else 0.0)) ** 2 for i in range(3))
        total_rps += _ranked_probability_score(probs, actual_idx)

        total_home_error += abs(pred["home_xg"] - actual_home_score)
        total_away_error += abs(pred["away_xg"] - actual_away_score)

        # Binary goal markets (probabilities come back as percentages).
        actual_total = actual_home_score + actual_away_score
        btts_market.add(
            res["btts_prob"] / 100.0,
            actual_home_score >= 1 and actual_away_score >= 1,
        )
        for line in GOAL_LINES:
            over_markets[line].add(res["over_lines"][line] / 100.0, actual_total > line)

        scored += 1

    if scored == 0:
        print("No matches could be scored (no prior data for any fixture).")
        return

    accuracy = (correct_outcomes / scored) * 100
    home_mae = total_home_error / scored
    away_mae = total_away_error / scored
    log_loss = total_log_loss / scored
    brier = total_brier / scored
    rps = total_rps / scored

    print("\n" + "=" * 50)
    print(f"BACKTEST RESULTS: {competition_id}  (leak-free / point-in-time)")
    print("=" * 50)
    print(f"Matches Scored:         {scored}  (skipped: {skipped})")
    print(f"Match Outcome Accuracy: {accuracy:.2f}%")
    print(f"Home Goals MAE:         {home_mae:.3f}")
    print(f"Away Goals MAE:         {away_mae:.3f}")
    print("-" * 50)
    print("Probability calibration (lower = better):")
    print(f"  Log Loss:  {log_loss:.4f}   (coin-flip 3-way ~1.099)")
    print(f"  Brier:     {brier:.4f}   (uniform guess ~0.667)")
    print(f"  RPS:       {rps:.4f}   (uniform guess ~0.222)")
    print("-" * 50)
    print(f"Binary goal markets ({btts_market.n} matches; binary Brier: 0.25 = coin flip):")
    print(btts_market.report_line())
    for line in GOAL_LINES:
        print(over_markets[line].report_line())
    print("=" * 50)

    run_corners_backtest(competition_id, matches)


def run_corners_backtest(competition_id, matches):
    """Backtest total-corner over/under lines on matches that have corner stats
    for both teams. Predictions are point-in-time (as_of_date), same as goals."""
    np.random.seed(43)

    with get_db() as db:
        cur = db.cursor()
        # match_id -> total corners, only when both teams have a stats row.
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

    if not corner_totals:
        print(f"\nNo corner data for {competition_id}; skipping corners backtest.")
        return

    corner_markets = {line: BinaryMarket(f"Over {line} corners") for line in CORNER_LINES}
    scored = 0
    skipped = 0

    for match in matches:
        actual_total = corner_totals.get(match["id"])
        if actual_total is None:
            continue

        try:
            pred = predict_corners(
                match["home_name"],
                match["away_name"],
                competition_id,
                as_of_date=match["date"],
                verbose=False,
            )
        except Exception as e:
            print(f"Error predicting corners {match['home_name']} vs {match['away_name']}: {e}")
            skipped += 1
            continue

        if not pred:
            skipped += 1
            continue

        over_lines = pred["results"]["over_lines"]
        for line in CORNER_LINES:
            corner_markets[line].add(over_lines[line] / 100.0, actual_total > line)
        scored += 1

    if scored == 0:
        print(f"\nNo corner matches could be scored for {competition_id}.")
        return

    print(f"\nCorner markets ({scored} matches with corner stats; skipped: {skipped}):")
    for line in CORNER_LINES:
        print(corner_markets[line].report_line())
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_backtest(sys.argv[1])
    else:
        run_backtest("world_cup_2022")
