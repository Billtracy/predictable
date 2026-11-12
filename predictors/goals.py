import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from models import fitted
from models.team_ratings import build_context
from models.poisson import simulate_match
from predictors.tips import has_lines, most_likely_tip

# Lines the goals tip chooses between when it has line probabilities. Only the
# widest line on each side is listed: Over 1.5 is always likelier than Over 2.5,
# so a narrower line could never be the most likely one.
GOAL_TIP_LINES = ("Over 1.5", "Under 3.5")


def matchup_goal_rates(context):
    home, away = context["home_ratings"], context["away_ratings"]
    average = context["avgs"]["avg_goals_per_team"]
    return (
        home["attack_goals"] * away["defense_goals"] * average * context["home_advantage"],
        away["attack_goals"] * home["defense_goals"] * average * context["away_advantage"],
    )


def predict_goals(home_team_name, away_team_name, competition_id, as_of_date=None, verbose=True):
    """Orchestrates goal prediction for a given matchup.

    When as_of_date is provided, team ratings and league averages are recomputed
    using only matches played strictly before that date (leak-free backtesting).
    Competitions with a "league" in config use the fitted model (models/fitted.py);
    the rest use the ratio ratings and the Monte Carlo simulation.
    """
    with get_db() as db:
        cur = db.cursor()

        # Resolve teams
        cur.execute("SELECT id, name, fifa_rank FROM teams WHERE name LIKE ?", (f"%{home_team_name}%",))
        home_team = cur.fetchone()

        cur.execute("SELECT id, name, fifa_rank FROM teams WHERE name LIKE ?", (f"%{away_team_name}%",))
        away_team = cur.fetchone()

        if not home_team or not away_team:
            print(f"Error: Could not find one or both teams ('{home_team_name}', '{away_team_name}').")
            return None

    if fitted.league_of(competition_id):
        pred = fitted.predict_goals(competition_id, home_team["id"], away_team["id"], as_of_date)
        if pred is None:
            if verbose:
                print(f"Error: Too little match history for {competition_id}")
            return None
        if verbose:
            print(f"Calculated Matchup xG -> {home_team['name']}: {pred['home_xg']:.2f} | {away_team['name']}: {pred['away_xg']:.2f}")
        return {"home_team": home_team["name"], "away_team": away_team["name"], **pred}

    # Build point-in-time (or cached) context: ratings, averages, home advantage
    ctx = build_context(competition_id, home_team["id"], away_team["id"], as_of_date)
    if ctx is None:
        print(f"Error: No match data for {competition_id}")
        return None

    home_ratings = ctx["home_ratings"]
    away_ratings = ctx["away_ratings"]
    if not home_ratings or not away_ratings:
        print("Error: Ratings not found. Run team_ratings.py first (or pass as_of_date).")
        return None

    home_xg, away_xg = matchup_goal_rates(ctx)

    if verbose:
        print(f"Calculated Matchup xG -> {home_team['name']}: {home_xg:.2f} | {away_team['name']}: {away_xg:.2f}")

    # Run Simulation
    results = simulate_match(
        home_xg, away_xg, home_team["fifa_rank"], away_team["fifa_rank"], n_sims=10000, is_corner=False
    )

    return {
        "home_team": home_team["name"],
        "away_team": away_team["name"],
        "home_xg": home_xg,
        "away_xg": away_xg,
        "results": results,
    }


def goal_tip(total_xg, over_lines=None):
    """The goals line we tip for a fixture. Given the fixture's over-line
    probabilities (pass them for fitted-model competitions, whose probabilities
    are calibrated), tip the line most likely to land. Without them, fall back to
    fixed bands on total expected goals."""
    if has_lines(GOAL_TIP_LINES, over_lines):
        return most_likely_tip(GOAL_TIP_LINES, over_lines)
    if total_xg > 3.0:
        return "Over 2.5"
    if total_xg > 2.5:
        return "Over 1.5"
    return "Under 3.5"
