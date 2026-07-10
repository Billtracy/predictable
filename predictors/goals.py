import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from models.team_ratings import build_context
from models.poisson import simulate_match


def predict_goals(home_team_name, away_team_name, competition_id, as_of_date=None, verbose=True):
    """Orchestrates goal prediction for a given matchup.

    When as_of_date is provided, team ratings and league averages are recomputed
    using only matches played strictly before that date (leak-free backtesting).
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

    global_avg_goals = ctx["avgs"]["avg_goals_per_team"]

    # Home advantage is fit from the data (neutral-venue tournaments collapse to ~1.0).
    home_advantage = ctx["home_advantage"]
    away_disadvantage = ctx["away_advantage"]

    home_xg = home_ratings["attack_goals"] * away_ratings["defense_goals"] * global_avg_goals * home_advantage
    away_xg = away_ratings["attack_goals"] * home_ratings["defense_goals"] * global_avg_goals * away_disadvantage

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
