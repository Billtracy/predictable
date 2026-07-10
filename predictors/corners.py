import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from models.team_ratings import build_context
from models.poisson import simulate_match


def predict_corners(home_team_name, away_team_name, competition_id, as_of_date=None, verbose=True):
    """Orchestrates corner prediction for a given matchup.

    When as_of_date is provided, ratings and league averages are recomputed using
    only matches played strictly before that date (leak-free backtesting).
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

    ctx = build_context(competition_id, home_team["id"], away_team["id"], as_of_date)
    if ctx is None:
        print(f"Error: No match data for {competition_id}")
        return None

    home_ratings = ctx["home_ratings"]
    away_ratings = ctx["away_ratings"]
    if not home_ratings or not away_ratings:
        print("Error: Ratings not found. Run team_ratings.py first (or pass as_of_date).")
        return None

    global_avg_corners = ctx["avgs"]["avg_corners_per_team"]
    if not global_avg_corners:
        print(f"Error: No corner data for {competition_id}")
        return None

    # Calculate Expected Corners
    home_expected = home_ratings["attack_corners"] * away_ratings["defense_corners"] * global_avg_corners
    away_expected = away_ratings["attack_corners"] * home_ratings["defense_corners"] * global_avg_corners

    if verbose:
        print(f"Calculated Expected Corners -> {home_team['name']}: {home_expected:.2f} | {away_team['name']}: {away_expected:.2f}")

    # Run Simulation
    results = simulate_match(
        home_expected, away_expected, home_team["fifa_rank"], away_team["fifa_rank"], n_sims=10000, is_corner=True
    )

    return {
        "home_team": home_team["name"],
        "away_team": away_team["name"],
        "home_expected": home_expected,
        "away_expected": away_expected,
        "results": results,
    }
