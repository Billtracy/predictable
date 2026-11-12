import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from models import fitted
from models.team_ratings import build_context
from models.poisson import simulate_match
from predictors.tips import has_lines, most_likely_tip


def matchup_corner_rates(context):
    home, away = context["home_ratings"], context["away_ratings"]
    average = context["avgs"]["avg_corners_per_team"]
    return (
        home["attack_corners"] * away["defense_corners"] * average,
        away["attack_corners"] * home["defense_corners"] * average,
    )


def predict_corners(home_team_name, away_team_name, competition_id, as_of_date=None, verbose=True):
    """Orchestrates corner prediction for a given matchup.

    When as_of_date is provided, ratings and league averages are recomputed using
    only matches played strictly before that date (leak-free backtesting).
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
        pred = fitted.predict_corners(competition_id, home_team["id"], away_team["id"], as_of_date)
        if pred is None:
            if verbose:
                print(f"Error: Too little corner history for {competition_id}")
            return None
        if verbose:
            print(f"Calculated Expected Corners -> {home_team['name']}: {pred['home_expected']:.2f} | {away_team['name']}: {pred['away_expected']:.2f}")
        return {"home_team": home_team["name"], "away_team": away_team["name"], **pred}

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

    home_expected, away_expected = matchup_corner_rates(ctx)

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


def recommend_corner_bet(total_expected, home_team_name, away_team_name):
    """Single-tip corner line for a fixture, keyed on xC rounded to 1dp (as displayed).

    Lines stay within the Over 7.5 - Under 12.5 band.
    xC <= 8.3  -> Under 11.5
    8.4 - 9.0  -> Under 12.5 (high-variance band, line shifted up)
    9.1 - 10.3 -> Over 7.5
    xC >= 10.4 -> Over 8.5
    """
    xc = round(total_expected, 1)
    if xc <= 8.3:
        return "Under 11.5"
    if xc <= 9.0:
        return "Under 12.5"
    if xc < 10.4:
        return "Over 7.5"
    return "Over 8.5"


def recommend_corner_bet_classic(total_expected, home_team_name, away_team_name):
    """Original corner rule: < 9.0 Under 11.5, < 10.0 Over 7.5, else Over 8.5."""
    if total_expected < 9.0:
        return "Under 11.5"
    if total_expected < 10.0:
        return "Over 7.5"
    return "Over 8.5"


# Lines the 'likeliest' strategy chooses between. Only the widest line on each
# side is listed: Over 7.5 is always likelier than Over 8.5, and Under 12.5 than
# Under 11.5, so a narrower line could never be the most likely one.
CORNER_TIP_LINES = ("Over 7.5", "Under 12.5")

# Selectable corner tip rules, keyed by the name the API/UI uses. A strategy with
# 'lines' tips whichever of them the model rates most likely, when the caller
# passes the fixture's line probabilities; 'fn' is its rule when it has none.
CORNER_STRATEGIES = {
    'likeliest': {
        'label': 'Most likely line',
        'description': 'Over 7.5 or Under 12.5, whichever the model rates more likely to land',
        'lines': CORNER_TIP_LINES,
        'fn': recommend_corner_bet,
    },
    'tight': {
        'label': 'Tight bands',
        'description': 'U11.5 ≤ 8.3 · U12.5 8.4–9.0 · O7.5 9.1–10.3 · O8.5 ≥ 10.4',
        'fn': recommend_corner_bet,
    },
    'classic': {
        'label': 'Classic',
        'description': 'U11.5 < 9.0 · O7.5 9.0–9.9 · O8.5 ≥ 10.0',
        'fn': recommend_corner_bet_classic,
    },
}
DEFAULT_CORNER_STRATEGY = 'likeliest'


def corner_tip(strategy, total_expected, home_team_name, away_team_name, over_lines=None):
    """Corner tip under a named strategy, falling back to the default for unknown names.

    over_lines is the fixture's {line: P(over) in percent}. Pass it for
    fitted-model competitions, whose probabilities are calibrated.
    """
    entry = CORNER_STRATEGIES.get(strategy) or CORNER_STRATEGIES[DEFAULT_CORNER_STRATEGY]
    if has_lines(entry.get('lines', ()), over_lines):
        return most_likely_tip(entry['lines'], over_lines)
    return entry['fn'](total_expected, home_team_name, away_team_name)
