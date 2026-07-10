import numpy as np
from collections import Counter
import math

# Dixon-Coles low-score dependency parameter. Real football has slightly more
# 0-0/1-1 draws and slightly fewer 1-0/0-1 results than independent Poisson
# predicts. rho < 0 nudges the four lowest-scoring cells accordingly.
# -0.05 matches the classic Dixon & Coles (1997) range and, per the seeded EPL
# A/B, leaves W/D/L calibration untouched while making the scoreline / over-under
# distribution more realistic (its main benefit). Stronger values trade a little
# probabilistic calibration for a marginal accuracy bump.
DIXON_COLES_RHO = -0.05

# Std of the per-match noise added to xG before drawing goals (over-dispersion).
# Fitted by experiments/fit_dispersion.py: swept 0.0-0.5 on the epl_2024 point-in-time
# backtest, and every goal market (BTTS / over-under) improved monotonically as the
# noise shrank. The old FIFA-rank guess (calculate_unpredictability) injected ~0.5 for
# club football, which the sweep confirmed was actively hurting totals. 0.05 is the
# argmin (statistically tied with pure Poisson at 0.0) and leaves W/D/L RPS untouched.
GOAL_DISPERSION = 0.05


def dixon_coles_tau(home_goals, away_goals, home_xg, away_xg, rho=DIXON_COLES_RHO):
    """Dixon-Coles correction factor for the four low-scoring scorelines.
    Returns 1.0 (no change) for every other scoreline."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - (home_xg * away_xg * rho)
    if home_goals == 0 and away_goals == 1:
        return 1.0 + (home_xg * rho)
    if home_goals == 1 and away_goals == 0:
        return 1.0 + (away_xg * rho)
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def calculate_unpredictability(fifa_rank):
    """
    Calculates the standard deviation for the Monte Carlo simulation based on FIFA rank.
    Lower ranked teams (higher rank number) have higher unpredictability.
    Rank 1-10 -> ~0.25
    Rank 50+ -> ~0.70
    """
    if fifa_rank is None or fifa_rank <= 0:
        return 0.5 # Default middle ground
        
    # Simple scaling: baseline std is 0.2, adds 0.01 for every rank up to a max of 0.8
    std = 0.2 + (min(fifa_rank, 80) * 0.007)
    return std

def analytical_poisson(expected_goals):
    """
    Standard Analytical Poisson distribution.
    P(x; λ) = (e^(-λ) * λ^x) / x!
    Returns dictionary of probabilities for 0 to 10 goals.
    """
    probs = {}
    for i in range(11):
        try:
            prob = (math.exp(-expected_goals) * (expected_goals ** i)) / math.factorial(i)
            probs[i] = prob
        except OverflowError:
            probs[i] = 0.0
    return probs

def simulate_match(home_xg, away_xg, home_rank, away_rank, n_sims=10000, is_corner=False):
    """
    Runs Monte Carlo simulation of a match using Poisson distribution 
    with unpredictability noise injected based on team rankings.
    """
    if home_xg <= 0 or away_xg <= 0:
        return {
            "home_win_prob": 0,
            "draw_prob": 0,
            "away_win_prob": 0,
            "btts_prob": 0,
            "most_likely_score": "0-0",
            "score_probs": {(0, 0): 1.0},
            "total_expected": 0,
            "over_lines": {}
        }

    if is_corner:
        # Corners keep the original wider noise (their own model is fit separately).
        home_std = away_std = 0.5 * 3
    else:
        # Goals use the fitted dispersion instead of the old FIFA-rank guess, which
        # the sweep showed was hurting BTTS/over-under. home_rank/away_rank are now
        # unused for goals but kept in the signature for API compatibility.
        home_std = away_std = GOAL_DISPERSION

    scorelines = Counter()
    total_goals_or_corners = []

    for _ in range(n_sims):
        # Inject noise (unpredictability)
        adj_home_xg = max(0.01, np.random.normal(home_xg, home_std))
        adj_away_xg = max(0.01, np.random.normal(away_xg, away_std))

        # Draw from Poisson distribution
        home_result = np.random.poisson(adj_home_xg)
        away_result = np.random.poisson(adj_away_xg)

        scorelines[(home_result, away_result)] += 1
        total_goals_or_corners.append(home_result + away_result)

    # Convert raw simulation counts into a scoreline probability distribution.
    score_probs = {score: count / n_sims for score, count in scorelines.items()}

    # Dixon-Coles correction (goals only): reweight the four low-scoring cells to
    # account for the real-world dependency independent Poisson misses, then
    # renormalize so the distribution still sums to 1.
    if not is_corner:
        for score in list(score_probs):
            h, a = score
            if h <= 1 and a <= 1:
                score_probs[score] *= dixon_coles_tau(h, a, home_xg, away_xg)
        norm = sum(score_probs.values()) or 1.0
        score_probs = {score: p / norm for score, p in score_probs.items()}

    # Derive win/draw/loss, BTTS and over/under lines from the (corrected)
    # scoreline distribution. Using the corrected distribution matters: the
    # Dixon-Coles reweighting shifts exactly the low-scoring cells that decide
    # BTTS and the 0.5/1.5/2.5 total-goals lines.
    home_win_prob = draw_prob = away_win_prob = 0.0
    btts_prob = 0.0
    goal_over_lines = {}
    if not is_corner:
        for (h, a), p in score_probs.items():
            if h > a:
                home_win_prob += p
            elif a > h:
                away_win_prob += p
            else:
                draw_prob += p
            if h >= 1 and a >= 1:
                btts_prob += p
        for threshold in (0.5, 1.5, 2.5, 3.5, 4.5):
            goal_over_lines[threshold] = (
                sum(p for (h, a), p in score_probs.items() if h + a > threshold) * 100
            )
        home_win_prob *= 100
        draw_prob *= 100
        away_win_prob *= 100
        btts_prob *= 100

    # Express scoreline probabilities as percentages for reporting.
    score_probs = {score: p * 100 for score, p in score_probs.items()}
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)

    total_probs = {}
    over_lines = {}
    if is_corner:
        total_counts = Counter(total_goals_or_corners)
        total_probs = {total: (count / n_sims) * 100 for total, count in total_counts.items()}
        top_totals = sorted(total_probs.items(), key=lambda x: x[1], reverse=True)
        most_likely = str(top_totals[0][0])

        for threshold in [7.5, 8.5, 9.5, 10.5, 11.5]:
            over_count = sum(1 for total in total_goals_or_corners if total > threshold)
            over_lines[threshold] = (over_count / n_sims) * 100
    else:
        most_likely = f"{top_scores[0][0][0]}-{top_scores[0][0][1]}"
        over_lines = goal_over_lines

    return {
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "btts_prob": btts_prob,
        "most_likely_score": most_likely,
        "score_probs": dict(top_scores[:10]), # Top 10 exact outcomes
        "total_probs": dict(sorted(total_probs.items(), key=lambda x: x[1], reverse=True)[:10]),
        "total_expected": home_xg + away_xg,
        "over_lines": over_lines
    }
