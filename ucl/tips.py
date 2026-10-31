"""One goals tip and one corners tip per fixture, the same rules the EPL page uses
(predictors/goals.goal_tip and predictors/corners.recommend_corner_bet, the 'tight'
default). Copied rather than imported so the UCL package never loads EPL code;
ucl/backtest.py reports how they score on UCL games.
"""


def goal_tip(total_xg):
    if total_xg > 3.0:
        return "Over 2.5"
    if total_xg > 2.5:
        return "Over 1.5"
    return "Under 3.5"


def corner_tip(total_corners):
    xc = round(total_corners, 1)
    if xc <= 8.3:
        return "Under 11.5"
    if xc <= 9.0:
        return "Under 12.5"
    if xc < 10.4:
        return "Over 7.5"
    return "Over 8.5"


def tip_won(tip, actual_total):
    """True/False for a settled 'Over x' / 'Under x' tip."""
    side, line = tip.split()
    return actual_total > float(line) if side == "Over" else actual_total < float(line)


def tip_prob(tip, over_probs):
    """Model probability that a tip lands, from {line: P(over)}."""
    side, line = tip.split()
    p_over = over_probs[float(line)]
    return p_over if side == "Over" else 1.0 - p_over
