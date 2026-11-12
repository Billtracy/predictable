"""Choosing an over/under tip from a fixture's line probabilities."""


def tip_probability(tip, over_lines):
    """Probability (percent) that a tip like 'Over 1.5' or 'Under 12.5' lands,
    from {line: P(over) in percent}."""
    side, line = tip.split()
    p_over = over_lines[float(line)]
    return p_over if side == "Over" else 100.0 - p_over


def tip_won(tip, actual_total):
    """Did a tip like 'Over 1.5' or 'Under 12.5' land on the actual total?"""
    side, line = tip.split()
    return actual_total > float(line) if side == "Over" else actual_total < float(line)


def has_lines(tips, over_lines):
    return bool(tips) and bool(over_lines) and all(float(tip.split()[1]) in over_lines for tip in tips)


def most_likely_tip(tips, over_lines):
    """The tip the model rates most likely to land; ties go to the earlier one."""
    return max(tips, key=lambda tip: tip_probability(tip, over_lines))
