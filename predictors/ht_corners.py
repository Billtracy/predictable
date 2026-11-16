"""Half-time corner tips: the full-time total-corners O/U call made at the HT whistle.

FT corners = HT corners (known) + C2, with C2 ~ NegBin(mean MU2, variance PHI * MU2).
Second-half corners are independent of the first half and of the pre-match corner
expectation in our EPL data, so MU2 is one league-wide number
(see experiments/ht_corner_pricer.py, which fits data/ht_corner_params.json).

The tip mirrors the pre-match 'likeliest' rule (Over 7.5 / Under 12.5 around a ~10
corner expectation): the likelier of the half-lines HT_TIP_MARGIN corners below and
above the expected FT total.
"""

import json
import math
import os

import numpy as np
from scipy.stats import nbinom, poisson

PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'ht_corner_params.json')
HT_TIP_MARGIN = 2.5
MAX_C2 = 30

_DEFAULT_PARAMS = {'a': 5.26, 'b': 0.0, 'phi': 1.08}


def load_params(path=PARAMS_PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return dict(_DEFAULT_PARAMS)


PARAMS = load_params()


def second_half_mean(params=PARAMS, pre_ft=0.0):
    return max(0.5, params['a'] + params.get('b', 0.0) * (pre_ft or 0.0))


def second_half_pmf(mu2, phi):
    k = np.arange(MAX_C2 + 1)
    pmf = nbinom(mu2 / (phi - 1.0), 1.0 / phi).pmf(k) if phi > 1.0 else poisson(mu2).pmf(k)
    return pmf / pmf.sum()


def tip_probability_at_ht(tip, ht_corners, params=PARAMS, pre_ft=0.0):
    """P (0-1) that an FT tip like 'Over 8.5' / 'Under 12.5' lands, given the HT count."""
    side, line = tip.split()
    line = float(line)
    pmf = second_half_pmf(second_half_mean(params, pre_ft), params['phi'])
    totals = ht_corners + np.arange(len(pmf))
    return float(pmf[totals > line].sum() if side == 'Over' else pmf[totals < line].sum())


def ht_corner_tip(ht_corners, params=PARAMS, pre_ft=0.0, margin=HT_TIP_MARGIN):
    """{'recommended_bet', 'prob', 'expected_total', 'ht_corners'} for an FT corner
    tip made at half-time."""
    mu2 = second_half_mean(params, pre_ft)
    expected = ht_corners + mu2
    over = f"Over {math.floor(expected - margin) + 0.5}"
    under = f"Under {math.ceil(expected + margin) - 0.5}"
    p_over = tip_probability_at_ht(over, ht_corners, params, pre_ft)
    p_under = tip_probability_at_ht(under, ht_corners, params, pre_ft)
    tip, prob = (over, p_over) if p_over >= p_under else (under, p_under)
    return {
        'ht_corners': ht_corners,
        'recommended_bet': tip,
        'prob': round(prob * 100, 1),
        'expected_total': round(expected, 1),
    }
