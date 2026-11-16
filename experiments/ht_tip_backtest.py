"""Leak-free backtest of the half-time corner tip vs the pre-match corner tip.

Each match is priced with parameters fitted only on earlier matches (expanding
second-half mean and dispersion), and compared with the pre-match 'likeliest'
tip on the same matches, as the site would have shown it before kickoff.
Odds are ignored: this scores hit rate and calibration only.

Usage: python experiments/ht_tip_backtest.py          # prints + writes data/ht_tip_backtest.json
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from db.database import get_db  # noqa: E402
from models import fitted  # noqa: E402
from predictors.corners import corner_tip, DEFAULT_CORNER_STRATEGY  # noqa: E402
from predictors.ht_corners import ht_corner_tip, tip_probability_at_ht  # noqa: E402
from predictors.tips import tip_won, tip_probability  # noqa: E402

OUT_PATH = os.path.join(ROOT, 'data', 'ht_tip_backtest.json')
MIN_HISTORY = 100  # matches of half-split history before the first tip


def load_matches():
    with get_db() as db:
        matches = db.execute("""
            SELECT id, competition_id, date, home_team_id, away_team_id
            FROM matches
            WHERE competition_id LIKE 'epl_%' AND status IN ('FT', 'AET', 'PEN')
            ORDER BY date, id
        """).fetchall()
        events = db.execute("""
            SELECT e.match_id, e.raw_data FROM match_events e
            JOIN matches m ON m.id = e.match_id
            WHERE m.competition_id LIKE 'epl_%' AND e.event_type = 'corner_kick'
        """).fetchall()
    halves = defaultdict(lambda: [0, 0])
    for row in events:
        period = json.loads(row['raw_data']).get('period')
        if period in (1, 2):
            halves[row['match_id']][period - 1] += 1
    return [(m, halves[m['id']]) for m in matches if m['id'] in halves]


def expanding_params(c2_history):
    c2 = np.array(c2_history, dtype=float)
    mu = c2.mean()
    return {'a': float(mu), 'b': 0.0, 'phi': float(max(1.0, c2.var() / mu))}


def rate(rows, key):
    n = len(rows)
    return {'won': int(sum(r[key] for r in rows)), 'graded': n,
            'hit_rate': round(100 * sum(r[key] for r in rows) / n, 1) if n else None}


def calibration(rows, p_key, won_key):
    buckets = defaultdict(list)
    for r in rows:
        buckets[min(95, int(r[p_key] // 5) * 5)].append(r)
    return {f"{b}-{b + 5}%": {'n': len(v), 'predicted': round(np.mean([r[p_key] for r in v]), 1),
                              'actual': round(100 * np.mean([r[won_key] for r in v]), 1)}
            for b, v in sorted(buckets.items())}


def main():
    history, rows = [], []
    for m, (c1, c2) in load_matches():
        if len(history) >= MIN_HISTORY:
            pred = fitted.predict_corners(m['competition_id'], m['home_team_id'], m['away_team_id'], m['date'])
            if pred:
                params = expanding_params(history)
                ol = pred['results']['over_lines']
                pre_tip = corner_tip(DEFAULT_CORNER_STRATEGY, pred['results']['total_expected'], '', '', ol)
                ht = ht_corner_tip(c1, params)
                ft = c1 + c2
                rows.append({
                    'season': m['competition_id'], 'match_id': m['id'], 'ht_corners': c1, 'ft_corners': ft,
                    'pre_tip': pre_tip, 'pre_p': tip_probability(pre_tip, ol), 'pre_won': tip_won(pre_tip, ft),
                    'pre_p_at_ht': 100 * tip_probability_at_ht(pre_tip, c1, params),
                    'ht_tip': ht['recommended_bet'], 'ht_p': ht['prob'], 'ht_won': tip_won(ht['recommended_bet'], ft),
                })
        history.append(c2)

    seasons = sorted({r['season'] for r in rows})
    by_line = defaultdict(list)
    for r in rows:
        by_line[f"HT+{float(r['ht_tip'].split()[1]) - r['ht_corners']} ({r['ht_tip'].split()[0]})"].append(r)
    # Re-check of the pre-match tip at HT: how often it's already settled / still alive.
    settled = [r for r in rows if r['pre_p_at_ht'] in (0.0, 100.0)]
    alive = [r for r in rows if r not in settled]
    confident = [r for r in alive if r['pre_p_at_ht'] >= 85]
    shaky = [r for r in alive if r['pre_p_at_ht'] < 60]

    report = {
        'matches': len(rows),
        'overall': {'pre_match': rate(rows, 'pre_won'), 'half_time': rate(rows, 'ht_won'),
                    'pre_match_mean_p': round(np.mean([r['pre_p'] for r in rows]), 1),
                    'half_time_mean_p': round(np.mean([r['ht_p'] for r in rows]), 1)},
        'by_season': {s: {'pre_match': rate([r for r in rows if r['season'] == s], 'pre_won'),
                          'half_time': rate([r for r in rows if r['season'] == s], 'ht_won')} for s in seasons},
        'ht_tip_lines': {k: rate(v, 'ht_won') for k, v in sorted(by_line.items())},
        'calibration': {'pre_match': calibration(rows, 'pre_p', 'pre_won'),
                        'half_time': calibration(rows, 'ht_p', 'ht_won')},
        'pre_tip_rechecked_at_ht': {
            'already_settled_at_ht': len(settled),
            'alive_p_ge_85': rate(confident, 'pre_won'),
            'alive_p_lt_60': rate(shaky, 'pre_won'),
        },
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    o = report['overall']
    print(f"Matches: {len(rows)} (EPL, leak-free; HT params from earlier matches only)\n")
    print(f"{'':<16}{'hit rate':>10}{'won/graded':>14}{'model said':>12}")
    for name, key, p in (('Pre-match tip', 'pre_match', 'pre_match_mean_p'), ('Half-time tip', 'half_time', 'half_time_mean_p')):
        print(f"{name:<16}{o[key]['hit_rate']:>9.1f}%{o[key]['won']:>8}/{o[key]['graded']:<5}{o[p]:>11.1f}%")
    print("\nBy season:")
    for s, v in report['by_season'].items():
        print(f"  {s}: pre-match {v['pre_match']['hit_rate']}% · half-time {v['half_time']['hit_rate']}%  (n={v['half_time']['graded']})")
    print("\nHalf-time tip lines:")
    for k, v in report['ht_tip_lines'].items():
        print(f"  {k:<18} {v['hit_rate']}%  ({v['won']}/{v['graded']})")
    print("\nCalibration (predicted -> actual):")
    for name, cal in report['calibration'].items():
        print(f"  {name}: " + ", ".join(f"{b} n={c['n']} {c['predicted']}->{c['actual']}" for b, c in cal.items() if c['n'] >= 20))
    pr = report['pre_tip_rechecked_at_ht']
    print(f"\nPre-match tip re-checked at HT: settled already {pr['already_settled_at_ht']}, "
          f"alive & >=85% {pr['alive_p_ge_85']['hit_rate']}% (n={pr['alive_p_ge_85']['graded']}), "
          f"alive & <60% {pr['alive_p_lt_60']['hit_rate']}% (n={pr['alive_p_lt_60']['graded']})")
    print(f"\n-> {os.path.relpath(OUT_PATH, ROOT)}")


if __name__ == '__main__':
    main()
