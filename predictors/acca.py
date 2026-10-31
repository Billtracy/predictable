"""Gameweek accumulator builders, shared by the API and the backtests.

Corner legs stay within the Over 7.5 - Under 12.5 band.
"""


def build_accas(fixtures):
    """Return (ultimate_acca, corner_acca) leg lists for one gameweek's fixtures.

    Each fixture needs match_id, home_team, away_team, goals.home_xg/away_xg and
    corners.total_expected.
    """
    acc_legs = []
    if fixtures:
        # unique_match_id constraint: every leg must come from a distinct fixture,
        # otherwise bookmakers reject it as a same-game multi (needs a Bet Builder).
        def unpicked(cands):
            picked_ids = {leg['match_id'] for leg in acc_legs}
            return [f for f in cands if f['match_id'] not in picked_ids]

        # 1. Safest Goal Over (highest xG, > 2.50)
        over_goal_cands = [f for f in fixtures if (f['goals']['home_xg'] + f['goals']['away_xg']) > 2.5]
        if over_goal_cands:
            best_goal_over = max(over_goal_cands, key=lambda f: f['goals']['home_xg'] + f['goals']['away_xg'])
            acc_legs.append({
                'match_id': best_goal_over['match_id'],
                'match': f"{best_goal_over['home_team']} vs {best_goal_over['away_team']}",
                'pick': "Over 1.5 Goals",
                'reason': f"Highest xG ({round(best_goal_over['goals']['home_xg'] + best_goal_over['goals']['away_xg'], 2)})"
            })

        # 2. Safest Corner Over (highest Exp Corners, >= 9.5)
        over_corner_cands = unpicked([f for f in fixtures if f['corners']['total_expected'] >= 9.5])
        if over_corner_cands:
            best_corner_over = max(over_corner_cands, key=lambda f: f['corners']['total_expected'])
            acc_legs.append({
                'match_id': best_corner_over['match_id'],
                'match': f"{best_corner_over['home_team']} vs {best_corner_over['away_team']}",
                'pick': "Over 7.5 Corners",
                'reason': f"High Expected Corners ({best_corner_over['corners']['total_expected']})"
            })

        # 3. Safest Corner Under (lowest Exp Corners, < 9.0)
        under_corner_cands = unpicked([f for f in fixtures if f['corners']['total_expected'] < 9.0])
        if under_corner_cands:
            best_corner_under = min(under_corner_cands, key=lambda f: f['corners']['total_expected'])
            acc_legs.append({
                'match_id': best_corner_under['match_id'],
                'match': f"{best_corner_under['home_team']} vs {best_corner_under['away_team']}",
                'pick': "Under 12.5 Corners",
                'reason': f"Low Expected Corners ({best_corner_under['corners']['total_expected']})"
            })

        # If we don't have 3 legs, fill with Goal Under (lowest xG, <= 2.5)
        if len(acc_legs) < 3:
            under_goal_cands = unpicked([f for f in fixtures if (f['goals']['home_xg'] + f['goals']['away_xg']) <= 2.5])
            if under_goal_cands:
                best_goal_under = min(under_goal_cands, key=lambda f: f['goals']['home_xg'] + f['goals']['away_xg'])
                acc_legs.append({
                    'match_id': best_goal_under['match_id'],
                    'match': f"{best_goal_under['home_team']} vs {best_goal_under['away_team']}",
                    'pick': "Under 4.5 Goals",
                    'reason': f"Lowest xG ({round(best_goal_under['goals']['home_xg'] + best_goal_under['goals']['away_xg'], 2)})"
                })

        # If still missing, fill with the safest Under 12.5 corners available
        if len(acc_legs) < 3:
            under_c_cands2 = unpicked([f for f in fixtures if f['corners']['total_expected'] < 9.5])
            if under_c_cands2:
                best_c2 = min(under_c_cands2, key=lambda f: f['corners']['total_expected'])
                acc_legs.append({
                    'match_id': best_c2['match_id'],
                    'match': f"{best_c2['home_team']} vs {best_c2['away_team']}",
                    'pick': "Under 12.5 Corners",
                    'reason': f"Low Expected Corners ({best_c2['corners']['total_expected']})"
                })

    acca = acc_legs[:3]

    corner_acca_legs = []
    if fixtures:
        # Sort by total expected corners
        sorted_by_corners = sorted(fixtures, key=lambda f: f['corners']['total_expected'])

        # 1. Safest Under
        if len(sorted_by_corners) > 0:
            c_under1 = sorted_by_corners[0]
            corner_acca_legs.append({
                'match_id': c_under1['match_id'],
                'match': f"{c_under1['home_team']} vs {c_under1['away_team']}",
                'pick': "Under 12.5 Corners",
                'reason': f"Lowest Expected Corners ({c_under1['corners']['total_expected']})"
            })

        # 2. 2nd Safest Under
        if len(sorted_by_corners) > 1:
            c_under2 = sorted_by_corners[1]
            corner_acca_legs.append({
                'match_id': c_under2['match_id'],
                'match': f"{c_under2['home_team']} vs {c_under2['away_team']}",
                'pick': "Under 12.5 Corners",
                'reason': f"Low Expected Corners ({c_under2['corners']['total_expected']})"
            })

        # 3. Safest Over
        if len(sorted_by_corners) > 2:
            c_over = sorted_by_corners[-1]
            corner_acca_legs.append({
                'match_id': c_over['match_id'],
                'match': f"{c_over['home_team']} vs {c_over['away_team']}",
                'pick': "Over 7.5 Corners",
                'reason': f"Highest Expected Corners ({c_over['corners']['total_expected']})"
            })

    return acca, corner_acca_legs


def grade_leg(leg, fixtures_by_id):
    """Grade a 3-word acca pick like 'Over 1.5 Goals' against the fixture's actual result."""
    fx = fixtures_by_id.get(leg.get('match_id'))
    if not fx or 'actual' not in fx:
        return None
    parts = leg['pick'].split()
    if len(parts) != 3:
        return None
    direction, category = parts[0].lower(), parts[2].lower()
    line = float(parts[1])
    if category == 'goals':
        total = fx['actual']['home_score'] + fx['actual']['away_score']
    elif category == 'corners':
        total = fx['actual']['corners']
    else:
        return None
    if total is None:
        return None
    return (total > line) if direction == 'over' else (total < line)
