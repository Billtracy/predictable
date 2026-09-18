import os
import sys
import re
import math
import uuid
from flask import Flask, jsonify, request, send_from_directory
from db.database import get_db
from predictors.goals import predict_goals
from predictors.corners import predict_corners

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory('static', 'epl.html')

@app.route('/epl')
def epl_page():
    return send_from_directory('static', 'epl.html')

@app.route('/calculator')
def calculator_page():
    return send_from_directory('static', 'calculator.html')


# ---------------------------------------------------------------------------
# EPL 2026/27 prediction endpoints
# ---------------------------------------------------------------------------
EPL_COMPETITION = 'epl_2026'
EPL_MATCHES_PER_GW = 10


@app.route('/api/epl/gameweeks')
def epl_gameweeks():
    """List gameweeks and detect the current one (first GW with upcoming matches)."""
    from datetime import datetime

    with get_db() as db:
        cur = db.cursor()
        cur.execute('''
            SELECT m.id, m.date, m.status
            FROM matches m
            WHERE m.competition_id = ?
            ORDER BY m.date
        ''', (EPL_COMPETITION,))
        all_matches = cur.fetchall()

    if not all_matches:
        return jsonify({'success': False, 'error': 'No EPL 2026/27 data. Run fetch first.'})

    total_gws = (len(all_matches) + EPL_MATCHES_PER_GW - 1) // EPL_MATCHES_PER_GW

    # Detect current gameweek: first GW that has at least one NS match
    now = datetime.utcnow().isoformat()
    current_gw = 1
    for gw in range(1, total_gws + 1):
        offset = (gw - 1) * EPL_MATCHES_PER_GW
        chunk = all_matches[offset:offset + EPL_MATCHES_PER_GW]
        has_upcoming = any(m['status'] == 'NS' for m in chunk)
        if has_upcoming:
            current_gw = gw
            break
    else:
        current_gw = total_gws  # all played

    return jsonify({
        'success': True,
        'total_gameweeks': total_gws,
        'current_gameweek': current_gw
    })


def _parse_line_bet(bet_label):
    """Parse 'Over 2.5' / 'Under 11.5' into (direction, line)."""
    if not bet_label:
        return None, None
    parts = bet_label.split()
    if len(parts) != 2:
        return None, None
    direction, line = parts[0].lower(), float(parts[1])
    return direction, line


def _bet_won(bet_label, actual_total):
    """Did the recommended over/under bet win against the actual total?"""
    if actual_total is None:
        return None
    direction, line = _parse_line_bet(bet_label)
    if direction is None:
        return None
    return (actual_total > line) if direction == 'over' else (actual_total < line)


def _grade_acca_leg(leg, fixtures_by_id):
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


@app.route('/api/epl/gameweek/<int:gw>')
def epl_gameweek_predictions(gw):
    """Return predictions for every fixture in a gameweek."""
    if gw < 1:
        return jsonify({'success': False, 'error': 'Invalid gameweek'})

    offset = (gw - 1) * EPL_MATCHES_PER_GW

    with get_db() as db:
        cur = db.cursor()
        cur.execute('''
            SELECT m.id, m.date, m.status,
                   ht.name AS home, at.name AS away,
                   m.home_score, m.away_score
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.competition_id = ?
            ORDER BY m.date
            LIMIT ? OFFSET ?
        ''', (EPL_COMPETITION, EPL_MATCHES_PER_GW, offset))
        matches = cur.fetchall()

        finished_ids = [m['id'] for m in matches if m['status'] in ('FT', 'AET', 'PEN')]
        corners_by_match = {}
        if finished_ids:
            placeholders = ','.join('?' for _ in finished_ids)
            cur.execute(f'''
                SELECT match_id, SUM(corners) AS c
                FROM match_stats
                WHERE match_id IN ({placeholders})
                GROUP BY match_id
            ''', finished_ids)
            corners_by_match = {row['match_id']: row['c'] for row in cur.fetchall()}

    if not matches:
        return jsonify({'success': False, 'error': 'No fixtures for this gameweek.'})

    fixtures = []
    for match in matches:
        home_name = match['home']
        away_name = match['away']

        try:
            gp = predict_goals(home_name, away_name, EPL_COMPETITION, verbose=False)
            cp = predict_corners(home_name, away_name, EPL_COMPETITION, verbose=False)
        except Exception as e:
            print(f"EPL predict error {home_name} v {away_name}: {e}")
            continue
        if not gp or not cp:
            continue

        res = gp['results']
        h, d, a = res['home_win_prob'], res['draw_prob'], res['away_win_prob']
        s = (h + d + a) or 1.0
        h, d, a = h / s * 100, d / s * 100, a / s * 100

        top_score = list(res['score_probs'].keys())[0]
        score_str = f"{top_score[0]}–{top_score[1]}"

        total_exp = cp['results']['total_expected']
        if total_exp < 9.0:
            rec_bet = "Under 11.5"
        elif total_exp < 10.0:
            rec_bet = "Over 7.5"
        else:
            rec_bet = "Over 8.5"

        total_goal_exp = gp['home_xg'] + gp['away_xg']
        if total_goal_exp > 3.0:
            rec_goal_bet = "Over 2.5"
        elif total_goal_exp > 2.5:
            rec_goal_bet = "Over 1.5"
        else:
            rec_goal_bet = "Under 3.5"

        fixture = {
            'match_id': match['id'],
            'date': match['date'],
            'status': match['status'],
            'home_team': home_name,
            'away_team': away_name,
            'goals': {
                'home_xg': round(gp['home_xg'], 2),
                'away_xg': round(gp['away_xg'], 2),
                'recommended_bet': rec_goal_bet,
                'win_prob': round(h, 1),
                'draw_prob': round(d, 1),
                'loss_prob': round(a, 1),
                'most_likely_score': score_str,
                'btts_prob': round(res.get('btts_prob', 0), 1),
                'over_2_5_prob': round(res.get('over_lines', {}).get(2.5, 0), 1),
            },
            'corners': {
                'home_expected': round(cp['home_expected'], 1),
                'away_expected': round(cp['away_expected'], 1),
                'total_expected': round(total_exp, 1),
                'recommended_bet': rec_bet,
                'over_8_5_prob': round(cp['results']['over_lines'].get(8.5, 0), 1),
                'over_9_5_prob': round(cp['results']['over_lines'].get(9.5, 0), 1),
            },
        }

        # Include actual result + bet grading if match is finished
        if match['status'] in ('FT', 'AET', 'PEN'):
            hs, as_ = match['home_score'], match['away_score']
            total_goals = hs + as_
            actual_corners = corners_by_match.get(match['id'])

            fixture['actual'] = {
                'home_score': hs,
                'away_score': as_,
                'corners': actual_corners,
            }
            fixture['goals']['bet_won'] = _bet_won(rec_goal_bet, total_goals)
            fixture['corners']['bet_won'] = _bet_won(rec_bet, actual_corners)

        fixtures.append(fixture)

    acc_legs = []
    if fixtures:
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
        over_corner_cands = [f for f in fixtures if f['corners']['total_expected'] >= 9.5]
        if over_corner_cands:
            best_corner_over = max(over_corner_cands, key=lambda f: f['corners']['total_expected'])
            acc_legs.append({
                'match_id': best_corner_over['match_id'],
                'match': f"{best_corner_over['home_team']} vs {best_corner_over['away_team']}",
                'pick': "Over 6.5 Corners",
                'reason': f"High Expected Corners ({best_corner_over['corners']['total_expected']})"
            })
            
        # 3. Safest Corner Under (lowest Exp Corners, < 9.0)
        under_corner_cands = [f for f in fixtures if f['corners']['total_expected'] < 9.0]
        if under_corner_cands:
            best_corner_under = min(under_corner_cands, key=lambda f: f['corners']['total_expected'])
            acc_legs.append({
                'match_id': best_corner_under['match_id'],
                'match': f"{best_corner_under['home_team']} vs {best_corner_under['away_team']}",
                'pick': "Under 13.5 Corners",
                'reason': f"Low Expected Corners ({best_corner_under['corners']['total_expected']})"
            })
            
        # If we don't have 3 legs, fill with Goal Under (lowest xG, <= 2.5)
        if len(acc_legs) < 3:
            under_goal_cands = [f for f in fixtures if (f['goals']['home_xg'] + f['goals']['away_xg']) <= 2.5]
            picked_matches = [leg['match'] for leg in acc_legs]
            under_goal_cands = [f for f in under_goal_cands if f"{f['home_team']} vs {f['away_team']}" not in picked_matches]
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
            under_c_cands2 = [f for f in fixtures if f['corners']['total_expected'] < 9.5]
            picked_matches = [leg['match'] for leg in acc_legs]
            under_c_cands2 = [f for f in under_c_cands2 if f"{f['home_team']} vs {f['away_team']}" not in picked_matches]
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

    fixtures_by_id = {f['match_id']: f for f in fixtures}
    for leg in acca + corner_acca_legs:
        leg['won'] = _grade_acca_leg(leg, fixtures_by_id)

    return jsonify({'success': True, 'gameweek': gw, 'fixtures': fixtures, 'acca': acca, 'corner_acca': corner_acca_legs})


@app.route('/api/predictions')
def get_predictions():
    competition_id = 'world_cup_2026'
    
    with get_db() as db:
        cur = db.cursor()
        cur.execute('''
            SELECT m.id, ht.name as home, at.name as away, m.date, m.status
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.competition_id = ? AND m.status = 'NS'
            ORDER BY m.date
        ''', (competition_id,))
        
        matches = cur.fetchall()
        
    predictions = []
    
    # Filter out placeholder teams (like W97, RU101)
    placeholder_pattern = re.compile(r'^(W|RU)\d+$')
    
    for match in matches:
        home_name = match['home']
        away_name = match['away']
        
        if placeholder_pattern.match(home_name) or placeholder_pattern.match(away_name):
            continue
            
        try:
            goal_pred = predict_goals(home_name, away_name, competition_id)
            corner_pred = predict_corners(home_name, away_name, competition_id)
            
            if goal_pred and corner_pred:
                total_exp = corner_pred['results']['total_expected']
                if total_exp < 9.0:
                    rec_bet = "Under 11.5"
                elif total_exp < 10.0:
                    rec_bet = "Over 7.5"
                else:
                    rec_bet = "Over 8.5"

                total_goal_exp = goal_pred['home_xg'] + goal_pred['away_xg']
                if total_goal_exp > 3.0:
                    rec_goal_bet = "Over 2.5"
                elif total_goal_exp > 2.5:
                    rec_goal_bet = "Over 1.5"
                else:
                    rec_goal_bet = "Under 3.5"

                predictions.append({
                    'id': match['id'],
                    'date': match['date'],
                    'home_team': home_name,
                    'away_team': away_name,
                    'goals': {
                        'home_xg': round(goal_pred['home_xg'], 2),
                        'away_xg': round(goal_pred['away_xg'], 2),
                        'recommended_bet': rec_goal_bet,
                        'win_prob': round(goal_pred['results']['home_win_prob'], 1),
                        'draw_prob': round(goal_pred['results']['draw_prob'], 1),
                        'loss_prob': round(goal_pred['results']['away_win_prob'], 1),
                        'most_likely_score': list(goal_pred['results']['score_probs'].keys())[0],
                        'btts_prob': round(goal_pred['results'].get('btts_prob', 0), 1),
                        'over_2_5_prob': round(goal_pred['results'].get('over_lines', {}).get(2.5, 0), 1)
                    },
                    'corners': {
                        'home_expected': round(corner_pred['home_expected'], 1),
                        'away_expected': round(corner_pred['away_expected'], 1),
                        'total_expected': round(total_exp, 1),
                        'recommended_bet': rec_bet,
                        'most_likely_total': corner_pred['results']['most_likely_score'],
                        'over_8_5_prob': round(corner_pred['results']['over_lines'].get(8.5, 0), 1),
                        'over_9_5_prob': round(corner_pred['results']['over_lines'].get(9.5, 0), 1)
                    }
                })
        except Exception as e:
            print(f"Error predicting {home_name} vs {away_name}: {e}")
            continue
            
    return jsonify({'success': True, 'predictions': predictions})

# ---------------------------------------------------------------------------
# "Beat the Model" game: guess a random past (2024/25) EPL fixture's markets and
# see how you did versus the model. The model prediction is point-in-time
# (as_of_date = match date) so it is leak-free — exactly what it would have said
# before kickoff. The actual result is only revealed server-side at grade time.
# ---------------------------------------------------------------------------
GAME_COMPETITION = 'epl_2024'
CORNER_LINES = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
_ROUNDS = {}  # round_id -> {match_id, model_picks}  (in-memory; fine for local play)


def _poisson_over(mean, line):
    """P(total > line) for a half-integer line, using a Poisson(mean)."""
    k_max = math.floor(line)
    cdf = sum(math.exp(-mean) * mean ** k / math.factorial(k) for k in range(k_max + 1))
    return max(0.0, min(1.0, 1.0 - cdf))


@app.route('/game')
def game_page():
    return send_from_directory('static', 'game.html')


@app.route('/api/game/fixture')
def game_fixture():
    """Return a random completed fixture + the model's leak-free prediction.
    Does NOT include the actual result."""
    with get_db() as db:
        cur = db.cursor()
        cur.execute('''
            SELECT m.id, m.date, ht.name AS home, at.name AS away
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN')
            ORDER BY RANDOM() LIMIT 8
        ''', (GAME_COMPETITION,))
        candidates = cur.fetchall()

    for match in candidates:
        try:
            gp = predict_goals(match['home'], match['away'], GAME_COMPETITION,
                               as_of_date=match['date'], verbose=False)
            cp = predict_corners(match['home'], match['away'], GAME_COMPETITION,
                                 as_of_date=match['date'], verbose=False)
        except Exception as e:
            print(f"Game predict error {match['home']} v {match['away']}: {e}")
            continue
        if not gp or not cp:
            continue

        res = gp['results']
        h, d, a = res['home_win_prob'], res['draw_prob'], res['away_win_prob']
        s = (h + d + a) or 1.0
        h, d, a = h / s * 100, d / s * 100, a / s * 100
        wdl = ['home', 'draw', 'away'][max(range(3), key=lambda i: (h, d, a)[i])]

        btts = res.get('btts_prob', 0.0)
        o15 = res.get('over_lines', {}).get(1.5, 0.0)
        o25 = res.get('over_lines', {}).get(2.5, 0.0)

        top_score = list(res['score_probs'].keys())[0]  # most likely scoreline (tuple)
        model_score = f"{top_score[0]}–{top_score[1]}"

        corner_mean = cp['results']['total_expected']
        corner_probs = {L: round(_poisson_over(corner_mean, L) * 100, 1) for L in CORNER_LINES}
        over50 = [L for L in CORNER_LINES if corner_probs[L] >= 50]
        model_corner_line = max(over50) if over50 else min(CORNER_LINES)

        model_picks = {
            'wdl': wdl,
            'gg': 'yes' if btts >= 50 else 'no',
            'o15': 'yes' if o15 >= 50 else 'no',
            'o25': 'yes' if o25 >= 50 else 'no',
            'corner_line': model_corner_line,
        }

        round_id = uuid.uuid4().hex
        _ROUNDS[round_id] = {'match_id': match['id'], 'model_picks': model_picks}
        # Cap memory for long sessions
        if len(_ROUNDS) > 500:
            _ROUNDS.pop(next(iter(_ROUNDS)))

        return jsonify({
            'success': True,
            'round_id': round_id,
            'fixture': {'home': match['home'], 'away': match['away'], 'date': match['date'][:10]},
            'model': {
                'home_xg': round(gp['home_xg'], 2), 'away_xg': round(gp['away_xg'], 2),
                'score': model_score,
                'home_win': round(h, 1), 'draw': round(d, 1), 'away_win': round(a, 1),
                'gg': round(btts, 1), 'o15': round(o15, 1), 'o25': round(o25, 1),
                'corner_expected': round(corner_mean, 1),
                'corner_probs': corner_probs,
                'picks': model_picks,
            },
        })

    return jsonify({'success': False, 'error': 'Could not build a fixture. Try again.'})


@app.route('/api/game/grade', methods=['POST'])
def game_grade():
    """Grade the user's picks (and the model's) against the real result."""
    data = request.get_json(force=True) or {}
    rnd = _ROUNDS.get(data.get('round_id'))
    if not rnd:
        return jsonify({'success': False, 'error': 'Round expired — start a new fixture.'}), 404

    user = data.get('picks', {})
    model = rnd['model_picks']

    with get_db() as db:
        cur = db.cursor()
        m = cur.execute(
            'SELECT home_score AS hs, away_score AS a_s FROM matches WHERE id = ?',
            (rnd['match_id'],)).fetchone()
        c = cur.execute(
            'SELECT SUM(corners) AS c FROM match_stats WHERE match_id = ?',
            (rnd['match_id'],)).fetchone()

    hs, as_ = m['hs'], m['a_s']
    total = hs + as_
    corners = c['c']
    outcome = 'home' if hs > as_ else 'away' if as_ > hs else 'draw'
    actual = {
        'wdl': outcome,
        'gg': 'yes' if (hs > 0 and as_ > 0) else 'no',
        'o15': 'yes' if total > 1.5 else 'no',
        'o25': 'yes' if total > 2.5 else 'no',
        'home_score': hs, 'away_score': as_, 'corners': corners,
    }

    def corner_over(line):
        return corners is not None and corners > line

    cats = []
    for key, label in [('wdl', '1X2'), ('gg', 'GG'), ('o15', 'Over 1.5'), ('o25', 'Over 2.5')]:
        cats.append({
            'label': label,
            'you': user.get(key), 'model': model.get(key), 'actual': actual[key],
            'you_correct': user.get(key) == actual[key],
            'model_correct': model.get(key) == actual[key],
        })
    # Corners: "over the chosen line" — correct if actual total exceeds it
    u_line = user.get('corner_line')
    cats.append({
        'label': 'Corners',
        'you': f"Over {u_line}" if u_line is not None else '—',
        'model': f"Over {model['corner_line']}",
        'actual': f"{corners} corners" if corners is not None else 'n/a',
        'you_correct': corner_over(float(u_line)) if u_line is not None else False,
        'model_correct': corner_over(model['corner_line']),
    })

    _ROUNDS.pop(data['round_id'], None)
    return jsonify({
        'success': True,
        'actual': actual,
        'categories': cats,
        'you_score': sum(1 for c in cats if c['you_correct']),
        'model_score': sum(1 for c in cats if c['model_correct']),
    })


@app.route('/game/gameweek')
def game_gameweek_page():
    return send_from_directory('static', 'game_gw.html')


@app.route('/api/game/gameweeks')
def game_gameweeks():
    with get_db() as db:
        cur = db.cursor()
        cur.execute('''
            SELECT id, date FROM matches 
            WHERE competition_id = ? AND status IN ('FT', 'AET', 'PEN')
            ORDER BY date
        ''', (GAME_COMPETITION,))
        all_matches = cur.fetchall()
    
    # Chunk into groups of 10
    total_gws = (len(all_matches) + 9) // 10
    return jsonify({'success': True, 'total_gameweeks': total_gws})


@app.route('/api/game/gameweek/<int:gw_id>')
def game_gameweek_fixtures(gw_id):
    if gw_id < 1:
        return jsonify({'success': False, 'error': 'Invalid Gameweek'})
        
    offset = (gw_id - 1) * 10
    limit = 10
    
    with get_db() as db:
        cur = db.cursor()
        cur.execute('''
            SELECT m.id, m.date, ht.name AS home, at.name AS away
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN')
            ORDER BY m.date
            LIMIT ? OFFSET ?
        ''', (GAME_COMPETITION, limit, offset))
        matches = cur.fetchall()

    if not matches:
        return jsonify({'success': False, 'error': 'No fixtures found for this gameweek.'})

    fixtures = []
    round_id = uuid.uuid4().hex
    gw_models = {}

    for match in matches:
        try:
            gp = predict_goals(match['home'], match['away'], GAME_COMPETITION,
                               as_of_date=match['date'], verbose=False)
            cp = predict_corners(match['home'], match['away'], GAME_COMPETITION,
                                 as_of_date=match['date'], verbose=False)
        except Exception as e:
            print(f"Game predict error {match['home']} v {match['away']}: {e}")
            continue
        if not gp or not cp:
            continue

        res = gp['results']
        h, d, a = res['home_win_prob'], res['draw_prob'], res['away_win_prob']
        s = (h + d + a) or 1.0
        h, d, a = h / s * 100, d / s * 100, a / s * 100
        wdl = ['home', 'draw', 'away'][max(range(3), key=lambda i: (h, d, a)[i])]

        btts = res.get('btts_prob', 0.0)
        o15 = res.get('over_lines', {}).get(1.5, 0.0)
        o25 = res.get('over_lines', {}).get(2.5, 0.0)

        top_score = list(res['score_probs'].keys())[0]  
        model_score = f"{top_score[0]}–{top_score[1]}"

        corner_mean = cp['results']['total_expected']
        corner_probs = {L: round(_poisson_over(corner_mean, L) * 100, 1) for L in CORNER_LINES}
        over50 = [L for L in CORNER_LINES if corner_probs[L] >= 50]
        model_corner_line = max(over50) if over50 else min(CORNER_LINES)

        model_picks = {
            'wdl': wdl,
            'gg': 'yes' if btts >= 50 else 'no',
            'o15': 'yes' if o15 >= 50 else 'no',
            'o25': 'yes' if o25 >= 50 else 'no',
            'corner_line': model_corner_line,
        }

        gw_models[str(match['id'])] = model_picks

        fixtures.append({
            'match_id': str(match['id']),
            'home': match['home'],
            'away': match['away'],
            'date': match['date'][:10],
            'model': {
                'home_xg': round(gp['home_xg'], 2), 'away_xg': round(gp['away_xg'], 2),
                'score': model_score,
                'home_win': round(h, 1), 'draw': round(d, 1), 'away_win': round(a, 1),
                'gg': round(btts, 1), 'o15': round(o15, 1), 'o25': round(o25, 1),
                'corner_expected': round(corner_mean, 1),
                'corner_probs': corner_probs,
                'picks': model_picks,
            }
        })

    _ROUNDS[round_id] = {'type': 'gameweek', 'fixtures': gw_models}
    if len(_ROUNDS) > 500:
        _ROUNDS.pop(next(iter(_ROUNDS)))

    return jsonify({
        'success': True,
        'round_id': round_id,
        'fixtures': fixtures
    })

@app.route('/api/game/grade_gameweek', methods=['POST'])
def game_grade_gameweek():
    data = request.get_json(force=True) or {}
    rnd = _ROUNDS.get(data.get('round_id'))
    if not rnd or rnd.get('type') != 'gameweek':
        return jsonify({'success': False, 'error': 'Round expired.'}), 404

    user_picks = data.get('picks', {})
    active_markets = data.get('markets', ['wdl', 'gg', 'o15', 'o25', 'corners'])
    gw_models = rnd['fixtures']

    match_ids = list(gw_models.keys())
    if not match_ids:
        return jsonify({'success': False, 'error': 'No matches found.'}), 404

    placeholders = ','.join('?' for _ in match_ids)
    
    with get_db() as db:
        cur = db.cursor()
        matches = cur.execute(f'''
            SELECT id, home_score, away_score
            FROM matches WHERE id IN ({placeholders})
        ''', match_ids).fetchall()
        
        c_stats = cur.execute(f'''
            SELECT match_id, SUM(corners) AS c 
            FROM match_stats WHERE match_id IN ({placeholders})
            GROUP BY match_id
        ''', match_ids).fetchall()
        
    corners_map = {str(c['match_id']): c['c'] for c in c_stats}
    match_actuals = {}
    
    for m in matches:
        mid = str(m['id'])
        hs, as_ = m['home_score'], m['away_score']
        total = hs + as_
        outcome = 'home' if hs > as_ else 'away' if as_ > hs else 'draw'
        c = corners_map.get(mid)
        match_actuals[mid] = {
            'wdl': outcome,
            'gg': 'yes' if (hs > 0 and as_ > 0) else 'no',
            'o15': 'yes' if total > 1.5 else 'no',
            'o25': 'yes' if total > 2.5 else 'no',
            'home_score': hs, 'away_score': as_, 'corners': c
        }

    results = []
    total_user_score = 0
    total_model_score = 0
    
    for mid in match_ids:
        actual = match_actuals.get(mid)
        if not actual: continue
        
        user = user_picks.get(mid, {})
        model = gw_models[mid]
        c = actual['corners']
        
        def corner_over(line):
            return c is not None and c > line

        cats = []
        for key, label in [('wdl', '1X2'), ('gg', 'GG'), ('o15', 'O1.5'), ('o25', 'O2.5')]:
            if key not in active_markets:
                continue
            u_correct = user.get(key) == actual[key] if user.get(key) else False
            m_correct = model.get(key) == actual[key]
            cats.append({
                'label': label,
                'you': user.get(key), 'model': model.get(key), 'actual': actual[key],
                'you_correct': u_correct, 'model_correct': m_correct
            })
            if u_correct: total_user_score += 1
            if m_correct: total_model_score += 1
            
        if 'corners' in active_markets:
            u_line = user.get('corner_line')
            u_c_correct = corner_over(float(u_line)) if u_line else False
            m_c_correct = corner_over(model['corner_line'])
            cats.append({
                'label': 'Corners',
                'you': f"Over {u_line}" if u_line else '—',
                'model': f"Over {model['corner_line']}",
                'actual': f"{c} corners" if c is not None else 'n/a',
                'you_correct': u_c_correct,
                'model_correct': m_c_correct
            })
            if u_c_correct: total_user_score += 1
            if m_c_correct: total_model_score += 1
        
        results.append({
            'match_id': mid,
            'actual_summary': f"{actual['home_score']}–{actual['away_score']} ({c} c)",
            'categories': cats
        })

    _ROUNDS.pop(data['round_id'], None)
    return jsonify({
        'success': True,
        'results': results,
        'you_score': total_user_score,
        'model_score': total_model_score
    })


if __name__ == '__main__':
    # Ensure static directory exists
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
