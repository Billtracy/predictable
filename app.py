import os
import sys
import re
from flask import Flask, jsonify, request, send_from_directory
from db.database import get_db
from predictors.goals import predict_goals, goal_tip
import json
from predictors.acca import build_accas, grade_leg
from predictors.corners import predict_corners, corner_tip, CORNER_STRATEGIES, DEFAULT_CORNER_STRATEGY

app = Flask(__name__, static_folder='static')

# Champions League: a separate model with its own database (ucl/), mounted at /ucl
from ucl.web import ucl_bp
app.register_blueprint(ucl_bp)

@app.after_request
def cache_static_images(response):
    # Crests never change: let browsers keep them for 30 days instead of re-checking each visit
    if request.path.startswith('/static/img/'):
        response.headers['Cache-Control'] = 'public, max-age=2592000, immutable'
    return response


@app.route('/')
def index():
    return send_from_directory('static', 'epl.html')

@app.route('/epl')
def epl_page():
    return send_from_directory('static', 'epl.html')

@app.route('/calculator')
def calculator_page():
    return send_from_directory('static', 'calculator.html')

@app.route('/backtest')
def backtest_page():
    return send_from_directory('static', 'backtest.html')


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


def _corner_strategy_arg():
    """The ?corner_strategy= query param, or the default if missing/unknown."""
    strategy = request.args.get('corner_strategy', DEFAULT_CORNER_STRATEGY)
    return strategy if strategy in CORNER_STRATEGIES else DEFAULT_CORNER_STRATEGY


CORNER_BACKTEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'corner_strategy_backtest.json')


@app.route('/api/corner-strategies')
def corner_strategies():
    """Available corner tip strategies, plus backtest results if the comparison has been run."""
    backtest = {}
    if os.path.exists(CORNER_BACKTEST_FILE):
        with open(CORNER_BACKTEST_FILE) as fh:
            backtest = json.load(fh)
    return jsonify({
        'success': True,
        'default': DEFAULT_CORNER_STRATEGY,
        'strategies': [
            {'id': key, 'label': s['label'], 'description': s['description'],
             'backtest': backtest.get('strategies', {}).get(key)}
            for key, s in CORNER_STRATEGIES.items()
        ],
        'backtest_meta': backtest.get('meta'),
    })


@app.route('/api/backtest')
def backtest_results():
    """Leak-free backtest results written by experiments/corner_strategy_compare.py."""
    if not os.path.exists(CORNER_BACKTEST_FILE):
        return jsonify({'success': False, 'error': 'No backtest yet. Run experiments/corner_strategy_compare.py.'})
    with open(CORNER_BACKTEST_FILE) as fh:
        results = json.load(fh)
    return jsonify({'success': True, 'default_corner_strategy': DEFAULT_CORNER_STRATEGY,
                    'corner_strategies': {k: s['label'] for k, s in CORNER_STRATEGIES.items()}, **results})


@app.route('/api/epl/gameweek/<int:gw>')
def epl_gameweek_predictions(gw):
    """Return predictions for every fixture in a gameweek."""
    if gw < 1:
        return jsonify({'success': False, 'error': 'Invalid gameweek'})
    strategy = _corner_strategy_arg()

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
        rec_bet = corner_tip(strategy, total_exp, home_name, away_name)

        rec_goal_bet = goal_tip(gp['home_xg'] + gp['away_xg'])

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

    acca, corner_acca_legs = build_accas(fixtures)

    fixtures_by_id = {f['match_id']: f for f in fixtures}
    for leg in acca + corner_acca_legs:
        leg['won'] = grade_leg(leg, fixtures_by_id)

    return jsonify({'success': True, 'gameweek': gw, 'corner_strategy': strategy, 'fixtures': fixtures, 'acca': acca, 'corner_acca': corner_acca_legs})


@app.route('/api/predictions')
def get_predictions():
    competition_id = 'world_cup_2026'
    strategy = _corner_strategy_arg()
    
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
                rec_bet = corner_tip(strategy, total_exp, home_name, away_name)

                rec_goal_bet = goal_tip(goal_pred['home_xg'] + goal_pred['away_xg'])

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


if __name__ == '__main__':
    # Ensure static directory exists
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
