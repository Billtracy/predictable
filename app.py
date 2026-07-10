import os
import sys
import re
from flask import Flask, jsonify, send_from_directory
from db.database import get_db
from predictors.goals import predict_goals
from predictors.corners import predict_corners

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

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
                predictions.append({
                    'id': match['id'],
                    'date': match['date'],
                    'home_team': home_name,
                    'away_team': away_name,
                    'goals': {
                        'home_xg': round(goal_pred['home_xg'], 2),
                        'away_xg': round(goal_pred['away_xg'], 2),
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
                        'total_expected': round(corner_pred['results']['total_expected'], 1),
                        'most_likely_total': corner_pred['results']['most_likely_score'],
                        'over_8_5_prob': round(corner_pred['results']['over_lines'].get('8.5', 0), 1),
                        'over_9_5_prob': round(corner_pred['results']['over_lines'].get('9.5', 0), 1)
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
