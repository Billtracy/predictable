import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from predictors.corners import predict_corners
from predictors.goals import predict_goals

def evaluate_pick(pick_name, actual_home, actual_away, actual_corners):
    if pick_name == "Over 1.5 Goals":
        return (actual_home + actual_away) > 1.5
    elif pick_name == "Under 4.5 Goals":
        return (actual_home + actual_away) < 4.5
    elif pick_name == "Over 6.5 Corners":
        return actual_corners > 6.5
    elif pick_name == "Under 12.5 Corners":
        return actual_corners < 12.5
    elif pick_name == "Under 13.5 Corners":
        return actual_corners < 13.5
    return False

def run_acca_backtest():
    competitions = ["epl_2024", "epl_2025"]
    
    total_accas = 0
    won_accas = 0
    lost_accas = 0
    
    for competition_id in competitions:
        print(f"\nProcessing acca backtest for {competition_id}...")
        with get_db() as db:
            cur = db.cursor()
            cur.execute(
                """
                SELECT m.id, m.date, ht.name as home_name, at.name as away_name, m.home_score, m.away_score
                FROM matches m
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN')
                ORDER BY m.date ASC
                """,
                (competition_id,),
            )
            matches = cur.fetchall()

            cur.execute(
                """
                SELECT ms.match_id, SUM(ms.corners) as total_corners, COUNT(*) as rows
                FROM match_stats ms
                JOIN matches m ON ms.match_id = m.id
                WHERE m.competition_id = ? AND ms.corners IS NOT NULL
                GROUP BY ms.match_id
                HAVING rows = 2
                """,
                (competition_id,),
            )
            corner_totals = {r["match_id"]: r["total_corners"] for r in cur.fetchall()}
            
        num_gws = len(matches) // 10
        
        for i in range(num_gws):
            gw_matches = matches[i*10 : (i+1)*10]
            
            fixtures = []
            
            for match in gw_matches:
                actual_corners = corner_totals.get(match["id"])
                actual_home = match["home_score"]
                actual_away = match["away_score"]
                
                if actual_corners is None or actual_home is None or actual_away is None:
                    continue

                try:
                    c_pred = predict_corners(
                        match["home_name"], match["away_name"], competition_id, as_of_date=match["date"], verbose=False
                    )
                    g_pred = predict_goals(
                        match["home_name"], match["away_name"], competition_id, as_of_date=match["date"], verbose=False
                    )
                except Exception:
                    continue

                if not c_pred or not g_pred:
                    continue
                    
                fixtures.append({
                    'home_team': match["home_name"],
                    'away_team': match["away_name"],
                    'actual_home': actual_home,
                    'actual_away': actual_away,
                    'actual_corners': actual_corners,
                    'goals': {
                        'home_xg': g_pred['home_xg'],
                        'away_xg': g_pred['away_xg'],
                    },
                    'corners': {
                        'total_expected': c_pred["results"]["total_expected"]
                    }
                })

            if len(fixtures) < 3:
                continue
                
            # Pick the 3 legs exactly as app.py does
            acc_legs = []
            
            over_goal_cands = [f for f in fixtures if (f['goals']['home_xg'] + f['goals']['away_xg']) > 2.5]
            if over_goal_cands:
                best_goal_over = max(over_goal_cands, key=lambda f: f['goals']['home_xg'] + f['goals']['away_xg'])
                acc_legs.append((best_goal_over, "Over 1.5 Goals"))
                
            over_corner_cands = [f for f in fixtures if f['corners']['total_expected'] >= 9.5]
            if over_corner_cands:
                best_corner_over = max(over_corner_cands, key=lambda f: f['corners']['total_expected'])
                acc_legs.append((best_corner_over, "Over 6.5 Corners"))
                
            under_corner_cands = [f for f in fixtures if f['corners']['total_expected'] < 9.0]
            if under_corner_cands:
                best_corner_under = min(under_corner_cands, key=lambda f: f['corners']['total_expected'])
                acc_legs.append((best_corner_under, "Under 13.5 Corners"))
                
            if len(acc_legs) < 3:
                under_goal_cands = [f for f in fixtures if (f['goals']['home_xg'] + f['goals']['away_xg']) <= 2.5]
                picked_matches = [leg[0]['home_team'] for leg in acc_legs]
                under_goal_cands = [f for f in under_goal_cands if f['home_team'] not in picked_matches]
                if under_goal_cands:
                    best_goal_under = min(under_goal_cands, key=lambda f: f['goals']['home_xg'] + f['goals']['away_xg'])
                    acc_legs.append((best_goal_under, "Under 4.5 Goals"))
                    
            if len(acc_legs) < 3:
                under_c_cands2 = [f for f in fixtures if f['corners']['total_expected'] < 9.5]
                picked_matches = [leg[0]['home_team'] for leg in acc_legs]
                under_c_cands2 = [f for f in under_c_cands2 if f['home_team'] not in picked_matches]
                if under_c_cands2:
                    best_c2 = min(under_c_cands2, key=lambda f: f['corners']['total_expected'])
                    acc_legs.append((best_c2, "Under 12.5 Corners"))
                    
            acca = acc_legs[:3]
            if len(acca) < 3:
                continue
                
            # Evaluate the Acca
            acca_won = True
            # print(f"GW {i+1} ({competition_id}):")
            for leg_fix, leg_pick in acca:
                leg_won = evaluate_pick(leg_pick, leg_fix['actual_home'], leg_fix['actual_away'], leg_fix['actual_corners'])
                if not leg_won:
                    acca_won = False
                    
            if acca_won:
                won_accas += 1
            else:
                lost_accas += 1
                
            total_accas += 1

    print("\n" + "=" * 50)
    print("🏆 ULTIMATE 3-LEG ACCA PERFORMANCE SUMMARY 🏆")
    print("=" * 50)
    print(f"Total Gameweeks Analyzed : {total_accas}")
    print(f"Total Accas WON          : {won_accas}")
    print(f"Total Accas LOST         : {lost_accas}")
    print(f"Overall Win Rate         : {(won_accas / total_accas * 100):.2f}%" if total_accas > 0 else "0.00%")

if __name__ == "__main__":
    run_acca_backtest()
