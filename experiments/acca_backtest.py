import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from predictors.corners import predict_corners
from predictors.goals import predict_goals
from predictors.acca import build_accas, grade_leg

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
                    'match_id': match["id"],
                    'home_team': match["home_name"],
                    'away_team': match["away_name"],
                    'actual': {'home_score': actual_home, 'away_score': actual_away, 'corners': actual_corners},
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
                
            acca, _ = build_accas(fixtures)
            if len(acca) < 3:
                continue

            # Evaluate the Acca
            acca_won = True
            # print(f"GW {i+1} ({competition_id}):")
            fixtures_by_id = {f['match_id']: f for f in fixtures}
            for leg in acca:
                if not grade_leg(leg, fixtures_by_id):
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
