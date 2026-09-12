import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from predictors.corners import predict_corners

def get_optimized_pick(expected_corners):
    if expected_corners < 9.0:
        return "Under 11.5", lambda actual: actual < 11.5
    elif expected_corners < 10.0:
        return "Over 7.5", lambda actual: actual > 7.5
    else:
        return "Over 8.5", lambda actual: actual > 8.5

def run_gw2_predictions():
    print("\n--- GW 2 PREDICTIONS (epl_2026) ---")
    competition_id = "epl_2026"
    with get_db() as db:
        cur = db.cursor()
        cur.execute(
            """
            SELECT m.id, m.date, ht.name as home_name, at.name as away_name, m.status
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.competition_id = ?
            ORDER BY m.date ASC
            """,
            (competition_id,),
        )
        matches = cur.fetchall()
        
    if len(matches) < 20:
        print("Not enough matches for GW 2")
        return
        
    gw2_matches = matches[10:20]
    
    for match in gw2_matches:
        try:
            pred = predict_corners(
                match["home_name"],
                match["away_name"],
                competition_id,
                as_of_date=match["date"],
                verbose=False,
            )
            if not pred:
                continue
            
            expected = pred["results"]["total_expected"]
            pick_name, _ = get_optimized_pick(expected)
            print(f"{match['date'][:10]} | {match['home_name']:20} vs {match['away_name']:20} | Exp: {expected:5.2f} -> BET: {pick_name}")
            
        except Exception as e:
            print(f"Error predicting {match['home_name']} vs {match['away_name']}: {e}")

def run_gw_backtest():
    competitions = ["epl_2024", "epl_2025"]
    
    gw_scores = {
        100: 0,
        90: 0,
        80: 0,
        70: 0,
        60: 0,
        50: 0,
        40: 0,
        30: 0,
        20: 0,
        10: 0,
        0: 0
    }
    
    total_gws = 0
    
    for competition_id in competitions:
        print(f"\nProcessing backtest for {competition_id}...")
        with get_db() as db:
            cur = db.cursor()
            cur.execute(
                """
                SELECT m.id, m.date, ht.name as home_name, at.name as away_name
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
            
            wins = 0
            valid_matches = 0
            
            for match in gw_matches:
                actual_total = corner_totals.get(match["id"])
                if actual_total is None:
                    continue

                try:
                    pred = predict_corners(
                        match["home_name"],
                        match["away_name"],
                        competition_id,
                        as_of_date=match["date"],
                        verbose=False,
                    )
                except Exception:
                    continue

                if not pred:
                    continue

                expected = pred["results"]["total_expected"]
                _, eval_func = get_optimized_pick(expected)
                
                if eval_func(actual_total):
                    wins += 1
                valid_matches += 1
                
            if valid_matches > 0:
                pct = int(round((wins / valid_matches) * 10)) * 10
                if pct in gw_scores:
                    gw_scores[pct] += 1
                    total_gws += 1
                    
    print("\n--- Gameweek Performance Breakdown ---")
    print(f"Total Gameweeks Analyzed: {total_gws}")
    print("How many Gameweeks hit each win rate:")
    for score in sorted(gw_scores.keys(), reverse=True):
        print(f"{score:>3}% Win Rate: {gw_scores[score]:>3} Gameweeks")


if __name__ == "__main__":
    run_gw2_predictions()
    run_gw_backtest()
