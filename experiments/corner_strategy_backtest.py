import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from predictors.corners import predict_corners

def run_strategy_backtest(competitions):
    results = {
        "<=8 (Under 11.5)": {"wins": 0, "losses": 0},
        "8-10 (Over 7.5)": {"wins": 0, "losses": 0},
        ">=10 (Over 8.5)": {"wins": 0, "losses": 0}
    }
    
    total_wins = 0
    total_losses = 0
    
    for competition_id in competitions:
        print(f"Running backtest for {competition_id}...")
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
            
        for match in matches:
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
            except Exception as e:
                continue

            if not pred:
                continue

            expected_corners = pred["results"]["total_expected"]
            
            won = False
            category = ""
            
            if expected_corners <= 8:
                category = "<=8 (Under 11.5)"
                if actual_total < 11.5:
                    won = True
            elif 8 < expected_corners < 10:
                category = "8-10 (Over 7.5)"
                if actual_total > 7.5:
                    won = True
            else:
                category = ">=10 (Over 8.5)"
                if actual_total > 8.5:
                    won = True
                    
            if won:
                results[category]["wins"] += 1
                total_wins += 1
            else:
                results[category]["losses"] += 1
                total_losses += 1

    print("\n--- Strategy Backtest Results ---")
    print(f"Competitions: {', '.join(competitions)}")
    print("-" * 65)
    for cat, stats in results.items():
        w = stats['wins']
        l = stats['losses']
        t = w + l
        wr = (w / t * 100) if t > 0 else 0
        print(f"{cat:20} | Wins: {w:4} | Losses: {l:4} | Total: {t:4} | Win Rate: {wr:.2f}%")
        
    print("-" * 65)
    tt = total_wins + total_losses
    twr = (total_wins / tt * 100) if tt > 0 else 0
    print(f"{'CUMULATIVE':20} | Wins: {total_wins:4} | Losses: {total_losses:4} | Total: {tt:4} | Win Rate: {twr:.2f}%")

if __name__ == "__main__":
    run_strategy_backtest(["epl_2024", "epl_2025"])
