import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from predictors.corners import predict_corners

def run_optimization():
    competitions = ["epl_2024", "epl_2025"]
    
    # Store tuples of (expected, actual)
    data_points = []
    
    for competition_id in competitions:
        print(f"Fetching data for {competition_id}...")
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
            except Exception:
                continue

            if not pred:
                continue

            expected_corners = pred["results"]["total_expected"]
            data_points.append((expected_corners, actual_total))

    print(f"\nCollected {len(data_points)} matches for optimization.")
    
    # Define bins for Expected Corners
    bins = [
        (0, 8.0, "<= 8.0"),
        (8.0, 9.0, "8.0 - 9.0"),
        (9.0, 10.0, "9.0 - 10.0"),
        (10.0, 11.0, "10.0 - 11.0"),
        (11.0, 100.0, ">= 11.0")
    ]
    
    markets = [
        ("Over 6.5", lambda a: a > 6.5),
        ("Over 7.5", lambda a: a > 7.5),
        ("Over 8.5", lambda a: a > 8.5),
        ("Over 9.5", lambda a: a > 9.5),
        ("Under 10.5", lambda a: a < 10.5),
        ("Under 11.5", lambda a: a < 11.5),
        ("Under 12.5", lambda a: a < 12.5),
        ("Under 13.5", lambda a: a < 13.5),
    ]

    print("\n--- Win Rates by Expected Corner Bins ---")
    
    best_strategy = []
    total_wins = 0
    total_matches = 0
    
    for b_min, b_max, b_name in bins:
        # Filter points in this bin
        bin_pts = [pt for pt in data_points if b_min < pt[0] <= b_max]
        if not bin_pts:
            continue
            
        print(f"\nBin: {b_name} (Matches: {len(bin_pts)})")
        
        best_market = None
        best_win_rate = 0
        best_wins = 0
        
        for m_name, m_func in markets:
            wins = sum(1 for pt in bin_pts if m_func(pt[1]))
            win_rate = wins / len(bin_pts) * 100
            
            # Highlight strong performers (>75%)
            marker = "*" if win_rate >= 75.0 else " "
            print(f"  {marker} {m_name:12}: {win_rate:5.1f}% ({wins}/{len(bin_pts)})")
            
            # To avoid taking ultra-low odds like Over 6.5 blindly, let's look for a balance
            # of decent lines (e.g. Over 7.5/8.5 or Under 11.5/10.5) that hit > 75%.
            # Let's just track the highest win rate.
            if win_rate > best_win_rate:
                # penalize very 'safe' lines slightly to find value if they are close
                # e.g., Over 6.5 will almost always win, but odds are terrible.
                # In real betting, Over 7.5 / Under 11.5 are usually the minimum acceptable odds ~1.3-1.4
                if m_name not in ["Over 6.5", "Under 13.5"] or win_rate > best_win_rate + 5:
                    best_win_rate = win_rate
                    best_market = m_name
                    best_wins = wins
                    
        # Find highest realistic line
        # We will pick a recommended market for this bin that has >= 75% win rate and the best value
        # Value proxy: 
        # For Overs: higher line is better value (e.g. Over 8.5 > Over 7.5)
        # For Unders: lower line is better value (e.g. Under 10.5 > Under 11.5)
        candidates = []
        for m_name, m_func in markets:
            wins = sum(1 for pt in bin_pts if m_func(pt[1]))
            wr = wins / len(bin_pts) * 100
            if wr >= 72.0: # threshold for 'good'
                candidates.append((m_name, wr, wins))
                
        # Sort candidates: we want a good win rate, but we prefer value.
        # It's tricky to automate "value", so we'll just print the top 3 by win rate.
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Pick a "Recommendation" that balances high WR with reasonable line
        # Filter out extreme lines (Over 6.5, Under 13.5) if possible
        filtered_cands = [c for c in candidates if c[0] not in ["Over 6.5", "Under 13.5", "Under 12.5"]]
        if filtered_cands:
            rec = filtered_cands[0]
        elif candidates:
            rec = candidates[0]
        else:
            rec = ("No good line", 0, 0)
            
        best_strategy.append((b_name, len(bin_pts), rec[0], rec[1], rec[2]))
        total_wins += rec[2]
        total_matches += len(bin_pts)

    print("\n" + "="*70)
    print("🏆 OPTIMIZED RECOMMENDED STRATEGY 🏆")
    print("="*70)
    print(f"{'Expected Corners':<15} | {'Recommended Bet':<15} | {'Win Rate':<10} | {'Record'}")
    print("-" * 70)
    for b_name, b_count, m_name, wr, wins in best_strategy:
        print(f"{b_name:<15} | {m_name:<15} | {wr:>6.2f}%    | {wins}/{b_count}")
    print("-" * 70)
    cum_wr = total_wins / total_matches * 100 if total_matches > 0 else 0
    print(f"{'CUMULATIVE':<15} | {'MIXED':<15} | {cum_wr:>6.2f}%    | {total_wins}/{total_matches}")
    print("="*70)


if __name__ == "__main__":
    run_optimization()
