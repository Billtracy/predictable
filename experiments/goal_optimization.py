import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from predictors.goals import predict_goals

def run_goal_optimization():
    competitions = ["epl_2024", "epl_2025"]
    
    # Store tuples: (total_xg, actual_home_score, actual_away_score, btts_prob)
    data_points = []
    
    for competition_id in competitions:
        print(f"Fetching data for {competition_id}...")
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
            
        for match in matches:
            actual_home = match["home_score"]
            actual_away = match["away_score"]
            if actual_home is None or actual_away is None:
                continue

            try:
                pred = predict_goals(
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

            total_xg = pred["home_xg"] + pred["away_xg"]
            btts_prob = pred["results"].get("btts_prob", 0)
            
            data_points.append((total_xg, actual_home, actual_away, btts_prob))

    print(f"\nCollected {len(data_points)} matches for goal optimization.")
    
    # Bins for Total Expected Goals (xG)
    bins = [
        (0.0, 2.5, "<= 2.50"),
        (2.5, 3.0, "2.51 - 3.00"),
        (3.0, 3.5, "3.01 - 3.50"),
        (3.5, 10.0, "> 3.50")
    ]
    
    markets = [
        ("Over 1.5", lambda h, a: (h + a) > 1.5),
        ("Over 2.5", lambda h, a: (h + a) > 2.5),
        ("Under 2.5", lambda h, a: (h + a) < 2.5),
        ("Under 3.5", lambda h, a: (h + a) < 3.5),
        ("Under 4.5", lambda h, a: (h + a) < 4.5),
        ("BTTS - Yes", lambda h, a: h > 0 and a > 0),
        ("BTTS - No", lambda h, a: h == 0 or a == 0),
    ]

    print("\n--- Win Rates by Total xG Bins ---")
    
    best_strategy = []
    total_wins = 0
    total_matches = 0
    
    for b_min, b_max, b_name in bins:
        bin_pts = [pt for pt in data_points if b_min < pt[0] <= b_max]
        if not bin_pts:
            continue
            
        print(f"\nBin: {b_name} (Matches: {len(bin_pts)})")
        
        candidates = []
        for m_name, m_func in markets:
            wins = sum(1 for pt in bin_pts if m_func(pt[1], pt[2]))
            win_rate = wins / len(bin_pts) * 100
            marker = "*" if win_rate >= 68.0 else " "
            print(f"  {marker} {m_name:12}: {win_rate:5.1f}% ({wins}/{len(bin_pts)})")
            
            # Record viable candidates
            if win_rate >= 60.0:
                candidates.append((m_name, win_rate, wins))
                
        # Sort by win rate to find recommendations
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out extreme "safe" bets (like Over 1.5 or Under 4.5) to find value
        # Usually Over 2.5, Under 3.5, Under 2.5, and BTTS provide good betting value
        filtered_cands = [c for c in candidates if c[0] not in ["Over 1.5", "Under 4.5"]]
        
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
    print("🏆 OPTIMIZED RECOMMENDED GOAL STRATEGY 🏆")
    print("="*70)
    print(f"{'Total xG':<15} | {'Recommended Bet':<15} | {'Win Rate':<10} | {'Record'}")
    print("-" * 70)
    for b_name, b_count, m_name, wr, wins in best_strategy:
        print(f"{b_name:<15} | {m_name:<15} | {wr:>6.2f}%    | {wins}/{b_count}")
    print("-" * 70)
    cum_wr = total_wins / total_matches * 100 if total_matches > 0 else 0
    print(f"{'CUMULATIVE':<15} | {'MIXED':<15} | {cum_wr:>6.2f}%    | {total_wins}/{total_matches}")
    print("="*70)


if __name__ == "__main__":
    run_goal_optimization()
