import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.database import get_db
from predictors.corners import predict_corners

def run_custom_corners_backtest(competition_id='epl_2024'):
    with get_db() as db:
        cur = db.cursor()
        # Fetch completed matches
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

        # Fetch actual corner sums for matches
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

    if not matches or not corner_totals:
        print("No match or corner data found.")
        return

    deductions = [0.5, 1.5, 2.5]
    
    n_matches = len(matches)
    p1 = n_matches // 3
    p2 = 2 * n_matches // 3
    
    phases = [
        ("Phase 1 (Start to 1/3)", matches[:p1]),
        ("Phase 2 (1/3 to 2/3)", matches[p1:p2]),
        ("Phase 3 (2/3 to End)", matches[p2:])
    ]
    
    print(f"Running custom backtest for {competition_id} (Split into 3 phases)...")
    
    for phase_name, phase_matches in phases:
        results = {d: {"pass": 0, "fail": 0} for d in deductions}
        scored = 0
        print(f"\n" + "=" * 50)
        print(f"--- Running {phase_name} ({len(phase_matches)} matches) ---")
        print("=" * 50)
        
        for match in phase_matches:
            actual_total = corner_totals.get(match["id"])
            if actual_total is None:
                continue

            try:
                # Predict leak-free expected corners for the match
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

            # Get the model's expected corners
            expected_corners = pred["results"]["total_expected"]

            for d in deductions:
                target_line = expected_corners - d
                # If the actual result is higher or equal, it passes
                if actual_total >= target_line:
                    results[d]["pass"] += 1
                else:
                    results[d]["fail"] += 1

            scored += 1
            
            # Output progress occasionally
            if scored % 50 == 0:
                print(f"[{phase_name}] Processed {scored} matches...")

        print(f"\n>>> {phase_name} RESULTS <<<")
        print(f"Total Matches Scored: {scored}")
        for d in deductions:
            p = results[d]["pass"]
            f = results[d]["fail"]
            total = p + f
            if total > 0:
                pct_pass = (p / total) * 100
                pct_fail = (f / total) * 100
                print(f"  Deduction {d:>3}: Passed {p:>3} ({pct_pass:>5.1f}%) | Failed {f:>3} ({pct_fail:>5.1f}%)")
            else:
                print(f"  Deduction {d:>3}: No data")

if __name__ == "__main__":
    run_custom_corners_backtest('epl_2024')
