import argparse
from predictors.goals import predict_goals
from predictors.corners import predict_corners

def print_goal_predictions(p):
    print(f"\n⚽ MATCH PREDICTION: {p['home_team']} vs {p['away_team']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📊 Based on 10,000 simulations\n")
    print("GOALS")
    print(f"  {p['home_team']} xG: {p['home_xg']:.2f}    {p['away_team']} xG: {p['away_xg']:.2f}")
    
    res = p['results']
    print(f"  Win: {p['home_team']} {res['home_win_prob']:.1f}% | Draw {res['draw_prob']:.1f}% | {p['away_team']} {res['away_win_prob']:.1f}%")
    
    print("\n  Most Likely Scorelines:")
    scores = list(res['score_probs'].items())
    
    # Print in two columns
    for i in range(0, min(6, len(scores)), 2):
        col1 = f"{scores[i][0][0]}-{scores[i][0][1]}  →  {scores[i][1]:.1f}%"
        if i + 1 < len(scores):
            col2 = f"{scores[i+1][0][0]}-{scores[i+1][0][1]}  →  {scores[i+1][1]:.1f}%"
            print(f"    {col1:<20} {col2}")
        else:
            print(f"    {col1}")

def print_corner_predictions(p):
    print(f"\n🔲 CORNERS")
    print(f"  {p['home_team']} Expected: {p['home_expected']:.1f}    {p['away_team']} Expected: {p['away_expected']:.1f}")
    
    res = p['results']
    print(f"  Total Expected: {res['total_expected']:.1f}")
    
    print("\n  Over/Under Lines:")
    over_lines = list(res['over_lines'].items())
    for i in range(0, min(4, len(over_lines)), 2):
        col1 = f"Over {over_lines[i][0]}:  {over_lines[i][1]:.1f}%"
        if i + 1 < len(over_lines):
            col2 = f"Over {over_lines[i+1][0]}:  {over_lines[i+1][1]:.1f}%"
            print(f"    {col1:<20} {col2}")
        else:
            print(f"    {col1}")
            
    likely_total = res['most_likely_score']
    total_probs = res.get('total_probs', {})
    likely_prob = total_probs.get(int(likely_total), 0) if likely_total.isdigit() else 0

    print(f"\n  Most Likely Total: {likely_total} corners ({likely_prob:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Predictable: Football Match Prediction Engine")
    subparsers = parser.add_subparsers(dest="command")

    # Fetch Command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch and update data for a competition")
    fetch_parser.add_argument("--competition", required=True, help="Competition ID (e.g., world_cup_2026)")

    # Predict Match Command
    match_parser = subparsers.add_parser("match", help="Predict a specific match")
    match_parser.add_argument("home_team", help="Home team name")
    match_parser.add_argument("away_team", help="Away team name")
    match_parser.add_argument("--competition", required=True, help="Competition ID (e.g., world_cup_2022)")
    match_parser.add_argument("--type", choices=['goals', 'corners', 'both'], default='both', help="Prediction type")

    args = parser.parse_args()

    if args.command == "fetch":
        from config import COMPETITIONS
        from data.api_sports import fetch_teams_from_fixtures, fetch_match_stats
        from data.rankings import fetch_fifa_rankings
        from data.sportradar import fetch_sportradar_events, fetch_sportradar_full
        from models.team_ratings import compute_team_ratings
        
        print("1. Initializing DB...")
        from db.database import init_db
        init_db()
        
        comp_config = COMPETITIONS.get(args.competition, {})
        is_sr_only = comp_config.get("sportradar_only", False)
        
        if is_sr_only:
            print(f"2. Fetching Full Data from SportRadar for {args.competition}...")
            fetch_sportradar_full(args.competition)
        else:
            print(f"2. Fetching Basic Stats for {args.competition}...")
            fetch_teams_from_fixtures(args.competition)
            fetch_match_stats(args.competition)

            print(f"3. Fetching SportRadar Events for {args.competition}...")
            fetch_sportradar_events(args.competition)

        print("4. Fetching FIFA Rankings...")
        fetch_fifa_rankings()
        
        print(f"5. Computing Team Ratings for {args.competition}...")
        compute_team_ratings(args.competition)
        
        print("\n✅ Fetch complete! Data is ready for predictions.")

    elif args.command == "match":
        if args.type in ['goals', 'both']:
            goal_preds = predict_goals(args.home_team, args.away_team, args.competition)
            if goal_preds:
                print_goal_predictions(goal_preds)
                
        if args.type in ['corners', 'both']:
            corner_preds = predict_corners(args.home_team, args.away_team, args.competition)
            if corner_preds:
                print_corner_predictions(corner_preds)

if __name__ == "__main__":
    main()
