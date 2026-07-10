import requests
import json
import os
import math
import time

# --- CONFIGURATION ---
API_KEY = "2f41c85435f7dcf94e1f9b33610249b9"
API_HOST = "v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

LEAGUE_ID = 1
SEASON = 2022
CACHE_FILE = "world_cup_stats_cache.json"

# --- DATA PIPELINE ---
def fetch_and_cache_tournament_data():
    print("🌐 Fetching tournament fixtures from live API...")
    url = f"https://{API_HOST}/fixtures"
    response = requests.get(url, headers=HEADERS, params={"league": LEAGUE_ID, "season": SEASON})
    
    response_json = response.json()
    
    # 1. Check for Top-Level API Errors (like invalid keys or exhausted daily quotas)
    if "errors" in response_json and response_json["errors"]:
        print(f"🛑 API Error: {response_json['errors']}")
        return None
        
    fixtures = response_json.get("response", [])
    print(f"✅ Found {len(fixtures)} total fixtures for Season {SEASON}.")
    
    if len(fixtures) == 0:
        print("⚠️ The API returned 0 fixtures. Try changing SEASON to 2022 to test with historical data.")
        return None

    # 2. Broaden the status filter to catch all completed match types
    valid_statuses = ["FT", "AET", "PEN"]
    matches_to_process = [m for m in fixtures if m["fixture"]["status"]["short"] in valid_statuses]
    
    print(f"📊 Found {len(matches_to_process)} completed matches. Fetching stats...")

    total_corners = 0
    total_matches_with_stats = 0
    team_stats = {}
    
    for match in matches_to_process:
        match_id = match["fixture"]["id"]
        
        # 3. Add a slight delay to prevent hitting API burst limits
        time.sleep(1) 
        
        stats_response = requests.get(f"https://{API_HOST}/fixtures/statistics", headers=HEADERS, params={"fixture": match_id})
        stats_json = stats_response.json()
        
        # Check for rate limits inside the loop
        if "errors" in stats_json and stats_json["errors"]:
            print(f"🛑 API Error hit on match {match_id}: {stats_json['errors']}")
            print("💾 Saving whatever data we have collected so far...")
            break 
            
        fixture_stats = stats_json.get("response", [])
        if len(fixture_stats) < 2:
            print(f"⚠️ Match {match_id} has no stats available. Skipping.")
            continue 
            
        home_team, away_team = fixture_stats[0]["team"], fixture_stats[1]["team"]
        home_id, home_name = home_team["id"], home_team["name"]
        away_id, away_name = away_team["id"], away_team["name"]
        
        def extract_corners(stats_list):
            for stat in stats_list:
                if stat["type"] == "Corner Kicks" and stat["value"] is not None:
                    return int(stat["value"])
            return 0

        home_corners = extract_corners(fixture_stats[0]["statistics"])
        away_corners = extract_corners(fixture_stats[1]["statistics"])
        
        for t_id, t_name in [(home_id, home_name), (away_id, away_name)]:
            if t_id not in team_stats:
                team_stats[t_id] = {"name": t_name, "corners_for": 0, "corners_against": 0, "matches": 0}
                
        team_stats[home_id]["corners_for"] += home_corners
        team_stats[home_id]["corners_against"] += away_corners
        team_stats[home_id]["matches"] += 1
        
        team_stats[away_id]["corners_for"] += away_corners
        team_stats[away_id]["corners_against"] += home_corners
        team_stats[away_id]["matches"] += 1
        
        total_corners += (home_corners + away_corners)
        total_matches_with_stats += 1
        print(f"   -> Parsed stats for match {match_id}")

    global_avg = total_corners / total_matches_with_stats if total_matches_with_stats > 0 else 0
    
    cache_data = {"global_avg": global_avg, "teams": team_stats}
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)
        
    print("\n💾 Data successfully saved to local cache!")
    return cache_data

def load_data():
    if os.path.exists(CACHE_FILE):
        print("📦 Found local cache file, checking contents...")
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
            # Force a refetch if the cache is empty
            if data.get("global_avg") == 0 and len(data.get("teams", {})) == 0:
                print("⚠️ Cache is empty. Triggering a fresh fetch...")
                return fetch_and_cache_tournament_data()
            return data
    return fetch_and_cache_tournament_data()

# --- MATH ENGINE ---
def calculate_poisson_probability(lam, k):
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def simulate_match_corners(home_stats, away_stats, global_avg):
    team_avg = global_avg / 2 

    home_attack = (home_stats["corners_for"] / home_stats["matches"]) / team_avg
    home_defense = (home_stats["corners_against"] / home_stats["matches"]) / team_avg
    
    away_attack = (away_stats["corners_for"] / away_stats["matches"]) / team_avg
    away_defense = (away_stats["corners_against"] / away_stats["matches"]) / team_avg

    home_expected = home_attack * away_defense * team_avg
    away_expected = away_attack * home_defense * team_avg
    total_expected = home_expected + away_expected
    
    probabilities = {corners: calculate_poisson_probability(total_expected, corners) * 100 for corners in range(16)}
    return total_expected, probabilities

# --- EXECUTION ---
# --- EXECUTION ---
def get_team_by_name(teams_dict, team_name):
    """Helper function to find a team's stats by their string name."""
    for t_id, stats in teams_dict.items():
        if stats["name"].lower() == team_name.lower():
            return stats
    return None

if __name__ == "__main__":
    data = load_data()
    
    if data and data["global_avg"] > 0:
        avg_corners = data["global_avg"]
        teams = data["teams"]
        
        print(f"\n🌍 TOURNAMENT BASELINE: {avg_corners:.2f} corners per match")
        
        # Define the matchup you want to predict
        home_team_name = "Argentina"
        away_team_name = "France"
        
        team_a = get_team_by_name(teams, home_team_name)
        team_b = get_team_by_name(teams, away_team_name)
        
        if not team_a or not team_b:
            print(f"\n❌ Error: Could not find one or both teams in the dataset.")
        else:
            expected, probs = simulate_match_corners(team_a, team_b, avg_corners)
            
            print(f"\n⚽ SIMULATION: {team_a['name']} vs {team_b['name']}")
            print(f"Expected Total Corners: {expected:.2f}")
            print("\nMost Likely Exact Outcomes:")
            
            # Sort probabilities from highest to lowest and print the top 5
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for corners, prob in sorted_probs[:5]:
                print(f"Exactly {corners} Corners: {prob:.2f}%")
                
            # Bonus: Calculate the popular "Over 8.5 Corners" betting line
            over_8_5 = sum(prob for corners, prob in probs.items() if corners > 8)
            print(f"\n📈 Probability of Over 8.5 Corners: {over_8_5:.2f}%")