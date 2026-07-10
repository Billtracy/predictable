import requests
import math

# Direct API-Sports Configuration
API_KEY = "2f41c85435f7dcf94e1f9b33610249b9"
API_HOST = "v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

# API-Football Constants for the 2026 World Cup
LEAGUE_ID = 1
SEASON = 2026

def get_tournament_fixtures():
    """Fetches all fixtures for the tournament to calculate corner statistics."""
    url = f"https://{API_HOST}/fixtures"
    querystring = {"league": LEAGUE_ID, "season": SEASON}
    
    print("Fetching tournament fixtures from API-Sports...")
    response = requests.get(url, headers=HEADERS, params=querystring)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch data (Status Code: {response.status_code})")
        return []
        
    return response.json().get("response", [])

def parse_corner_baselines(fixtures):
    """Parses fixture statistics to calculate global and team specific corner averages."""
    total_corners = 0
    total_matches_with_stats = 0
    team_stats = {}
    
    for match in fixtures:
        if match["fixture"]["status"]["short"] != "FT":
            continue
            
        match_id = match["fixture"]["id"]
        stats_url = f"https://{API_HOST}/fixtures/statistics"
        stats_response = requests.get(stats_url, headers=HEADERS, params={"fixture": match_id})
        
        if stats_response.status_code != 200:
            continue
            
        fixture_stats = stats_response.json().get("response", [])
        if len(fixture_stats) < 2:
            continue 
            
        home_team, away_team = fixture_stats[0]["team"], fixture_stats[1]["team"]
        home_id, home_name = home_team["id"], home_team["name"]
        away_id, away_name = away_team["id"], away_team["name"]
        
        def extract_corners(stats_list):
            for stat in stats_list:
                if stat["type"] == "Corner Kicks":
                    return int(stat["value"]) if stat["value"] is not None else 0
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

    global_average = total_corners / total_matches_with_stats if total_matches_with_stats > 0 else 0
    return global_average, team_stats

def calculate_poisson_probability(lam, k):
    """Standard Poisson formula: P(x; λ) = (e^(-λ) * λ^x) / x!"""
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def simulate_match_corners(home_stats, away_stats, global_avg):
    """Calculates Attack/Defense coefficients and outputs Poisson probabilities."""
    if global_avg == 0 or home_stats["matches"] == 0 or away_stats["matches"] == 0:
        return None
        
    team_avg = global_avg / 2 

    # Attack and Defense Strength Coefficients
    home_attack = (home_stats["corners_for"] / home_stats["matches"]) / team_avg
    home_defense = (home_stats["corners_against"] / home_stats["matches"]) / team_avg
    
    away_attack = (away_stats["corners_for"] / away_stats["matches"]) / team_avg
    away_defense = (away_stats["corners_against"] / away_stats["matches"]) / team_avg

    # Expected Corners (Lambda)
    home_expected = home_attack * away_defense * team_avg
    away_expected = away_attack * home_defense * team_avg
    total_expected = home_expected + away_expected
    
    # Generate Probability Matrix (0 to 15 corners)
    probabilities = {}
    for corners in range(16):
        prob = calculate_poisson_probability(total_expected, corners)
        probabilities[corners] = prob * 100 
        
    return total_expected, probabilities

if __name__ == "__main__":
    fixtures_data = get_tournament_fixtures()
    
    if fixtures_data:
        # Slicing the last 10 matches to preserve your daily 100 API limit during testing
        avg_corners, teams = parse_corner_baselines(fixtures_data[:10])
        
        print(f"\n--- TOURNAMENT BASELINE ---")
        print(f"Global Corner Average per Match: {avg_corners:.2f}")
        
        # Example Simulation: Grabbing the first two teams parsed to test the engine
        if len(teams) >= 2:
            team_ids = list(teams.keys())
            team_a, team_b = teams[team_ids[0]], teams[team_ids[1]]
            
            expected, probs = simulate_match_corners(team_a, team_b, avg_corners)
            
            print(f"\n--- MATCH SIMULATION: {team_a['name']} vs {team_b['name']} ---")
            print(f"Total Expected Corners: {expected:.2f}")
            print("\nProbabilities:")
            for corners, prob in probs.items():
                if prob > 1.0:  # Only show somewhat likely outcomes (>1%)
                    print(f"Exactly {corners} Corners: {prob:.2f}%")