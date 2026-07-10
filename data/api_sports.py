import requests
import time
import sys
import os

# Add parent directory to path to allow importing config and db
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import API_SPORTS_KEY, API_SPORTS_HOST, COMPETITIONS
from db.database import get_db

HEADERS = {
    "x-apisports-key": API_SPORTS_KEY,
    "x-rapidapi-host": API_SPORTS_HOST
}

def fetch_teams_from_fixtures(competition_name):
    """Fetch fixtures and insert teams if they don't exist, and matches."""
    comp_config = COMPETITIONS.get(competition_name)
    if not comp_config:
        print(f"Competition {competition_name} not found in config.")
        return

    league_id = comp_config["api_sports"]["league"]
    season = comp_config["api_sports"]["season"]

    url = f"https://{API_SPORTS_HOST}/fixtures"
    params = {"league": league_id, "season": season}
    
    print(f"Fetching fixtures for {competition_name} (League: {league_id}, Season: {season})...")
    response = _get_with_retries(url, params)
    if response is None:
        return

    if response.status_code != 200:
        print(f"Error fetching fixtures: HTTP {response.status_code}")
        print(f"  Body: {response.text[:300]}")
        return

    _print_quota(response)
    data = response.json()
    errors = data.get("errors") or {}

    # API-Sports returns HTTP 200 with an errors payload (and empty response) when
    # the daily quota is spent — distinguish that from a genuinely empty season so
    # the message isn't misleading.
    if errors:
        if "requests" in errors:
            print("\n" + "!" * 60)
            print(f"DAILY QUOTA EXHAUSTED: {errors['requests']}")
            print("Re-run after the quota resets.")
            print("!" * 60)
        else:
            print(f"API error: {errors}")
        return

    fixtures = data.get("response", [])
    if not fixtures:
        print(f"No fixtures returned for season {season} (likely outside this plan's coverage).")
        return

    print(f"Found {len(fixtures)} fixtures. Saving to database...")
    
    with get_db() as db:
        for f in fixtures:
            fixture = f["fixture"]
            league = f["league"]
            teams = f["teams"]
            goals = f["goals"]
            
            # Insert teams
            home_team = teams["home"]
            away_team = teams["away"]
            
            db.execute("INSERT OR IGNORE INTO teams (id, name) VALUES (?, ?)", (home_team["id"], home_team["name"]))
            db.execute("INSERT OR IGNORE INTO teams (id, name) VALUES (?, ?)", (away_team["id"], away_team["name"]))
            
            # Insert match
            status = fixture["status"]["short"]
            db.execute('''
                INSERT OR REPLACE INTO matches 
                (id, competition_id, home_team_id, away_team_id, home_score, away_score, status, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fixture["id"], 
                competition_name,
                home_team["id"], 
                away_team["id"], 
                goals["home"], 
                goals["away"], 
                status,
                fixture["date"]
            ))

def _print_quota(response):
    """API-Sports reports remaining quota in response headers on every call."""
    daily = response.headers.get("x-ratelimit-requests-remaining")
    minute = response.headers.get("X-RateLimit-Remaining")
    print(f"  [quota] daily remaining: {daily} | this minute: {minute}")


def _get_with_retries(url, params):
    """GET with retries for transient connection errors. Returns None on failure."""
    for attempt in range(3):
        try:
            return requests.get(url, headers=HEADERS, params=params, timeout=30)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < 2:
                wait = 10 * (attempt + 1)
                print(f"  Connection error: {e}. Retrying in {wait}s... (attempt {attempt+2}/3)")
                time.sleep(wait)
    print("  Failed after 3 attempts.")
    return None


def fetch_match_stats(competition_name):
    """Fetch statistics for completed matches that don't have stats yet.

    Resumable: only matches missing from match_stats are requested, so re-running
    after a daily-quota stop picks up where it left off.
    """
    with get_db() as db:
        # Get completed matches without stats
        cur = db.cursor()
        cur.execute('''
            SELECT m.id
            FROM matches m
            LEFT JOIN match_stats ms ON m.id = ms.match_id
            WHERE m.competition_id = ?
              AND m.status IN ('FT', 'AET', 'PEN')
              AND ms.match_id IS NULL
        ''', (competition_name,))

        matches_to_fetch = [row['id'] for row in cur.fetchall()]

    print(f"Found {len(matches_to_fetch)} completed matches needing stats.")

    url = f"https://{API_SPORTS_HOST}/fixtures/statistics"
    saved = 0
    idx = 0
    while idx < len(matches_to_fetch):
        match_id = matches_to_fetch[idx]
        print(f"Fetching stats for match {match_id}... ({idx+1}/{len(matches_to_fetch)})")
        params = {"fixture": match_id}

        response = _get_with_retries(url, params)
        if response is None:
            idx += 1
            continue

        if response.status_code == 429:
            print("  Rate limit hit (429). Waiting 65 seconds, then retrying this match...")
            time.sleep(65)
            continue  # same match, not skipped

        _print_quota(response)

        if response.status_code != 200:
            print(f"  Error fetching stats for {match_id}: HTTP {response.status_code}")
            print(f"  Body: {response.text[:300]}")
            idx += 1
            continue

        data = response.json()
        errors = data.get("errors") or {}

        if errors:
            # API-Sports signals quota problems with HTTP 200 + an errors payload.
            # Per-minute limit ("rateLimit") is recoverable; the daily cap
            # ("requests") is not — stop cleanly instead of burning the queue.
            if "rateLimit" in errors:
                print(f"  Per-minute rate limit: {errors['rateLimit']}")
                print("  Waiting 65 seconds, then retrying this match...")
                time.sleep(65)
                continue
            if "requests" in errors:
                remaining = len(matches_to_fetch) - idx
                print("\n" + "!" * 60)
                print(f"DAILY QUOTA EXHAUSTED: {errors['requests']}")
                print(f"Saved {saved} matches this run; {remaining} still missing.")
                print("Re-run the same command after the quota resets (resumes automatically).")
                print("!" * 60)
                return
            print(f"  API error for match {match_id}: {errors}")
            idx += 1
            continue

        stats_response = data.get("response", [])
        
        if not stats_response or len(stats_response) < 2:
            print(f"  No stats available for match {match_id}")
            # Insert dummy record to avoid fetching again? We'll leave it for now.
            time.sleep(1)
            idx += 1
            continue


        with get_db() as db:
            for team_stats in stats_response:
                team_id = team_stats["team"]["id"]
                stats = team_stats["statistics"]
                
                def extract_stat(type_name):
                    for stat in stats:
                        if stat["type"] == type_name and stat["value"] is not None:
                            val = stat["value"]
                            # Handle percentages like "54%"
                            if isinstance(val, str) and "%" in val:
                                return float(val.replace("%", ""))
                            return val
                    return 0
                
                shots_total = extract_stat("Total Shots")
                shots_on_target = extract_stat("Shots on Goal")
                shots_off_target = extract_stat("Shots off Goal")
                blocked_shots = extract_stat("Blocked Shots")
                corners = extract_stat("Corner Kicks")
                possession = extract_stat("Ball Possession")
                passes_total = extract_stat("Total passes")
                passes_accurate = extract_stat("Passes accurate")
                fouls = extract_stat("Fouls")
                offsides = extract_stat("Offsides")
                saves = extract_stat("Goalkeeper Saves")
                
                db.execute('''
                    INSERT OR REPLACE INTO match_stats 
                    (match_id, team_id, shots_total, shots_on_target, shots_off_target, blocked_shots,
                     corners, possession, passes_total, passes_accurate, fouls, offsides, saves)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    match_id, team_id, shots_total, shots_on_target, shots_off_target, blocked_shots,
                    corners, possession, passes_total, passes_accurate, fouls, offsides, saves
                ))
                
        saved += 1
        idx += 1
        print(f"  Saved stats for match {match_id}.")
        time.sleep(7) # Respect 10 req/min rate limit on free tier

    print(f"\nDone: saved stats for {saved} matches; nothing left to fetch for {competition_name}.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        comp = sys.argv[1]
        fetch_teams_from_fixtures(comp)
        fetch_match_stats(comp)
    else:
        print("Usage: python data/api_sports.py <competition_name>")
