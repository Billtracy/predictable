import json
import os
import re
import sys
import time
from datetime import datetime

import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SPORTRADAR_KEY, COMPETITIONS
from db.database import get_db

BASE_URL = "https://api.sportradar.com/soccer-extended/trial/v4/en"

EVENT_TYPES = {
    "shot_off_target",
    "shot_on_target",
    "shot_saved",
    "shot_blocked",
    "goal",
    "score_change",
    "corner_kick",
}

TEAM_ALIASES = {
    "usa": "united states",
    "u s a": "united states",
    "usmnt": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "wolves": "wolverhampton",
    "wolverhampton wanderers": "wolverhampton",
    "newcastle utd": "newcastle united",
    "nottm forest": "nottingham forest",
}


def normalize_name(name):
    """Normalize provider team names enough for cross-source matching."""
    if not name:
        return ""

    normalized = name.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    words = [
        word
        for word in normalized.split()
        if word not in {"fc", "afc", "cf", "club", "football", "soccer", "team"}
    ]
    normalized = " ".join(words).strip()
    return TEAM_ALIASES.get(normalized, normalized)


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10]


def get_competition_year(competition_name):
    comp_config = COMPETITIONS.get(competition_name, {})
    return comp_config.get("api_sports", {}).get("season")


def get_competitors(sport_event):
    competitors = sport_event.get("competitors") or []
    by_qualifier = {c.get("qualifier"): c for c in competitors if c.get("qualifier")}
    home = by_qualifier.get("home") or (competitors[0] if competitors else None)
    away = by_qualifier.get("away") or (competitors[1] if len(competitors) > 1 else None)
    return home, away


def fetch_sportradar_seasons(competition_name):
    """Return the SportRadar season that matches the configured competition year."""
    comp_config = COMPETITIONS.get(competition_name)
    if not comp_config or "sportradar" not in comp_config:
        print(f"SportRadar config not found for {competition_name}")
        return None

    comp_urn = comp_config["sportradar"]["urn"]
    target_year = get_competition_year(competition_name)

    url = f"{BASE_URL}/competitions/{comp_urn}/seasons.json"
    params = {"api_key": SPORTRADAR_KEY}

    print(f"Fetching seasons for {comp_urn}...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching seasons: {response.status_code}")
        print(response.text[:500])
        return None

    seasons = response.json().get("seasons", [])
    if not seasons:
        print("No seasons found.")
        return None

    for season in seasons[:8]:
        print(
            f"  Season: {season.get('name', 'N/A')} | "
            f"ID: {season.get('id', 'N/A')} | Year: {season.get('year', 'N/A')}"
        )

    if target_year:
        # A configured year like 2024 can appear as "2024" (tournaments) or as a
        # cross-year league season: "24/25", "2024/25", "2024/2025".
        y = int(target_year)
        candidates = {
            str(y),
            f"{y % 100:02d}/{(y + 1) % 100:02d}",
            f"{y}/{(y + 1) % 100:02d}",
            f"{y}/{y + 1}",
        }
        for season in seasons:
            year = str(season.get("year", ""))
            name = str(season.get("name", ""))
            if year in candidates or any(c in name for c in candidates):
                print(f"Matched season: {season.get('name')} ({season.get('id')})")
                return season

        print(f"ABORT: no SportRadar season matches year {target_year} for {competition_name}.")
        print("Check the season list above and adjust config.py if needed.")
        return None

    return seasons[0]


def fetch_season_schedule(season_id):
    """Get all sport events for a season."""
    url = f"{BASE_URL}/seasons/{season_id}/schedules.json"
    params = {"api_key": SPORTRADAR_KEY}

    print(f"Fetching schedule for season {season_id}...")
    time.sleep(1.5)
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching schedule: {response.status_code}")
        print(response.text[:500])
        return []

    return response.json().get("schedules", [])


def _print_sr_quota(response):
    """SportRadar trial keys report monthly quota usage in response headers."""
    allotted = response.headers.get("x-plan-quota-allotted")
    used = response.headers.get("x-plan-quota-current")
    if allotted and used:
        try:
            print(f"  [quota] monthly: {used}/{allotted} used ({int(allotted) - int(used)} left)")
        except ValueError:
            print(f"  [quota] monthly: {used}/{allotted} used")


def fetch_match_timeline(sport_event_id):
    """Fetch play-by-play timeline for a specific match."""
    url = f"{BASE_URL}/sport_events/{sport_event_id}/timeline.json"
    params = {"api_key": SPORTRADAR_KEY}

    for attempt in range(3):
        response = requests.get(url, params=params)
        if response.status_code != 429:
            break
        print(f"  Rate limited (429). Waiting 30s... (attempt {attempt + 1}/3)")
        time.sleep(30)

    _print_sr_quota(response)

    if response.status_code != 200:
        print(f"  Error fetching timeline for {sport_event_id}: HTTP {response.status_code}")
        print(f"  Body: {response.text[:300]}")
        if response.status_code == 403:
            print("  (403 on a trial key usually means the monthly quota is exhausted.)")
        return None

    return response.json()


def resolve_team_id(db, sr_competitor):
    """Resolve a SportRadar competitor to an API-Sports team row."""
    if not sr_competitor:
        return None

    sr_id = sr_competitor.get("id")
    name = sr_competitor.get("name", "")

    if sr_id:
        row = db.execute("SELECT id FROM teams WHERE sportradar_id = ?", (sr_id,)).fetchone()
        if row:
            return row["id"]

    wanted = normalize_name(name)
    rows = db.execute("SELECT id, name FROM teams").fetchall()
    for row in rows:
        existing = normalize_name(row["name"])
        if existing == wanted or existing in wanted or wanted in existing:
            if sr_id:
                db.execute("UPDATE teams SET sportradar_id = ? WHERE id = ?", (sr_id, row["id"]))
            return row["id"]

    return None


def resolve_match_id(db, competition_name, sport_event):
    """Resolve a SportRadar event to an existing API-Sports fixture."""
    sr_event_id = sport_event.get("id")
    if not sr_event_id:
        return None, {}

    row = db.execute("SELECT id FROM matches WHERE sportradar_id = ?", (sr_event_id,)).fetchone()
    if row:
        home, away = get_competitors(sport_event)
        return row["id"], {
            home.get("id"): resolve_team_id(db, home) if home else None,
            away.get("id"): resolve_team_id(db, away) if away else None,
        }

    home, away = get_competitors(sport_event)
    home_team_id = resolve_team_id(db, home)
    away_team_id = resolve_team_id(db, away)
    scheduled_date = parse_date(sport_event.get("start_time") or sport_event.get("scheduled"))

    if not home_team_id or not away_team_id:
        print(
            "  Could not map teams for "
            f"{home.get('name') if home else 'unknown'} vs {away.get('name') if away else 'unknown'}"
        )
        return None, {}

    params = [competition_name, home_team_id, away_team_id, away_team_id, home_team_id]
    date_clause = ""
    if scheduled_date:
        date_clause = "AND substr(date, 1, 10) = ?"
        params.append(scheduled_date)

    row = db.execute(
        f"""
        SELECT id
        FROM matches
        WHERE competition_id = ?
          AND (
                (home_team_id = ? AND away_team_id = ?)
             OR (home_team_id = ? AND away_team_id = ?)
          )
          {date_clause}
        ORDER BY date
        LIMIT 1
        """,
        params,
    ).fetchone()

    if not row and scheduled_date:
        row = db.execute(
            """
            SELECT id
            FROM matches
            WHERE competition_id = ?
              AND (
                    (home_team_id = ? AND away_team_id = ?)
                 OR (home_team_id = ? AND away_team_id = ?)
              )
            ORDER BY date
            LIMIT 1
            """,
            (competition_name, home_team_id, away_team_id, away_team_id, home_team_id),
        ).fetchone()

    if not row:
        print(
            "  Could not map SportRadar match to local fixture: "
            f"{home.get('name')} vs {away.get('name')} on {scheduled_date or 'unknown date'}"
        )
        return None, {}

    db.execute("UPDATE matches SET sportradar_id = ? WHERE id = ?", (sr_event_id, row["id"]))
    return row["id"], {home.get("id"): home_team_id, away.get("id"): away_team_id}


def get_event_competitor_id(event):
    """Return the competitor qualifier ('home' or 'away') from a timeline event.
    SportRadar timeline events use 'home'/'away' strings, not IDs."""
    competitor = event.get("competitor") or event.get("team")
    if isinstance(competitor, dict):
        return competitor.get("qualifier") or competitor.get("id")
    return competitor  # Already 'home' or 'away' string


def get_event_outcome(event_type, event):
    if event_type == "goal":
        return "goal"
    return event.get("outcome") or event.get("status") or event_type



def fetch_sportradar_events(competition_name):
    """Fetch SportRadar timelines and store mapped shot/corner events.
    Use when API-Sports data already provides teams and matches."""
    season = fetch_sportradar_seasons(competition_name)
    if not season:
        return

    schedules = fetch_season_schedule(season["id"])
    if not schedules:
        print("No schedules found for this season.")
        return

    completed = []
    for entry in schedules:
        event = entry.get("sport_event", {})
        status = entry.get("sport_event_status", {})
        if status.get("status") == "closed":
            completed.append(event)

    print(f"Found {len(completed)} completed matches. Fetching timelines...")

    consecutive_failures = 0
    fetched = 0
    for i, event in enumerate(completed):
        sr_event_id = event.get("id")
        if not sr_event_id:
            continue

        with get_db() as db:
            match_id, sr_team_mapping = resolve_match_id(db, competition_name, event)
            if not match_id:
                continue
            
            # Build qualifier-based mapping: 'home' -> team_id, 'away' -> team_id
            home_comp, away_comp = get_competitors(event)
            competitor_team_ids = {}
            if home_comp:
                competitor_team_ids["home"] = sr_team_mapping.get(home_comp.get("id"))
            if away_comp:
                competitor_team_ids["away"] = sr_team_mapping.get(away_comp.get("id"))

            already_fetched = db.execute(
                "SELECT COUNT(*) as cnt FROM match_events WHERE match_id = ?",
                (match_id,),
            ).fetchone()["cnt"]
            if already_fetched > 0:
                continue

        print(f"  Fetching timeline for {sr_event_id}... ({i + 1}/{len(completed)})")
        time.sleep(1.5)

        tl_data = fetch_match_timeline(sr_event_id)
        if not tl_data:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("\n" + "!" * 60)
                print("ABORT: 5 timeline fetches failed in a row — quota likely exhausted.")
                print(f"Fetched {fetched} timelines this run. Re-run this command later;")
                print("it resumes automatically (matches with saved events are skipped).")
                print("!" * 60)
                return
            continue
        consecutive_failures = 0
        fetched += 1

        timeline = tl_data.get("timeline", [])
        saved_count = 0
        # Aggregate shot/corner counts per qualifier straight from the timeline so
        # the same fetch that gives xG coordinates also fills match_stats. Counts
        # every shot/corner (coordinates optional), unlike the event rows below.
        tally = {
            "home": {"shots": 0, "on": 0, "off": 0, "blocked": 0, "corners": 0},
            "away": {"shots": 0, "on": 0, "off": 0, "blocked": 0, "corners": 0},
        }

        with get_db() as db:
            for ev in timeline:
                event_type = ev.get("type", "")
                if event_type not in EVENT_TYPES:
                    continue

                competitor_id = get_event_competitor_id(ev)

                # Stat tally (independent of whether coordinates are present)
                if competitor_id in tally:
                    t = tally[competitor_id]
                    if event_type in ("shot_on_target", "shot_saved", "score_change"):
                        t["shots"] += 1
                        t["on"] += 1
                    elif event_type == "shot_off_target":
                        t["shots"] += 1
                        t["off"] += 1
                    elif event_type == "shot_blocked":
                        t["shots"] += 1
                        t["blocked"] += 1
                    elif event_type == "corner_kick":
                        t["corners"] += 1

                x_coord = ev.get("x")
                y_coord = ev.get("y")
                if x_coord is None or y_coord is None:
                    continue

                team_id = competitor_team_ids.get(competitor_id)
                if not team_id:
                    continue

                event_id = ev.get("id") or f"{saved_count}"
                db.execute(
                    """
                    INSERT OR IGNORE INTO match_events
                    (id, match_id, team_id, event_type, minute, x_coord, y_coord, outcome, body_part, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{sr_event_id}_{event_id}",
                        match_id,
                        team_id,
                        event_type,
                        ev.get("match_time"),
                        x_coord,
                        y_coord,
                        get_event_outcome(event_type, ev),
                        ev.get("body_part", ""),
                        json.dumps(ev),
                    ),
                )
                saved_count += 1

            # Write derived match_stats non-destructively: OR IGNORE keeps any
            # existing (e.g. API-Sports) row for this match/team untouched, and
            # only fills the shots/corners fields the model actually reads.
            stats_written = 0
            for qualifier in ("home", "away"):
                team_id = competitor_team_ids.get(qualifier)
                if not team_id:
                    continue
                t = tally[qualifier]
                if t["shots"] == 0 and t["corners"] == 0:
                    continue  # empty/unusable timeline — don't write a zero row
                db.execute(
                    """
                    INSERT OR IGNORE INTO match_stats
                    (match_id, team_id, shots_total, shots_on_target, shots_off_target,
                     blocked_shots, corners, possession, passes_total, passes_accurate,
                     fouls, offsides, saves)
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL)
                    """,
                    (match_id, team_id, t["shots"], t["on"], t["off"], t["blocked"], t["corners"]),
                )
                stats_written += 1

        if saved_count > 0 or stats_written > 0:
            h, a = tally["home"], tally["away"]
            print(
                f"  Saved {saved_count} events, {stats_written} stat rows | "
                f"H {h['shots']}S/{h['corners']}C  A {a['shots']}S/{a['corners']}C"
            )

    print(f"\nDone: fetched {fetched} new timelines for {competition_name}.")


def ingest_sportradar_schedule(competition_name):
    """Create teams + matches (with final scores) from the SportRadar season schedule.
    No timelines/stats — cheap (~2 API calls). Shared by the full import and by
    scores-only fetches (cross-season rating priors need goals only). Team resolution
    is by season-independent sr:competitor id / name, so a prior season maps onto the
    same team rows. Returns the completed-match count, or None on failure."""
    season = fetch_sportradar_seasons(competition_name)
    if not season:
        return None

    schedules = fetch_season_schedule(season["id"])
    if not schedules:
        print("No schedules found for this season.")
        return None

    # Create teams and matches from schedule
    print(f"Found {len(schedules)} total matches. Ingesting teams and matches...")
    
    # Generate synthetic team IDs starting from 100000 to avoid clashing with API-Sports IDs
    team_name_to_id = {}
    next_id = 100000
    
    with get_db() as db:
        # Check existing teams to get highest ID
        cur = db.cursor()
        cur.execute("SELECT MAX(id) as max_id FROM teams")
        row = cur.fetchone()
        if row and row["max_id"] and row["max_id"] >= next_id:
            next_id = row["max_id"] + 1
        
        # Also check existing team names 
        cur.execute("SELECT id, name FROM teams")
        for r in cur.fetchall():
            norm = normalize_name(r["name"])
            team_name_to_id[norm] = r["id"]

    completed_count = 0
    with get_db() as db:
        for entry in schedules:
            event = entry.get("sport_event", {})
            status = entry.get("sport_event_status", {})
            
            home_comp, away_comp = get_competitors(event)
            if not home_comp or not away_comp:
                continue
            
            # Resolve or create teams
            for comp in [home_comp, away_comp]:
                name = comp.get("name", "Unknown")
                norm = normalize_name(name)
                if norm not in team_name_to_id:
                    team_name_to_id[norm] = next_id
                    db.execute(
                        "INSERT OR IGNORE INTO teams (id, name, sportradar_id) VALUES (?, ?, ?)",
                        (next_id, name, comp.get("id"))
                    )
                    next_id += 1
                else:
                    # Update sportradar_id if missing
                    sr_id = comp.get("id")
                    if sr_id:
                        db.execute(
                            "UPDATE teams SET sportradar_id = ? WHERE id = ? AND (sportradar_id IS NULL OR sportradar_id = '')",
                            (sr_id, team_name_to_id[norm])
                        )
            
            home_id = team_name_to_id[normalize_name(home_comp.get("name", ""))]
            away_id = team_name_to_id[normalize_name(away_comp.get("name", ""))]
            
            home_score = status.get("home_score")
            away_score = status.get("away_score")
            match_status = "FT" if status.get("status") == "closed" else "NS"
            
            # Use a synthetic match ID derived from SportRadar event ID
            sr_event_id = event.get("id", "")
            # Extract numeric part from sr:sport_event:XXXXX
            match_id = int(sr_event_id.split(":")[-1]) if ":" in sr_event_id else hash(sr_event_id) % 10000000
            
            db.execute('''
                INSERT OR REPLACE INTO matches 
                (id, competition_id, home_team_id, away_team_id, home_score, away_score, status, date, sportradar_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id,
                competition_name,
                home_id,
                away_id,
                home_score,
                away_score,
                match_status,
                event.get("start_time", ""),
                sr_event_id
            ))
            
            if match_status == "FT":
                completed_count += 1
    
    print(f"Inserted {len(team_name_to_id)} teams and {completed_count} completed matches.")
    return completed_count


def fetch_sportradar_full(competition_name):
    """Full data ingestion from SportRadar: schedule (teams+matches+scores) then
    timelines (events + derived stats). Use when API-Sports data is unavailable."""
    if ingest_sportradar_schedule(competition_name) is None:
        return

    # Phase 2: Fetch timelines for completed matches and extract events + stats
    print("Fetching timelines for completed matches...")
    with get_db() as db:
        cur = db.cursor()
        cur.execute('''
            SELECT id, sportradar_id, home_team_id, away_team_id 
            FROM matches 
            WHERE competition_id = ? AND status IN ('FT', 'AET', 'PEN')
            ORDER BY date
        ''', (competition_name,))
        completed_matches = cur.fetchall()
    
    for i, match in enumerate(completed_matches):
        match_id = match["id"]
        sr_event_id = match["sportradar_id"]
        home_team_id = match["home_team_id"]
        away_team_id = match["away_team_id"]
        
        # Check if already fetched
        with get_db() as db:
            already = db.execute(
                "SELECT COUNT(*) as cnt FROM match_events WHERE match_id = ?",
                (match_id,)
            ).fetchone()["cnt"]
            if already > 0:
                continue
        
        print(f"  Fetching timeline ({i+1}/{len(completed_matches)})...")
        time.sleep(1.5)
        
        tl_data = fetch_match_timeline(sr_event_id)
        if not tl_data:
            continue
        
        timeline = tl_data.get("timeline", [])
        
        # Extract events and compute basic stats
        home_stats = {"shots": 0, "shots_on": 0, "shots_off": 0, "corners": 0, "goals": 0}
        away_stats = {"shots": 0, "shots_on": 0, "shots_off": 0, "corners": 0, "goals": 0}
        saved_events = 0
        
        with get_db() as db:
            for ev in timeline:
                event_type = ev.get("type", "")
                qualifier = ev.get("competitor", "")
                
                # Track stats
                if qualifier in ("home", "away"):
                    stats = home_stats if qualifier == "home" else away_stats
                    if event_type in ("shot_on_target", "shot_saved"):
                        stats["shots"] += 1
                        stats["shots_on"] += 1
                    elif event_type == "shot_off_target":
                        stats["shots"] += 1
                        stats["shots_off"] += 1
                    elif event_type == "shot_blocked":
                        stats["shots"] += 1
                    elif event_type == "corner_kick":
                        stats["corners"] += 1
                    elif event_type == "score_change":
                        stats["goals"] += 1
                        stats["shots"] += 1
                        stats["shots_on"] += 1
                
                # Save shot/corner events with coordinates
                if event_type not in EVENT_TYPES:
                    continue
                
                x_coord = ev.get("x")
                y_coord = ev.get("y")
                if x_coord is None or y_coord is None:
                    continue
                
                team_id = home_team_id if qualifier == "home" else away_team_id if qualifier == "away" else None
                if not team_id:
                    continue
                
                event_id = ev.get("id") or f"{saved_events}"
                db.execute(
                    """
                    INSERT OR IGNORE INTO match_events
                    (id, match_id, team_id, event_type, minute, x_coord, y_coord, outcome, body_part, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{sr_event_id}_{event_id}",
                        match_id,
                        team_id,
                        event_type,
                        ev.get("match_time"),
                        x_coord,
                        y_coord,
                        get_event_outcome(event_type, ev),
                        ev.get("body_part", ""),
                        json.dumps(ev),
                    ),
                )
                saved_events += 1
            
            # Insert basic match stats derived from timeline
            for qualifier, stats, team_id in [("home", home_stats, home_team_id), ("away", away_stats, away_team_id)]:
                db.execute('''
                    INSERT OR REPLACE INTO match_stats 
                    (match_id, team_id, shots_total, shots_on_target, shots_off_target, blocked_shots,
                     corners, possession, passes_total, passes_accurate, fouls, offsides, saves)
                    VALUES (?, ?, ?, ?, ?, 0, ?, 50.0, 0, 0, 0, 0, 0)
                ''', (
                    match_id, team_id, stats["shots"], stats["shots_on"], stats["shots_off"], stats["corners"]
                ))
        
        if saved_events > 0:
            print(f"    Saved {saved_events} events | Home: {home_stats['shots']}S/{home_stats['corners']}C | Away: {away_stats['shots']}S/{away_stats['corners']}C")
    
    print(f"\nSportRadar full import complete for {competition_name}.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        comp = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "events"
        if mode == "full":
            fetch_sportradar_full(comp)
        elif mode == "scores":
            ingest_sportradar_schedule(comp)  # teams + matches + scores only, no timelines
        else:
            fetch_sportradar_events(comp)
    else:
        print("Usage: python data/sportradar.py <competition_name> [full|events|scores]")

