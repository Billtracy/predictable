import os

NBA_DIR = os.path.dirname(os.path.abspath(__file__))
NBA_DB_PATH = os.environ.get("NBA_DB_PATH", os.path.join(os.path.dirname(NBA_DIR), "nba.db"))

# Season keys use the START year, same convention as the EPL and UCL configs
# ("2025" = 2025-26). ESPN labels seasons by their END year; ingest converts.
# ESPN's odds feed carries closing lines from 2022-23 and opening lines from 2023-24.
SEASONS = (2022, 2023, 2024, 2025, 2026)
CURRENT_SEASON = 2026

# Sportsbooks to take the line from, in order. DraftKings is present in every season
# ESPN covers; the rest are fallbacks for the odd game DK is missing.
LINE_PROVIDERS = ("DraftKings", "ESPN BET", "consensus", "Caesars Sportsbook", "MGM")

# Standard -110 juice: a tip must hit more often than this to break even.
BREAK_EVEN = 110 / 210
