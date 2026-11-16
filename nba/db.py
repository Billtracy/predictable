import sqlite3
from contextlib import contextmanager

from nba.config import NBA_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id TEXT PRIMARY KEY,          -- ESPN team id ('1'..'30')
    abbreviation TEXT NOT NULL,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,          -- ESPN event id
    season INTEGER NOT NULL,      -- start year
    date TEXT NOT NULL,           -- ISO tip-off time (UTC)
    game_type TEXT NOT NULL,      -- 'regular', 'playin' or 'playoff'
    neutral INTEGER NOT NULL DEFAULT 0,
    home_id TEXT NOT NULL REFERENCES teams(id),
    away_id TEXT NOT NULL REFERENCES teams(id),
    status TEXT NOT NULL,         -- 'FT' or 'NS'
    home_pts INTEGER,             -- final score incl. overtime (what totals settle on)
    away_pts INTEGER,
    home_reg INTEGER,             -- score after 4 quarters
    away_reg INTEGER,
    periods INTEGER,              -- 4, or 5+ with overtime
    -- team box totals, for possessions = FGA - OREB + TOV + 0.44 * FTA
    home_fga INTEGER, home_oreb INTEGER, home_tov INTEGER, home_fta INTEGER,
    away_fga INTEGER, away_oreb INTEGER, away_tov INTEGER, away_fta INTEGER
);

-- One row per game: the line from the first provider in config.LINE_PROVIDERS.
-- spread is the HOME handicap: -6.5 means home favoured by 6.5.
CREATE TABLE IF NOT EXISTS lines (
    game_id TEXT PRIMARY KEY REFERENCES games(id),
    provider TEXT NOT NULL,
    total REAL,                   -- closing (latest) total
    spread REAL,
    total_open REAL,
    spread_open REAL,
    over_odds REAL,               -- American odds on the closing total
    under_odds REAL,
    fetched_at TEXT NOT NULL      -- a line fetched before tip-off is refetched once the game is final
);

CREATE TABLE IF NOT EXISTS player_games (
    game_id TEXT NOT NULL REFERENCES games(id),
    team_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    name TEXT,
    starter INTEGER,
    minutes INTEGER,              -- NULL / 0 when DNP
    points INTEGER,
    plus_minus INTEGER,
    dnp_reason TEXT,
    PRIMARY KEY (game_id, player_id)
);

-- Append-only record of what was shown for a game before it was played. The site
-- reads played games from here and never re-predicts them.
CREATE TABLE IF NOT EXISTS published_predictions (
    game_id TEXT NOT NULL REFERENCES games(id),
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    pred_total REAL,
    pred_margin REAL,             -- home minus away
    line_total REAL,
    line_spread REAL,
    p_over REAL,
    p_home_cover REAL,
    total_tip TEXT,               -- 'over' / 'under' / NULL
    spread_tip TEXT,              -- 'home' / 'away' / NULL
    PRIMARY KEY (game_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_player_games_team ON player_games(team_id, game_id);
"""


def init_db():
    with get_db() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)


def load_games(with_lines=True):
    """All games as dicts, oldest first, with line columns (line_total, line_spread,
    line_total_open, line_spread_open) joined in when with_lines."""
    sql = "SELECT g.* FROM games g ORDER BY g.date"
    if with_lines:
        sql = ("SELECT g.*, l.total AS line_total, l.spread AS line_spread, "
               "l.total_open AS line_total_open, l.spread_open AS line_spread_open, "
               "l.provider AS line_provider "
               "FROM games g LEFT JOIN lines l ON l.game_id = g.id ORDER BY g.date")
    with get_db() as db:
        return [dict(r) for r in db.execute(sql)]


def team_info():
    with get_db() as db:
        return {r["id"]: dict(r) for r in db.execute("SELECT * FROM teams")}


@contextmanager
def get_db():
    conn = sqlite3.connect(NBA_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
