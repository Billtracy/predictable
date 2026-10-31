import sqlite3
from contextlib import contextmanager

from ucl.config import UCL_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id TEXT PRIMARY KEY,          -- sr:competitor id, stable across competitions/seasons
    name TEXT NOT NULL,
    country TEXT,
    abbreviation TEXT             -- SR 3-letter code, used for badges
);

CREATE TABLE IF NOT EXISTS matches (
    id TEXT PRIMARY KEY,          -- sr:sport_event id
    competition TEXT NOT NULL,    -- key from ucl.config.COMPETITIONS
    season INTEGER NOT NULL,      -- start year
    date TEXT NOT NULL,           -- ISO kickoff time (UTC)
    home_id TEXT NOT NULL REFERENCES teams(id),
    away_id TEXT NOT NULL REFERENCES teams(id),
    status TEXT NOT NULL,         -- 'FT' or 'NS'
    home_goals INTEGER,           -- 90-minute score: what the model fits and markets settle on
    away_goals INTEGER,
    home_final INTEGER,           -- final score incl. extra time (= 90-min score otherwise)
    away_final INTEGER,
    decided_by TEXT,              -- NULL, 'aet' or 'pen'
    winner_id TEXT,               -- match winner incl. extra time/penalties (NULL = draw)
    tie_winner_id TEXT,           -- knockout: who went through (set on the deciding match)
    stage TEXT,                   -- SR stage phase, e.g. 'league', 'playoffs', 'qualification'
    round_name TEXT,              -- e.g. 'round_of_16', 'final'
    round_number INTEGER,         -- league-phase matchday
    tie_id TEXT,                  -- SR cup_round id: shared by both legs of a knockout tie
    leg INTEGER,                  -- 1 or 2 for two-legged ties, NULL otherwise
    neutral INTEGER NOT NULL DEFAULT 0,
    home_corners INTEGER,         -- NULL until fetched (ucl.ingest corners / epl-corners)
    away_corners INTEGER
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_comp ON matches(competition, season);
CREATE INDEX IF NOT EXISTS idx_matches_tie ON matches(tie_id);
"""


def init_db():
    with get_db() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        # Columns added after the first ingest; CREATE IF NOT EXISTS won't add them.
        for table, col, typ in (("teams", "abbreviation", "TEXT"), ("matches", "round_number", "INTEGER"),
                                ("matches", "home_corners", "INTEGER"), ("matches", "away_corners", "INTEGER")):
            if col not in {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def load_matches(competitions=None):
    """All matches as dicts, oldest first. competitions: optional iterable filter."""
    with get_db() as db:
        rows = [dict(r) for r in db.execute("SELECT * FROM matches ORDER BY date")]
    if competitions is not None:
        wanted = set(competitions)
        rows = [r for r in rows if r["competition"] in wanted]
    return rows


def team_names():
    with get_db() as db:
        return {r["id"]: r["name"] for r in db.execute("SELECT id, name FROM teams")}


def team_info():
    with get_db() as db:
        return {r["id"]: dict(r) for r in db.execute("SELECT * FROM teams")}


@contextmanager
def get_db():
    conn = sqlite3.connect(UCL_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
