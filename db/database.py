import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "predictable.db")

def init_db():
    """Initializes the SQLite database with the required schema."""
    with get_db() as db:
        db.execute("PRAGMA journal_mode=WAL")
        
        # Teams table
        db.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                fifa_rank INTEGER,
                fifa_points REAL,
                sportradar_id TEXT
            )
        ''')
        
        # Matches table
        db.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY,
                competition_id TEXT NOT NULL,
                home_team_id INTEGER NOT NULL,
                away_team_id INTEGER NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                status TEXT,
                date TEXT,
                sportradar_id TEXT,
                FOREIGN KEY(home_team_id) REFERENCES teams(id),
                FOREIGN KEY(away_team_id) REFERENCES teams(id)
            )
        ''')
        
        # Match Stats table (from API-Sports)
        db.execute('''
            CREATE TABLE IF NOT EXISTS match_stats (
                match_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                shots_total INTEGER,
                shots_on_target INTEGER,
                shots_off_target INTEGER,
                blocked_shots INTEGER,
                corners INTEGER,
                possession REAL,
                passes_total INTEGER,
                passes_accurate INTEGER,
                fouls INTEGER,
                offsides INTEGER,
                saves INTEGER,
                PRIMARY KEY (match_id, team_id),
                FOREIGN KEY(match_id) REFERENCES matches(id),
                FOREIGN KEY(team_id) REFERENCES teams(id)
            )
        ''')
        
        # Match Events table (from SportRadar)
        db.execute('''
            CREATE TABLE IF NOT EXISTS match_events (
                id TEXT PRIMARY KEY,
                match_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                minute INTEGER,
                x_coord REAL,
                y_coord REAL,
                outcome TEXT,
                body_part TEXT,
                raw_data TEXT, -- JSON fallback for extra fields
                FOREIGN KEY(match_id) REFERENCES matches(id),
                FOREIGN KEY(team_id) REFERENCES teams(id)
            )
        ''')
        
        # Cached Team Ratings
        db.execute('''
            CREATE TABLE IF NOT EXISTS team_ratings (
                team_id INTEGER NOT NULL,
                competition_id TEXT NOT NULL,
                attack_goals REAL,
                defense_goals REAL,
                attack_corners REAL,
                defense_corners REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, competition_id),
                FOREIGN KEY(team_id) REFERENCES teams(id)
            )
        ''')
        
        # Cached Predictions
        db.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                prediction_type TEXT, -- 'goals' or 'corners'
                home_expected REAL,
                away_expected REAL,
                home_win_prob REAL,
                draw_prob REAL,
                away_win_prob REAL,
                most_likely_score TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indices
        db.execute('CREATE INDEX IF NOT EXISTS idx_matches_comp ON matches(competition_id)')
        db.execute('CREATE INDEX IF NOT EXISTS idx_match_events_type ON match_events(event_type)')
        db.execute('CREATE INDEX IF NOT EXISTS idx_match_events_match ON match_events(match_id)')

@contextmanager
def get_db():
    """Context manager for SQLite connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
