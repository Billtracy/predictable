import requests
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import API_SPORTS_KEY, API_SPORTS_HOST
from db.database import get_db

HEADERS = {
    "x-apisports-key": API_SPORTS_KEY,
    "x-rapidapi-host": API_SPORTS_HOST
}

def fetch_fifa_rankings():
    """Fetch FIFA rankings from API-Sports and update the database."""
    print("Fetching FIFA rankings...")
    # API-Sports FIFA rankings endpoint doesn't exist directly like this in v3, 
    # it's under teams/statistics or we might need to use a general rankings approach.
    # Actually, API-Sports v3 has a GET /rankings endpoint or we can get points.
    # Let's assume there is a way or we can mock/fetch it from an alternative source if it's missing.
    # Actually, API-football doesn't have a direct /rankings for FIFA international teams. 
    # It has /standings for tournaments. 
    # Let's simulate FIFA rankings updating based on a predefined mapping or fetching from /teams if they have ranking fields.
    
    # In API-Football v3, /teams endpoint returns national=True/False, but not fifa_ranking directly.
    # For the sake of the model, we can assign a base ranking or a simplified proxy ranking 
    # if the API doesn't provide it, or use a known static dictionary for 2022/2026 teams to bootstrap.
    # For a robust implementation, you might want to hit a free FIFA ranking API.
    # Here, we will do a proxy: if no ranking, default to 50.
    
    print("Note: API-Football does not natively provide FIFA rankings in standard endpoints.")
    print("Assigning default proxy rankings based on World Cup 2022 cache/standings...")
    
    # Bootstrapping some top team rankings for demonstration
    static_rankings = {
        "Argentina": (1, 1855),
        "France": (2, 1845),
        "England": (3, 1800),
        "Belgium": (4, 1798),
        "Brazil": (5, 1784),
        "Netherlands": (6, 1745),
        "Portugal": (7, 1745),
        "Spain": (8, 1732),
        "Italy": (9, 1718),
        "Croatia": (10, 1717),
        "USA": (11, 1681),
        "Morocco": (12, 1661),
        "Colombia": (14, 1655),
        "Mexico": (15, 1652),
        "Uruguay": (15, 1652),
        "Germany": (16, 1631),
        "Senegal": (17, 1624),
        "Japan": (18, 1620),
        "Switzerland": (19, 1616),
        "Iran": (20, 1610),
        "Denmark": (21, 1602),
        "Australia": (23, 1539),
        "Poland": (28, 1520),
        "Tunisia": (41, 1493),
        "Ecuador": (31, 1519),
        "Qatar": (34, 1504),
        "Saudi Arabia": (53, 1441),
        "Wales": (29, 1520),
        "Cameroon": (43, 1484),
        "Canada": (41, 1475),
        "Costa Rica": (31, 1500),
        "Ghana": (60, 1393),
        "Serbia": (25, 1549),
        "South Korea": (28, 1530),
    }
    
    with get_db() as db:
        for name, (rank, points) in static_rankings.items():
            db.execute('''
                UPDATE teams
                SET fifa_rank = ?, fifa_points = ?
                WHERE name = ?
            ''', (rank, points, name))
            
        # Update any NULL rankings to a default of 50
        db.execute('''
            UPDATE teams
            SET fifa_rank = 50, fifa_points = 1400
            WHERE fifa_rank IS NULL
        ''')
        
    print("FIFA rankings updated in database.")

if __name__ == "__main__":
    fetch_fifa_rankings()
