import os

def _load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Minimal .env loader (KEY=VALUE lines) so secrets stay out of git.
    Real environment variables take precedence over the file."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass

_load_dotenv()

# API-Sports — key comes from .env (see .env.example)
API_SPORTS_KEY = os.environ.get("API_SPORTS_KEY", "")
API_SPORTS_HOST = "v3.football.api-sports.io"

# SportRadar — key comes from .env (see .env.example)
SPORTRADAR_KEY = os.environ.get("SPORTRADAR_KEY", "")
SPORTRADAR_HOST = "api.sportradar.com"
SPORTRADAR_ACCESS_LEVEL = "trial"
SPORTRADAR_VERSION = "v4"

# Competitions to track
COMPETITIONS = {
    "world_cup_2026": {
        "api_sports": {"league": 1, "season": 2026},
        "sportradar": {"urn": "sr:competition:16"}, # SportRadar World Cup URN
        "difficulty_weight": 1.0,  # Highest difficulty
        "sportradar_only": True,
    },
    "world_cup_2022": {
        "api_sports": {"league": 1, "season": 2022},
        "sportradar": {"urn": "sr:competition:16"},
        "difficulty_weight": 1.0,
    },
    "epl_2024": {
        "api_sports": {"league": 39, "season": 2024},
        "sportradar": {"urn": "sr:competition:17"}, # SportRadar Premier League URN
        "difficulty_weight": 0.95,  # EPL is very competitive
        "prior_competition": "epl_2023",  # seed early-season ratings from last season
    },
    # Prior season, used only to seed cross-season rating priors for epl_2024's
    # early games (fixes the cold-start hole). Needs goal results only, not shot
    # stats/events. 2023 is within the API-Sports free-tier coverage window.
    "epl_2023": {
        "api_sports": {"league": 39, "season": 2023},
        "sportradar": {"urn": "sr:competition:17"},
        "difficulty_weight": 0.95,
    },
    # 2025/26 season: most recent complete season. Full SportRadar ingest (scores +
    # shot timelines) — it's the xG-training boost AND the prior for 2026/27, and the
    # strongest dress-rehearsal set. SR trial exposes 25/26, so no API-Sports needed.
    "epl_2025": {
        "api_sports": {"league": 39, "season": 2025},
        "sportradar": {"urn": "sr:competition:17"},
        "difficulty_weight": 0.95,
        "prior_competition": "epl_2024",
    },
}
