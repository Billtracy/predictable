import os

UCL_DIR = os.path.dirname(os.path.abspath(__file__))
UCL_DB_PATH = os.environ.get("UCL_DB_PATH", os.path.join(os.path.dirname(UCL_DIR), "ucl.db"))

# Season keys use the START year, same convention as the EPL config ("2025" = 25/26).
# The SR trial key exposes 24/25, 25/26 and 26/27 only.
SEASONS = (2024, 2025, 2026)
CURRENT_SEASON = 2026

UCL = "ucl"
UEL = "uel"

# Competitions ingested into ucl.db. The domestic leagues are scores-only inputs:
# they rate teams between UCL games, and the European meetings between them put the
# leagues on one scale. The Europa League is a link-only input: it is never
# predicted, but it roughly triples the cross-league games, which halved the league-
# strength error on synthetic data. Add a league here (e.g. Portugal, Netherlands)
# to give its clubs domestic form instead of the "other" bucket.
COMPETITIONS = {
    UCL:          {"urn": "sr:competition:7",   "name": "UEFA Champions League"},
    UEL:          {"urn": "sr:competition:679", "name": "UEFA Europa League"},
    "epl":        {"urn": "sr:competition:17",  "name": "Premier League"},
    "laliga":     {"urn": "sr:competition:8",   "name": "LaLiga"},
    "seriea":     {"urn": "sr:competition:23",  "name": "Serie A"},
    "bundesliga": {"urn": "sr:competition:35",  "name": "Bundesliga"},
    "ligue1":     {"urn": "sr:competition:34",  "name": "Ligue 1"},
}
EUROPEAN = (UCL, UEL)
DOMESTIC = tuple(c for c in COMPETITIONS if c not in EUROPEAN)

# League bucket for teams with no domestic data in ucl.db (Benfica, PSV, Celtic...).
OTHER_LEAGUE = "other"

# --- Model parameters (tune via ucl/backtest.py) ---

# Tuned by ucl/tune.py on the 25/26 walk-forward (189 matches). The surface is a
# flat plateau over half-life 365-730 x team ridge 8-16 (binary Brier within
# 0.0002); these are its middle, and beat the first-guess 180/2/5 on the 26/27
# holdout too (binary 0.1988 vs 0.2022, RPS 0.1744 vs 0.1764).

# Exponential time decay on match weight. A long half-life wins: a club's level
# changes slowly relative to how noisy a few months of results are.
HALF_LIFE_DAYS = 365.0

# L2 penalty on each team's deviation from its league mean (log scale). Teams with
# few matches stay near their league average; well-observed teams are free to move.
TEAM_RIDGE = 8.0

# L2 penalty on league strength effects. The cross-league link (European meetings)
# is thin, so league effects are noisy. Real-data results barely move from 2 to 10.
LEAGUE_RIDGE = 5.0

# Corners model: same joint structure, fitted on matches with corner counts (UCL
# league phase onward + EPL). Tuned by `ucl/tune.py --target corners` on 25/26:
# mean over-line Brier 0.2026 vs base-rate 0.2080. Corners want more shrinkage
# than goals: club corner styles are steadier than a few games suggest.
CORNER_HALF_LIFE_DAYS = 365.0
CORNER_TEAM_RIDGE = 16.0
CORNER_LEAGUE_RIDGE = 10.0

# Variance / mean of total corners. EPL runs 1.06-1.17, UCL 1.24-1.30; on the UCL
# walk-forward 1.0-1.4 all score within 0.001 Brier, so this is a measured middle.
# Tips use expected totals only, so they don't depend on it.
CORNER_DISPERSION = 1.2

# Lines the corner markets are priced at; tips come from these.
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)

# Dixon-Coles low-score correction, same value as the EPL model.
DIXON_COLES_RHO = -0.05

# Largest goal count per team in the scoreline grid.
MAX_GOALS = 10
