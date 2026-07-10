# Predictable ⚽📊

A Poisson-based football match prediction engine that predicts **scorelines** and **corner totals** using attack/defense strength ratings, expected goals (xG), FIFA rankings, and Monte Carlo simulation.

Inspired by [this approach](https://www.youtube.com/): fetch detailed match event data (including X/Y coordinates of shots), build per-team attack/defense profiles, then run 10,000 Poisson-distributed simulations to generate win/draw/loss probabilities and most likely scorelines.

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| Config + Database schema | ✅ Done | SQLite (`predictable.db`), auto-created on first run |
| API-Sports data fetcher | ✅ Done | Fetches fixtures + per-match stats (corners, shots, possession, etc.) |
| SportRadar data fetcher | ✅ Done | Fetches X/Y coordinate event data (shots, corners) for xG model |
| FIFA Rankings loader | ✅ Done | Static bootstrap applied after teams are loaded |
| xG Model | ✅ Done | Logistic regression on shot distance/angle, with proxy fallback |
| Team Ratings calculator | ✅ Done | Attack/defense scores for goals + corners, competition-weighted |
| Poisson Monte Carlo engine | ✅ Done | 10K simulations with unpredictability factor per FIFA rank |
| Goal + Corner predictors | ✅ Done | Orchestrators that wire everything together |
| CLI interface (`predict.py`) | ✅ Done | `fetch` and `match` commands |
| World Cup 2022 data | ✅ Done | All 64 match stats and events fetched, backtested at 51.56% accuracy |
| World Cup 2026 data | ✅ Done | Fetched via SportRadar full import (API-Sports free tier limit bypassed) |
| EPL 2024 data | ⏳ In Progress | Fetching 380 matches... (Takes time due to API-Sports 10 req/min limit) |
| SportRadar events | ✅ Done | Events downloaded for WC 2022 & 2026. xG model trained on 3399 real shots. |

## Project Structure

```
Predictable/
├── config.py                  # API keys and competition definitions
├── predict.py                 # CLI entry point (fetch data + run predictions)
├── predictable.db             # SQLite database (auto-created)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── db/
│   ├── __init__.py
│   └── database.py            # SQLite schema + connection manager
│
├── data/
│   ├── __init__.py
│   ├── api_sports.py          # Fetches from API-Football (basic stats: corners, shots, possession)
│   ├── sportradar.py          # Fetches from SportRadar Extended (X/Y coordinate events)
│   └── rankings.py            # Loads FIFA rankings into teams table
│
├── models/
│   ├── __init__.py
│   ├── xg_model.py            # Expected Goals model (logistic regression on shot coords)
│   ├── team_ratings.py        # Computes attack/defense strength per team per competition
│   └── poisson.py             # Monte Carlo Poisson simulation engine
│
├── predictors/
│   ├── __init__.py
│   ├── goals.py               # Goal prediction orchestrator
│   └── corners.py             # Corner prediction orchestrator
│
├── corner_predictor.py        # [LEGACY] Original corner predictor (kept for reference)
├── index.py                   # [LEGACY] Original index script (kept for reference)
└── test.py                    # [LEGACY] API connection test
```

---

## Setup

### 1. Prerequisites
- Python 3.10+
- A virtual environment (already created in `venv/`)

### 2. Install Dependencies

```bash
# If venv already exists:
venv/bin/pip install -r requirements.txt

# If starting fresh:
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### 3. API Keys

Both keys are already set in `config.py`:

| Service | Key Location | Purpose |
|---|---|---|
| **API-Sports** (API-Football) | `config.py → API_SPORTS_KEY` | Basic match stats (goals, corners, shots, possession) |
| **SportRadar Extended** | `config.py → SPORTRADAR_KEY` | X/Y coordinate event data for the xG model |

> **SportRadar free trial is 30 days.** All data gets cached in SQLite so you only fetch each match once.

---

## Usage

### Step 1: Fetch Data for a Competition

```bash
# Fetch World Cup 2022 data (for backtesting)
venv/bin/python predict.py fetch --competition world_cup_2022

# Fetch World Cup 2026 data (for live predictions)
venv/bin/python predict.py fetch --competition world_cup_2026

# Fetch EPL 2024 data (for gameweek predictions - free tier max season is 2024)
venv/bin/python predict.py fetch --competition epl_2024
```

The `fetch` command does the following in order:
1. Initializes the SQLite database
2. Fetches all fixtures from API-Sports and saves teams + matches
3. Fetches per-match statistics (corners, shots, possession, etc.) — **7-second delay between calls** to respect the 10 req/min free tier limit
4. Loads FIFA rankings after teams exist in the database
5. Fetches SportRadar event data (X/Y coordinates for shots/corners)
6. Computes team attack/defense ratings

> ⚠️ **API-Sports free tier = 10 requests/minute.** Fetching stats for 64 World Cup matches takes ~8 minutes. The fetcher auto-retries on rate limits with a 60-second backoff.

### Step 2: Predict a Match

```bash
# Predict goals only
venv/bin/python predict.py match "Argentina" "France" --competition world_cup_2022 --type goals

# Predict corners only
venv/bin/python predict.py match "England" "Iran" --competition world_cup_2022 --type corners

# Predict both goals and corners
venv/bin/python predict.py match "Argentina" "France" --competition world_cup_2022 --type both
```

### Example Output

```
⚽ MATCH PREDICTION: Argentina vs France
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Based on 10,000 simulations

GOALS
  Argentina xG: 2.67    France xG: 2.09
  Win: Argentina 51.1% | Draw 17.6% | France 31.3%

  Most Likely Scorelines:
    2-1  →  6.5%         2-2  →  6.3%
    1-1  →  5.1%         3-1  →  5.1%
    3-2  →  4.9%         1-2  →  4.5%

🔲 CORNERS
  Argentina Expected: 5.8    France Expected: 4.4
  Total Expected: 10.2

  Over/Under Lines:
    Over 7.5:  78.4%     Over 8.5:  68.1%
    Over 9.5:  54.2%     Over 10.5: 38.7%

  Most Likely Total: 10 corners (12.1%)
```

---

## How the Model Works

### 1. Data Collection
- **API-Sports** provides aggregate stats per match: goals, corners, shots (total/on target/off target), possession, passes, fouls, saves
- **SportRadar Extended** provides per-event X/Y coordinates on the pitch — specifically shot locations used to build the xG model

### 2. Expected Goals (xG) Model (`models/xg_model.py`)
- Extracts all shot events with X/Y coordinates from the database
- Calculates **distance to goal** and **visible angle of goal** for each shot
- Trains a **logistic regression** model: `P(Goal) = σ(β₀ + β₁·distance + β₂·angle)`
- Falls back to hardcoded baseline coefficients if < 100 shots in the database
- Also provides a **proxy xG** for teams where only aggregate stats are available: `proxy_xG = shots_on_target × 0.3`

### 3. Team Ratings (`models/team_ratings.py`)
For each team in a competition, calculates:

| Metric | Formula |
|---|---|
| **Attack (Goals)** | `(0.4 × goals_per_match + 0.6 × proxy_xG) / league_avg × difficulty_weight` |
| **Defense (Goals)** | `goals_conceded_per_match / league_avg × difficulty_weight` |
| **Attack (Corners)** | `corners_won_per_match / league_avg × difficulty_weight` |
| **Defense (Corners)** | `corners_conceded_per_match / league_avg × difficulty_weight` |

Values > 1.0 = above average, < 1.0 = below average. Capped between 0.1 and 3.0.

### 4. Monte Carlo Poisson Simulation (`models/poisson.py`)

1. Calculate **expected goals** for the matchup:
   - `home_xG = home_attack × away_defense × league_avg × home_advantage`
   - `away_xG = away_attack × home_defense × league_avg × away_disadvantage`
2. For each of **10,000 simulations**:
   - Add random noise based on **unpredictability factor** (lower-ranked teams → higher std deviation → more upset potential)
   - Draw actual goals from a Poisson distribution with the noisy lambda
3. Aggregate results → win/draw/loss probabilities, most likely scorelines

The same engine works for corners — swap xG for expected corners.

### 5. Unpredictability Factor
```
std_dev = 0.2 + (min(fifa_rank, 80) × 0.007)
```
- Rank 1 team: std = 0.207 (very consistent)
- Rank 50 team: std = 0.55 (more volatile)
- Rank 80+ team: std = 0.76 (high upset potential)

---

## Competitions Configured

Defined in `config.py`:

| Key | League | Season | Difficulty Weight |
|---|---|---|---|
| `world_cup_2022` | 1 (FIFA World Cup) | 2022 | 1.0 |
| `world_cup_2026` | 1 (FIFA World Cup) | 2026 | 1.0 | (SportRadar Only)
| `epl_2024` | 39 (Premier League) | 2024 | 0.95 |

To add more competitions, add entries to the `COMPETITIONS` dict in `config.py`. You'll need the API-Sports league ID (find via `test.py` or the [API-Football docs](https://www.api-football.com/documentation-v3)).

---

## Database Schema

SQLite file: `predictable.db` (auto-created on first run)

| Table | Purpose |
|---|---|
| `teams` | Team ID, name, FIFA rank, FIFA points, SportRadar ID |
| `matches` | Fixture data: teams, scores, date, competition, status |
| `match_stats` | Per-match aggregate stats from API-Sports |
| `match_events` | Per-event X/Y coordinate data from SportRadar |
| `team_ratings` | Cached computed attack/defense scores |
| `predictions` | Stored predictions for accuracy tracking |

You can inspect the database directly:
```bash
sqlite3 predictable.db
.tables
SELECT name, fifa_rank FROM teams ORDER BY fifa_rank;
SELECT COUNT(*) FROM match_stats;
.quit
```

---

## Backtesting

You can run a backtest on completed competitions to evaluate the engine's accuracy:
```bash
venv/bin/python backtest.py world_cup_2022
venv/bin/python backtest.py epl_2024
```

The backtest is **leak-free / point-in-time**: each fixture is predicted using only
matches played *strictly before* its date, so a team's ratings never "see" results
that hadn't happened yet. It reports outcome accuracy plus probability-calibration
metrics (Log Loss, Brier, RPS — lower is better) so you can judge whether the
*probabilities* themselves are honest, not just the top pick.

*EPL 2024 (leak-free): ~49% outcome accuracy, Log Loss ~1.006 / Brier ~0.602 / RPS ~0.209 — all beating the uniform-guess baselines. The World Cup sits near chance because group-stage teams have almost no prior matches to rate them on (an earlier ~51% figure was inflated by look-ahead leakage).*

---

## Rate Limits

| API | Free Tier Limit | Our Delay | Notes |
|---|---|---|---|
| API-Sports | 10 requests/minute | 7 seconds between calls | Auto-retries with 60s backoff on 429 |
| SportRadar | ~1 request/second (trial) | 1.5 seconds between calls | 30-day trial |

The fetcher skips matches that are already in the database, so re-running a fetch is safe and will only fetch what's missing.

---

## Dependencies

```
requests        # HTTP client for API calls
numpy           # Fast random sampling for Monte Carlo simulation
scikit-learn    # Logistic regression for the xG model
```

`sqlite3` is built into Python — no install needed.

---

## Adding a New Competition

1. Find the league ID on [API-Football](https://www.api-football.com/documentation-v3) (or run `test.py` with a search query)
2. Find the SportRadar competition URN from their [coverage matrix](https://coverage-matrix.sportradar.com/)
3. Add to `config.py`:

```python
COMPETITIONS["la_liga_2025"] = {
    "api_sports": {"league": 140, "season": 2025},
    "sportradar": {"urn": "sr:competition:8"},
    "difficulty_weight": 0.90,
}
```

4. Fetch: `venv/bin/python predict.py fetch --competition la_liga_2025`
5. Predict: `venv/bin/python predict.py match "Real Madrid" "Barcelona" --competition la_liga_2025`
