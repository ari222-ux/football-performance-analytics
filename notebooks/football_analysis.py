import pandas as pd
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load datasets safely
fixtures = pd.read_csv(BASE_DIR / "data" / "fixtures.csv")
team_stats = pd.read_csv(BASE_DIR / "data" / "teamStats.csv")
teams = pd.read_csv(BASE_DIR / "data" / "teams.csv")


# View first rows
print(fixtures.head())
print(team_stats.head())
print(teams.head())

print("\n--- FIXTURES INFO ---")
fixtures.info()

print("\n--- TEAM STATS INFO ---")
team_stats.info()

print("\n--- TEAMS INFO ---")
teams.info()
# =========================
# STEP 4.1: COLUMN SELECTION
# =========================

# Select relevant columns from fixtures
fixtures_clean = fixtures[
    [
        "eventId",
        "date",
        "homeTeamId",
        "awayTeamId",
        "homeTeamWinner",
        "attendance"
    ]
].copy()

# Select relevant columns from teamStats
team_stats_clean = team_stats[
    [
        "eventId",
        "teamId",
        "possessionPct",
        "totalShots",
        "shotsOnTarget",
        "foulsCommitted",
        "yellowCards",
        "redCards"
    ]
].copy()

# Select relevant columns from teams
teams_clean = teams[
    [
        "teamId",
        "name",
        "location"
    ]
].copy()
print("\n--- FIXTURES CLEAN ---")
print(fixtures_clean.head())

print("\n--- TEAM STATS CLEAN ---")
print(team_stats_clean.head())

print("\n--- TEAMS CLEAN ---")
print(teams_clean.head())
# =========================
# STEP 4.2: ADD TEAM NAMES
# =========================

team_stats_named = team_stats_clean.merge(
    teams_clean,
    on="teamId",
    how="left"
)
print("\n--- TEAM STATS WITH NAMES ---")
print(team_stats_named.head())

# STEP 4.3: ADD MATCH CONTEXT


# Merge team stats with fixture info
team_match = team_stats_named.merge(
    fixtures_clean,
    on="eventId",
    how="left"
)

# Determine home or away
team_match["isHome"] = team_match["teamId"] == team_match["homeTeamId"]

# Determine if team won
team_match["isWinner"] = (
    (team_match["isHome"] & team_match["homeTeamWinner"]) |
    (~team_match["isHome"] & ~team_match["homeTeamWinner"])
)
print("\n--- TEAM MATCH DATA ---")
print(team_match[[
    "eventId",
    "name",
    "isHome",
    "homeTeamWinner",
    "isWinner"
]].head(10))
# =========================
# STEP 4.4: FEATURE ENGINEERING
# =========================

# Shot accuracy
team_match["shotAccuracy"] = (
    team_match["shotsOnTarget"] / team_match["totalShots"]
)

# Discipline score (higher = worse discipline)
team_match["disciplineScore"] = (
    team_match["foulsCommitted"]
    + team_match["yellowCards"] * 2
    + team_match["redCards"] * 5
)

# Possession buckets
team_match["possessionBucket"] = pd.cut(
    team_match["possessionPct"],
    bins=[0, 40, 50, 60, 100],
    labels=["Low", "Medium", "High", "Very High"]
)
print("\n--- FEATURE ENGINEERED DATA ---")
print(team_match[
    ["name", "possessionPct", "possessionBucket", "shotAccuracy", "disciplineScore"]
].head(10))
win_rate_by_possession = (
    team_match
    .groupby("possessionBucket")["isWinner"]
    .mean()
    .sort_index()
)

# =========================
# STEP 5.2: TEAM PERFORMANCE ANALYSIS
# =========================

team_win_rate = (
    team_match
    .groupby("name")["isWinner"]
    .mean()
    .sort_values(ascending=False)
)

print("\n--- TOP TEAMS BY WIN RATE ---")
print(team_win_rate.head(10))
shot_accuracy_vs_win = (
    team_match
    .groupby("isWinner")["shotAccuracy"]
    .mean()
)

print("\n--- SHOT ACCURACY VS MATCH RESULT ---")
print(shot_accuracy_vs_win)
discipline_vs_win = (
    team_match
    .groupby("isWinner")["disciplineScore"]
    .mean()
)

print("\n--- DISCIPLINE SCORE VS MATCH RESULT ---")
print(discipline_vs_win)
# =========================
# STEP 5.3: INSIGHT SUMMARY DATA
# =========================

insight_summary = {
    "Metric": [
        "Win rate (Low possession)",
        "Win rate (Medium possession)",
        "Win rate (High possession)",
        "Win rate (Very High possession)",
        "Avg shot accuracy (Wins)",
        "Avg shot accuracy (Losses)",
        "Avg discipline score (Wins)",
        "Avg discipline score (Losses)"
    ],
    "Value": [
        win_rate_by_possession.get("Low"),
        win_rate_by_possession.get("Medium"),
        win_rate_by_possession.get("High"),
        win_rate_by_possession.get("Very High"),
        shot_accuracy_vs_win.get(True),
        shot_accuracy_vs_win.get(False),
        discipline_vs_win.get(True),
        discipline_vs_win.get(False)
    ]
}

insight_df = pd.DataFrame(insight_summary)

print("\n--- FINAL INSIGHT SUMMARY ---")
print(insight_df)
# =========================
# STEP 6.1: EXPORT FOR POWER BI
# =========================

team_match.to_csv(
    "team_match_analysis.csv",
    index=False
)

print("Data exported successfully for Power BI")
