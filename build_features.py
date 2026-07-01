"""
World Cup 2026 — Commentary Feature Engineering Pipeline
=========================================================
Input  : output/*.json  (25 scraped match files)
Output : master_commentary.csv   — one row per commentary entry, cleaned + classified
         pressure_index.csv      — per-match per-minute pressure index per team
"""

import json
import os
import re
import glob
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_DIR  = "output"
OUT_MASTER = "master_commentary.csv"
OUT_PRESS  = "pressure_index.csv"

# ── 1. EVENT TYPE CLASSIFICATION ─────────────────────────────────────────────
# Rules are checked in ORDER — first match wins.
# Patterns are written against LiveScore's exact sentence structures.
# Weight = base pressure contribution of that event (0 = admin/neutral).

EVENT_RULES = [
    # ── GOALS ──────────────────────────────────────────────────────────────
    # "Goal! Argentina 2, Austria 0. [Player] left footed shot..."
    (r"(?<!\w)Goal!",                                       "goal",             10),

    # Own goal: "Own Goal by [Player] ([Team]),"
    (r"\bOwn Goal\b",                                       "own_goal",         10),

    # ── PENALTIES ──────────────────────────────────────────────────────────
    # Missed: "Penalty missed. [Player]..." or "PEN\nPenalty missed..."
    (r"\bPenalty missed\b",                                 "penalty_miss",      7),

    # Saved: "Penalty saved! [Player]..."
    (r"\bPenalty saved\b",                                  "penalty_saved",     7),

    # Awarded to team: "Penalty [Team]. [Player] draws a foul..."
    # "Penalty conceded by [Player] ([Team]) after a foul in the penalty area."
    (r"\bPenalty conceded\b",                               "penalty_conceded",  5),
    (r"^Penalty\s+\w[\w\s]*\.",                             "penalty_awarded",   5),

    # VAR review leading to penalty or overturning: "VAR Decision: Penalty..."
    (r"\bVAR Decision\b",                                   "VAR",               4),

    # ── CARDS ──────────────────────────────────────────────────────────────
    # "is shown the red card" / "is shown the red card for a bad foul"
    (r"\bshown the red card\b",                             "red_card",          8),

    # "is shown the yellow card" / "is shown the yellow card for a bad foul"
    (r"\bshown the yellow card\b",                          "yellow_card",       3),

    # ── SHOT ATTEMPTS ──────────────────────────────────────────────────────
    # "Attempt saved. [Player] right footed shot / header from [position]
    #  is saved in [location] by [GK]."
    (r"\bAttempt saved\b",                                  "attempt_saved",     4),

    # "Attempt missed. [Player] right footed shot / header from [position]
    #  misses to the left / is high and wide / is close but misses."
    (r"\bAttempt missed\b",                                 "attempt_missed",    3),

    # "Attempt blocked. [Player] right footed shot from [position] is blocked."
    (r"\bAttempt blocked\b",                                "attempt_blocked",   3),

    # ── SET PIECES ─────────────────────────────────────────────────────────
    # "Corner, [Team]. Conceded by [Player]."
    (r"^Corner,\s+\w",                                      "corner",            2),

    # "[Player] ([Team]) wins a free kick in/on the [half/wing]."
    # "Foul by [Player] ([Team])." — paired with free kick, but foul is the action
    (r"\bwins a free kick\b",                               "free_kick_won",     1),
    (r"\bFoul by\b",                                        "foul",              1),

    # Handball: "Handball by [Player] ([Team])."
    (r"\bHandball by\b",                                    "handball",          2),

    # Offside: "Offside, [Team]. [Player] is caught offside."
    (r"^Offside,\s+\w",                                     "offside",           1),

    # ── SUBSTITUTIONS ──────────────────────────────────────────────────────
    # "Substitution, [Team]. [Player A] replaces [Player B]."
    # "Substitution, [Team]. [Player A] replaces [Player B] because of an injury."
    (r"^Substitution,\s+\w",                                "substitution",      0),

    # ── DELAYS & INJURIES ──────────────────────────────────────────────────
    # "Delay in match because of an injury [Player] ([Team])."
    (r"\bDelay in match because of an injury\b",            "injury_delay",      0),

    # "Delay in match for a drinks break."
    (r"\bDelay in match for a drinks break\b",              "drinks_break",      0),

    # "Delay in match ([Team])." — generic team delay
    (r"\bDelay in match\b",                                 "delay",             0),

    # "Delay over. They are ready to continue."
    (r"\bDelay over\b",                                     "delay_over",        0),

    # ── MATCH STRUCTURE ────────────────────────────────────────────────────
    # "First Half begins." / "Second Half begins Argentina 1, Austria 0."
    (r"\bFirst Half begins\b|\bSecond Half begins\b",       "kickoff",           0),

    # "First Half ends, [Team] X, [Team] Y." / "Second Half ends..." / "Match ends..."
    (r"\bFirst Half ends\b|\bSecond Half ends\b|\bMatch ends\b", "half_end",     0),

    # "Fourth official has announced 5 minutes of added time."
    (r"\bFourth official has announced\b",                  "added_time",        0),

    # "Lineups are announced and players are warming up."
    (r"\bLineups are announced\b|\bwarming up\b",           "pre_match",         0),
]

def classify_event(text):
    for pattern, etype, weight in EVENT_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return etype, weight
    return "other", 0


# ── 2. MINUTE PARSING ─────────────────────────────────────────────────────────
def parse_minute(minute_str):
    """
    Convert minute strings to float:
      ""       → NaN
      "4'"     → 4.0
      "90+5'"  → 95.0
      "45+7'"  → 52.0
    """
    if not minute_str or minute_str.strip() == "":
        return np.nan
    clean = minute_str.replace("'", "").strip()
    if "+" in clean:
        base, extra = clean.split("+")
        return float(base) + float(extra)
    try:
        return float(clean)
    except ValueError:
        return np.nan


# ── 3. TEAM EXTRACTION ────────────────────────────────────────────────────────
def extract_team(text, team_a, team_b):
    """
    Determine which team is the 'acting' team from the text.
    Uses the team names parsed from match_name as reference.
    Returns team_a, team_b, or 'neutral'
    """
    ta = team_a.lower().replace("_", " ")
    tb = team_b.lower().replace("_", " ")
    t  = text.lower()

    hit_a = ta in t
    hit_b = tb in t

    # Parenthetical team tag e.g. "(Argentina)"
    paren = re.search(r"\(([^)]+)\)", text)
    if paren:
        p = paren.group(1).lower()
        if ta in p:
            return team_a
        if tb in p:
            return team_b

    if hit_a and not hit_b:
        return team_a
    if hit_b and not hit_a:
        return team_b
    return "neutral"


# ── 4. LOAD + CLEAN ALL MATCHES ───────────────────────────────────────────────
def load_all_matches(input_dir):
    files = glob.glob(os.path.join(input_dir, "*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in '{input_dir}/'")

    analyzer = SentimentIntensityAnalyzer()
    all_rows = []

    for fpath in sorted(files):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        match_name = data["match_name"]

        # Parse team names from match_name e.g. "Argentina_vs_Austria"
        parts = match_name.replace("_R32", "").split("_vs_")
        team_a = parts[0].strip() if len(parts) == 2 else "TeamA"
        team_b = parts[1].strip() if len(parts) == 2 else "TeamB"

        entries = data.get("raw_entries", [])
        # Reverse so entries go chronologically (kickoff → full time)
        entries = list(reversed(entries))

        for i, entry in enumerate(entries):
            raw_minute = entry.get("minute", "")
            text       = entry.get("text", "").strip()
            if not text:
                continue

            minute_float           = parse_minute(raw_minute)
            event_type, base_weight = classify_event(text)
            team                   = extract_team(text, team_a, team_b)

            # VADER sentiment — gives compound score in [-1, 1]
            # For our use: treat magnitude (abs) as urgency signal
            vader  = analyzer.polarity_scores(text)
            urgency = abs(vader["compound"])  # 0 = neutral, 1 = max intensity

            # Final entry weight = base_weight boosted by urgency
            weight = round(base_weight * (1 + urgency), 3)

            all_rows.append({
                "match_name":   match_name,
                "team_a":       team_a,
                "team_b":       team_b,
                "entry_index":  i,
                "raw_minute":   raw_minute,
                "minute":       minute_float,
                "event_type":   event_type,
                "acting_team":  team,
                "base_weight":  base_weight,
                "urgency":      round(urgency, 3),
                "weight":       weight,
                "text":         text,
            })

        print(f"  Loaded {len(entries):>3} entries — {match_name}")

    return pd.DataFrame(all_rows)


# ── 5. PRESSURE INDEX CALCULATION ─────────────────────────────────────────────
def build_pressure_index(df, window=5):
    """
    For each match × team, compute rolling pressure index over a
    `window`-minute rolling window of event weights.

    Output: one row per (match, team, minute) with pressure_index value.
    """
    results = []

    for match_name, match_df in df.groupby("match_name"):
        team_a = match_df["team_a"].iloc[0]
        team_b = match_df["team_b"].iloc[0]

        # Drop entries with no minute (pre-match / half-end admin rows)
        md = match_df.dropna(subset=["minute"]).copy()

        # Create a full minute grid 1-120 (covers extra time)
        minutes = np.arange(1, 121, dtype=float)

        for team in [team_a, team_b, "neutral"]:
            team_df = md[md["acting_team"] == team]

            # Sum weights landing on each exact minute
            minute_weights = team_df.groupby("minute")["weight"].sum()
            minute_series  = pd.Series(0.0, index=minutes)
            for min_val, w in minute_weights.items():
                if min_val in minute_series.index:
                    minute_series[min_val] = w

            # Rolling sum over window minutes → Pressure Index
            rolling = minute_series.rolling(window=window, min_periods=1).sum()

            for minute, press in rolling.items():
                results.append({
                    "match_name":     match_name,
                    "team":           team,
                    "minute":         int(minute),
                    "pressure_index": round(press, 3),
                })

    return pd.DataFrame(results)


# ── 6. MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("Loading and cleaning matches...")
    df = load_all_matches(INPUT_DIR)
    print(f"\nTotal entries across all matches: {len(df)}")

    # Save master CSV
    df.to_csv(OUT_MASTER, index=False)
    print(f"Saved: {OUT_MASTER}")

    # Event type distribution
    print("\nEvent type counts:")
    print(df["event_type"].value_counts().to_string())

    # Build pressure index
    print("\nBuilding pressure index...")
    press_df = build_pressure_index(df, window=5)
    press_df.to_csv(OUT_PRESS, index=False)
    print(f"Saved: {OUT_PRESS}")
    print(f"Pressure index rows: {len(press_df)}")

    print("\nDone. Next step: dashboard (Streamlit).")


if __name__ == "__main__":
    main()