import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# Setup
MASTER_CSV = "master_commentary.csv"
PRESS_CSV  = "pressure_index.csv"
VISUALS    = "visuals"
os.makedirs(VISUALS, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

GOAL_COLOR    = "#f78166"   
PRESS_COLOR   = "#58a6ff" 
NEUTRAL_COLOR = "#8b949e"  
ACCENT_COLOR  = "#3fb950"  

def save(fig, name):
    path = os.path.join(VISUALS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# Load Data
print("Loading data...")
master = pd.read_csv(MASTER_CSV)
press  = pd.read_csv(PRESS_CSV)

# Goal events only (acting team is the scorer)
goals = master[master["event_type"] == "goal"].copy()
goals = goals.dropna(subset=["minute"])
print(f"  Total goals found: {len(goals)}")
print(f"  Matches: {master['match_name'].nunique()}")


# Helper: get pressure for a team in a match at a minute
def get_pressure(match_name, team, minute):
    row = press[
        (press["match_name"] == match_name) &
        (press["team"] == team) &
        (press["minute"] == int(minute))
    ]
    return row["pressure_index"].values[0] if len(row) else 0.0

def get_pressure_window(match_name, team, minute, window=5):
    """Average pressure in the N minutes before a given minute."""
    minutes = range(max(1, int(minute) - window), int(minute))
    vals = [get_pressure(match_name, team, m) for m in minutes]
    return np.mean(vals) if vals else 0.0

def get_match_avg(match_name, team):
    """Overall average pressure for a team across the full match."""
    rows = press[
        (press["match_name"] == match_name) &
        (press["team"] == team)
    ]
    return rows["pressure_index"].mean() if len(rows) else 0.0


# Build per-goal stats
print("Computing per-goal pressure stats...")
records = []
for _, row in goals.iterrows():
    mn        = row["match_name"]
    team      = row["acting_team"]
    minute    = row["minute"]
    pre5      = get_pressure_window(mn, team, minute, window=5)
    pre10     = get_pressure_window(mn, team, minute, window=10)
    match_avg = get_match_avg(mn, team)
    at_goal   = get_pressure(mn, team, minute)

    records.append({
        "match":       mn,
        "team":        team,
        "minute":      minute,
        "pre5_avg":    round(pre5, 3),
        "pre10_avg":   round(pre10, 3),
        "match_avg":   round(match_avg, 3),
        "at_goal":     round(at_goal, 3),
        "uplift":      round(pre5 - match_avg, 3),
        "label":       f"{team}\n{mn.replace('_',' ')}\n{int(minute)}'"
    })

goal_df = pd.DataFrame(records)

SET_PIECE_KEYWORDS   = ["following a corner", "following a set piece", "set piece situation",
                        "free kick", "direct free kick"]
COUNTER_KEYWORDS     = ["fast break", "on the break", "counter attack", "counterattack",
                        "on the counter"]
PENALTY_KEYWORDS     = ["penalty spot", "from the spot", "converts the penalty"]
HEADER_KEYWORDS      = ["header"]

def classify_goal_type(goal_row, master_df):
    mn     = goal_row["match"]
    minute = goal_row["minute"]
    team   = goal_row["team"]

    # Goal entry text itself — this is the PRIMARY classification signal
    goal_text = master_df[
        (master_df["match_name"] == mn) &
        (master_df["minute"] == minute) &
        (master_df["event_type"] == "goal")
    ]["text"].str.lower().str.cat(sep=" ")

    # Narrow pre-window (3 minutes) — used ONLY for penalty event detection
    pre_window = master_df[
        (master_df["match_name"] == mn) &
        (master_df["minute"] >= minute - 3) &
        (master_df["minute"] < minute)
    ]

    
    penalty_events = pre_window[
        pre_window["event_type"].isin(["penalty_awarded", "penalty_conceded"])
    ]
    var_penalty_events = pre_window[
        (pre_window["event_type"] == "VAR") &
        pre_window["text"].str.contains("penalty", case=False, na=False)
    ]
    if any(k in goal_text for k in PENALTY_KEYWORDS) or len(penalty_events) > 0 or len(var_penalty_events) > 0:
        return "penalty"

    # Counter-attack: keywords in goal text only
    if any(k in goal_text for k in COUNTER_KEYWORDS):
        return "counter_attack"

    # Set piece: keywords in goal text only
    if any(k in goal_text for k in SET_PIECE_KEYWORDS):
        return "set_piece"

    return "open_play"

goal_df["goal_type"] = goal_df.apply(
    lambda r: classify_goal_type(r, master), axis=1
)

goal_df.to_csv("goal_pressure_stats.csv", index=False)
print(f"  Saved goal_pressure_stats.csv ({len(goal_df)} goals)")
print(f"  Goal type breakdown:\n{goal_df['goal_type'].value_counts().to_string()}")

# VISUAL 1 — Pre-goal pressure vs match average (bar chart per goal)

print("\n[1] Pre-goal pressure bar chart...")
fig, ax = plt.subplots(figsize=(16, 6))

x     = np.arange(len(goal_df))
width = 0.35

bars1 = ax.bar(x - width/2, goal_df["pre5_avg"],  width, color=GOAL_COLOR,    alpha=0.85, label="Avg pressure (5 min before goal)")
bars2 = ax.bar(x + width/2, goal_df["match_avg"], width, color=NEUTRAL_COLOR, alpha=0.65, label="Full-match avg pressure")

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{r['team']}\n{int(r['minute'])}''" for _, r in goal_df.iterrows()],
    fontsize=6.5, rotation=45, ha="right"
)
ax.set_ylabel("Pressure Index", fontsize=11)
ax.set_title("Pre-Goal Pressure vs Full-Match Average  |  WC 2026 (All Goals)", fontsize=13, pad=15)
ax.legend(fontsize=9)
ax.grid(axis="y")
ax.set_ylim(0, max(goal_df[["pre5_avg","match_avg"]].max()) * 1.25)

# Annotate uplift
for i, (_, row) in enumerate(goal_df.iterrows()):
    color = ACCENT_COLOR if row["uplift"] > 0 else GOAL_COLOR
    ax.text(i, max(row["pre5_avg"], row["match_avg"]) + 0.3,
            f"{'+' if row['uplift']>=0 else ''}{row['uplift']:.1f}",
            ha="center", fontsize=6, color=color, fontweight="bold")

fig.tight_layout()
save(fig, "pre_goal_pressure_bar.png")

# VISUAL 2 — Scatter: pre-goal pressure vs match average

print("[2] Scatter: pre-goal vs match average...")
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(goal_df["match_avg"], goal_df["pre5_avg"],
           color=GOAL_COLOR, s=80, alpha=0.8, zorder=3, edgecolors="white", linewidths=0.4)

# Diagonal y=x line (pre-goal == match avg, no uplift)
lim = float(goal_df[["match_avg","pre5_avg"]].max().max()) * 1.1
ax.plot([0, lim], [0, lim], color=NEUTRAL_COLOR, linewidth=1.2,
        linestyle="--", label="No uplift (y = x)")

# Best-fit line (needs at least 3 unique x values)
if goal_df["match_avg"].nunique() >= 3:
    m, b, r, p, _ = stats.linregress(goal_df["match_avg"], goal_df["pre5_avg"])
    xfit = np.linspace(0, lim, 100)
    ax.plot(xfit, m*xfit + b, color=ACCENT_COLOR, linewidth=1.8,
            label=f"Trend (r={r:.2f}, p={p:.3f})")
else:
    r, p = float('nan'), float('nan')
    print("  [!] Not enough data for regression — run with all 25 matches")

ax.set_xlabel("Full-Match Avg Pressure", fontsize=11)
ax.set_ylabel("Avg Pressure (5 min before goal)", fontsize=11)
ax.set_title("Do Teams Build Pressure Before Goals?\nScatter of Pre-Goal vs Match Average  |  WC 2026", fontsize=12, pad=12)
ax.legend(fontsize=9)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.grid(True)

# Shade "above average" region
ax.fill_between([0, lim], [0, lim], [lim, lim],
                alpha=0.05, color=ACCENT_COLOR, label="Pre-goal > match avg")

fig.tight_layout()
save(fig, "pressure_vs_match_avg.png")


# VISUAL 3 — Heatmap: pressure around all goal minutes (-10 to +5)

print("[3] Heatmap: pressure window around goals...")
BEFORE, AFTER = 10, 5
heat_rows = []

for _, row in goal_df.iterrows():
    mn, team, gmin = row["match"], row["team"], row["minute"]
    label = f"{team} {int(gmin)}'"
    vals  = {}
    for offset in range(-BEFORE, AFTER+1):
        m = int(gmin) + offset
        if 1 <= m <= 120:
            vals[offset] = get_pressure(mn, team, m)
        else:
            vals[offset] = np.nan
    vals["label"] = label
    heat_rows.append(vals)

heat_df = pd.DataFrame(heat_rows).set_index("label")
heat_df = heat_df[[c for c in heat_df.columns if isinstance(c, int)]]

fig, ax = plt.subplots(figsize=(14, max(6, len(heat_df) * 0.35)))
sns.heatmap(
    heat_df, ax=ax, cmap="YlOrRd",
    linewidths=0.3, linecolor="#21262d",
    cbar_kws={"label": "Pressure Index", "shrink": 0.6},
    annot=False
)
ax.axvline(x=BEFORE, color=GOAL_COLOR, linewidth=2.5, label="Goal minute")
ax.set_xlabel("Minutes relative to goal  (0 = goal minute)", fontsize=11)
ax.set_ylabel("Goal event", fontsize=11)
ax.set_title(f"Pressure Heatmap Around Every Goal  |  {BEFORE} min before → {AFTER} min after\nWC 2026 — All {len(goal_df)} Goals", fontsize=12, pad=12)
ax.legend(loc="upper left", fontsize=9)
xtick_labels = [str(i) for i in range(-BEFORE, AFTER+1)]
ax.set_xticklabels(xtick_labels, fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
fig.tight_layout()
save(fig, "goal_window_heatmap.png")

# VISUAL 4 — Pressure timeline for 4 high-drama matches

print("[4] Pressure timelines for high-drama matches...")

# Pick 4 matches with most goals (most dramatic)
goal_counts = goal_df.groupby("match").size().sort_values(ascending=False)
top_matches = goal_counts.head(4).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, match_name in enumerate(top_matches):
    ax = axes[idx]
    match_press = press[press["match_name"] == match_name]
    match_goals = goal_df[goal_df["match"] == match_name]

    teams = [t for t in match_press["team"].unique() if t != "neutral"]
    colors_map = {teams[0]: PRESS_COLOR, teams[1]: GOAL_COLOR} if len(teams) >= 2 else {}

    for team in teams:
        td = match_press[match_press["team"] == team].sort_values("minute")
        ax.plot(td["minute"], td["pressure_index"],
                label=team, color=colors_map.get(team, NEUTRAL_COLOR),
                linewidth=1.8, alpha=0.9)
        ax.fill_between(td["minute"], td["pressure_index"],
                        alpha=0.08, color=colors_map.get(team, NEUTRAL_COLOR))

    # Mark goals
    for _, g in match_goals.iterrows():
        gcolor = colors_map.get(g["team"], ACCENT_COLOR)
        ax.axvline(x=g["minute"], color=gcolor, linewidth=1.5,
                   linestyle=":", alpha=0.9)
        ax.text(g["minute"] + 0.5,
                ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 5,
                f"⚽ {g['team'][:3].upper()} {int(g['minute'])}'",
                fontsize=6.5, color=gcolor, rotation=90, va="top")

    # Halftime line
    ax.axvline(x=45, color=NEUTRAL_COLOR, linewidth=1, linestyle="--", alpha=0.5)
    ax.text(45.5, 0.5, "HT", fontsize=7, color=NEUTRAL_COLOR, va="bottom")

    ax.set_title(match_name.replace("_", " "), fontsize=10, pad=8)
    ax.set_xlabel("Minute", fontsize=8)
    ax.set_ylabel("Pressure Index", fontsize=8)
    ax.legend(fontsize=7.5)
    ax.grid(True)
    ax.set_xlim(0, 100)

fig.suptitle("Pressure Index Timeline — Top 4 High-Drama Matches  |  WC 2026",
             fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "top_matches_timeline.png")

# VISUAL 5 — Statistical test summary card

print("[5] T-test summary card...")

pre_goal_vals = goal_df["pre5_avg"].values
match_avg_vals = goal_df["match_avg"].values
if len(goal_df) >= 5:
    t_stat, p_val = stats.ttest_rel(pre_goal_vals, match_avg_vals)
else:
    t_stat, p_val = float('nan'), float('nan')
    print("  [!] Not enough goals for t-test — run with all 25 matches")
mean_uplift   = np.mean(pre_goal_vals - match_avg_vals)
pct_above     = (pre_goal_vals > match_avg_vals).mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.axis("off")

result_color = ACCENT_COLOR if p_val < 0.05 else GOAL_COLOR
result_text  = "STATISTICALLY SIGNIFICANT ✓" if p_val < 0.05 else "NOT SIGNIFICANT ✗"

stats_lines = [
    ("Goals analysed",              f"{len(goal_df)}"),
    ("Matches",                     f"{master['match_name'].nunique()}"),
    ("Mean pre-goal pressure",      f"{pre_goal_vals.mean():.3f}"),
    ("Mean full-match pressure",    f"{match_avg_vals.mean():.3f}"),
    ("Mean uplift (pre − avg)",     f"{mean_uplift:+.3f}"),
    ("Goals where pre > avg",       f"{pct_above:.1f}%"),
    ("Paired t-statistic",          f"{t_stat:.4f}"),
    ("p-value",                     f"{p_val:.4f}"),
    ("Result",                      result_text),
]

ax.text(0.5, 0.97,
        "Pressure Index Validation  —  Paired t-test\nDo teams build pressure before scoring?",
        ha="center", va="top", fontsize=13, fontweight="bold",
        transform=ax.transAxes, color="#e6edf3")

for i, (label, val) in enumerate(stats_lines):
    y = 0.80 - i * 0.085
    color = result_color if label == "Result" else "#e6edf3"
    weight = "bold" if label in ("Result", "p-value", "Mean uplift (pre − avg)") else "normal"
    size   = 11 if label == "Result" else 10
    ax.text(0.25, y, label + ":", ha="right", va="center",
            transform=ax.transAxes, fontsize=size,
            color=NEUTRAL_COLOR, fontweight=weight)
    ax.text(0.28, y, val, ha="left", va="center",
            transform=ax.transAxes, fontsize=size,
            color=color, fontweight=weight)

ax.add_patch(mpatches.FancyBboxPatch(
    (0.03, 0.03), 0.94, 0.94,
    boxstyle="round,pad=0.02", linewidth=1.5,
    edgecolor=result_color, facecolor="#161b22",
    transform=ax.transAxes, zorder=0
))

fig.tight_layout()
save(fig, "ttest_summary.png")

# VISUAL 6 — Event weight distribution by type (boxplot)

print("[6] Event weight distribution...")

plot_df = master[master["weight"] > 0].copy()
order   = plot_df.groupby("event_type")["weight"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(
    data=plot_df, x="event_type", y="weight",
    order=order, ax=ax,
    palette="YlOrRd", linewidth=0.8,
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4}
)
ax.set_xlabel("Event Type", fontsize=11)
ax.set_ylabel("Pressure Weight", fontsize=11)
ax.set_title("Pressure Weight Distribution by Event Type  |  WC 2026", fontsize=12, pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
ax.grid(axis="y")
fig.tight_layout()
save(fig, "event_weight_distribution.png")


# VISUAL 7 — Pressure uplift by goal type (the key new finding)

print("[7] Pressure uplift by goal type...")

TYPE_COLORS = {
    "open_play":      "#58a6ff",
    "set_piece":      "#e3b341",
    "counter_attack": "#f78166",
    "penalty":        "#bc8cff",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: Boxplot of uplift by goal type
ax = axes[0]
goal_types  = goal_df["goal_type"].unique()
type_order  = ["open_play", "counter_attack", "set_piece", "penalty"]
type_order  = [t for t in type_order if t in goal_types]

box_data   = [goal_df[goal_df["goal_type"] == t]["uplift"].values for t in type_order]
box_colors = [TYPE_COLORS.get(t, NEUTRAL_COLOR) for t in type_order]

bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                medianprops={"color": "white", "linewidth": 2},
                whiskerprops={"color": NEUTRAL_COLOR},
                capprops={"color": NEUTRAL_COLOR},
                flierprops={"marker": "o", "markersize": 4,
                            "markerfacecolor": NEUTRAL_COLOR, "alpha": 0.5})

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.axhline(0, color="white", linewidth=1.2, linestyle="--", alpha=0.6,
           label="No uplift (0)")
ax.set_xticks(range(1, len(type_order)+1))
ax.set_xticklabels([t.replace("_", "\n") for t in type_order], fontsize=10)
ax.set_ylabel("Pressure Uplift (pre-goal − match avg)", fontsize=10)
ax.set_title("Pressure Uplift Distribution\nby Goal Type", fontsize=11, pad=10)
ax.legend(fontsize=8)
ax.grid(axis="y")

# Annotate n per group
for i, t in enumerate(type_order):
    n = len(goal_df[goal_df["goal_type"] == t])
    ax.text(i+1, ax.get_ylim()[0] * 0.92, f"n={n}",
            ha="center", fontsize=8, color=NEUTRAL_COLOR)

# Right: Mean uplift bar + per-type t-test p-values
ax = axes[1]
type_stats = []
for t in type_order:
    sub = goal_df[goal_df["goal_type"] == t]
    mean_up = sub["uplift"].mean()
    if len(sub) >= 3:
        t_s, p_v = stats.ttest_1samp(sub["uplift"], popmean=0)
    else:
        t_s, p_v = float("nan"), float("nan")
    type_stats.append({
        "type":    t,
        "n":       len(sub),
        "mean_uplift": mean_up,
        "t_stat":  t_s,
        "p_value": p_v,
    })

ts_df = pd.DataFrame(type_stats)

bar_colors = [TYPE_COLORS.get(t, NEUTRAL_COLOR) for t in ts_df["type"]]
bars = ax.bar(range(len(ts_df)), ts_df["mean_uplift"],
              color=bar_colors, alpha=0.8, edgecolor="white", linewidth=0.5)

ax.axhline(0, color="white", linewidth=1.2, linestyle="--", alpha=0.6)
ax.set_xticks(range(len(ts_df)))
ax.set_xticklabels([t.replace("_", "\n") for t in ts_df["type"]], fontsize=10)
ax.set_ylabel("Mean Pressure Uplift", fontsize=10)
ax.set_title("Mean Pressure Uplift by Goal Type\n(with t-test p-values)", fontsize=11, pad=10)
ax.grid(axis="y")

# Annotate mean value and p-value on each bar
for i, row in ts_df.iterrows():
    ypos = row["mean_uplift"] + (0.15 if row["mean_uplift"] >= 0 else -0.4)
    ax.text(i, ypos,
            f"{row['mean_uplift']:+.2f}\np={row['p_value']:.3f}" if not np.isnan(row["p_value"])
            else f"{row['mean_uplift']:+.2f}\n(n<3)",
            ha="center", fontsize=8.5, color="white", fontweight="bold")

fig.suptitle("Does Goal Type Determine Whether Pressure Builds Beforehand?\nWC 2026 — Goal Type Breakdown",
             fontsize=13, y=1.02)
fig.tight_layout()
save(fig, "goal_type_uplift_breakdown.png")

# Print breakdown summary
print("\n  Goal type t-test results:")
print(ts_df[["type","n","mean_uplift","p_value"]].to_string(index=False))


# Summary
print(f"""
{'='*50}
VALIDATION SUMMARY
{'='*50}
Goals analysed     : {len(goal_df)}
Mean pre-goal (5m) : {pre_goal_vals.mean():.3f}
Mean match avg     : {match_avg_vals.mean():.3f}
Mean uplift        : {mean_uplift:+.3f}
Goals where pre>avg: {pct_above:.1f}%
t-statistic        : {t_stat:.4f}
p-value            : {p_val:.4f}
Result             : {'SIGNIFICANT (p < 0.05)' if p_val < 0.05 else 'not significant'}
{'='*50}
All visuals saved to visuals/
""")