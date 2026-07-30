"""
Plot latest benchmark metrics per prompt, grouped by profile (fast / safe / eval).
Saves: benchmark_history/latest_metrics.png
Run from repo root: python tests/plot_latest_metrics.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    sys.exit("matplotlib / numpy not found. Run: pip install matplotlib numpy")

PROMPTS_DIR = Path(__file__).parent.parent / "benchmark_history" / "prompts"
OUT_FILE = Path(__file__).parent.parent / "benchmark_history" / "latest_metrics.png"

PROFILES = ["fast", "safe", "eval"]
PROFILE_COLORS = {"fast": "#4CAF50", "safe": "#2196F3", "eval": "#FF9800"}

METRICS = {
    "speedup":        "Speedup (×)",
    "accuracy_proxy": "Accuracy proxy (0–1)",
    "consistent":     "Consistency (0–1)",
}

# ── 1. Load latest row per (prompt_slug, profile) ──────────────────────────

from datetime import datetime, timezone, timedelta

latest: dict[tuple, dict] = {}  # (prompt_slug, profile) → row
latest_ts: dict[tuple, str] = {}  # (slug, profile) → timestamp string

for csv_path in sorted(PROMPTS_DIR.glob("*_trend.csv")):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        continue
    by_profile: dict[str, list] = defaultdict(list)
    for row in rows:
        by_profile[row.get("profile", "").strip()].append(row)
    for profile, prows in by_profile.items():
        if profile not in PROFILES:
            continue
        latest_row = max(prows, key=lambda r: r.get("timestamp", ""))
        slug = csv_path.stem.replace("_trend", "")
        latest[(slug, profile)] = latest_row
        latest_ts[(slug, profile)] = latest_row.get("timestamp", "")

if not latest:
    sys.exit("No data found in " + str(PROMPTS_DIR))

# ── 2. Filter to latest session only ────────────────────────────────────────
# Find each slug's most recent timestamp, sort descending, detect session
# boundaries by gaps > 4 h. Merge from the top until we have >= 5 prompts.

from datetime import datetime, timezone, timedelta

def parse_ts(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)

SESSION_GAP = timedelta(hours=1)
MIN_PROMPTS  = 5

# Latest timestamp per slug (across all profiles)
slug_max_ts: dict[str, datetime] = {}
for (slug, profile), ts_str in latest_ts.items():
    t = parse_ts(ts_str)
    if t > slug_max_ts.get(slug, datetime.min.replace(tzinfo=timezone.utc)):
        slug_max_ts[slug] = t

sorted_slugs = sorted(slug_max_ts, key=lambda s: slug_max_ts[s], reverse=True)

# Walk through sorted slugs, split on gaps > SESSION_GAP, accumulate sessions
sessions: list[list[str]] = []
current: list[str] = []
for i, slug in enumerate(sorted_slugs):
    current.append(slug)
    if i + 1 < len(sorted_slugs):
        gap = slug_max_ts[slug] - slug_max_ts[sorted_slugs[i + 1]]
        if gap > SESSION_GAP:
            sessions.append(current)
            current = []
if current:
    sessions.append(current)

# Merge latest sessions until we reach MIN_PROMPTS
session_slugs: set[str] = set()
for sess in sessions:
    session_slugs.update(sess)
    if len(session_slugs) >= MIN_PROMPTS:
        break

max_ts = max(slug_max_ts[s] for s in session_slugs)

# ── 3. Collect prompts that have at least one profile entry ─────────────────

all_slugs = sorted(session_slugs, key=lambda s: slug_max_ts[s])
n_prompts = len(all_slugs)
print(f"Latest session: {len(sessions)} session(s) detected, merged top {len([s for s in sessions if any(sl in session_slugs for sl in s)])} → {n_prompts} prompts")
print(f"Most recent entry: {max_ts.strftime('%Y-%m-%d %H:%M')} UTC")
slug_labels = [s.replace("_", " ")[:38] for s in all_slugs]

# ── 4. Build metric arrays  [n_metrics × n_profiles × n_prompts] ─────────────

def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

data = {}
for metric in METRICS:
    data[metric] = {
        profile: [
            safe_float(latest.get((slug, profile), {}).get(metric))
            for slug in all_slugs
        ]
        for profile in PROFILES
    }

# ── 5. Plot ──────────────────────────────────────────────────────────────────

n_metrics = len(METRICS)
fig, axes = plt.subplots(
    n_metrics, 1,
    figsize=(max(14, n_prompts * 0.9), 4.5 * n_metrics),
    facecolor="#1a1a2e",
)
fig.suptitle(
    f"HydrusOPT — Latest session metrics  ({max_ts.strftime('%Y-%m-%d')})",
    color="white", fontsize=15, fontweight="bold", y=1.01,
)

x = np.arange(n_prompts)
bar_width = 0.26
offsets = [-bar_width, 0, bar_width]

for ax, (metric, ylabel) in zip(axes, METRICS.items()):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#334155")
    ax.yaxis.label.set_color("white")
    ax.set_ylabel(ylabel, fontsize=9)

    for i, profile in enumerate(PROFILES):
        vals = data[metric][profile]
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=bar_width,
            color=PROFILE_COLORS[profile],
            alpha=0.85,
            label=profile,
            zorder=3,
        )
        # value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="white", rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(slug_labels, rotation=40, ha="right", fontsize=7.5, color="#cbd5e1")
    ax.grid(axis="y", color="#334155", linewidth=0.6, zorder=0)

    legend_patches = [
        mpatches.Patch(color=PROFILE_COLORS[p], label=p) for p in PROFILES
    ]
    ax.legend(handles=legend_patches, fontsize=8, facecolor="#1a1a2e",
              labelcolor="white", framealpha=0.7)

plt.tight_layout()
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT_FILE}")

# Open in default viewer
import subprocess, platform
if platform.system() == "Windows":
    os.startfile(OUT_FILE)
elif platform.system() == "Darwin":
    subprocess.run(["open", OUT_FILE])
else:
    subprocess.run(["xdg-open", OUT_FILE])
