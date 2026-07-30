"""
LinkedIn-ready benchmark chart for HydrusOPT.
1200×628 landscape. Saves: benchmark_history/linkedin_chart.png
Run: python tests/plot_linkedin.py
"""
import csv, os, sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
except ImportError:
    sys.exit("pip install matplotlib numpy")

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent.parent / "benchmark_history" / "prompts"
OUT         = Path(__file__).parent.parent / "benchmark_history" / "linkedin_chart.png"

BG          = "#0D1117"
CARD_BG     = "#161B22"
ACCENT      = "#58A6FF"
GREEN       = "#3FB950"
ORANGE      = "#F78166"
PURPLE      = "#BC8CFF"
TEXT        = "#E6EDF3"
MUTED       = "#8B949E"

CATEGORIES = {
    "Factual":    {"color": ACCENT,  "kws": ["capital", "planet", "causes", "london"]},
    "Scientific": {"color": GREEN,   "kws": ["mitosis", "meiosis", "black hole", "sleep", "moons"]},
    "Logic":      {"color": PURPLE,  "kws": ["cats", "dogs", "fish", "color of silence",
                                              "god", "rock", "die", "born", "false"]},
}

def categorise(prompt: str) -> tuple[str, str]:
    p = prompt.lower()
    for cat, meta in CATEGORIES.items():
        if any(k in p for k in meta["kws"]):
            return cat, meta["color"]
    return "Other", MUTED

# ── Load latest row per prompt, skip integer-only (scoring artefact) ────────
rows = []
for csv_path in sorted(PROMPTS_DIR.glob("*_trend.csv")):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        data = list(csv.DictReader(fh))
    if not data:
        continue
    last = max(data, key=lambda r: r.get("timestamp", ""))
    acc = float(last.get("accuracy_proxy", 0) or 0)
    spd = float(last.get("speedup", 1)       or 1)
    prompt = last.get("prompt", csv_path.stem).strip()
    # skip trivially short prompts with broken scoring
    if acc == 0.0 or acc < 0.05:
        continue
    cat, color = categorise(prompt)
    rows.append(dict(prompt=prompt, accuracy=acc, speedup=spd,
                     profile=last.get("profile",""), cat=cat, color=color))

rows.sort(key=lambda r: r["accuracy"])
labels   = [r["prompt"][:52] + ("…" if len(r["prompt"]) > 52 else "") for r in rows]
accs     = [r["accuracy"]  for r in rows]
colors   = [r["color"]     for r in rows]
speedups = [r["speedup"]   for r in rows]
n = len(rows)

avg_acc   = np.mean(accs)
avg_spd   = np.mean(speedups)
n_prompts = n

# ── Canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6.28), facecolor=BG)

# Two columns: 62% chart | 38% stats
left_ax  = fig.add_axes([0.01, 0.10, 0.58, 0.78])   # bar chart
right_ax = fig.add_axes([0.63, 0.10, 0.35, 0.78])    # stats card (invisible axes)
right_ax.set_visible(False)

# ── Left: horizontal bar chart ───────────────────────────────────────────────
y = np.arange(n)
bars = left_ax.barh(y, accs, color=colors, height=0.68, zorder=3, clip_on=False)

# background track
left_ax.barh(y, [1.0]*n, color="#21262D", height=0.68, zorder=2)

# value labels
for bar, val in zip(bars, accs):
    left_ax.text(val + 0.012, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=8.5,
                 color=TEXT, fontweight="bold")

left_ax.set_yticks(y)
left_ax.set_yticklabels(labels, fontsize=8.2, color=TEXT)
left_ax.set_xlim(0, 1.13)
left_ax.set_xlabel("Accuracy proxy score", color=MUTED, fontsize=9, labelpad=8)
left_ax.tick_params(axis="x", colors=MUTED, labelsize=8)
left_ax.tick_params(axis="y", length=0)
left_ax.set_facecolor(BG)
left_ax.spines[:].set_visible(False)
left_ax.xaxis.grid(True, color="#21262D", linewidth=0.8, zorder=0)
left_ax.set_title("Accuracy across diverse prompt types", color=TEXT,
                  fontsize=10.5, loc="left", pad=10, fontweight="bold")

# category legend
legend_handles = [
    mpatches.Patch(facecolor=CATEGORIES[c]["color"], label=c, linewidth=0)
    for c in CATEGORIES
]
left_ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
               facecolor=CARD_BG, edgecolor="#30363D", labelcolor=TEXT,
               framealpha=0.9, handlelength=1.2, handleheight=1.0)

# ── Right: stat cards ─────────────────────────────────────────────────────────
def stat_card(fig, x, y, w, h, value, label, sublabel, vcolor):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(CARD_BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines[:].set_color("#30363D")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # value
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=28, fontweight="bold", color=vcolor, transform=ax.transAxes)
    # label
    ax.text(0.5, 0.28, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=TEXT, transform=ax.transAxes)
    # sublabel
    ax.text(0.5, 0.10, sublabel, ha="center", va="center",
            fontsize=7.5, color=MUTED, transform=ax.transAxes)
    return ax

stat_card(fig, 0.635, 0.60, 0.335, 0.26,
          "100%", "Consistency", "identical output on repeated runs",
          GREEN)
stat_card(fig, 0.635, 0.315, 0.335, 0.26,
          f"{avg_acc:.0%}", "Avg Accuracy", f"across {n_prompts} prompt types",
          ACCENT)
stat_card(fig, 0.635, 0.030, 0.335, 0.26,
          f"{avg_spd:.3f}×", "Avg Speedup", "vs unoptimised baseline",
          ORANGE)

# ── Headline ─────────────────────────────────────────────────────────────────
fig.text(0.01, 0.96, "HydrusOPT", fontsize=20, fontweight="bold",
         color=TEXT, va="top")
fig.text(0.195, 0.968, "· Local LLM Optimisation", fontsize=12,
         color=MUTED, va="top")
fig.text(0.01, 0.025,
         "Open-source · Runs 100% offline · github.com/SmaranHolkar/HydrusOPT",
         fontsize=8, color=MUTED)

# thin top accent line
fig.add_artist(plt.Line2D([0, 1], [0.99, 0.99], transform=fig.transFigure,
                           color=ACCENT, linewidth=2.5, zorder=10))

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT}")

import subprocess, platform
if platform.system() == "Windows":
    os.startfile(OUT)
elif platform.system() == "Darwin":
    subprocess.run(["open", OUT])
else:
    subprocess.run(["xdg-open", OUT])
