"""Create an immediate, model-independent UPT results summary figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"

plays = pd.read_parquet(TABLES / "play_metrics_base.parquet")
sensitivity = pd.read_csv(TABLES / "sensitivity.csv")
rng = np.random.default_rng(2026)
games = plays.gameId.unique()


def summarize(frame: pd.DataFrame) -> np.ndarray:
    pressure = frame.pressure.eq(1)
    any_upt = frame.UPT.gt(0)
    return np.array([
        frame.loc[~pressure, "UPT"].mean(),
        frame.loc[pressure, "UPT"].mean(),
        frame.loc[~any_upt, "pressure"].mean(),
        frame.loc[any_upt, "pressure"].mean(),
    ])


point = summarize(plays)
# Fast cluster bootstrap: resample game-level sufficient statistics, not rows.
cluster = []
for _, game in plays.groupby("gameId", sort=False):
    pressure = game.pressure.eq(1)
    any_upt = game.UPT.gt(0)
    cluster.append([
        game.loc[~pressure, "UPT"].sum(), (~pressure).sum(),
        game.loc[pressure, "UPT"].sum(), pressure.sum(),
        game.loc[~any_upt, "pressure"].sum(), (~any_upt).sum(),
        game.loc[any_upt, "pressure"].sum(), any_upt.sum(),
    ])
cluster = np.asarray(cluster, dtype=float)
draws = rng.integers(0, len(cluster), size=(500, len(cluster)))
aggregated = cluster[draws].sum(axis=1)
boot = np.column_stack([
    aggregated[:, 0] / aggregated[:, 1],
    aggregated[:, 2] / aggregated[:, 3],
    aggregated[:, 4] / aggregated[:, 5],
    aggregated[:, 6] / aggregated[:, 7],
])
low, high = np.quantile(boot, [0.025, 0.975], axis=0)

fig = plt.figure(figsize=(15, 9), facecolor="white")
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.36, wspace=0.25)

# Mean UPT by outcome.
ax = fig.add_subplot(grid[0, 0])
x = np.arange(2)
ax.bar(x, point[:2], color=[BLUE, ORANGE], width=0.62)
ax.errorbar(x, point[:2], yerr=[point[:2] - low[:2], high[:2] - point[:2]],
            fmt="none", ecolor="black", capsize=6, linewidth=1.5)
ax.set_xticks(x, ["No pressure", "Pressure"])
ax.set_ylabel("Mean Unseen Pressure Time (seconds)")
ax.set_title("A. Pressured plays contain more unseen-pressure time", loc="left", fontweight="bold")
for i, value in enumerate(point[:2]):
    ax.text(i, value + 0.055, f"{value:.3f} s", ha="center", fontweight="bold")
ax.text(0.5, 0.07, "Difference: +0.424 s\n95% CI: +0.393 to +0.457",
        transform=ax.transAxes, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=.45", facecolor="#FFF4E6", edgecolor=ORANGE))
ax.spines[["top", "right"]].set_visible(False)

# Pressure rate by any UPT.
ax = fig.add_subplot(grid[0, 1])
rates = point[2:] * 100
rate_low, rate_high = low[2:] * 100, high[2:] * 100
ax.bar(x, rates, color=[GREEN, PURPLE], width=0.62)
ax.errorbar(x, rates, yerr=[rates - rate_low, rate_high - rates],
            fmt="none", ecolor="black", capsize=6, linewidth=1.5)
ax.set_xticks(x, ["Zero UPT", "Any UPT"])
ax.set_ylabel("Plays ending in pressure (%)")
ax.set_title("B. Pressure is about twice as common with any UPT", loc="left", fontweight="bold")
for i, value in enumerate(rates):
    ax.text(i, value + 1.8, f"{value:.1f}%", ha="center", fontweight="bold")
ax.text(0.5, 0.07, "Difference: +19.9 percentage points\n95% CI: +16.9 to +22.7",
        transform=ax.transAxes, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=.45", facecolor="#F5ECF4", edgecolor=PURPLE))
ax.set_ylim(0, max(rate_high) + 10)
ax.spines[["top", "right"]].set_visible(False)

# Full 36-definition sensitivity heatmap.
ax = fig.add_subplot(grid[1, :])
sensitivity["definition"] = sensitivity.apply(
    lambda row: f"{row['distance']:.0f} yd / {row['closing']:g} yd/s", axis=1
)
pivot = sensitivity.pivot(index="cone", columns="definition", values="difference")
ordered_columns = [
    f"{distance} yd / {closing:g} yd/s"
    for distance in [5, 7, 9]
    for closing in [1.0, 1.5, 2.5]
]
pivot = pivot.reindex(index=[45, 60, 75, 90], columns=ordered_columns)
image = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0)
ax.set_xticks(np.arange(len(pivot.columns)), [label.replace(" / ", "\n") for label in pivot.columns])
ax.set_yticks(np.arange(len(pivot.index)), [f"{v}°" for v in pivot.index])
ax.set_xlabel("Threat distance / minimum closing speed")
ax.set_ylabel("Body-facing cone half-angle")
ax.set_title("C. UPT difference stays positive under all 36 reasonable definitions", loc="left", fontweight="bold")
for row in range(pivot.shape[0]):
    for col in range(pivot.shape[1]):
        value = pivot.iloc[row, col]
        ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                color="white" if value > 0.44 else "black", fontsize=8)
colorbar = fig.colorbar(image, ax=ax, pad=0.015)
colorbar.set_label("Pressure minus no-pressure mean UPT (seconds)")

fig.suptitle("Unseen Pressure Time (UPT): full 2021 Weeks 1–8 sample",
             fontsize=18, fontweight="bold", y=0.98)
fig.text(0.5, 0.935,
         "8,557 passing plays · 122 games · pressure = PFF hurry, hit, or sack · whiskers = 95% game-bootstrap intervals",
         ha="center", fontsize=11)
fig.text(0.5, 0.015,
         "Takeaway: unseen pass-rush exposure is consistently associated with pressure, but this observational comparison does not prove causation.",
         ha="center", fontsize=11, style="italic")
fig.subplots_adjust(top=0.88, bottom=0.16, left=0.08, right=0.94)

output = FIGURES / "upt_results_summary.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
plt.close(fig)
print(output)
