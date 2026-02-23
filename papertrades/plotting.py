from dataclasses import dataclass

import matplotlib.pyplot as plt

COLORS = ["blue", "green", "orange", "cyan", "magenta"]


@dataclass
class PlotEntry:
    """Holds data needed for plotting a single strategy result."""
    label: str
    dates: list
    norm: list[float]
    interval: str


def _plot_panel(ax, entries, colors, title, extra_overlay=None):
    """Plot a list of PlotEntry on a single axes panel."""
    for i, entry in enumerate(entries):
        ax.plot(
            entry.dates, entry.norm,
            label=entry.label,
            color=colors[i % len(colors)],
            linewidth=2 if "Daily" in entry.label else 1.2,
        )
    if extra_overlay:
        for i, entry in enumerate(extra_overlay):
            ax.plot(
                entry.dates, entry.norm,
                label=entry.label,
                color=colors[i % len(colors)],
                linewidth=2, alpha=0.7,
                marker="o", markersize=3,
            )
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="50-50 Hold (baseline)")
    ax.set_ylabel("Value vs Baseline")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_results(plot_entries: list[PlotEntry]):
    """Plot strategy results."""
    daily = [r for r in plot_entries if r.interval == "daily"]
    hourly = [r for r in plot_entries if r.interval == "hourly"]

    n_panels = bool(daily) + bool(hourly)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    title_parts = [r.label for r in plot_entries]
    fig.suptitle(
        "Strategy Comparison: " + " vs ".join(title_parts),
        fontsize=13, fontweight="bold",
    )

    panel = 0
    if daily:
        _plot_panel(axes[panel], daily, COLORS, "Daily Strategies (normalized to baseline = 1.0)")
        panel += 1

    if hourly:
        _plot_panel(
            axes[panel], hourly, COLORS[2:] + COLORS[:2],
            "All Strategies on Hourly Timescale",
            extra_overlay=daily,
        )
        axes[panel].set_xlabel("Date")

    plt.tight_layout()
    plt.show()
