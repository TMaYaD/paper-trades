from dataclasses import dataclass

import matplotlib.pyplot as plt

from .wallet import Activity

COLORS = ["blue", "green", "orange", "cyan", "magenta"]


@dataclass
class PlotEntry:
    """Holds data needed for plotting a single strategy result."""
    label: str
    dates: list
    norm: list[float]
    activity: list[Activity] | None = None


def _plot_panel(ax, entries, colors):
    """Plot a list of PlotEntry on a single axes panel."""
    for i, entry in enumerate(entries):
        ax.plot(
            entry.dates, entry.norm,
            label=entry.label,
            color=colors[i % len(colors)],
            linewidth=1.2,
        )
    for entry in entries:
        if entry.activity:
            date_to_idx = {d: i for i, d in enumerate(entry.dates)}
            for act in entry.activity:
                idx = date_to_idx.get(act.timestamp)
                if idx is not None:
                    is_buy = act.buy_token < act.sell_token  # buy base token
                    ax.plot(
                        entry.dates[idx], entry.norm[idx],
                        marker=("^" if is_buy else "v"),
                        color=("green" if is_buy else "red"),
                        markersize=8, zorder=5,
                    )
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="50-50 Hold (baseline)")
    ax.set_ylabel("Value vs Baseline")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_results(plot_entries: list[PlotEntry]):
    """Plot strategy results."""
    if not plot_entries:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    title_parts = [r.label for r in plot_entries]
    fig.suptitle(
        "Strategy Comparison: " + " vs ".join(title_parts),
        fontsize=13, fontweight="bold",
    )

    _plot_panel(ax, plot_entries, COLORS)

    plt.tight_layout()
    plt.show()
