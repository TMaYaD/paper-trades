from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .wallet import Wallet


def _max_drawdown(values) -> float:
    peak = values[0]
    mdd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd
    return mdd


@dataclass
class StrategyStats:
    label: str
    final_value: float
    vs_baseline: float
    total_return: float
    alpha: float
    volatility: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    wallet: Wallet | None = field(default=None, repr=False)
    norm: list[float] = field(default_factory=list, repr=False)

    @property
    def trade_count(self) -> int:
        return len(self.wallet.activity) if self.wallet else 0

    @classmethod
    def compute(cls, history: list[float], baseline: list[float],
                label: str,
                wallet: Wallet | None = None) -> "StrategyStats":
        periods_per_year = 365 * 24  # hourly data
        norm = [v / b for v, b in zip(history, baseline)]
        total_ret = (history[-1] / history[0] - 1) * 100
        base_ret = (baseline[-1] / baseline[0] - 1) * 100
        alpha = total_ret - base_ret
        returns = pd.Series(history).pct_change().dropna()
        vol = returns.std() * np.sqrt(periods_per_year) * 100
        sharpe = (
            (returns.mean() / returns.std() * np.sqrt(periods_per_year))
            if returns.std() > 0
            else 0
        )
        mdd = _max_drawdown(history) * 100
        wins = sum(1 for i in range(1, len(norm)) if norm[i] > norm[i - 1])
        win_rate = wins / (len(norm) - 1) * 100 if len(norm) > 1 else 0

        return cls(
            label=label,
            final_value=history[-1],
            vs_baseline=norm[-1],
            total_return=total_ret,
            alpha=alpha,
            volatility=vol,
            sharpe=sharpe,
            max_drawdown=mdd,
            win_rate=win_rate,
            wallet=wallet,
            norm=norm,
        )


def print_results_table(stats_list: list[StrategyStats], baseline_history: list[float]):
    base_ret = (baseline_history[-1] / baseline_history[0] - 1) * 100

    columns = ["50-50 Hold"] + [s.label for s in stats_list]
    col_width = max(15, max(len(c) for c in columns) + 2)

    header = f"{'Metric':<30}"
    for c in columns:
        header += f" {c:>{col_width}}"

    width = 30 + (col_width + 1) * len(columns)
    print("\n" + "=" * width)
    print("                           STRATEGY COMPARISON RESULTS")
    print("=" * width)
    print(header)
    print("-" * width)

    def row(label, baseline_val, strat_vals):
        line = f"{label:<30} {baseline_val:>{col_width}}"
        for v in strat_vals:
            line += f" {v:>{col_width}}"
        print(line)

    row("Final Value (Token A)",
        f"{baseline_history[-1]:.4f}",
        [f"{s.final_value:.4f}" for s in stats_list])
    row("vs Baseline (ratio)",
        "1.0000",
        [f"{s.vs_baseline:.4f}" for s in stats_list])
    row("Total Return (%)",
        f"{base_ret:.2f}%",
        [f"{s.total_return:.2f}%" for s in stats_list])
    row("Alpha vs Hold (%)",
        "0.00%",
        [f"{s.alpha:.2f}%" for s in stats_list])
    row("Ann. Volatility (%)",
        "\u2014",
        [f"{s.volatility:.2f}%" for s in stats_list])
    row("Sharpe Ratio (ann.)",
        "\u2014",
        [f"{s.sharpe:.3f}" for s in stats_list])
    row("Max Drawdown (%)",
        "\u2014",
        [f"{s.max_drawdown:.2f}%" for s in stats_list])
    row("Trade Count",
        "0",
        [str(s.trade_count) for s in stats_list])
    row("Win Rate vs Baseline",
        "\u2014",
        [f"{s.win_rate:.1f}%" for s in stats_list])

    all_vs = [(s.vs_baseline, s.label) for s in stats_list]
    winner = max(all_vs, key=lambda x: x[0])
    print(f"\n  >>> Winner: {winner[1]} ({winner[0]:.4f}x baseline)")
    print("=" * width)
