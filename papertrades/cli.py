import click

from .engine import BacktestEngine, LiveEngine
from .plotting import PlotEntry, plot_results
from .stats import StrategyStats, print_results_table
from .strategies import REGISTRY


@click.group()
def cli():
    """papertrades — backtest and paper-trade crypto strategies."""
    pass


@cli.command()
@click.option("--token-a",
              default="So11111111111111111111111111111111111111112",
              help="Mint address for Base Token (e.g., SOL).")
@click.option("--token-b",
              default="SKRbvo6Gf7GondiT3BbTfuRDPqLWei4j2Qy2NPGZhW3",
              help="Mint address for Quote Token (e.g., SKR).")
@click.option("--start-date",
              type=click.DateTime(formats=["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]),
              default="2026-01-21 09:00:00",
              help="Start datetime (YYYY-MM-DD HH:MM:SS)")
@click.option("--strategy", "-s", "strategies", multiple=True,
              help="Strategy name(s) to run. Omit to run all.")
@click.option("--interval", "-i",
              type=click.Choice(["hourly", "daily"], case_sensitive=False),
              default=None,
              help="Interval: hourly or daily. Omit to run both.")
@click.option("--swap-fee", type=float, default=0.005, help="Swap fee (default 0.005)")
def backtest(token_a, token_b, start_date, strategies, interval, swap_fee):
    """Run backtests against historical data."""
    if strategies:
        strat_names = list(strategies)
        for name in strat_names:
            if name not in REGISTRY:
                raise click.BadParameter(
                    f"Unknown strategy '{name}'. Available: {', '.join(REGISTRY.keys())}")
    else:
        strat_names = [name for name in REGISTRY if name != "hold"]

    intervals = [interval] if interval else ["daily", "hourly"]

    print("--- Strategy Backtest ---")
    print(f"Base Token A: {token_a}")
    print(f"Quote Token B: {token_b}")
    print(f"Start Date:   {start_date}")
    print(f"Strategies:   {', '.join(strat_names)}")
    print(f"Intervals:    {', '.join(intervals)}")

    engine = BacktestEngine(token_a, token_b, swap_fee)
    hold = REGISTRY["hold"]()

    print("\nRunning strategies...")

    all_stats = []
    plot_data = []

    for iv in intervals:
        periods_per_year = 365 * 24 if iv == "hourly" else 365

        baseline_result = engine.run(hold, start_date, iv)

        for sname in strat_names:
            strategy = REGISTRY[sname]()
            result = engine.run(strategy, start_date, iv)

            label = f"{iv.title()} {sname}"
            stats = StrategyStats.compute(
                result.value_history, baseline_result.value_history,
                label, periods_per_year, result.trade_count,
            )
            all_stats.append(stats)
            plot_data.append(PlotEntry(
                label=label,
                dates=result.dates,
                norm=stats.norm,
                interval=iv,
            ))

    if "daily" in intervals:
        table_baseline = engine.run(hold, start_date, "daily").value_history
    else:
        table_baseline = engine.run(hold, start_date, "hourly").value_history

    print_results_table(all_stats, table_baseline)
    plot_results(plot_data)


@cli.command()
@click.option("--token-a",
              default="So11111111111111111111111111111111111111112",
              help="Mint address for Base Token (e.g., SOL).")
@click.option("--token-b",
              default="SKRbvo6Gf7GondiT3BbTfuRDPqLWei4j2Qy2NPGZhW3",
              help="Mint address for Quote Token (e.g., SKR).")
@click.option("--strategy", "-s", required=True,
              help="Strategy name to trade with.")
@click.option("--interval", "-i",
              type=click.Choice(["hourly", "daily"], case_sensitive=False),
              default="hourly",
              help="Polling interval (default: hourly)")
@click.option("--swap-fee", type=float, default=0.005, help="Swap fee (default 0.005)")
def trade(token_a, token_b, strategy, interval, swap_fee):
    """Run live paper trading — polls prices and simulates trades."""
    if strategy not in REGISTRY:
        raise click.BadParameter(
            f"Unknown strategy '{strategy}'. Available: {', '.join(REGISTRY.keys())}")

    engine = LiveEngine(token_a, token_b, REGISTRY[strategy](), swap_fee, interval)
    engine.run()


def main():
    cli()


if __name__ == "__main__":
    main()
