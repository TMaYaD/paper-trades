import click

from ..engine import BacktestEngine
from ..plotting import PlotEntry, plot_results
from ..stats import StrategyStats, print_results_table
from ..strategies import lookup, REGISTRY


@click.command()
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
@click.option("--swap-fee", type=float, default=0.005, help="Swap fee (default 0.005)")
def backtest(token_a, token_b, start_date, strategies, swap_fee):
    """Run backtests against historical data."""
    print("--- Strategy Backtest ---")
    print(f"Base Token A: {token_a}")
    print(f"Quote Token B: {token_b}")
    print(f"Start Date:   {start_date}")
    print(f"Strategies:   {', '.join(strategies)}")

    engine = BacktestEngine(token_a, token_b, swap_fee)
    hold = REGISTRY["hold"]()

    print("\nRunning strategies...")

    baseline_result = engine.run(hold, start_date)

    all_stats = []
    plot_data = []

    for strategy in lookup(*strategies):
        result = engine.run(strategy, start_date)

        stats = StrategyStats.compute(
            result.value_history, baseline_result.value_history,
            strategy.name, wallet=result.wallet,
        )
        all_stats.append(stats)
        plot_data.append(PlotEntry(
            label=strategy.name,
            dates=result.dates,
            norm=stats.norm,
            activity=result.wallet.activity,
        ))

    print_results_table(all_stats, baseline_result.value_history)
    # plot_results(plot_data)  # TODO: re-enable when non-blocking
