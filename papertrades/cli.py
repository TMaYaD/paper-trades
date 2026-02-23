import click

from .engine import BacktestEngine, LiveEngine
from .plotting import PlotEntry, plot_results
from .stats import StrategyStats, print_results_table
from .strategies import REGISTRY, lookup


@click.group()
@click.option("-S", "--list-strategies", is_flag=True, is_eager=True,
              expose_value=False, callback=lambda ctx, _param, value: (
                  click.echo("\n".join(sorted(REGISTRY))) or ctx.exit()
              ) if value else None,
              help="List available strategies and exit.")
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
@click.option("--swap-fee", type=float, default=0.005, help="Swap fee (default 0.005)")
def trade(token_a, token_b, strategy, swap_fee):
    """Run live paper trading — polls prices and simulates trades."""
    engine = LiveEngine(token_a, token_b, lookup(strategy)[0], swap_fee)
    engine.run()


def main():
    cli()


if __name__ == "__main__":
    main()
