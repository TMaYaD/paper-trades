import click

from ..engine import LiveEngine
from ..strategies import lookup


@click.command()
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
