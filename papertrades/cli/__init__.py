import logging

import click

from ..strategies import REGISTRY
from .backtest import backtest
from .trade import trade


@click.group()
@click.option("-S", "--list-strategies", is_flag=True, is_eager=True,
              expose_value=False, callback=lambda ctx, _param, value: (
                  click.echo("\n".join(sorted(REGISTRY))) or ctx.exit()
              ) if value else None,
              help="List available strategies and exit.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose):
    """papertrades — backtest and paper-trade crypto strategies."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


cli.add_command(backtest)
cli.add_command(trade)


def main():
    cli()
