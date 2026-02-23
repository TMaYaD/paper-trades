from . import Strategy


class HoldStrategy(Strategy):
    name = "hold"

    def step(self, wallet, history_a, history_b, tick=None) -> float:
        return 0
