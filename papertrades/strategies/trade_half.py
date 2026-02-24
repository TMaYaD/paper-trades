from datetime import timedelta

from . import Strategy


class TradeHalfStrategy(Strategy):
    name = "trade-half"

    def __init__(self, lookback=timedelta(hours=1)):
        self.lookback = lookback

    def step(self, wallet, history_a, history_b, tick=None) -> float:
        if tick is None:
            return 0

        try:
            prev_a = history_a.price_at(tick - self.lookback)
            prev_b = history_b.price_at(tick - self.lookback)
        except ValueError:
            return 0

        cur_a = history_a.price_at(tick)
        cur_b = history_b.price_at(tick)
        ret_a = (cur_a - prev_a) / prev_a
        ret_b = (cur_b - prev_b) / prev_b

        if ret_a > ret_b:
            return -(wallet.balance(history_a.token) / 2)
        elif ret_b > ret_a:
            return +(wallet.balance(history_b.token) / 2)
        return 0
