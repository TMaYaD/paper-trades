import numpy as np
import pandas as pd

from . import Strategy


class EMAMomentumStrategy(Strategy):
    name = "ema-momentum"

    def __init__(self, fast_span=12, slow_span=26, sensitivity=7.0,
                 max_tilt=0.3, rebalance_band=0.05, zscore_window=50,
                 zscore_limit=2.0, threshold_window=20, threshold_mult=0.5):
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.sensitivity = sensitivity
        self.max_tilt = max_tilt
        self.rebalance_band = rebalance_band
        self.zscore_window = zscore_window
        self.zscore_limit = zscore_limit
        self.threshold_window = threshold_window
        self.threshold_mult = threshold_mult

    def _target_weight_a(self, ratio_series) -> float:
        """Compute target weight for token A from a price-ratio series."""
        fast = ratio_series.ewm(span=self.fast_span, adjust=False).mean()
        slow = ratio_series.ewm(span=self.slow_span, adjust=False).mean()
        raw_signal = (fast - slow) / slow

        std = raw_signal.rolling(self.threshold_window, min_periods=1).std()
        signal = raw_signal.where(raw_signal.abs() >= std * self.threshold_mult, 0.0)

        ratio_sma = ratio_series.rolling(self.zscore_window, min_periods=1).mean()
        ratio_std = ratio_series.rolling(self.zscore_window, min_periods=1).std()
        zscore = ((ratio_series - ratio_sma) / ratio_std.replace(0, np.nan)).fillna(0)

        weight = 0.5 + self.max_tilt * np.tanh(signal * self.sensitivity)
        extreme = zscore.abs() > self.zscore_limit
        weight = weight.where(~extreme, 0.5 + (weight - 0.5) * 0.5)

        return float(weight.iloc[-1])

    def step(self, wallet, history_a, history_b, tick=None) -> float:
        prices_a = pd.Series(dict(history_a.prices(end=tick)))
        prices_b = pd.Series(dict(history_b.prices(end=tick)))
        if len(prices_a) < 2 or len(prices_b) < 2:
            return 0

        ratio = (prices_a / prices_b).dropna()
        if len(ratio) < 2:
            return 0

        target = self._target_weight_a(ratio)
        price_a = history_a.price_at(tick)
        price_b = history_b.price_at(tick)
        bal_a = wallet.balance(history_a.token)
        bal_b = wallet.balance(history_b.token)
        portfolio = bal_a + (bal_b * price_b) / price_a
        if portfolio <= 0:
            return 0

        actual = bal_a / portfolio
        drift = target - actual
        if abs(drift) <= self.rebalance_band:
            return 0

        value_to_move = abs(drift) * portfolio  # in token-A units
        if drift > 0:
            sell_b = (value_to_move * price_a) / price_b
            return min(sell_b, bal_b)
        else:
            return -min(value_to_move, bal_a)
