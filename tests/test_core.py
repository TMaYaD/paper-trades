import numpy as np
import pandas as pd
import pytest

from papertrades.dex import SimulatedDex
from papertrades.wallet import SimulatedWallet
from papertrades.price_history import PriceHistory
from papertrades.engine import BacktestEngine, _portfolio_value
from papertrades.stats import StrategyStats
from papertrades.strategies import REGISTRY


# --- SimulatedWallet ---

class TestSimulatedWallet:
    def test_balances(self):
        w = SimulatedWallet({'A': 10.0, 'B': 5.0})
        assert w.balance('A') == 10.0
        assert w.balance('B') == 5.0

    def test_balances_dict(self):
        w = SimulatedWallet({'A': 10.0, 'B': 5.0})
        assert w.balances == {'A': 10.0, 'B': 5.0}

    def test_missing_token(self):
        w = SimulatedWallet({'A': 10.0})
        assert w.balance('C') == 0.0

    def test_balances_is_copy(self):
        w = SimulatedWallet({'A': 10.0})
        b = w.balances
        b['A'] = 999
        assert w.balance('A') == 10.0

    def test_swap_no_fee(self):
        dex = SimulatedDex({'A': 2.0, 'B': 1.0}, fee=0.0)
        w = SimulatedWallet({'A': 10.0, 'B': 10.0})
        # sell 5 B at price_b=1, buy A at price_a=2 → receive 5*1/2 = 2.5
        received = w.swap('B', 5.0, 'A', dex)
        assert w.balance('B') == pytest.approx(5.0)
        assert w.balance('A') == pytest.approx(12.5)
        assert received == pytest.approx(2.5)

    def test_swap_with_fee(self):
        dex = SimulatedDex({'A': 1.0, 'B': 1.0}, fee=0.01)
        w = SimulatedWallet({'A': 0.0, 'B': 100.0})
        received = w.swap('B', 100.0, 'A', dex)
        assert w.balance('B') == pytest.approx(0.0)
        assert w.balance('A') == pytest.approx(99.0)
        assert received == pytest.approx(99.0)

    def test_balanced_factory(self):
        dex = SimulatedDex({'A': 100.0, 'B': 50.0})
        w = SimulatedWallet.balanced('A', 'B', dex)
        assert w.balance('A') == 1.0
        assert w.balance('B') == pytest.approx(2.0)  # 100/50


# --- PriceHistory ---

def _make_history(prices, freq='h'):
    dates = pd.date_range('2025-01-01', periods=len(prices), freq=freq)
    series = pd.Series(prices, index=dates, dtype=float)
    return PriceHistory.from_series('TOK', series), dates


class TestPriceHistory:
    def test_current_price_at_end(self):
        ph, _ = _make_history([1.0, 2.0, 3.0])
        assert ph.current_price == 3.0

    def test_set_cursor(self):
        ph, _ = _make_history([1.0, 2.0, 3.0])
        ph.set_cursor(1)
        assert ph.current_price == 2.0

    def test_price_at(self):
        ph, dates = _make_history([10.0, 20.0, 30.0])
        assert ph.price_at(dates[1]) == 20.0

    def test_price_at_between_timestamps(self):
        ph, dates = _make_history([10.0, 20.0, 30.0])
        mid = dates[0] + (dates[1] - dates[0]) / 2
        assert ph.price_at(mid) == 10.0

    def test_prices_since(self):
        ph, dates = _make_history([1.0, 2.0, 3.0, 4.0, 5.0])
        series = ph.prices_since(dates[2])
        assert len(series) == 3
        assert list(series.values) == [3.0, 4.0, 5.0]

    def test_all_prices_respects_cursor(self):
        ph, _ = _make_history([1.0, 2.0, 3.0, 4.0])
        ph.set_cursor(2)
        assert list(ph.all_prices().values) == [1.0, 2.0, 3.0]

    def test_append_live_mode(self):
        ph = PriceHistory('TOK', 'solana', None)
        ph.append('2025-06-01 12:00', 100.0)
        ph.append('2025-06-01 13:00', 105.0)
        assert ph.current_price == 105.0
        assert len(ph.all_prices()) == 2

    def test_price_at_no_data_raises(self):
        ph = PriceHistory('TOK', 'solana', None)
        with pytest.raises(ValueError):
            ph.price_at('2025-01-01')


# --- Registry ---

class TestRegistry:
    def test_all_registered(self):
        assert set(REGISTRY) >= {'hold', 'trade-half', 'ema-momentum'}


# --- StrategyStats ---

class TestStrategyStats:
    def test_compute(self):
        history = [1.0, 1.1, 1.2, 1.15, 1.3]
        baseline = [1.0, 1.05, 1.1, 1.08, 1.2]
        stats = StrategyStats.compute(history, baseline, 'test', 365, trade_count=3)
        assert stats.label == 'test'
        assert stats.trade_count == 3
        assert stats.total_return == pytest.approx(30.0)
        assert len(stats.norm) == 5


# --- Integration ---

def _make_bars(n=20, seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2025-01-01', periods=n, freq='h')
    return pd.DataFrame({
        'price_a': 100.0 + np.cumsum(np.random.randn(n) * 2),
        'price_b': 50.0 + np.cumsum(np.random.randn(n) * 1),
    }, index=dates)


class TestBacktestEngine:
    def _run(self, strategy_name, n=20, df_hourly=None):
        df = _make_bars(n)
        engine = BacktestEngine('token_a', 'token_b', swap_fee=0.005)
        result = engine.run(
            REGISTRY[strategy_name](), '2025-01-01', 'hourly',
            df_bars=df, df_hourly=df_hourly if df_hourly is not None else df,
        )
        return result, df

    def test_hold_no_trades(self):
        result, df = self._run('hold')
        assert result.trade_count == 0
        assert len(result.value_history) == len(df)

    def test_trade_half_produces_trades(self):
        result, _ = self._run('trade-half', n=50)
        assert result.trade_count > 0

    def test_trade_half_signal_is_absolute(self):
        df = _make_bars(50)
        strat = REGISTRY['trade-half']()
        p0 = df.iloc[0]
        dex = SimulatedDex({'t_a': p0['price_a'], 't_b': p0['price_b']}, fee=0.005)
        wallet = SimulatedWallet.balanced('t_a', 't_b', dex)
        ha = PriceHistory.from_series('t_a', df['price_a'])
        hb = PriceHistory.from_series('t_b', df['price_b'])
        ha.set_cursor(1)
        hb.set_cursor(1)
        signal = strat.step(wallet, ha, hb, tick=df.index[1])
        if signal > 0:
            assert signal == pytest.approx(wallet.balance('t_b') / 2)
        elif signal < 0:
            assert abs(signal) == pytest.approx(wallet.balance('t_a') / 2)

    def test_ema_momentum_runs(self):
        result, df = self._run('ema-momentum', n=100, df_hourly=_make_bars(100))
        assert len(result.value_history) == len(df)

    def test_history_length_matches_bars(self):
        df = _make_bars(30)
        engine = BacktestEngine('token_a', 'token_b', swap_fee=0.005)
        for name in REGISTRY:
            result = engine.run(
                REGISTRY[name](), '2025-01-01', 'hourly',
                df_bars=df, df_hourly=df,
            )
            assert len(result.value_history) == len(df), f"{name}: length mismatch"

    def test_balances_never_negative(self):
        df = _make_bars(50)
        for name in ['trade-half', 'ema-momentum']:
            strat = REGISTRY[name]()
            p0 = df.iloc[0]
            dex = SimulatedDex(
                {'t_a': p0['price_a'], 't_b': p0['price_b']},
                fee=0.005,
            )
            wallet = SimulatedWallet.balanced('t_a', 't_b', dex)
            ha = PriceHistory.from_series('t_a', df['price_a'])
            hb = PriceHistory.from_series('t_b', df['price_b'])
            for i in range(1, len(df)):
                ha.set_cursor(i)
                hb.set_cursor(i)
                dex.prices['t_a'] = df['price_a'].iloc[i]
                dex.prices['t_b'] = df['price_b'].iloc[i]
                signal = strat.step(wallet, ha, hb, tick=df.index[i])
                if signal != 0:
                    if signal > 0:
                        wallet.swap('t_b', signal, 't_a', dex)
                    else:
                        wallet.swap('t_a', abs(signal), 't_b', dex)
                assert wallet.balance('t_a') >= -1e-12, f"{name}: t_a negative at {i}"
                assert wallet.balance('t_b') >= -1e-12, f"{name}: t_b negative at {i}"
