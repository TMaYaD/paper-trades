import numpy as np
import pandas as pd
import pytest

from papertrades.dex import SimulatedDex
from papertrades.wallet import Activity, SimulatedWallet
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

    def test_activity_empty_initially(self):
        w = SimulatedWallet({'A': 10.0})
        assert w.activity == []

    def test_set_time_and_swap_records_activity(self):
        dex = SimulatedDex({'A': 2.0, 'B': 1.0}, fee=0.0)
        w = SimulatedWallet({'A': 10.0, 'B': 10.0})
        w.set_time('2025-01-01T00:00')
        received = w.swap('B', 5.0, 'A', dex)
        assert len(w.activity) == 1
        act = w.activity[0]
        assert act == Activity('2025-01-01T00:00', 'B', 5.0, 'A', received)

    def test_activity_is_copy(self):
        dex = SimulatedDex({'A': 1.0, 'B': 1.0}, fee=0.0)
        w = SimulatedWallet({'A': 10.0, 'B': 10.0})
        w.swap('A', 1.0, 'B', dex)
        acts = w.activity
        acts.clear()
        assert len(w.activity) == 1

    def test_swap_without_set_time_records_none_timestamp(self):
        dex = SimulatedDex({'A': 1.0, 'B': 1.0}, fee=0.0)
        w = SimulatedWallet({'A': 10.0, 'B': 10.0})
        w.swap('A', 1.0, 'B', dex)
        assert w.activity[0].timestamp is None

    def test_swap_activity_populates_on_existing_swap_tests(self):
        """Existing swap path: verify activity is populated as side-effect."""
        dex = SimulatedDex({'A': 1.0, 'B': 1.0}, fee=0.01)
        w = SimulatedWallet({'A': 0.0, 'B': 100.0})
        w.set_time(42)
        received = w.swap('B', 100.0, 'A', dex)
        assert len(w.activity) == 1
        act = w.activity[0]
        assert act.timestamp == 42
        assert act.sell_token == 'B'
        assert act.sell_amount == pytest.approx(100.0)
        assert act.buy_token == 'A'
        assert act.received == pytest.approx(received)


# --- Stub client ---

class _StubClient:
    """Serves canned OHLCV data — no network, no disk."""

    def __init__(self, token_data):
        self._data = token_data

    def get_top_pools(self, network, token):
        return [{"attributes": {"address": f"pool_{token}"}}]

    def get_ohlcv(self, network, pool, timeframe, token,
                  limit=1000, before_timestamp=None):
        return self._data.get(token, [])


# --- PriceHistory ---

def _make_history(prices, freq='h'):
    dates = pd.date_range('2025-01-01', periods=len(prices), freq=freq)
    ohlcv = [[int(d.timestamp()), p] for d, p in zip(dates, prices)]
    client = _StubClient({'TOK': ohlcv})
    return PriceHistory('TOK', 'solana', client, start_date='2025-01-01'), dates


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


def _stub_client_from_bars(df):
    token_data = {}
    for col, token in [('price_a', 'token_a'), ('price_b', 'token_b')]:
        token_data[token] = [
            [int(ts.timestamp()), price]
            for ts, price in zip(df.index, df[col])
        ]
    return _StubClient(token_data)


class TestBacktestEngine:
    def _run(self, strategy_name, n=20):
        df = _make_bars(n)
        engine = BacktestEngine('token_a', 'token_b', swap_fee=0.005,
                                client=_stub_client_from_bars(df))
        result = engine.run(REGISTRY[strategy_name](), '2025-01-01', 'hourly')
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
        client = _StubClient({
            't_a': [[int(ts.timestamp()), p] for ts, p in zip(df.index, df['price_a'])],
            't_b': [[int(ts.timestamp()), p] for ts, p in zip(df.index, df['price_b'])],
        })
        strat = REGISTRY['trade-half']()
        p0 = df.iloc[0]
        dex = SimulatedDex({'t_a': p0['price_a'], 't_b': p0['price_b']}, fee=0.005)
        wallet = SimulatedWallet.balanced('t_a', 't_b', dex)
        ha = PriceHistory('t_a', 'solana', client, start_date='2025-01-01')
        hb = PriceHistory('t_b', 'solana', client, start_date='2025-01-01')
        ha.set_cursor(1)
        hb.set_cursor(1)
        signal = strat.step(wallet, ha, hb, tick=df.index[1])
        if signal > 0:
            assert signal == pytest.approx(wallet.balance('t_b') / 2)
        elif signal < 0:
            assert abs(signal) == pytest.approx(wallet.balance('t_a') / 2)

    def test_ema_momentum_runs(self):
        result, df = self._run('ema-momentum', n=100)
        assert len(result.value_history) == len(df)

    def test_history_length_matches_bars(self):
        df = _make_bars(30)
        engine = BacktestEngine('token_a', 'token_b', swap_fee=0.005,
                                client=_stub_client_from_bars(df))
        for name in REGISTRY:
            result = engine.run(REGISTRY[name](), '2025-01-01', 'hourly')
            assert len(result.value_history) == len(df), f"{name}: length mismatch"

    def test_balances_never_negative(self):
        df = _make_bars(50)
        client = _StubClient({
            't_a': [[int(ts.timestamp()), p] for ts, p in zip(df.index, df['price_a'])],
            't_b': [[int(ts.timestamp()), p] for ts, p in zip(df.index, df['price_b'])],
        })
        for name in ['trade-half', 'ema-momentum']:
            strat = REGISTRY[name]()
            p0 = df.iloc[0]
            dex = SimulatedDex(
                {'t_a': p0['price_a'], 't_b': p0['price_b']},
                fee=0.005,
            )
            wallet = SimulatedWallet.balanced('t_a', 't_b', dex)
            ha = PriceHistory('t_a', 'solana', client, start_date='2025-01-01')
            hb = PriceHistory('t_b', 'solana', client, start_date='2025-01-01')
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
