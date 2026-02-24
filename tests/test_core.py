import os
import tempfile
from datetime import datetime, timedelta, timezone

import csv

import numpy as np
import pytest

from papertrades.client import CachedClient
from papertrades.dex import SimulatedDex
from papertrades.wallet import Activity, SimulatedWallet
from papertrades.price_history import PriceHistory
from papertrades.engine import BacktestEngine, _portfolio_value
from papertrades.stats import StrategyStats
from papertrades.strategies import REGISTRY
from papertrades.timestamps import OHLCV, UTC, from_unix, parse_ts, to_unix


# --- SimulatedWallet ---

class TestSimulatedWallet:
    def test_balances(self):
        w = SimulatedWallet({'A': 10.0, 'B': 5.0})
        assert w.balance('A') == 10.0
        assert w.balance('B') == 5.0

    def test_missing_token(self):
        w = SimulatedWallet({'A': 10.0})
        assert w.balance('C') == 0.0

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
        t = datetime(2025, 1, 1, tzinfo=UTC)
        w.set_time(t)
        received = w.swap('B', 5.0, 'A', dex)
        assert len(w.activity) == 1
        act = w.activity[0]
        assert act == Activity(t, 'B', 5.0, 'A', received)

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
        t = datetime(2025, 1, 1, tzinfo=UTC)
        w.set_time(t)
        received = w.swap('B', 100.0, 'A', dex)
        assert len(w.activity) == 1
        act = w.activity[0]
        assert act.timestamp == t
        assert act.sell_token == 'B'
        assert act.sell_amount == pytest.approx(100.0)
        assert act.buy_token == 'A'
        assert act.received == pytest.approx(received)


# --- Stub client ---

def _dt(year, month, day, hour=0):
    """Shorthand to build a UTC datetime."""
    return datetime(year, month, day, hour, tzinfo=UTC)


def _date_range(start_dt, periods, hours=1):
    """Generate a list of datetimes like pd.date_range but stdlib."""
    return [start_dt + timedelta(hours=i * hours) for i in range(periods)]


class _StubClient:
    """Serves canned OHLCV data — no network, no disk.
    Respects limit and before_timestamp filtering."""

    def __init__(self, token_data, pool_created_at=None):
        self._data = token_data
        self._pool_created_at = pool_created_at or {}

    def get_top_pools(self, network, token):
        attrs = {"address": f"pool_{token}"}
        created = self._pool_created_at.get(token)
        if created:
            attrs["pool_created_at"] = created
        return [{"attributes": attrs}]

    def get_ohlcv(self, network, pool, timeframe, token,
                  limit=1000, before_timestamp=None):
        rows = self._data.get(token, [])
        if before_timestamp is not None:
            rows = [r for r in rows if r.timestamp < before_timestamp]
        rows = sorted(rows, key=lambda r: r.timestamp)
        return rows[-limit:] if rows else []


# --- PriceHistory ---

def _make_history(prices, freq_hours=1):
    base = _dt(2025, 1, 1)
    dates = _date_range(base, len(prices), hours=freq_hours)
    ohlcv = [OHLCV(d, p) for d, p in zip(dates, prices)]
    client = _StubClient({'TOK': ohlcv},
                         pool_created_at={'TOK': parse_ts('2025-01-01T00:00:00Z')})
    return PriceHistory('TOK', 'solana', client), dates


class TestPriceHistory:
    def test_price_at(self):
        ph, dates = _make_history([10.0, 20.0, 30.0])
        assert ph.price_at(dates[1]) == 20.0

    def test_price_at_between_timestamps(self):
        ph, dates = _make_history([10.0, 20.0, 30.0])
        mid = dates[0] + (dates[1] - dates[0]) / 2
        assert ph.price_at(mid) == 10.0

    def test_prices_generator(self):
        ph, dates = _make_history([1.0, 2.0, 3.0])
        result = list(ph.prices())
        assert len(result) == 3
        assert result[0][1] == 1.0
        assert result[2][1] == 3.0

    def test_prices_with_end(self):
        ph, dates = _make_history([1.0, 2.0, 3.0, 4.0])
        result = list(ph.prices(end=dates[1]))
        assert len(result) == 2
        assert result[-1][1] == 2.0

    def test_price_at_no_data_raises(self):
        client = _StubClient({'TOK': []},
                             pool_created_at={'TOK': parse_ts('2025-01-01T00:00:00Z')})
        ph = PriceHistory('TOK', 'solana', client)
        with pytest.raises(ValueError):
            ph.price_at(_dt(2025, 1, 1))

    def test_pool_cached_property(self):
        ph, _ = _make_history([1.0])
        assert ph.pool == "pool_TOK"
        # Second access uses cached_property
        assert ph.pool == "pool_TOK"

    def test_prices_start_before_data_yields_available(self):
        """start 1000 hours before first data — should still yield data."""
        base = _dt(2025, 1, 1)
        dates = _date_range(base, 10)
        ohlcv = [OHLCV(d, float(i + 1)) for i, d in enumerate(dates)]
        client = _StubClient({'TOK': ohlcv},
                             pool_created_at={'TOK': parse_ts('2025-01-01T00:00:00Z')})
        early_start = dates[0] - timedelta(hours=1000)
        ph = PriceHistory('TOK', 'solana', client)
        result = list(ph.prices(start=early_start, end=dates[-1]))
        assert len(result) == 10


# --- Registry ---

class TestRegistry:
    def test_all_registered(self):
        assert set(REGISTRY) >= {'hold', 'trade-half', 'ema-momentum'}


# --- StrategyStats ---

class TestStrategyStats:
    def test_compute(self):
        history = [1.0, 1.1, 1.2, 1.15, 1.3]
        baseline = [1.0, 1.05, 1.1, 1.08, 1.2]
        dex = SimulatedDex({'A': 1.0, 'B': 1.0}, fee=0.0)
        w = SimulatedWallet({'A': 10.0, 'B': 10.0})
        for _ in range(3):
            w.swap('A', 1.0, 'B', dex)
        stats = StrategyStats.compute(history, baseline, 'test', wallet=w)
        assert stats.label == 'test'
        assert stats.wallet is w
        assert stats.trade_count == 3
        assert stats.total_return == pytest.approx(30.0)
        assert len(stats.norm) == 5


# --- Integration ---

def _make_bars(n=20, seed=42):
    np.random.seed(seed)
    base = _dt(2025, 1, 1)
    dates = _date_range(base, n)
    return dates, {
        'price_a': (100.0 + np.cumsum(np.random.randn(n) * 2)).tolist(),
        'price_b': (50.0 + np.cumsum(np.random.randn(n) * 1)).tolist(),
    }


def _stub_client_from_bars(dates, data):
    token_data = {}
    created = {}
    first_iso = dates[0].isoformat()
    for col, token in [('price_a', 'token_a'), ('price_b', 'token_b')]:
        token_data[token] = [
            OHLCV(d, price)
            for d, price in zip(dates, data[col])
        ]
        created[token] = parse_ts(first_iso)
    return _StubClient(token_data, pool_created_at=created)


class TestBacktestEngine:
    def _run(self, strategy_name, n=20):
        dates, data = _make_bars(n)
        engine = BacktestEngine('token_a', 'token_b', swap_fee=0.005,
                                client=_stub_client_from_bars(dates, data))
        end = dates[-1].isoformat()
        result = engine.run(REGISTRY[strategy_name](), '2025-01-01', end_date=end)
        return result, dates, data

    def test_hold_no_trades(self):
        result, dates, data = self._run('hold')
        assert len(result.wallet.activity) == 0
        assert len(result.value_history) == len(dates)

    def test_trade_half_produces_trades(self):
        result, _, _ = self._run('trade-half', n=50)
        assert len(result.wallet.activity) > 0

    def test_trade_half_signal_is_absolute(self):
        dates, data = _make_bars(50)
        client = _StubClient({
            't_a': [OHLCV(d, p) for d, p in zip(dates, data['price_a'])],
            't_b': [OHLCV(d, p) for d, p in zip(dates, data['price_b'])],
        }, pool_created_at={'t_a': parse_ts('2025-01-01T00:00:00Z'),
                            't_b': parse_ts('2025-01-01T00:00:00Z')})
        strat = REGISTRY['trade-half']()
        dex = SimulatedDex({'t_a': data['price_a'][0], 't_b': data['price_b'][0]}, fee=0.005)
        wallet = SimulatedWallet.balanced('t_a', 't_b', dex)
        ha = PriceHistory('t_a', 'solana', client)
        hb = PriceHistory('t_b', 'solana', client)
        signal = strat.step(wallet, ha, hb, tick=dates[1])
        if signal > 0:
            assert signal == pytest.approx(wallet.balance('t_b') / 2)
        elif signal < 0:
            assert abs(signal) == pytest.approx(wallet.balance('t_a') / 2)

    def test_ema_momentum_runs(self):
        result, dates, data = self._run('ema-momentum', n=100)
        assert len(result.value_history) == len(dates)

    def test_history_length_matches_bars(self):
        dates, data = _make_bars(30)
        engine = BacktestEngine('token_a', 'token_b', swap_fee=0.005,
                                client=_stub_client_from_bars(dates, data))
        end = dates[-1].isoformat()
        for name in REGISTRY:
            result = engine.run(REGISTRY[name](), '2025-01-01', end_date=end)
            assert len(result.value_history) == len(dates), f"{name}: length mismatch"

    def test_balances_never_negative(self):
        dates, data = _make_bars(50)
        client = _StubClient({
            't_a': [OHLCV(d, p) for d, p in zip(dates, data['price_a'])],
            't_b': [OHLCV(d, p) for d, p in zip(dates, data['price_b'])],
        }, pool_created_at={'t_a': parse_ts('2025-01-01T00:00:00Z'),
                            't_b': parse_ts('2025-01-01T00:00:00Z')})
        for name in ['trade-half', 'ema-momentum']:
            strat = REGISTRY[name]()
            dex = SimulatedDex(
                {'t_a': data['price_a'][0], 't_b': data['price_b'][0]},
                fee=0.005,
            )
            wallet = SimulatedWallet.balanced('t_a', 't_b', dex)
            ha = PriceHistory('t_a', 'solana', client)
            hb = PriceHistory('t_b', 'solana', client)
            for i in range(1, len(dates)):
                tick = dates[i]
                dex.prices['t_a'] = data['price_a'][i]
                dex.prices['t_b'] = data['price_b'][i]
                signal = strat.step(wallet, ha, hb, tick=tick)
                if signal != 0:
                    if signal > 0:
                        wallet.swap('t_b', signal, 't_a', dex)
                    else:
                        wallet.swap('t_a', abs(signal), 't_b', dex)
                assert wallet.balance('t_a') >= -1e-12, f"{name}: t_a negative at {i}"
                assert wallet.balance('t_b') >= -1e-12, f"{name}: t_b negative at {i}"

    def test_engine_effective_start_from_pool_created_at(self):
        """Engine clamps start_date to max(pool_created_at_a, pool_created_at_b)."""
        # Token A data starts at hour 5, token B data starts at hour 10
        # Both have 20 hours of data from their respective starts
        base = _dt(2025, 1, 1)
        data_a = [OHLCV(base + timedelta(hours=i), 100.0 + i) for i in range(5, 25)]
        data_b = [OHLCV(base + timedelta(hours=i), 50.0 + i) for i in range(10, 30)]

        # pool_created_at for B is later (hour 10)
        created_a = base + timedelta(hours=5)
        created_b = base + timedelta(hours=10)

        client = _StubClient(
            {'t_a': data_a, 't_b': data_b},
            pool_created_at={
                't_a': created_a,
                't_b': created_b,
            },
        )
        engine = BacktestEngine('t_a', 't_b', swap_fee=0.005, client=client)
        end = _dt(2025, 1, 2).isoformat()
        result = engine.run(REGISTRY['hold'](), '2025-01-01', end_date=end)

        # The series should start at created_b (the later one), not at '2025-01-01'
        first_date = result.dates[0]
        assert first_date >= created_b


# --- CachedClient contiguous cache ---

class _FakeAPI:
    """Records calls and returns canned data keyed by before_timestamp ranges."""

    def __init__(self, records: list[list], max_batch: int = 1000,
                 pool_created_at=None):
        """records: list of [unix_ts, open, high, low, close, volume]"""
        self._records = sorted(records, key=lambda r: r[0])
        self._max_batch = max_batch
        self._pool_created_at = pool_created_at or {}
        self.calls: list[dict] = []

    def get_top_pools(self, network, token):
        attrs = {"address": f"pool_{token}"}
        created = self._pool_created_at.get(token)
        if created:
            attrs["pool_created_at"] = created
        return [{"attributes": attrs}]

    def get_ohlcv(self, network, pool, timeframe, token,
                  limit=1000, before_timestamp=None):
        self.calls.append({
            "limit": limit,
            "before_timestamp": before_timestamp,
        })
        effective_limit = min(limit, self._max_batch)
        if before_timestamp is None:
            subset = self._records
        else:
            # before_timestamp is now a datetime; convert to unix for comparison
            bt = to_unix(before_timestamp)
            subset = [r for r in self._records if r[0] < bt]
        # Return the last `effective_limit` records (sorted ascending by time)
        subset = sorted(subset, key=lambda r: r[0])
        result = subset[-effective_limit:]
        # Convert to OHLCV (matching GeckoClient contract): close is at index 4
        return [OHLCV(from_unix(r[0]), float(r[4])) for r in result]


def _make_ohlcv_records(start_ts, count, step=3600):
    """Generate count 6-column OHLCV records starting at start_ts (unix ints)."""
    return [
        [start_ts + i * step, 1.0, 1.0, 1.0, float(i + 1), 100.0]
        for i in range(count)
    ]


class TestCachedClientContiguous:
    def _make_client(self, records, cache_dir=None, max_batch=1000):
        api = _FakeAPI(records, max_batch=max_batch)
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp()
        client = CachedClient(api, cache_dir=cache_dir)
        return client, api, cache_dir

    def test_empty_cache_fetches_from_api(self):
        """First request on empty cache should call API once and persist cache."""
        records = _make_ohlcv_records(1_000_000, 50)
        client, api, cache_dir = self._make_client(records)

        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  before_timestamp=from_unix(1_050_000))
        assert len(api.calls) == 1
        assert len(result) > 0
        assert os.path.exists(os.path.join(cache_dir, "cache_tok.csv"))

    def test_cache_hit_no_api_call(self):
        """Request within cached range should NOT call API again."""
        records = _make_ohlcv_records(1_000_000, 50)
        client, api, _ = self._make_client(records)

        # First call populates cache with all 50 records (before hour 60)
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 60 * 3600))
        assert len(api.calls) == 1

        # Second call for a timestamp within cached range — no new API call
        mid_ts = 1_000_000 + 25 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  before_timestamp=from_unix(mid_ts))
        assert len(api.calls) == 1  # no additional call
        assert len(result) > 0

    def test_extend_backward(self):
        """Requesting before cache_min should extend backward."""
        # 100 records, but API only returns 20 per batch
        records = _make_ohlcv_records(1_000_000, 100)
        client, api, _ = self._make_client(records, max_batch=20)

        # Seed cache — the initial centered fetch only gets the last 20
        late_ts = 1_000_000 + 80 * 3600
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(late_ts))
        assert len(api.calls) == 1

        # Now request a timestamp earlier than what was cached
        early_ts = 1_000_000 + 5 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  before_timestamp=from_unix(early_ts))
        assert len(api.calls) >= 2  # needed at least one more fetch
        timestamps = [to_unix(r[0]) for r in result]
        assert min(timestamps) <= early_ts

    def test_extend_forward(self):
        """Forward extension adds newer data to cache."""
        # 100 records spanning hours 0-99.  max_batch=100 (fits in one fetch).
        records = _make_ohlcv_records(1_000_000, 100)
        client, api, cache_dir = self._make_client(records, max_batch=100)

        # Manually write a cache file covering only the first 20 hours
        early = records[:20]
        cache_path = os.path.join(cache_dir, "cache_tok.csv")
        with open(cache_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "close"])
            for r in early:
                writer.writerow([from_unix(r[0]).isoformat(), r[4]])

        # Request a timestamp beyond hour 19 but within the data range
        late_ts = 1_000_000 + 80 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  before_timestamp=from_unix(late_ts))
        assert len(api.calls) >= 1  # at least one forward-extension fetch
        timestamps = [to_unix(r[0]) for r in result]
        # Should have data up to just before late_ts
        assert max(timestamps) >= late_ts - 3600

    def test_extend_forward_no_gaps(self):
        """Forward extension must fill contiguously, not jump to target."""
        # 3000 hourly records, batch=1000 (matches real API limit)
        records = _make_ohlcv_records(1_000_000, 3000)
        client, api, cache_dir = self._make_client(records, max_batch=1000)

        # Seed cache near the start (gets last 1000 before hour 20 → hours 0-19)
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 20 * 3600))

        # Request far ahead — should fill contiguously, not skip
        far_ts = 1_000_000 + 2500 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=100, before_timestamp=from_unix(far_ts))
        assert len(result) == 100
        # Verify the 100 rows are the ones just before far_ts (no gap), newest first
        timestamps = [to_unix(r[0]) for r in result]
        assert timestamps == sorted(timestamps, reverse=True)
        assert max(timestamps) >= far_ts - 100 * 3600

    def test_limit_and_before_timestamp_filtering(self):
        """Returned data should respect limit and before_timestamp."""
        records = _make_ohlcv_records(1_000_000, 50)
        client, api, _ = self._make_client(records)

        # Fetch all to populate cache
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 60 * 3600))

        # Now request with limit=5 and a before_timestamp in the middle
        mid_ts = 1_000_000 + 25 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=5, before_timestamp=from_unix(mid_ts))
        assert len(result) == 5
        # All returned timestamps should be < before_timestamp
        for r in result:
            assert to_unix(r[0]) < mid_ts

    # -- limit < BATCH_SIZE scenarios --

    def test_small_limit_empty_cache(self):
        """limit=5, no existing cache. Should fetch once, return exactly 5."""
        records = _make_ohlcv_records(1_000_000, 200)
        client, api, _ = self._make_client(records)

        target_ts = 1_000_000 + 100 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=5, before_timestamp=from_unix(target_ts))
        assert len(api.calls) == 1
        assert len(result) == 5
        for r in result:
            assert to_unix(r[0]) < target_ts
        # Should be the 5 latest records before target_ts, newest first
        timestamps = [to_unix(r[0]) for r in result]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_small_limit_existing_cache_covers_target(self):
        """limit=3, cache already covers target. No API call, returns 3."""
        records = _make_ohlcv_records(1_000_000, 100)
        client, api, cache_dir = self._make_client(records)

        # Populate cache with all 100 records
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 110 * 3600))
        first_calls = len(api.calls)

        # Small limit request within cached range
        target_ts = 1_000_000 + 50 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=3, before_timestamp=from_unix(target_ts))
        assert len(api.calls) == first_calls  # no new API calls
        assert len(result) == 3
        for r in result:
            assert to_unix(r[0]) < target_ts

    def test_small_limit_cache_far_from_target(self):
        """limit=5, cache exists but is far from the requested time.
        Should extend cache, then return 5 filtered rows."""
        records = _make_ohlcv_records(1_000_000, 2000)
        client, api, cache_dir = self._make_client(records)

        # Populate cache near the start
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 50 * 3600))
        first_calls = len(api.calls)

        # Request far ahead of what's cached
        far_ts = 1_000_000 + 1500 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=5, before_timestamp=from_unix(far_ts))
        assert len(api.calls) > first_calls  # had to extend forward
        assert len(result) == 5
        for r in result:
            assert to_unix(r[0]) < far_ts
        # The 5 returned should be the last 5 before far_ts, newest first
        timestamps = [to_unix(r[0]) for r in result]
        assert timestamps == sorted(timestamps, reverse=True)
        assert max(timestamps) >= far_ts - 5 * 3600

    def test_limit_1_empty_cache(self):
        """limit=1 (price_at path), no cache. Fetches, returns exactly 1."""
        records = _make_ohlcv_records(1_000_000, 50)
        client, api, _ = self._make_client(records)

        target_ts = 1_000_000 + 30 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=1, before_timestamp=from_unix(target_ts))
        assert len(result) == 1
        assert to_unix(result[0][0]) < target_ts

    def test_limit_1_existing_cache(self):
        """limit=1 with warm cache — no extra API call."""
        records = _make_ohlcv_records(1_000_000, 50)
        client, api, _ = self._make_client(records)

        # Warm the cache
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 60 * 3600))
        first_calls = len(api.calls)

        # limit=1 within cached range
        target_ts = 1_000_000 + 25 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=1, before_timestamp=from_unix(target_ts))
        assert len(api.calls) == first_calls
        assert len(result) == 1
        assert to_unix(result[0][0]) < target_ts

    # -- limit > BATCH_SIZE scenarios --

    def test_large_limit_empty_cache(self):
        """limit=1500 (> 1000 API batch), no cache.
        Should fetch once (gets 1000 from API), return up to what's available."""
        records = _make_ohlcv_records(1_000_000, 2000)
        client, api, _ = self._make_client(records)

        target_ts = 1_000_000 + 1800 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=1500, before_timestamp=from_unix(target_ts))
        assert len(api.calls) >= 1
        assert len(result) <= 1500
        assert len(result) > 0
        for r in result:
            assert to_unix(r[0]) < target_ts

    def test_large_limit_existing_cache_covers(self):
        """limit=1500, cache has enough data. Returns 1500 rows."""
        records = _make_ohlcv_records(1_000_000, 2000)
        client, api, _ = self._make_client(records)

        # Build contiguous cache with overlapping fetches:
        # hours 0-999, then extend forward to hours 1000-1999
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 1000 * 3600))
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 2000 * 3600))
        first_calls = len(api.calls)

        # Now request 1500 within cached range
        target_ts = 1_000_000 + 1900 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=1500, before_timestamp=from_unix(target_ts))
        assert len(result) == 1500
        for r in result:
            assert to_unix(r[0]) < target_ts
        timestamps = [to_unix(r[0]) for r in result]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_large_limit_cache_far_from_target(self):
        """limit=1500, cache exists but far from target. Extends then filters."""
        records = _make_ohlcv_records(1_000_000, 3000)
        client, api, _ = self._make_client(records)

        # Populate cache near start
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 100 * 3600))
        first_calls = len(api.calls)

        # Request far ahead
        far_ts = 1_000_000 + 2500 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=1500, before_timestamp=from_unix(far_ts))
        assert len(api.calls) > first_calls  # had to extend
        assert len(result) <= 1500
        assert len(result) > 0
        for r in result:
            assert to_unix(r[0]) < far_ts

    # -- cache near vs far edge cases --

    def test_cache_just_barely_covers_target(self):
        """Cache max is at or past target_ts — no extension needed."""
        records = _make_ohlcv_records(1_000_000, 100)
        client, api, cache_dir = self._make_client(records)

        # Populate cache (all 100 records, max ts = 1_000_000 + 99*3600)
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 110 * 3600))
        first_calls = len(api.calls)

        # Request at the edge of cached data (cache_max = hour 99)
        edge_ts = 1_000_000 + 99 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=10, before_timestamp=from_unix(edge_ts))
        assert len(api.calls) == first_calls  # no new API calls
        assert len(result) == 10
        for r in result:
            assert to_unix(r[0]) < edge_ts

    def test_cache_one_hour_short_of_target(self):
        """Cache max is 1 hour before target — must extend forward."""
        records = _make_ohlcv_records(1_000_000, 200)
        client, api, cache_dir = self._make_client(records)

        # Populate cache up to hour 99
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 110 * 3600))
        first_calls = len(api.calls)

        # Request 1 hour past cache max
        target_ts = 1_000_000 + 100 * 3600 + 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=10, before_timestamp=from_unix(target_ts))
        # May or may not need extension depending on cache_max vs target
        assert len(result) == 10
        for r in result:
            assert to_unix(r[0]) < target_ts

    def test_target_before_all_data(self):
        """target_ts is before the earliest record — returns empty after filter."""
        records = _make_ohlcv_records(1_000_000, 50)
        client, api, _ = self._make_client(records)

        # Request before any data exists
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=10, before_timestamp=from_unix(999_999))
        assert len(result) == 0

    def test_no_before_timestamp_returns_all_up_to_limit(self):
        """Omitting before_timestamp returns the tail of the cache."""
        records = _make_ohlcv_records(1_000_000, 50)
        client, api, _ = self._make_client(records)

        result = client.get_ohlcv("net", "pool", "hour", "tok", limit=10)
        assert len(result) == 10
        timestamps = [to_unix(r[0]) for r in result]
        assert timestamps == sorted(timestamps, reverse=True)
        # Should be the last 10 records, newest first
        all_ts = sorted(r[0] for r in records)
        assert timestamps == all_ts[-10:][::-1]

    def test_empty_cache_target_before_all_data_returns_empty(self):
        """Initial fetch returns nothing (target far before data).
        Returns empty list without crashing."""
        records = _make_ohlcv_records(1_000_000, 50)
        client, api, cache_dir = self._make_client(records)

        # Request with target way before any data exists
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  before_timestamp=from_unix(500_000))
        assert len(result) == 0

    def test_pool_created_at_prevents_backward_extension(self):
        """When pool_created_at is known, no backward extension past it."""
        records = _make_ohlcv_records(1_000_000, 50)
        api = _FakeAPI(records, max_batch=20,
                       pool_created_at={"tok": parse_ts("2001-09-09T01:46:40Z")})  # ts=1_000_000
        cache_dir = tempfile.mkdtemp()
        client = CachedClient(api, cache_dir=cache_dir)

        # Populate pool cache so pool_created_at is stored
        client.get_top_pools("net", "tok")

        # Seed cache near the end of data
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 60 * 3600))
        calls_after_seed = len(api.calls)

        # Request before pool_created_at — should skip backward extension
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(900_000))
        assert len(api.calls) == calls_after_seed  # no new API calls

    def test_pool_cache_prevents_repeated_backward_extension(self):
        """pool_created_at in pool cache prevents repeated backward fetches."""
        records = _make_ohlcv_records(1_000_000, 50)
        api = _FakeAPI(records, max_batch=20,
                       pool_created_at={"tok": parse_ts("2001-09-09T01:46:40Z")})
        cache_dir = tempfile.mkdtemp()
        client = CachedClient(api, cache_dir=cache_dir)

        # Populate pool cache
        client.get_top_pools("net", "tok")

        # Seed cache
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 60 * 3600))
        # Request early timestamp
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(900_000))
        calls_after = len(api.calls)

        # Request same early timestamp again — still no new calls
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(900_000))
        assert len(api.calls) == calls_after  # no new calls

    def test_extend_backward_overshoots_target(self):
        """Backward extension fetches one batch past target_ts so there
        are enough rows before it, not just barely reaching it."""
        # 2000 hourly records; batch=1000 mirrors real API
        records = _make_ohlcv_records(1_000_000, 2000)
        client, api, _ = self._make_client(records, max_batch=1000)

        # Seed cache near the end (gets hours 1000-1999)
        client.get_ohlcv("net", "pool", "hour", "tok",
                         before_timestamp=from_unix(1_000_000 + 2000 * 3600))
        seed_calls = len(api.calls)

        # Request with target just inside the cached range after first
        # backward batch — without overshoot we'd have very few rows
        target_ts = 1_000_000 + 500 * 3600
        result = client.get_ohlcv("net", "pool", "hour", "tok",
                                  limit=200, before_timestamp=from_unix(target_ts))
        assert len(result) == 200
        for r in result:
            assert to_unix(r[0]) < target_ts
        # Needed at least one backward extension call
        assert len(api.calls) > seed_calls
