import time as _time
from datetime import datetime

import pandas as pd


class PriceHistory:
    """Domain-specific price accessor.

    Backtest: pass start_date — history is lazy-loaded on first access.
    Live: omit start_date — data arrives via poll()/append().
    """

    def __init__(self, token: str, network: str, client,
                 start_date: str | None = None):
        self.token = token
        self.network = network
        self._client = client
        self._start_date = start_date
        self._pool = None
        self._series = pd.Series(dtype=float)
        self._series.index = pd.DatetimeIndex([])
        self._cursor = -1
        self._loaded = start_date is None  # nothing to lazy-load in live mode

    def _ensure_loaded(self):
        if not self._loaded:
            self._loaded = True
            self._fetch_history()

    def _resolve_pool(self):
        if self._pool is None:
            pools = self._client.get_top_pools(self.network, self.token)
            if not pools:
                raise RuntimeError(f"Could not locate a pool for {self.token}")
            self._pool = pools[0]["attributes"]["address"]
        return self._pool

    @property
    def current_price(self) -> float:
        self._ensure_loaded()
        if self._cursor < 0:
            raise ValueError(f"No current price for {self.token}")
        return float(self._series.iloc[self._cursor])

    def price_at(self, time) -> float:
        """Closest price at or before the given time."""
        self._ensure_loaded()
        time = pd.Timestamp(time)
        mask = self._series.index[self._series.index <= time]
        if len(mask) == 0:
            raise ValueError(f"No price data at or before {time} for {self.token}")
        return float(self._series.loc[mask[-1]])

    def prices_since(self, time) -> pd.Series:
        """All prices from time up to current cursor position."""
        self._ensure_loaded()
        time = pd.Timestamp(time)
        end = self._series.index[self._cursor]
        return self._series.loc[time:end].copy()

    def all_prices(self) -> pd.Series:
        """All prices up to current cursor position."""
        self._ensure_loaded()
        return self._series.iloc[: self._cursor + 1].copy()

    def set_cursor(self, idx: int):
        """Advance the current tick position (called by engine)."""
        self._ensure_loaded()
        self._cursor = idx

    def append(self, timestamp, price):
        """Add a new price point (live mode)."""
        self._series.loc[pd.Timestamp(timestamp)] = price
        self._cursor = len(self._series) - 1

    def _fetch_history(self):
        """Fetch historical data through client and populate internal series."""
        pool = self._resolve_pool()
        target_ts = int(pd.to_datetime(self._start_date).timestamp())

        all_rows = []
        cursor = None

        while True:
            ohlcv = self._client.get_ohlcv(
                self.network, pool, "hour", self.token,
                limit=1000, before_timestamp=cursor,
            )
            if not ohlcv:
                break

            all_rows.extend(ohlcv)
            batch_oldest = min(row[0] for row in ohlcv)
            print(f"  -> Loaded {len(ohlcv)} records down to {pd.to_datetime(batch_oldest, unit='s')}")

            if batch_oldest <= target_ts:
                break
            if cursor is not None and batch_oldest >= cursor:
                break
            cursor = batch_oldest

        if not all_rows:
            raise ValueError(f"No historical data for {self.token}")

        # Build series from [timestamp, close] pairs
        data = {pd.to_datetime(row[0], unit="s"): row[1] for row in all_rows}
        self._series = pd.Series(data, dtype=float).sort_index()
        self._series = self._series[~self._series.index.duplicated(keep="last")]
        self._series = self._series.loc[self._start_date:]
        self._cursor = len(self._series) - 1

    def poll(self) -> float:
        """Fetch current price from client, append to series. For live mode."""
        pool = self._resolve_pool()
        ohlcv = self._client.get_ohlcv(
            self.network, pool, "hour", self.token, limit=1,
        )
        if not ohlcv:
            raise RuntimeError(f"Could not fetch current price for {self.token}")
        price = float(ohlcv[0][4]) if len(ohlcv[0]) > 4 else float(ohlcv[0][1])
        self.append(datetime.utcnow(), price)
        return price
