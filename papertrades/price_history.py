from collections.abc import Iterator
from datetime import timedelta
from functools import cached_property

from .timestamps import utcnow

_BATCH_SIZE = 1000


class PriceHistory:
    """Thin stateless translation layer.

    Translates domain time-range queries into correctly-sized client calls
    and yields results.  No in-memory state, no cursor, no lazy-loading.
    """

    def __init__(self, token: str, network: str, client):
        self.token = token
        self.network = network
        self._client = client

    @cached_property
    def _pool_attrs(self):
        pools = self._client.get_top_pools(self.network, self.token)
        if not pools:
            raise RuntimeError(f"Could not locate a pool for {self.token}")
        return pools[0]["attributes"]

    @cached_property
    def pool(self):
        return self._pool_attrs["address"]

    @cached_property
    def pool_created_at(self):
        return self._pool_attrs.get("pool_created_at")

    @property
    def current_price(self) -> float:
        return self.price_at(utcnow())

    def price_at(self, t) -> float:
        """Single price at or before *t*.  Calls client with limit=1."""
        ohlcv = self._client.get_ohlcv(
            self.network, self.pool, "hour", self.token,
            limit=1, before_timestamp=t + timedelta(seconds=1),
        )
        if not ohlcv:
            raise ValueError(f"No price data at or before {t}")
        return float(ohlcv[0][1])

    def prices(self, *, start=None, end=None) -> Iterator[tuple]:
        """Yield (datetime, price) from *start* to *end*.

        *start* defaults to pool_created_at.  Iterates backward from
        *end* in _BATCH_SIZE chunks, then yields in chronological order.
        """
        if start is None:
            start = self.pool_created_at
        if start is None:
            raise ValueError("start is required when pool_created_at is unknown")
        if end is None:
            end = utcnow()

        # get_ohlcv returns the newest N rows before a timestamp,
        # so walk backward from end, collecting chunks.
        collected = {}
        cursor = end + timedelta(seconds=1)
        while True:
            ohlcv = self._client.get_ohlcv(
                self.network, self.pool, "hour", self.token,
                limit=_BATCH_SIZE, before_timestamp=cursor,
            )
            if not ohlcv:
                break
            for row in ohlcv:
                ts = row[0]
                if start <= ts <= end:
                    collected[ts] = float(row[1])
            oldest = ohlcv[-1][0]
            if oldest <= start or len(ohlcv) < _BATCH_SIZE:
                break
            cursor = oldest

        for ts in sorted(collected):
            yield (ts, collected[ts])
