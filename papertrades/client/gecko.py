import time
from datetime import timedelta

import requests

from ..timestamps import OHLCV, from_unix, parse_ts, to_unix, utcnow

RATE_LIMIT_DELAY = 1.5


class GeckoClient:
    """Mirrors the GeckoTerminal API. No caching, no domain logic."""

    BASE_URL = "https://api.geckoterminal.com/api/v2"
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    def _rate_limited_get(self, url: str) -> dict:
        time.sleep(RATE_LIMIT_DELAY)
        while True:
            res = requests.get(url, headers=self.HEADERS)
            if res.status_code == 429:
                print("[!] HTTP 429: Rate limit hit. Sleeping for 30 seconds...")
                time.sleep(30)
                continue
            return res.json()

    def get_top_pools(self, network: str, token: str) -> list[dict]:
        url = f"{self.BASE_URL}/networks/{network}/tokens/{token}/pools"
        res = self._rate_limited_get(url)
        pools = res.get("data", [])
        for pool in pools:
            attrs = pool.get("attributes", {})
            if "pool_created_at" in attrs and attrs["pool_created_at"] is not None:
                attrs["pool_created_at"] = parse_ts(attrs["pool_created_at"])
        return pools

    def get_ohlcv(
        self,
        network: str,
        pool: str,
        timeframe: str,
        token: str,
        limit: int = 1000,
        before_timestamp=None,
    ) -> list[OHLCV]:
        before_timestamp, limit = self._adjust_time_range(before_timestamp, limit, timeframe)
        if limit == 0:
            return []
        url = (
            f"{self.BASE_URL}/networks/{network}/pools/{pool}"
            f"/ohlcv/{timeframe}?limit={limit}&token={token}&currency=usd"
        )
        if before_timestamp is not None:
            url += f"&before_timestamp={to_unix(before_timestamp)}"
        res = self._rate_limited_get(url)
        try:
            ohlcv_list = res["data"]["attributes"]["ohlcv_list"]
        except KeyError:
            raise RuntimeError(f"API Error for pool {pool}. Response: {res}")
        # API returns [ts, open, high, low, close, volume]; extract timestamp + close
        return [OHLCV(from_unix(row[0]), float(row[4])) for row in ohlcv_list]

    def _adjust_time_range(self, before_timestamp, limit, timeframe):
        if before_timestamp is None:
            return before_timestamp, limit

        # 1. Map the timeframe string directly to a timedelta object
        timeframe_map = {
            "minute": timedelta(minutes=1),
            "hour": timedelta(hours=1),
            "day": timedelta(days=1)
        }
        tf_delta = timeframe_map.get(timeframe.lower())
        if tf_delta is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # 3. Define the absolute 180-day wall and the evaluation boundaries
        earliest_allowed_ts = utcnow() - timedelta(days=180)

        # if before_timestamp itself is beyond 180 days, we need to adjust it
        if before_timestamp < earliest_allowed_ts:
            print("[!] before_timestamp is beyond 180 days. Adjusting...")
            before_timestamp = earliest_allowed_ts

        # Multiplying a timedelta by an int yields the total duration
        requested_oldest_time = before_timestamp - (limit * tf_delta)

        # 4. Adjust the limit based on the evaluation boundaries
        if requested_oldest_time < earliest_allowed_ts:
            print("[!] limit is going beyond 180 days. Adjusting...")
            limit = (before_timestamp - earliest_allowed_ts) // tf_delta

        return before_timestamp, limit
