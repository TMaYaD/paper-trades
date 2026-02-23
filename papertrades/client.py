import json
import os
import time

import pandas as pd
import requests

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
        return res.get("data", [])

    def get_ohlcv(
        self,
        network: str,
        pool: str,
        timeframe: str,
        token: str,
        limit: int = 1000,
        before_timestamp: int | None = None,
    ) -> list[list]:
        url = (
            f"{self.BASE_URL}/networks/{network}/pools/{pool}"
            f"/ohlcv/{timeframe}?limit={limit}&token={token}&currency=usd"
        )
        if before_timestamp is not None:
            url += f"&before_timestamp={before_timestamp}"
        res = self._rate_limited_get(url)
        try:
            return res["data"]["attributes"]["ohlcv_list"]
        except KeyError:
            raise RuntimeError(f"API Error for pool {pool}. Response: {res}")


class CachedClient:
    """Same interface as GeckoClient. Checks file cache first, fills gaps from API."""

    POOL_CACHE_FILE = "pool_cache.json"

    def __init__(self, api: GeckoClient, cache_dir: str = "caches"):
        self._api = api
        self._cache_dir = cache_dir

    def _pool_cache_path(self) -> str:
        return os.path.join(self._cache_dir, self.POOL_CACHE_FILE)

    def _ohlcv_cache_path(self, token: str) -> str:
        return os.path.join(self._cache_dir, f"cache_{token}.csv")

    def _load_pool_cache(self) -> dict:
        path = self._pool_cache_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_pool_cache(self, cache: dict):
        with open(self._pool_cache_path(), "w") as f:
            json.dump(cache, f, indent=2)

    def get_top_pools(self, network: str, token: str) -> list[dict]:
        cache = self._load_pool_cache()
        if token in cache:
            print(f"  Using cached pool for {token[:8]}...")
            return [{"attributes": {"address": cache[token]}}]
        pools = self._api.get_top_pools(network, token)
        if pools:
            address = pools[0]["attributes"]["address"]
            cache[token] = address
            self._save_pool_cache(cache)
        return pools

    def get_ohlcv(
        self,
        network: str,
        pool: str,
        timeframe: str,
        token: str,
        limit: int = 1000,
        before_timestamp: int | None = None,
    ) -> list[list]:
        cache_file = self._ohlcv_cache_path(token)
        target_ts = before_timestamp

        # Load existing cache
        if os.path.exists(cache_file):
            df_cache = pd.read_csv(cache_file)
            df_cache["timestamp"] = pd.to_datetime(df_cache["timestamp"])
            cache_oldest_ts = int(df_cache["timestamp"].min().timestamp())
            cache_newest_ts = int(df_cache["timestamp"].max().timestamp())
        else:
            df_cache = None
            cache_oldest_ts = float("inf")
            cache_newest_ts = float("-inf")

        now_ts = int(time.time())

        # If cache is fresh enough and covers the range, serve from cache
        if (
            df_cache is not None
            and cache_newest_ts >= now_ts - 3600
            and (target_ts is None or cache_oldest_ts <= target_ts)
        ):
            print("  -> Cache is up to date. No API calls needed.")
            return self._df_to_ohlcv_list(df_cache)

        # Otherwise fetch from API, merge with cache
        all_dfs = [df_cache] if df_cache is not None else []
        cursor = before_timestamp

        while True:
            ohlcv_list = self._api.get_ohlcv(
                network, pool, timeframe, token, limit=limit, before_timestamp=cursor
            )
            if not ohlcv_list:
                break

            df = pd.DataFrame(
                ohlcv_list,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            batch_oldest_ts = int(df["timestamp"].min())
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            all_dfs.append(df[["timestamp", "close"]])
            print(f"  -> Fetched {len(df)} records down to {df['timestamp'].min()}")

            if target_ts is not None and batch_oldest_ts <= target_ts:
                break
            if cursor is not None and batch_oldest_ts >= cursor:
                break

            # Skip through cached data if possible
            if batch_oldest_ts <= cache_newest_ts and df_cache is not None:
                if cache_oldest_ts <= (target_ts or 0):
                    print("  -> Cache covers the remaining history. Stopping fetch.")
                    break
                else:
                    print(f"  -> Skipping cursor through cached data to {pd.to_datetime(cache_oldest_ts, unit='s')}")
                    cursor = cache_oldest_ts
                    continue

            cursor = batch_oldest_ts

        if not all_dfs:
            raise ValueError("No historical data could be retrieved.")

        final_df = pd.concat(all_dfs).drop_duplicates(subset=["timestamp"])
        final_df = final_df.sort_values("timestamp").reset_index(drop=True)
        final_df.to_csv(cache_file, index=False)

        return self._df_to_ohlcv_list(final_df)

    @staticmethod
    def _df_to_ohlcv_list(df: pd.DataFrame) -> list[list]:
        """Convert a cached DataFrame back to ohlcv_list format [ts, close]."""
        rows = []
        for _, row in df.iterrows():
            ts = int(pd.Timestamp(row["timestamp"]).timestamp())
            rows.append([ts, float(row["close"])])
        return rows
