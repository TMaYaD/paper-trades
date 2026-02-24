import json
import os

from ..timestamps import OHLCV, parse_ts
from .token_cache import TokenCache


class CachedClient:
    """Same interface as GeckoClient. Checks file cache first, fills gaps from API."""

    POOL_CACHE_FILE = "pool_cache.json"

    def __init__(self, api, cache_dir: str = "caches"):
        self._api = api
        self._cache_dir = cache_dir

    def _pool_cache_path(self) -> str:
        return os.path.join(self._cache_dir, self.POOL_CACHE_FILE)

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
            entry = cache[token]
            attrs = {"address": entry["address"]}
            if "pool_created_at" in entry:
                attrs["pool_created_at"] = parse_ts(entry["pool_created_at"])
            return [{"attributes": attrs}]
        pools = self._api.get_top_pools(network, token)
        if pools:
            attrs = pools[0]["attributes"]
            entry = {"address": attrs["address"]}
            if "pool_created_at" in attrs and attrs["pool_created_at"] is not None:
                # Store as ISO string in JSON cache; attrs already has datetime from API
                entry["pool_created_at"] = attrs["pool_created_at"].isoformat()
            cache[token] = entry
            self._save_pool_cache(cache)
        return pools

    def _pool_created_ts(self, token: str):
        pool_cache = self._load_pool_cache()
        entry = pool_cache.get(token)
        if isinstance(entry, dict) and "pool_created_at" in entry:
            return parse_ts(entry["pool_created_at"])
        return None

    def get_ohlcv(
        self,
        network: str,
        pool: str,
        timeframe: str,
        token: str,
        limit: int = 1000,
        before_timestamp=None,
    ) -> list[OHLCV]:
        cache = TokenCache(
            api=self._api,
            network=network,
            pool=pool,
            timeframe=timeframe,
            token=token,
            cache_dir=self._cache_dir,
            pool_created_ts=self._pool_created_ts(token),
        )
        return cache.get(before_timestamp, limit)
