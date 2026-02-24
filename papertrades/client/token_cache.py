import csv
import logging
import os
from datetime import timedelta

from ..timestamps import OHLCV, parse_ts, utcnow

log = logging.getLogger(__name__)


class TokenCache:
    """Per-token OHLCV cache backed by a CSV file."""

    def __init__(self, *, api, network, pool, timeframe, token,
                 cache_dir, pool_created_ts):
        self._api = api
        self._network = network
        self._pool = pool
        self._timeframe = timeframe
        self._token = token
        self._tag = token[:8]
        self._cache_file = os.path.join(cache_dir, f"cache_{token}.csv")
        self._floor_file = os.path.join(cache_dir, f"cache_{token}.floor")
        self._ceiling_file = os.path.join(cache_dir, f"cache_{token}.ceiling")
        self._pool_created_ts = pool_created_ts
        self._rows = None  # lazy
        self._cache_min = None
        self._cache_max = None
        self._hit_floor = os.path.exists(self._floor_file)
        self._hit_ceiling = self._load_ceiling()
        log.debug("[%s] init: timeframe=%s pool_created=%s hit_floor=%s hit_ceiling=%s",
                  self._tag, timeframe, pool_created_ts, self._hit_floor, self._hit_ceiling)

    @property
    def rows(self) -> list[OHLCV] | None:
        if self._rows is None:
            self._rows = self._load()
            self._update_bounds()
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = value
        self._update_bounds()

    def _update_bounds(self):
        if self._rows:
            self._cache_min = self._rows[-1].timestamp
            self._cache_max = self._rows[0].timestamp
        else:
            self._cache_min = None
            self._cache_max = None

    def get(self, before_timestamp, limit) -> list[OHLCV]:
        target_ts = before_timestamp if before_timestamp is not None else utcnow()
        log.debug("[%s] get: before_timestamp=%s limit=%d target_ts=%s",
                  self._tag, before_timestamp, limit, target_ts)

        if self.rows is None:
            log.debug("[%s] get: no rows loaded, seeding with target_ts=%s", self._tag, target_ts)
            self._seed(target_ts)
            if self.rows is None:
                log.debug("[%s] get: seed returned nothing, returning []", self._tag)
                return []
        else:
            log.debug("[%s] get: rows loaded (%d), cache_min=%s cache_max=%s, extending toward target_ts=%s",
                      self._tag, len(self.rows), self._cache_min, self._cache_max, target_ts)
            self._extend(target_ts)

        rows = self.rows
        if before_timestamp is not None:
            rows = [r for r in rows if r.timestamp < before_timestamp]
        result = rows[:limit]
        log.debug("[%s] get: returning %d rows (filtered from %d)", self._tag, len(result), len(rows))
        return result

    # -- cache lifecycle ---------------------------------------------------

    def _load(self) -> list[OHLCV] | None:
        if not os.path.exists(self._cache_file):
            log.debug("[%s] _load: no cache file", self._tag)
            return None
        with open(self._cache_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = [OHLCV(parse_ts(row["timestamp"]), float(row["close"]))
                    for row in reader]
        log.debug("[%s] _load: loaded %d rows from CSV", self._tag, len(rows))
        return rows

    def _save(self):
        with open(self._cache_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "close"])
            for r in self.rows:
                writer.writerow([r.timestamp.isoformat(), r.close])

    def _seed(self, target_ts):
        log.debug("[%s] _seed: target_ts=%s", self._tag, target_ts)
        try:
            self.rows = self._fetch(target_ts)
        except ValueError:
            log.debug("[%s] _seed: fetch raised ValueError, no data", self._tag)
            return

        log.debug("[%s] _seed: got %d rows, cache_min=%s cache_max=%s",
                  self._tag, len(self.rows), self._cache_min, self._cache_max)
        self._save()

    def _extend(self, target_ts):
        if target_ts < self._cache_min:
            log.debug("[%s] _extend: target_ts=%s < cache_min=%s -> backward",
                      self._tag, target_ts, self._cache_min)
            self._extend_backward(target_ts)
        elif target_ts > self._cache_max:
            log.debug("[%s] _extend: target_ts=%s > cache_max=%s -> forward",
                      self._tag, target_ts, self._cache_max)
            self._extend_forward(target_ts)
        else:
            log.debug("[%s] _extend: target_ts=%s within [%s, %s], no extension needed",
                      self._tag, target_ts, self._cache_min, self._cache_max)

    def _mark_floor(self):
        self._hit_floor = True
        with open(self._floor_file, "w") as f:
            f.write("")

    def _load_ceiling(self) -> bool:
        if not os.path.exists(self._ceiling_file):
            return False
        ts = parse_ts(open(self._ceiling_file).read().strip())
        # Ceiling expires after one time interval has passed
        return utcnow() < ts + self._time_interval()

    def _mark_ceiling(self):
        self._hit_ceiling = True
        with open(self._ceiling_file, "w") as f:
            f.write(utcnow().isoformat())

    def _extend_backward(self, target_ts):
        log.debug("[%s] _extend_backward: target_ts=%s cache_min=%s hit_floor=%s pool_created=%s",
                  self._tag, target_ts, self._cache_min, self._hit_floor, self._pool_created_ts)
        if self._hit_floor:
            log.debug("[%s] _extend_backward: hit_floor=True, skipping", self._tag)
            return
        if self._pool_created_ts is not None and target_ts < self._pool_created_ts:
            log.debug("[%s] _extend_backward: target_ts < pool_created_ts, skipping", self._tag)
            return

        def _fetch_and_filter():
            try:
                rows = self._fetch(self._cache_min)
            except ValueError:
                log.debug("[%s] _extend_backward: fetch raised ValueError", self._tag)
                return []
            before_filter = len(rows)
            rows = [r for r in rows if r.timestamp < self._cache_min]
            log.debug("[%s] _extend_backward: fetched %d, after filter %d (< %s)",
                      self._tag, before_filter, len(rows), self._cache_min)
            return rows

        iteration = 0
        while self._cache_min > target_ts:
            iteration += 1
            prev_min = self._cache_min
            log.info("[%s] _extend_backward[%d]: cache_min=%s target_ts=%s",
                     self._tag, iteration, self._cache_min, target_ts)
            rows = _fetch_and_filter()
            if len(rows) == 0:
                log.debug("[%s] _extend_backward: no new rows, marking floor", self._tag)
                self._mark_floor()
                self._save()
                return
            self.rows = self.rows + rows
            log.debug("[%s] _extend_backward: appended %d rows, cache_min %s -> %s",
                      self._tag, len(rows), prev_min, self._cache_min)
            if self._cache_min >= prev_min:
                log.debug("[%s] _extend_backward: no progress, marking floor", self._tag)
                self._mark_floor()
                break

        # One more fetch so there are up to 1000 extra rows before target_ts.
        log.debug("[%s] _extend_backward: bonus fetch for padding", self._tag)
        rows = _fetch_and_filter()
        self.rows = self.rows + rows

        self._save()

    _TIMEFRAME_INTERVALS = {"minute": timedelta(minutes=1), "hour": timedelta(hours=1), "day": timedelta(days=1)}

    def _time_interval(self):
        return self._TIMEFRAME_INTERVALS[self._timeframe]

    def _floor(self, ts):
        if self._timeframe == "minute":
            return ts.replace(second=0, microsecond=0)
        elif self._timeframe == "hour":
            return ts.replace(minute=0, second=0, microsecond=0)
        elif self._timeframe == "day":
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)

    def _extend_forward(self, target_ts):
        log.debug("[%s] _extend_forward: target_ts=%s cache_max=%s hit_ceiling=%s",
                  self._tag, target_ts, self._cache_max, self._hit_ceiling)
        if self._hit_ceiling:
            log.debug("[%s] _extend_forward: hit_ceiling=True, skipping", self._tag)
            return
        now = utcnow()
        original_target = target_ts
        target_ts = self._floor(max(now, target_ts))
        log.debug("[%s] _extend_forward: now=%s original_target=%s clamped_target=%s",
                  self._tag, now, original_target, target_ts)

        iteration = 0
        while self._cache_max < target_ts:
            iteration += 1
            prev_max = self._cache_max
            fetch_before = self._cache_max + 1001 * self._time_interval()
            log.info("[%s] _extend_forward[%d]: cache_max=%s target_ts=%s fetch_before=%s",
                     self._tag, iteration, self._cache_max, target_ts, fetch_before)
            try:
                rows = self._fetch(fetch_before)
            except ValueError:
                log.debug("[%s] _extend_forward: fetch raised ValueError, marking ceiling", self._tag)
                self._mark_ceiling()
                break
            before_filter = len(rows)
            rows = [r for r in rows if r.timestamp > self._cache_max]
            log.debug("[%s] _extend_forward: fetched %d, after filter %d (> %s)",
                      self._tag, before_filter, len(rows), self._cache_max)
            if len(rows) == 0:
                log.debug("[%s] _extend_forward: no new rows, marking ceiling", self._tag)
                self._mark_ceiling()
                break
            self.rows = rows + self.rows
            self._update_bounds()
            log.debug("[%s] _extend_forward: appended %d rows, cache_max %s -> %s",
                      self._tag, len(rows), prev_max, self._cache_max)
            if self._cache_max <= prev_max:
                log.debug("[%s] _extend_forward: no progress, marking ceiling", self._tag)
                self._mark_ceiling()
                break

        self._save()

    def _fetch(self, before_ts) -> list[OHLCV]:
        fetched = self._api.get_ohlcv(
            self._network, self._pool, self._timeframe, self._token,
            limit=1000, before_timestamp=before_ts,
        )
        if not fetched:
            raise ValueError("No historical data could be retrieved.")

        # Enforce descending (newest-first) order regardless of API sort
        fetched.sort(key=lambda r: r.timestamp, reverse=True)

        print(f"  -> Fetched {len(fetched)} records for {self._token[:8]}... from {fetched[-1].timestamp} to {fetched[0].timestamp}")
        return fetched
