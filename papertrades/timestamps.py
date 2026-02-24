from datetime import datetime, timedelta, timezone
from typing import NamedTuple

UTC = timezone.utc


class OHLCV(NamedTuple):
    timestamp: datetime
    close: float


def utcnow():
    return datetime.now(UTC)


def parse_ts(val):
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(val)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def from_unix(ts):
    return datetime.fromtimestamp(ts, tz=UTC)


def to_unix(dt):
    return int(dt.timestamp())


def ceil_hour(dt):
    if dt.minute == dt.second == dt.microsecond == 0:
        return dt
    return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
