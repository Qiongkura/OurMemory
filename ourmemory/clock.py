# -*- coding: utf-8 -*-
"""Timezone-aware day boundaries and timestamp rendering.

Timestamps are stored as UTC epoch seconds, so the data itself is
timezone-agnostic.  What *is* timezone-dependent is the answer to "what counts
as today" and how a timestamp is printed.  Both go through this module so a
server in UTC can serve a user in Asia/Shanghai without the day boundary
drifting eight hours.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone, tzinfo

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - very old interpreters
    ZoneInfo = None  # type: ignore[assignment]


def resolve_tz(tz: str | tzinfo | None) -> tzinfo | None:
    """Accept ``None`` (system local time), an IANA name, or a tzinfo object."""
    if tz is None or isinstance(tz, tzinfo):
        return tz
    name = str(tz).strip()
    if not name:
        return None
    if name.upper() in ("UTC", "Z"):
        return timezone.utc
    if ZoneInfo is None:
        raise ValueError(
            "zoneinfo is unavailable on this interpreter; pass a tzinfo object"
        )
    try:
        return ZoneInfo(name)
    except Exception as exc:  # ZoneInfoNotFoundError, or missing tzdata
        raise ValueError(
            f"unknown timezone {name!r}. On Windows the IANA database is not "
            "bundled with Python: run `pip install tzdata`, or pass a tzinfo "
            "object such as `datetime.timezone(timedelta(hours=8))`."
        ) from exc


def _local(ts: float, tz) -> datetime:
    zone = resolve_tz(tz)
    if zone is None:
        return datetime.fromtimestamp(float(ts))
    return datetime.fromtimestamp(float(ts), zone)


def stamp(ts: float, tz=None, pattern: str = "%m-%d %H:%M") -> str:
    """Format an epoch timestamp in ``tz`` (system local time when None)."""
    return _local(ts, tz).strftime(pattern)


def day_bounds(now: float, tz=None,
               offset_days: int = 0) -> tuple[float, float]:
    """Epoch bounds of the calendar day ``offset_days`` away from ``now``.

    ``offset_days=0`` is today, ``-1`` yesterday.  Uses real calendar
    midnights, so DST transitions and non-24-hour days are handled by the
    tz database instead of by adding 86400 seconds.
    """
    current = _local(now, tz)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    start = start + timedelta(days=offset_days)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def shift_days(now: float, days: float) -> float:
    """Plain epoch arithmetic for coarse windows ("last 7 days")."""
    return float(now) + float(days) * 86400.0


def now_epoch() -> float:
    return time.time()
