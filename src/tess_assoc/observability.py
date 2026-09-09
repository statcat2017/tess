"""Validated cadence evidence for light-curve event measurements."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from statistics import median
from typing import Any

from tess_assoc._validate import (
    is_finite_number,
    is_strict_int,
    require_finite,
    require_positive_finite,
)


def _cadence_threshold(time: tuple[float, ...]) -> float | None:
    gaps = sorted(b - a for a, b in zip(time, time[1:]) if b > a)
    if not gaps:
        return None
    return median(gaps[: max(1, (len(gaps) + 1) // 2)]) * 5.0


def _derive_windows(
    time: tuple[float, ...],
    usable: tuple[bool, ...],
    max_gap_days: float | None = None,
) -> tuple[tuple[float, float], ...]:
    threshold = _cadence_threshold(time) if max_gap_days is None else max_gap_days
    if threshold is None:
        return ()

    windows: list[tuple[float, float]] = []
    start: float | None = None
    previous: float | None = None
    for current, is_usable in zip(time, usable):
        if not is_usable:
            if start is not None and previous is not None and previous > start:
                windows.append((start, previous))
            start = None
            previous = None
            continue
        if start is None:
            start = current
        elif previous is not None and current - previous > threshold:
            if previous > start:
                windows.append((start, previous))
            start = current
        previous = current
    if start is not None and previous is not None and previous > start:
        windows.append((start, previous))
    return tuple(windows)


@dataclass(frozen=True)
class CadenceEvidence:
    """Raw cadence availability and quality state aligned by cadence index."""

    time: tuple[float, ...]
    usable: tuple[bool, ...]
    quality_flags: tuple[int, ...]
    observing_windows: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.time, (list, tuple)):
            raise ValueError("cadence time must be a list/tuple")
        if not isinstance(self.usable, (list, tuple)):
            raise ValueError("cadence usable mask must be a list/tuple")
        if not isinstance(self.quality_flags, (list, tuple)):
            raise ValueError("cadence quality flags must be a list/tuple")
        if not isinstance(self.observing_windows, (list, tuple)):
            raise ValueError("observing_windows must be a list/tuple")

        times = tuple(self.time)
        usable = tuple(self.usable)
        quality_flags = tuple(self.quality_flags)
        if not times:
            raise ValueError("cadence evidence must contain at least one cadence")
        if len(times) != len(usable) or len(times) != len(quality_flags):
            raise ValueError("cadence evidence fields must have equal length")
        for value in times:
            require_finite("cadence time", value)
        if any(not b > a for a, b in zip(times, times[1:])):
            raise ValueError("cadence time must be strictly increasing")
        if any(not isinstance(value, bool) for value in usable):
            raise ValueError("cadence usable mask must contain bool values")
        for value in quality_flags:
            if not is_strict_int(value) or value < 0:
                raise ValueError("cadence quality flags must be non-negative ints")

        derived = _derive_windows(times, usable)
        supplied = tuple(tuple(window) for window in self.observing_windows)
        for window in supplied:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ValueError("each observing window must be a [start, end] pair")
            require_finite("observing window start", window[0])
            require_finite("observing window end", window[1])
            if window[1] <= window[0]:
                raise ValueError("observing window end must be after start")
        if supplied and supplied != derived:
            raise ValueError("observing_windows do not match cadence evidence")

        object.__setattr__(self, "time", times)
        object.__setattr__(self, "usable", usable)
        object.__setattr__(self, "quality_flags", quality_flags)
        object.__setattr__(self, "observing_windows", derived)

    @property
    def usable_time(self) -> tuple[float, ...]:
        return tuple(t for t, is_usable in zip(self.time, self.usable) if is_usable)

    @property
    def quality_flagged(self) -> int:
        return sum(flag != 0 for flag in self.quality_flags)

    @property
    def missing(self) -> int:
        return sum(not is_usable for is_usable in self.usable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": list(self.time),
            "usable": list(self.usable),
            "quality_flags": list(self.quality_flags),
            "observing_windows": [list(window) for window in self.observing_windows],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CadenceEvidence:
        if not isinstance(value, dict):
            raise ValueError("cadence evidence must be a dict")
        required = {"time", "usable", "quality_flags", "observing_windows"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"cadence evidence missing keys: {missing}")
        extra = sorted(set(value) - required)
        if extra:
            raise ValueError(f"cadence evidence unknown keys: {extra}")
        return cls(
            time=value["time"],
            usable=value["usable"],
            quality_flags=value["quality_flags"],
            observing_windows=value["observing_windows"],
        )


@dataclass(frozen=True)
class LightCurve:
    """Good flux samples plus the raw cadence evidence they came from."""

    time: tuple[float, ...]
    flux: tuple[float, ...]
    evidence: CadenceEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.time, (list, tuple)) or not isinstance(
            self.flux, (list, tuple)
        ):
            raise ValueError("light-curve time and flux must be lists/tuples")
        if not isinstance(self.evidence, CadenceEvidence):
            raise ValueError("light-curve evidence must be CadenceEvidence")
        time = tuple(self.time)
        flux = tuple(self.flux)
        if not time or len(time) != len(flux):
            raise ValueError("light-curve time and flux must be non-empty and equal length")
        if any(not is_finite_number(value) for value in (*time, *flux)):
            raise ValueError("light-curve time and flux must be finite")
        expected_time = self.evidence.usable_time
        if time != expected_time:
            raise ValueError("light-curve samples must match usable cadence evidence")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "flux", flux)

    def __iter__(self) -> Iterator[list[float]]:
        """Keep the historical ``time, flux = load_lightcurve(...)`` seam."""
        yield list(self.time)
        yield list(self.flux)
