"""Validated cadence evidence for light-curve event measurements."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from statistics import median
from typing import Any

from tess_assoc._validate import (
    is_finite_number,
    require_finite,
    require_positive_finite,
    require_strict_int,
)


def _cadence_threshold(time: tuple[float, ...]) -> float | None:
    gaps = sorted(b - a for a, b in zip(time, time[1:]) if b > a)
    if not gaps:
        return None
    return median(gaps[: max(1, (len(gaps) + 1) // 2)]) * 5.0


def coverage_windows(
    time: list[float] | tuple[float, ...],
    max_gap_days: float | None = None,
    excluded_windows: list[tuple[float, float]] | None = None,
    *,
    usable: list[bool] | tuple[bool, ...] | None = None,
) -> list[tuple[float, float]]:
    """Return contiguous usable spans without crossing gaps or masks."""
    if not isinstance(time, (list, tuple)):
        raise ValueError("time must be a list/tuple")
    times = tuple(time)
    if any(not is_finite_number(value) for value in times):
        raise ValueError("time must contain finite values")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("time must be strictly increasing")
    if usable is None:
        mask = (True,) * len(times)
    else:
        if not isinstance(usable, (list, tuple)) or len(usable) != len(times):
            raise ValueError("usable mask must match time length")
        if any(not isinstance(value, bool) for value in usable):
            raise ValueError("usable mask must contain bool values")
        mask = tuple(usable)
    if max_gap_days is not None:
        require_positive_finite("max_gap_days", max_gap_days)
    threshold = _cadence_threshold(times) if max_gap_days is None else max_gap_days
    spans: list[tuple[float, float]] = []
    if threshold is not None:
        start: float | None = None
        previous: float | None = None
        for current, is_usable in zip(times, mask):
            if not is_usable:
                if start is not None and previous is not None and previous > start:
                    spans.append((start, previous))
                start = None
                previous = None
                continue
            if start is None:
                start = current
            elif previous is not None and current - previous > threshold:
                if previous > start:
                    spans.append((start, previous))
                start = current
            previous = current
        if start is not None and previous is not None and previous > start:
            spans.append((start, previous))

    if not excluded_windows:
        return spans
    for excluded in excluded_windows:
        if not isinstance(excluded, (list, tuple)) or len(excluded) != 2:
            raise ValueError("each excluded window must be a [start, end] pair")
        excluded_start, excluded_end = excluded
        require_finite("excluded window start", excluded_start)
        require_finite("excluded window end", excluded_end)
        if excluded_end <= excluded_start:
            raise ValueError("excluded window end must be after start")
        remainder: list[tuple[float, float]] = []
        for start, end in spans:
            if excluded_end <= start or excluded_start >= end:
                remainder.append((start, end))
                continue
            if start < excluded_start:
                remainder.append((start, min(end, excluded_start)))
            if excluded_end < end:
                remainder.append((max(start, excluded_end), end))
        spans = [span for span in remainder if span[1] > span[0]]
    return spans


@dataclass(frozen=True)
class CadenceEvidence:
    """Raw cadence availability and quality state aligned by cadence index."""

    time: tuple[float, ...]
    usable: tuple[bool, ...]
    quality_flags: tuple[int, ...]
    observing_windows: tuple[tuple[float, float], ...] = ()
    source_product: SourceProduct | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time, (list, tuple)):
            raise ValueError("cadence time must be a list/tuple")
        if not isinstance(self.usable, (list, tuple)):
            raise ValueError("cadence usable mask must be a list/tuple")
        if not isinstance(self.quality_flags, (list, tuple)):
            raise ValueError("cadence quality flags must be a list/tuple")
        if not isinstance(self.observing_windows, (list, tuple)):
            raise ValueError("observing_windows must be a list/tuple")
        if self.source_product is not None and not isinstance(
            self.source_product, SourceProduct
        ):
            raise ValueError("source_product must be SourceProduct or None")

        times = tuple(self.time)
        usable = tuple(self.usable)
        quality_flags = tuple(self.quality_flags)
        if not times:
            raise ValueError("cadence evidence must contain at least one cadence")
        if len(times) != len(usable) or len(times) != len(quality_flags):
            raise ValueError("cadence evidence fields must have equal length")
        for value in times:
            require_finite("cadence time", value)
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("cadence time must be strictly increasing")
        if any(not isinstance(value, bool) for value in usable):
            raise ValueError("cadence usable mask must contain bool values")
        for value in quality_flags:
            require_strict_int("cadence quality flag", value, minimum=0)

        derived = tuple(coverage_windows(times, usable=usable))
        supplied: list[tuple[float, float]] = []
        for window in self.observing_windows:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ValueError("each observing window must be a [start, end] pair")
            require_finite("observing window start", window[0])
            require_finite("observing window end", window[1])
            if window[1] <= window[0]:
                raise ValueError("observing window end must be after start")
            supplied.append((window[0], window[1]))
        if supplied and tuple(supplied) != derived:
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
            "source_product": (
                None if self.source_product is None else self.source_product.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CadenceEvidence:
        if not isinstance(value, dict):
            raise ValueError("cadence evidence must be a dict")
        required = {
            "time", "usable", "quality_flags", "observing_windows", "source_product"
        }
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
            source_product=(
                None
                if value["source_product"] is None
                else SourceProduct.from_dict(value["source_product"])
            ),
        )


@dataclass(frozen=True)
class SourceProduct:
    """Stable archive identity, excluding machine-local cache paths."""

    provider: str
    product: str
    data_uri: str
    retrieved_utc: str

    def __post_init__(self) -> None:
        for name, value in (
            ("provider", self.provider),
            ("product", self.product),
            ("data_uri", self.data_uri),
            ("retrieved_utc", self.retrieved_utc),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"source product {name} must be a non-empty str")

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "product": self.product,
            "data_uri": self.data_uri,
            "retrieved_utc": self.retrieved_utc,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceProduct:
        if not isinstance(value, dict):
            raise ValueError("source product must be a dict")
        required = {"provider", "product", "data_uri", "retrieved_utc"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"source product missing keys: {missing}")
        extra = sorted(set(value) - required)
        if extra:
            raise ValueError(f"source product unknown keys: {extra}")
        return cls(
            provider=value["provider"],
            product=value["product"],
            data_uri=value["data_uri"],
            retrieved_utc=value["retrieved_utc"],
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
        if len(time) != len(flux):
            raise ValueError("light-curve time and flux must have equal length")
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
