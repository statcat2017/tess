"""Real-data event extraction (issue #3).

Enumerates catalog-predicted transits of a known planet inside downloaded
TESS-SPOC FFI light curves and builds EventRecords on a fixed resampled
phase grid — the same contract as fixtures, with real morphology.
This is a replay of known systems (labels declared), not discovery.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tess_assoc.archive import ArchiveProduct, ArchiveUnavailable
from tess_assoc.event import EventRecord
from tess_assoc._validate import require_positive_finite
from tess_assoc.manifest import ReplaySystem

BTJD_OFFSET = 2457000.0


@dataclass(frozen=True)
class ExtractedEvent:
    record: EventRecord
    predicted_t0_btjd: float
    measured_depth: float
    n_points: int


@dataclass(frozen=True)
class SkippedTransit:
    predicted_t0_btjd: float
    reason: str


def _require_deps() -> None:
    try:
        import astropy  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        raise ArchiveUnavailable(
            "replay needs the 'replay' extra: pip install tess-assoc[replay] "
            f"({e})"
        ) from e


def predicted_transits(
    t0_bjd_tdb: float, period_days: float, tmin_btjd: float, tmax_btjd: float
) -> list[float]:
    """All ephemeris transit times (BTJD) within [tmin, tmax]."""
    if period_days <= 0:
        raise ValueError("period_days must be > 0")
    t0 = t0_bjd_tdb - BTJD_OFFSET
    k_min = math.ceil((tmin_btjd - t0) / period_days)
    k_max = math.floor((tmax_btjd - t0) / period_days)
    return [t0 + k * period_days for k in range(k_min, k_max + 1)]


def load_lightcurve(product: ArchiveProduct) -> tuple[list[float], list[float]]:
    """Return good-cadence (TIME, flux) lists; PDCSAP preferred, SAP fallback."""
    _require_deps()
    import numpy as np
    from astropy.io import fits

    data = None
    with fits.open(product.local_path) as handle:
        data = handle[1].data
    time = np.asarray(data["TIME"], dtype=float)
    flux = np.asarray(
        data["PDCSAP_FLUX"] if "PDCSAP_FLUX" in data.columns.names else data["SAP_FLUX"],
        dtype=float,
    )
    quality = np.asarray(data["QUALITY"])
    good = np.isfinite(time) & np.isfinite(flux) & (quality == 0)
    # Cast to Python floats: list(np_array) would leak np.float64 scalars,
    # which pass isinstance(x, float) yet poison comparisons into np.bool_.
    return [float(v) for v in time[good]], [float(v) for v in flux[good]]


def _phase_distance(
    times: Sequence[float], center: float, period_days: float
) -> list[float]:
    """Signed distance of each time to center, folded into [-P/2, P/2)."""
    half = period_days / 2.0
    return [((t - center) % period_days + half) % period_days - half for t in times]


def refine_epoch(
    time: list[float],
    flux: list[float],
    period_days: float,
    t0_guess_btjd: float,
    duration_days: float,
) -> float:
    """Shift the predicted epoch to the local flux minimum.

    Catalog T0 uncertainties accumulate over hundreds of orbits, so the
    predicted phase can sit aperture-widths off the real dip. The shift is
    bounded to ±15% of the period (no cycle ambiguity) on a duration/8 grid
    and recorded in provenance — deterministic, no fitting beyond one shift.
    """
    import numpy as np

    tarr, farr = np.array(time), np.array(flux)
    half = duration_days / 2.0
    span = 0.15 * period_days
    step = max(duration_days / 8.0, 1e-4)
    grid = np.arange(-span, span + step / 2.0, step)
    best_shift, best_level = 0.0, float("inf")
    qualified = False
    for shift in grid:
        inside = np.abs(_phase_distance(tarr, t0_guess_btjd + shift, period_days)) <= half
        if int(np.sum(inside)) < 5:
            continue
        qualified = True
        level = float(np.median(farr[inside]))
        if level < best_level:
            best_level, best_shift = level, float(shift)
    if not qualified:
        raise ArchiveUnavailable(
            f"epoch refinement found no usable cadence near {t0_guess_btjd}"
        )
    return t0_guess_btjd + best_shift


def coverage_windows(
    time: list[float], max_gap_days: float = 0.5
) -> list[tuple[float, float]]:
    """Contiguous observed spans; splits on gaps (real window function).

    Degenerate zero-width spans (isolated single cadences) are dropped:
    no transit window fits inside them, and downstream window checks
    treat every span as observable.
    """
    if not time:
        return []
    spans: list[tuple[float, float]] = []
    start = prev = time[0]
    for t in time[1:]:
        if t - prev > max_gap_days:
            if prev > start:
                spans.append((start, prev))
            start = t
        prev = t
    if prev > start:
        spans.append((start, prev))
    return spans


def extract_at(
    time: list[float],
    flux: list[float],
    t_center: float,
    duration_days: float,
    *,
    tic_id: int,
    sector: int,
    half_span_days: float = 0.6,
    resample_samples: int = 61,
    quality: dict | None = None,
) -> EventRecord | SkippedTransit:
    """Measure one window into an EventRecord (shared ephemeris/blind core)."""
    _require_deps()
    import numpy as np

    require_positive_finite("duration_days", duration_days)
    tarr = np.array(time, dtype=float)
    step = (2.0 * half_span_days) / (resample_samples - 1)
    phases = [-half_span_days + i * step for i in range(resample_samples)]
    grid = [t_center + ph for ph in phases]
    if grid[0] < tarr[0] or grid[-1] > tarr[-1]:
        return SkippedTransit(t_center, "window truncated at data edge")
    interp = np.interp(grid, tarr, np.array(flux, dtype=float))
    half = duration_days / 2.0
    inside = np.abs(np.array(phases)) <= half
    outside = (np.abs(np.array(phases)) > duration_days) & (
        np.abs(np.array(phases)) <= half_span_days
    )
    if int(np.sum(inside)) < 3 or int(np.sum(outside)) < 10:
        return SkippedTransit(t_center, "too few points in/out of transit")
    f0 = float(np.median(interp[outside]))
    depth = 1.0 - float(np.median(interp[inside])) / f0
    if not depth > 0:
        return SkippedTransit(t_center, "non-positive measured depth")
    resid = interp[outside] / f0 - 1.0
    scatter = float(np.std(resid)) or 1e-9
    snr = depth / scatter * (float(np.sum(inside)) ** 0.5)
    return EventRecord(
        tic_id=tic_id,
        sector=sector,
        t0=float(t_center),
        local_time=[float(v) for v in grid],
        local_flux=[float(v) for v in interp / f0],
        depth=depth,
        duration_days=duration_days,
        snr=snr,
        stellar_meta={},
        quality=dict(quality or {}),
    )
def extract_events(
    product: ArchiveProduct,
    system: ReplaySystem,
    half_span_days: float = 0.6,
    resample_samples: int = 61,
) -> tuple[list[ExtractedEvent], list[SkippedTransit], list[tuple[float, float]]]:
    """Extract one EventRecord per predicted transit with full window coverage."""
    _require_deps()

    period = system.period_days
    duration_days = system.duration_hours / 24.0
    time, flux = load_lightcurve(product)
    if not time:
        raise ArchiveUnavailable(f"no good cadences in {product.local_path}")
    windows = coverage_windows(time)
    quality_base = {
        "provider": "archive",
        "product": "TESS-SPOC FFI",
        "data_uri": product.data_uri,
        "retrieved_utc": product.retrieved_utc,
        "ephemeris": f"{system.name} P={period}d T0={system.t0_bjd_tdb}",
        "role": "predicted-transit",
    }

    extracted: list[ExtractedEvent] = []
    skipped: list[SkippedTransit] = []
    for t_pred in predicted_transits(system.t0_bjd_tdb, period, time[0], time[-1]):
        try:
            t_ref = refine_epoch(time, flux, period, t_pred, duration_days)
        except ArchiveUnavailable:
            skipped.append(
                SkippedTransit(t_pred, "epoch refinement found no usable cadence")
            )
            continue
        result = extract_at(
            time,
            flux,
            t_ref,
            duration_days,
            tic_id=product.tic_id,
            sector=product.sector,
            half_span_days=half_span_days,
            resample_samples=resample_samples,
            quality={
                **quality_base,
                "predicted_t0_btjd": t_pred,
                "epoch_shift_days": t_ref - t_pred,
            },
        )
        if isinstance(result, SkippedTransit):
            skipped.append(SkippedTransit(t_pred, result.reason))
        else:
            extracted.append(
                ExtractedEvent(
                    record=result,
                    predicted_t0_btjd=t_pred,
                    measured_depth=result.depth,
                    n_points=resample_samples,
                )
            )
    return extracted, skipped, windows
