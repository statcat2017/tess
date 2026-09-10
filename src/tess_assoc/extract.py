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
from tess_assoc.observability import (
    CadenceEvidence,
    LightCurve,
    SourceProduct,
)
from tess_assoc.window import samples_in_windows
from tess_assoc._validate import require_positive_finite, require_strict_int
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
    observability: CadenceEvidence | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "t0": self.predicted_t0_btjd,
            "reason": self.reason,
            "observability": (
                None if self.observability is None else self.observability.to_dict()
            ),
        }


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


def load_lightcurve(product: ArchiveProduct) -> LightCurve:
    """Return good samples and raw cadence evidence; PDCSAP preferred."""
    _require_deps()
    import numpy as np
    from astropy.io import fits

    with fits.open(product.local_path) as handle:
        data = handle[1].data
    columns = getattr(getattr(data, "columns", None), "names", None)
    if not isinstance(columns, (list, tuple)):
        raise ValueError("light-curve table has no named columns")
    flux_column = "PDCSAP_FLUX" if "PDCSAP_FLUX" in columns else "SAP_FLUX"
    missing = [name for name in ("TIME", "QUALITY", flux_column) if name not in columns]
    if missing:
        raise ValueError(f"light-curve missing columns: {missing}")
    try:
        raw_time = np.asarray(data["TIME"])
        raw_flux = np.asarray(data[flux_column])
        quality = np.asarray(data["QUALITY"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("light-curve table contains malformed columns") from error
    if raw_time.dtype.kind not in "fiu" or raw_flux.dtype.kind not in "fiu":
        raise ValueError("light-curve TIME and flux columns must be numeric")
    if quality.dtype.kind not in "iu":
        raise ValueError("light-curve QUALITY column must contain integer flags")
    if len(raw_time) != len(raw_flux) or len(raw_time) != len(quality):
        raise ValueError("light-curve columns must have equal length")
    time = raw_time.astype(float, copy=False)
    flux = raw_flux.astype(float, copy=False)
    finite_time = np.isfinite(time)
    invalid_time_count = int((~finite_time).sum())
    time = time[finite_time]
    flux = flux[finite_time]
    quality = quality[finite_time]
    usable = np.isfinite(flux) & (quality == 0)
    # Cast to Python floats: list(np_array) would leak np.float64 scalars,
    # which pass isinstance(x, float) yet poison comparisons into np.bool_.
    quality_values = quality.tolist()
    for value in quality_values:
        require_strict_int("light-curve quality flag", value, minimum=0)
    evidence = CadenceEvidence(
        time=tuple(float(v) for v in time),
        usable=tuple(bool(v) for v in usable),
        quality_flags=tuple(quality_values),
        source_product=SourceProduct(
            provider="MAST",
            product="TESS-SPOC FFI",
            data_uri=product.data_uri,
            retrieved_utc=product.retrieved_utc,
        ),
        invalid_time_count=invalid_time_count,
    )
    return LightCurve(
        time=tuple(float(v) for v in time[usable]),
        flux=tuple(float(v) for v in flux[usable]),
        evidence=evidence,
    )


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
    observability: CadenceEvidence | None = None,
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
        return SkippedTransit(t_center, "window truncated at data edge", observability)
    interp = np.interp(grid, tarr, np.array(flux, dtype=float))
    half = duration_days / 2.0
    inside = np.abs(np.array(phases)) <= half
    outside = (np.abs(np.array(phases)) > duration_days) & (
        np.abs(np.array(phases)) <= half_span_days
    )
    if int(np.sum(inside)) < 3 or int(np.sum(outside)) < 10:
        return SkippedTransit(t_center, "too few points in/out of transit", observability)
    f0 = float(np.median(interp[outside]))
    depth = 1.0 - float(np.median(interp[inside])) / f0
    if not depth > 0:
        return SkippedTransit(t_center, "non-positive measured depth", observability)
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
        observability=observability,
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
    curve = load_lightcurve(product)
    time, flux = list(curve.time), list(curve.flux)
    windows = list(curve.evidence.observing_windows)
    if not curve.evidence.time:
        raise ArchiveUnavailable(f"no finite cadences in {product.local_path}")
    if not time:
        skipped = [
            SkippedTransit(t_pred, "no usable cadence", curve.evidence)
            for t_pred in predicted_transits(
                system.t0_bjd_tdb,
                period,
                curve.evidence.time[0],
                curve.evidence.time[-1],
            )
        ]
        return [], skipped, windows
    quality_base = {
        "source_product": (
            None
            if curve.evidence.source_product is None
            else curve.evidence.source_product.to_dict()
        ),
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
                SkippedTransit(
                    t_pred,
                    "epoch refinement found no usable cadence",
                    curve.evidence,
                )
            )
            continue
        if not any(start <= t_ref <= end for start, end in windows):
            skipped.append(
                SkippedTransit(
                    t_pred,
                    "refined epoch outside observing window",
                    curve.evidence,
                )
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
            observability=curve.evidence,
        )
        if isinstance(result, SkippedTransit):
            skipped.append(SkippedTransit(t_pred, result.reason, curve.evidence))
        elif not samples_in_windows(result.local_time, windows):
            skipped.append(
                SkippedTransit(
                    t_pred,
                    "insufficient full observing window coverage",
                    curve.evidence,
                )
            )
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
