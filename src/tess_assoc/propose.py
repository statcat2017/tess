"""High-recall blind event proposer (issue #4).

Detrended local-dip detection over raw light curves. Takes only
(time, flux) — no orbital period, no ephemeris. Deliberately impure:
the matcher and window filter downstream do the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

from tess_assoc._validate import require_finite, require_positive_finite
from tess_assoc.event import EventRecord
from tess_assoc.extract import SkippedTransit, extract_at
from tess_assoc.observability import CadenceEvidence, coverage_windows
from tess_assoc.window import samples_in_windows


PROPOSER_SNR_THRESHOLD = 4.0


@dataclass(frozen=True)
class Proposal:
    t0_guess: float
    depth_guess: float
    duration_guess_days: float
    snr_guess: float
    n_points: int


def detrend(
    time: list[float],
    flux: list[float],
    trend_span_days: float = 1.5,
    *,
    observing_windows: list[tuple[float, float]] | None = None,
) -> tuple[list[float], float]:
    """Divide out a rolling-median trend; return (detrended, robust sigma)."""
    require_positive_finite("trend_span_days", trend_span_days)
    if len(time) != len(flux) or not time:
        raise ValueError("time and flux must be non-empty and equal length")
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    tarr = np.array(time, dtype=float)
    farr = np.array(flux, dtype=float)
    windows = coverage_windows(time) if observing_windows is None else observing_windows
    if not windows:
        windows = [(time[0], time[-1])]
    detrended = np.ones(len(time), dtype=float)
    for start, end in windows:
        indices = [i for i, value in enumerate(time) if start <= value <= end]
        if not indices:
            continue
        segment_flux = farr[indices]
        if len(indices) < 3:
            trend = segment_flux
        else:
            cadence = float(np.median(np.diff(tarr[indices])))
            width = max(int(round(trend_span_days / cadence)) | 1, 3)
            width = min(width, len(indices) if len(indices) % 2 else len(indices) - 1)
            pad = width // 2
            padded = np.pad(segment_flux, pad, mode="edge")
            trend = np.median(sliding_window_view(padded, width), axis=1)
        detrended[indices] = segment_flux / trend
    sigma = float(1.4826 * np.median(np.abs(detrended - 1.0))) or 1e-9
    return [float(v) for v in detrended], sigma


def _window_index(
    value: float, windows: list[tuple[float, float]]
) -> int | None:
    for index, (start, end) in enumerate(windows):
        if start <= value <= end:
            return index
    return None


def center_on_minimum(
    time: list[float], flux: list[float], t_guess: float, radius_days: float
) -> float:
    """Time of minimum flux within radius of the guess (data-located center)."""
    require_positive_finite("radius_days", radius_days)
    best_t, best_f = t_guess, float("inf")
    for t, f in zip(time, flux):
        if abs(t - t_guess) <= radius_days and f < best_f:
            best_t, best_f = t, f
    return best_t


def find_dips(
    time: list[float],
    detrended: list[float],
    sigma: float,
    snr_threshold: float = 4.0,
    min_points: int = 2,
    merge_gap_points: int = 2,
    min_duration_days: float = 0.02,
    max_duration_days: float = 0.6,
    *,
    observing_windows: list[tuple[float, float]] | None = None,
) -> list[Proposal]:
    """Contiguous above-threshold runs → dip proposals (period-free)."""
    require_finite("snr_threshold", snr_threshold)
    if snr_threshold <= 0:
        raise ValueError("snr_threshold must be > 0")
    if len(time) != len(detrended) or not time:
        raise ValueError("time and detrended must be non-empty and equal length")
    windows = coverage_windows(time) if observing_windows is None else observing_windows
    above = [(1.0 - f) / sigma > snr_threshold for f in detrended]
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    segment: int | None = None
    for i, flag in enumerate(above):
        current_segment = _window_index(time[i], windows)
        if current_segment is None or current_segment != segment:
            if start is not None and segment is not None:
                runs.append((start, i - 1, segment))
            start = None
            segment = current_segment
        if current_segment is None:
            continue
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1, segment))
            start = None
    if start is not None:
        runs.append((start, len(above) - 1, segment))
    merged: list[list[int]] = []
    merged_segments: list[int] = []
    for s, e, run_segment in runs:
        if (
            merged
            and merged_segments[-1] == run_segment
            and s - merged[-1][1] - 1 <= merge_gap_points
        ):
            merged[-1][1] = e
        else:
            merged.append([s, e])
            merged_segments.append(run_segment)
    proposals: list[Proposal] = []
    for s, e in merged:
        if e - s + 1 < min_points:
            continue
        duration = time[e] - time[s]
        if not min_duration_days <= duration <= max_duration_days:
            continue
        window = detrended[s : e + 1]
        depth = 1.0 - min(window)
        if not depth > 0:
            continue
        t0_guess = time[s + window.index(min(window))]
        proposals.append(
            Proposal(
                t0_guess=t0_guess,
                depth_guess=depth,
                duration_guess_days=duration,
                snr_guess=depth / sigma * ((e - s + 1) ** 0.5),
                n_points=e - s + 1,
            )
        )
    return proposals


def dip_snr_at(
    time: list[float],
    detrended: list[float],
    sigma: float,
    t_center: float,
    half_width_days: float,
) -> float:
    """Strongest dip signal within half_width of t_center (may be negative)."""
    require_finite("t_center", t_center)
    require_positive_finite("half_width_days", half_width_days)
    if len(time) != len(detrended):
        raise ValueError("time and detrended must be equal length")
    if not sigma > 0:
        raise ValueError("sigma must be > 0")
    best = float("-inf")
    for t, f in zip(time, detrended):
        if abs(t - t_center) <= half_width_days:
            snr = (1.0 - f) / sigma
            if snr > best:
                best = snr
    if best == float("-inf"):
        raise ValueError("no cadences within half_width of t_center")
    return best


def propose_with_detail(
    time: list[float],
    flux: list[float],
    snr_threshold: float = PROPOSER_SNR_THRESHOLD,
    *,
    observability: CadenceEvidence | None = None,
) -> tuple[list[Proposal], list[float], float]:
    """Blind proposals plus the detrended curve and sigma behind them."""
    windows = None if observability is None else list(observability.observing_windows)
    detrended, sigma = detrend(time, flux, observing_windows=windows)
    proposals = find_dips(
        time,
        detrended,
        sigma,
        snr_threshold=snr_threshold,
        observing_windows=windows,
    )
    return proposals, detrended, sigma


def propose_events(
    time: list[float],
    flux: list[float],
    snr_threshold: float = PROPOSER_SNR_THRESHOLD,
    *,
    observability: CadenceEvidence | None = None,
) -> list[Proposal]:
    """Blind proposals from raw light curves. No period, no ephemeris."""
    proposals, _, _ = propose_with_detail(
        time, flux, snr_threshold, observability=observability
    )
    return proposals


def records_from_proposals(
    time: list[float],
    flux: list[float],
    proposals: list[Proposal],
    *,
    tic_id: int,
    sector: int,
    half_span_days: float = 0.6,
    resample_samples: int = 61,
    quality_base: dict | None = None,
    observing_windows: list[tuple[float, float]] | None = None,
    observability: CadenceEvidence | None = None,
) -> tuple[dict[str, EventRecord], list[SkippedTransit]]:
    """Measure proposal windows through the shared extract_at core."""
    records: dict[str, EventRecord] = {}
    skipped: list[SkippedTransit] = []
    base = dict(quality_base or {})
    if observing_windows is None:
        windows = (
            list(observability.observing_windows)
            if observability is not None
            else coverage_windows(time)
        )
    else:
        windows = observing_windows
    for i, p in enumerate(proposals):
        t_center = center_on_minimum(time, flux, p.t0_guess, p.duration_guess_days)
        result = extract_at(
            time,
            flux,
            t_center,
            p.duration_guess_days,
            tic_id=tic_id,
            sector=sector,
            half_span_days=half_span_days,
            resample_samples=resample_samples,
            observability=observability,
            quality={
                **base,
                "role": "blind-proposal",
                "proposal_t0_guess": p.t0_guess,
                "proposal_snr_guess": p.snr_guess,
            },
        )
        if isinstance(result, SkippedTransit):
            skipped.append(result)
        elif not samples_in_windows(result.local_time, windows):
            skipped.append(
                SkippedTransit(
                    p.t0_guess,
                    "insufficient full observing window coverage",
                    observability,
                )
            )
        else:
            records[f"S{sector}-{i:03d}"] = result
    return records, skipped
