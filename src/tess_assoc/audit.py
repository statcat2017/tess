"""Evidence measurements for single-event follow-up candidates."""

from __future__ import annotations

import math
import statistics
from html import escape
from pathlib import Path
from typing import Any, Sequence

from tess_assoc._validate import require_finite, require_positive_finite


def _masks(
    time: Sequence[float], t0: float, duration_days: float, half_span_days: float
) -> tuple[list[bool], list[bool]]:
    require_finite("t0", t0)
    require_positive_finite("duration_days", duration_days)
    require_positive_finite("half_span_days", half_span_days)
    if half_span_days <= duration_days:
        raise ValueError("half_span_days must exceed duration_days")
    inside = [abs(t - t0) <= duration_days / 2 for t in time]
    outside = [duration_days < abs(t - t0) <= half_span_days for t in time]
    return inside, outside


def _finite(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _robust_sigma(values: Sequence[float]) -> float:
    values = _finite(values)
    if not values:
        return float("nan")
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    return 1.4826 * mad or 1e-12


def measure_flux_channel(
    time: Sequence[float],
    flux: Sequence[float],
    t0: float,
    duration_days: float,
    *,
    half_span_days: float = 0.6,
) -> dict[str, Any]:
    """Measure a local event from one flux channel without searching for events."""
    if len(time) != len(flux):
        raise ValueError("time and flux must have equal length")
    inside, outside = _masks(time, t0, duration_days, half_span_days)
    in_flux = _finite([value for value, keep in zip(flux, inside) if keep])
    out_flux = _finite([value for value, keep in zip(flux, outside) if keep])
    if len(in_flux) < 3 or len(out_flux) < 10:
        return {"status": "insufficient-data", "n_inside": len(in_flux), "n_outside": len(out_flux)}
    baseline = statistics.median(out_flux)
    if not math.isfinite(baseline) or baseline <= 0:
        return {"status": "invalid-baseline"}
    normalized_out = [value / baseline for value in out_flux]
    depth = 1.0 - statistics.median(in_flux) / baseline
    sigma = _robust_sigma([value - 1.0 for value in normalized_out])
    snr = depth / sigma * math.sqrt(len(in_flux))
    return {
        "status": "measured",
        "n_inside": len(in_flux),
        "n_outside": len(out_flux),
        "baseline": baseline,
        "depth": depth,
        "sigma": sigma,
        "snr": snr,
    }


def audit_flux_channels(
    time: Sequence[float],
    channels: dict[str, Sequence[float]],
    t0: float,
    duration_days: float,
    *,
    half_span_days: float = 0.6,
) -> dict[str, Any]:
    """Compare SAP and PDCSAP evidence for the already-selected event."""
    measured = {
        name: measure_flux_channel(
            time, flux, t0, duration_days, half_span_days=half_span_days
        )
        for name, flux in channels.items()
    }
    usable = [result for result in measured.values() if result.get("status") == "measured"]
    stable = None
    if len(usable) >= 2:
        depths = [float(result["depth"]) for result in usable]
        reference = max(abs(depths[0]), 1e-12)
        stable = abs(depths[0] - depths[1]) / reference <= 0.25
    return {"channels": measured, "stable": stable}


def measure_event_shape(
    time: Sequence[float],
    flux: Sequence[float],
    t0: float,
    duration_days: float,
    *,
    half_span_days: float = 0.6,
) -> dict[str, Any]:
    """Describe flat-bottom versus V-shaped structure in the selected event."""
    if len(time) != len(flux):
        raise ValueError("time and flux must have equal length")
    inside, outside = _masks(time, t0, duration_days, half_span_days)
    baseline_values = _finite([value for value, keep in zip(flux, outside) if keep])
    if len(baseline_values) < 10:
        return {"status": "insufficient-data"}
    baseline = statistics.median(baseline_values)
    if baseline <= 0:
        return {"status": "invalid-baseline"}
    center = [
        float(value) / baseline
        for value, t in zip(flux, time)
        if abs(t - t0) <= duration_days / 4 and math.isfinite(float(value))
    ]
    left = [
        float(value) / baseline
        for value, t in zip(flux, time)
        if -duration_days / 2 <= t - t0 < -duration_days / 4 and math.isfinite(float(value))
    ]
    right = [
        float(value) / baseline
        for value, t in zip(flux, time)
        if duration_days / 4 < t - t0 <= duration_days / 2 and math.isfinite(float(value))
    ]
    edge = left + right
    if len(center) < 3 or len(edge) < 3:
        return {"status": "insufficient-event-shape", "n_center": len(center), "n_edge": len(edge)}
    center_depth = 1.0 - statistics.median(center)
    edge_depth = 1.0 - statistics.median(edge)
    ratio = edge_depth / center_depth if center_depth > 0 else None
    return {
        "status": "measured",
        "n_center": len(center),
        "n_edge": len(edge),
        "center_depth": center_depth,
        "edge_depth": edge_depth,
        "edge_to_center_depth": ratio,
        "left_right_asymmetry": (
            abs(statistics.median(left) - statistics.median(right))
            if left and right else None
        ),
        "shape": (
            "flat-bottomed" if ratio is not None and ratio >= 0.5 else "V-shaped"
        ),
    }


def measure_centroid(
    time: Sequence[float],
    x: Sequence[float],
    y: Sequence[float],
    t0: float,
    duration_days: float,
    *,
    half_span_days: float = 0.6,
) -> dict[str, Any]:
    """Measure in-event versus out-of-event centroid displacement."""
    if not (len(time) == len(x) == len(y)):
        raise ValueError("centroid arrays must have equal length")
    inside, outside = _masks(time, t0, duration_days, half_span_days)
    in_points = [
        (float(px), float(py))
        for px, py, keep in zip(x, y, inside)
        if keep and math.isfinite(float(px)) and math.isfinite(float(py))
    ]
    out_points = [
        (float(px), float(py))
        for px, py, keep in zip(x, y, outside)
        if keep and math.isfinite(float(px)) and math.isfinite(float(py))
    ]
    if len(in_points) < 3 or len(out_points) < 10:
        return {"status": "insufficient-data", "n_inside": len(in_points), "n_outside": len(out_points)}
    in_x = statistics.median(point[0] for point in in_points)
    in_y = statistics.median(point[1] for point in in_points)
    out_x = statistics.median(point[0] for point in out_points)
    out_y = statistics.median(point[1] for point in out_points)
    # Do not divide by sqrt(N): spacecraft pointing and aperture systematics
    # are correlated, so the observed centroid scatter is the honest floor.
    uncertainty_x = math.hypot(
        _robust_sigma([point[0] for point in out_points]),
        _robust_sigma([point[0] for point in in_points]),
    )
    uncertainty_y = math.hypot(
        _robust_sigma([point[1] for point in out_points]),
        _robust_sigma([point[1] for point in in_points]),
    )
    shift = math.hypot(in_x - out_x, in_y - out_y)
    uncertainty = math.hypot(uncertainty_x, uncertainty_y)
    significance = shift / uncertainty if uncertainty > 0 else float("inf")
    return {
        "status": "measured",
        "n_inside": len(in_points),
        "n_outside": len(out_points),
        "in_event": [in_x, in_y],
        "out_event": [out_x, out_y],
        "shift_pixels": shift,
        "uncertainty_pixels": uncertainty,
        "significance_sigma": significance,
        "shift_status": "significant" if significance >= 3 else "not-significant",
    }


def audit_lightcurve_file(
    path: str | Path,
    *,
    sector: int,
    t0: float,
    duration_days: float,
    half_span_days: float = 0.6,
) -> dict[str, Any]:
    """Audit one cached TESS-SPOC light-curve FITS file."""
    try:
        import numpy as np
        from astropy.io import fits
    except ImportError as e:
        raise RuntimeError("single-event audit needs the replay extra") from e
    with fits.open(path) as handle:
        data = handle[1].data
        names = set(data.columns.names)
        time = np.asarray(data["TIME"], dtype=float)
        quality = np.asarray(data["QUALITY"], dtype=int)
        good = np.isfinite(time) & (quality == 0)
        time = time[good]
        channels = {
            name: np.asarray(data[name], dtype=float)[good]
            for name in ("SAP_FLUX", "PDCSAP_FLUX")
            if name in names
        }
        flux_audit = audit_flux_channels(
            time.tolist(),
            {name: values.tolist() for name, values in channels.items()},
            t0,
            duration_days,
            half_span_days=half_span_days,
        )
        shape = (
            measure_event_shape(
                time.tolist(), channels["PDCSAP_FLUX"].tolist(),
                t0, duration_days, half_span_days=half_span_days,
            )
            if "PDCSAP_FLUX" in channels else {"status": "unavailable"}
        )
        centroids = {}
        for prefix in ("PSF", "MOM"):
            x_name, y_name = f"{prefix}_CENTR1", f"{prefix}_CENTR2"
            if x_name in names and y_name in names:
                centroids[prefix.lower()] = measure_centroid(
                    time.tolist(),
                    np.asarray(data[x_name], dtype=float)[good].tolist(),
                    np.asarray(data[y_name], dtype=float)[good].tolist(),
                    t0,
                    duration_days,
                    half_span_days=half_span_days,
                )
    return {
        "sector": int(sector),
        # Keep reports portable; raw archive files remain outside the repo.
        "path": Path(path).name,
        "cadences": int(len(time)),
        "time_range": [float(time[0]), float(time[-1])] if len(time) else None,
        "flux": flux_audit,
        "shape": shape,
        "centroid": centroids,
    }


def build_single_event_audit(
    tic_id: int,
    event: dict[str, Any],
    products: Sequence[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
    catalogs: dict[str, Any] | None = None,
    neighbors: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Combine light-curve measurements and external checks into an audit."""
    if int(tic_id) < 1:
        raise ValueError("tic_id must be positive")
    sector = int(event["sector"])
    t0 = float(event["t0"])
    duration = float(event["duration_days"])
    event_products = [p for p in products if int(p["sector"]) == sector]
    if not event_products:
        raise ValueError(f"no product supplied for event sector {sector}")
    curves = [
        audit_lightcurve_file(
            p["local_path"], sector=sector, t0=t0, duration_days=duration
        )
        for p in event_products
    ]
    curve = curves[0]
    flux_stable = curve["flux"].get("stable")
    centroid_statuses = [
        result.get("shift_status")
        for result in curve["centroid"].values()
        if result.get("status") == "measured"
    ]
    centroid_status = (
        "significant" if "significant" in centroid_statuses
        else "not-significant" if centroid_statuses
        else "inconclusive"
    )
    metadata = dict(metadata or {})
    catalogs = dict(catalogs or {})
    neighbors = [dict(neighbor) for neighbor in (neighbors or [])]
    target_tmag = metadata.get("tmag")
    bright_neighbors = []
    for neighbor in neighbors:
        try:
            separation = float(neighbor["separation_arcsec"])
            g_mag = float(neighbor["g_mag"])
            if target_tmag is not None and separation <= 30 and g_mag <= float(target_tmag) + 3:
                bright_neighbors.append(neighbor)
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "audit_version": 1,
        "tic_id": int(tic_id),
        "event": dict(event),
        "metadata": metadata,
        "catalogs": catalogs,
        "available_sectors": metadata.get("available_sectors", []),
        "event_product": curve,
        "gaia_neighbors": neighbors,
        "checks": {
            "sap_pdcsap": (
                "pass" if flux_stable is True
                else "fail" if flux_stable is False
                else "inconclusive"
            ),
            "centroid_shift": centroid_status,
            "bright_gaia_neighbor": "review" if bright_neighbors else "none-found",
            "transit_shape": curve["shape"].get("shape", "inconclusive"),
            "period": "unconstrained",
            "pixel_difference_image": "required",
            "radial_velocity": "not-obtained",
            "ground_multicolor": "not-obtained",
        },
    }


def render_single_event_audit(reports: Sequence[dict[str, Any]]) -> str:
    """Render a compact standalone HTML evidence report."""
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>TESS single-event audit</title>",
        "<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#17213a}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #ccd3e0;padding:.5rem;text-align:left}"
        ".warn{background:#fff2d6}.pass{background:#e7f7ed}.muted{color:#56627a}img{max-width:680px;width:100%}</style>",
        "</head><body><h1>Single-event evidence audit</h1>",
        '<p class="muted">Measurements establish robustness and source-localisation evidence; they do not establish an orbital period or confirm a planet.</p>',
    ]
    for report in reports:
        event = report["event"]
        curve = report["event_product"]
        pdcsap = curve["flux"]["channels"].get("PDCSAP_FLUX", {})
        parts.append(f"<section><h2>TIC {report['tic_id']}</h2>")
        parts.append(
            f"<p>Sector {event['sector']} at BTJD {event['t0']:.5f}; "
            f"depth {event.get('depth', 0):.4f}; duration {event['duration_days'] * 24:.2f} h; "
            f"SNR {event.get('snr', 0):.1f}. Period: <b>unconstrained</b>.</p>"
        )
        parts.append("<h3>Measurements</h3><table><tr><th>Evidence</th><th>Result</th></tr>")
        for name, result in curve["flux"]["channels"].items():
            depth = result.get("depth")
            parts.append(
                f"<tr><td>{escape(name)}</td><td>depth "
                f"{depth:.5f} · SNR {result.get('snr', float('nan')):.1f} · "
                f"{result.get('n_inside', 0)} in / {result.get('n_outside', 0)} out points</td></tr>"
            )
        parts.append(
            f"<tr><td>Reduction consistency</td><td>{report['checks']['sap_pdcsap']}</td></tr>"
            f"<tr><td>PDCSAP shape</td><td>{report['checks']['transit_shape']}</td></tr>"
            f"<tr><td>Centroid shift</td><td>{report['checks']['centroid_shift']}</td></tr>"
            f"<tr><td>Bright Gaia neighbour</td><td>{report['checks']['bright_gaia_neighbor']}</td></tr>"
            "</table>"
        )
        parts.append("<h3>False-positive checklist</h3><table><tr><th>Check</th><th>Status</th></tr>")
        for check, status in report["checks"].items():
            klass = "pass" if status in ("pass", "none-found", "not-significant") else "warn"
            parts.append(f'<tr class="{klass}"><td>{escape(check)}</td><td>{escape(str(status))}</td></tr>')
        parts.append("</table>")
        parts.append(f'<img src="../assets/single_{report["tic_id"]}_s{event["sector"]}.svg" alt="single event">')
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


__all__ = [
    "audit_flux_channels",
    "audit_lightcurve_file",
    "build_single_event_audit",
    "measure_centroid",
    "measure_event_shape",
    "measure_flux_channel",
    "render_single_event_audit",
]
