"""Pixel-level TESSCut difference-image evidence."""

from __future__ import annotations

import math
import statistics
from html import escape
from pathlib import Path
from typing import Any, Sequence

from tess_assoc.audit import measure_flux_channel
from tess_assoc._validate import require_finite, require_positive_finite


def _event_masks(
    time: Sequence[float], t0: float, duration_days: float, half_span_days: float
) -> tuple[list[bool], list[bool]]:
    require_finite("t0", t0)
    require_positive_finite("duration_days", duration_days)
    require_positive_finite("half_span_days", half_span_days)
    if half_span_days <= duration_days:
        raise ValueError("half_span_days must exceed duration_days")
    return (
        [abs(t - t0) <= duration_days / 2 for t in time],
        [duration_days < abs(t - t0) <= half_span_days for t in time],
    )


def difference_image(
    time: Sequence[float],
    flux_cube: Sequence[Sequence[Sequence[float]]],
    t0: float,
    duration_days: float,
    *,
    quality: Sequence[int] | None = None,
    half_span_days: float = 0.6,
) -> dict[str, Any]:
    """Return out-of-event minus in-event median pixels for one event."""
    import numpy as np

    cube = np.asarray(flux_cube, dtype=float)
    if cube.ndim != 3 or cube.shape[0] != len(time):
        raise ValueError("flux_cube must have shape (time, y, x)")
    good = np.isfinite(np.asarray(time, dtype=float))
    if quality is not None:
        if len(quality) != len(time):
            raise ValueError("quality and time must have equal length")
        good &= np.asarray(quality) == 0
    inside, outside = _event_masks(time, t0, duration_days, half_span_days)
    in_mask = good & np.asarray(inside)
    out_mask = good & np.asarray(outside)
    in_cube = cube[in_mask]
    out_cube = cube[out_mask]
    if len(in_cube) < 3 or len(out_cube) < 10:
        return {
            "status": "insufficient-data",
            "n_inside": int(len(in_cube)),
            "n_outside": int(len(out_cube)),
            "n_inside_all": int(np.asarray(inside).sum()),
            "n_outside_all": int(np.asarray(outside).sum()),
        }
    in_median = np.nanmedian(in_cube, axis=0)
    out_median = np.nanmedian(out_cube, axis=0)
    diff = out_median - in_median
    if not np.isfinite(diff).all():
        return {"status": "non-finite-difference"}
    return {
        "status": "measured",
        "n_inside": int(len(in_cube)),
        "n_outside": int(len(out_cube)),
        "n_inside_all": int(np.asarray(inside).sum()),
        "n_outside_all": int(np.asarray(outside).sum()),
        "difference_image": diff.tolist(),
    }


def difference_centroid(
    image: Sequence[Sequence[float]], target_pixel: tuple[float, float]
) -> dict[str, Any]:
    """Flux-weight the positive difference image and compare with target pixel."""
    import numpy as np

    diff = np.asarray(image, dtype=float)
    if diff.ndim != 2 or not np.isfinite(diff).all():
        raise ValueError("image must be a finite 2-D array")
    background = float(np.median(diff))
    signal = np.clip(diff - background, 0.0, None)
    total = float(signal.sum())
    if total <= 0:
        return {"status": "no-positive-signal"}
    yy, xx = np.indices(diff.shape, dtype=float)
    centroid = (float((signal * xx).sum() / total), float((signal * yy).sum() / total))
    offset = math.hypot(centroid[0] - target_pixel[0], centroid[1] - target_pixel[1])
    return {
        "status": "measured",
        "target_pixel": [float(target_pixel[0]), float(target_pixel[1])],
        "difference_centroid": list(centroid),
        "offset_pixels": offset,
        "signal_sum": total,
    }


def aperture_depths(
    time: Sequence[float],
    flux_cube: Sequence[Sequence[Sequence[float]]],
    target_pixel: tuple[float, float],
    t0: float,
    duration_days: float,
    *,
    quality: Sequence[int] | None = None,
    half_span_days: float = 0.6,
    half_widths: Sequence[int] = (0, 1, 2),
) -> list[dict[str, Any]]:
    """Measure event depth in square apertures centred on the target pixel."""
    import numpy as np

    cube = np.asarray(flux_cube, dtype=float)
    _, out = _event_masks(time, t0, duration_days, half_span_days)
    good = np.isfinite(np.asarray(time, dtype=float))
    if quality is not None:
        good &= np.asarray(quality) == 0
    results = []
    x0, y0 = target_pixel
    for half_width in half_widths:
        x_min = max(0, int(math.floor(x0 - half_width)))
        x_max = min(cube.shape[2], int(math.floor(x0 + half_width + 1)))
        y_min = max(0, int(math.floor(y0 - half_width)))
        y_max = min(cube.shape[1], int(math.floor(y0 + half_width + 1)))
        series = cube[:, y_min:y_max, x_min:x_max].sum(axis=(1, 2))
        result = measure_flux_channel(
            np.asarray(time)[good].tolist(),
            series[good].tolist(),
            t0,
            duration_days,
            half_span_days=half_span_days,
        )
        results.append({"pixels": int((x_max - x_min) * (y_max - y_min)), **result})
    return results


def audit_tesscut_file(
    path: str | Path,
    *,
    ra_deg: float,
    dec_deg: float,
    sector: int,
    t0: float,
    duration_days: float,
    half_span_days: float = 0.6,
) -> dict[str, Any]:
    """Run difference-image and aperture checks on a MAST TESSCut FITS file."""
    try:
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as e:
        raise RuntimeError("pixel audit needs the replay extra") from e
    with fits.open(path) as handle:
        data = handle[1].data
        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data["FLUX"], dtype=float)
        quality = np.asarray(data["QUALITY"], dtype=int)
        wcs = WCS(handle[2].header).celestial
        target_pixel = tuple(float(v) for v in wcs.world_to_pixel_values(ra_deg, dec_deg))
    difference = difference_image(
        time.tolist(), flux, t0, duration_days,
        quality=quality.tolist(), half_span_days=half_span_days,
    )
    result: dict[str, Any] = {
        "path": Path(path).name,
        "sector": int(sector),
        "cadences": int(len(time)),
        "time_range": [float(time[0]), float(time[-1])] if len(time) else None,
        "target_pixel": list(target_pixel),
        "difference": difference,
    }
    if difference["status"] == "measured":
        result["centroid"] = difference_centroid(
            difference["difference_image"], target_pixel
        )
        result["apertures"] = aperture_depths(
            time.tolist(), flux, target_pixel, t0, duration_days,
            quality=quality.tolist(), half_span_days=half_span_days,
        )
    return result


def download_tesscut(
    ra_deg: float,
    dec_deg: float,
    sector: int,
    *,
    size: int = 9,
    directory: str | Path = "/tmp/tesscut_cache",
    name: str | None = None,
) -> Path:
    """Download or reuse an arbitrary-target MAST TESSCut FFI cutout."""
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Tesscut

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    filename = name or f"tesscut_s{int(sector):04d}_{ra_deg:.6f}_{dec_deg:.6f}.fits"
    path = output / filename
    if path.exists():
        return path
    cutouts = Tesscut.get_cutouts(
        coordinates=SkyCoord(ra_deg, dec_deg, unit="deg"),
        size=size,
        sector=int(sector),
    )
    if not cutouts:
        raise RuntimeError(f"MAST TESSCut returned no cutout for sector {sector}")
    cutouts[0].writeto(path)
    return path


def render_pixel_audit_html(reports: Sequence[dict[str, Any]]) -> str:
    """Render standalone HTML for pixel-level audit results."""
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<title>TESSCut pixel audit</title>",
        "<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#17213a}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #ccd3e0;padding:.5rem;text-align:left}"
        ".warn{background:#fff2d6}.pass{background:#e7f7ed}svg{max-width:520px;width:100%}</style></head>",
        "<body><h1>TESSCut pixel audit</h1>",
        "<p>Difference images use Sector 69 TESSCut FFIs. A positive centroid result is evidence for a blend, not a planet confirmation.</p>",
    ]
    for report in reports:
        parts.append(f"<h2>TIC {report['tic_id']}</h2>")
        pixel = report["pixel"]
        centroid = pixel.get("centroid", {})
        parts.append(
            f"<p>Sector {pixel['sector']} · target pixel "
            f"({pixel['target_pixel'][0]:.2f}, {pixel['target_pixel'][1]:.2f}) · "
            f"difference status <b>{escape(str(pixel['difference'].get('status')))}</b> · "
            f"clean event cadences {pixel['difference'].get('n_inside', 0)}/"
            f"{pixel['difference'].get('n_inside_all', 0)} · "
            f"difference centroid offset "
            f"{centroid.get('offset_pixels', float('nan')):.3f} px.</p>"
        )
        parts.append("<table><tr><th>Aperture</th><th>Depth</th><th>SNR</th></tr>")
        for aperture in pixel.get("apertures", []):
            parts.append(
                f"<tr><td>{aperture['pixels']} pixels</td><td>"
                f"{aperture.get('depth', float('nan')):.5f}</td><td>"
                f"{aperture.get('snr', float('nan')):.1f}</td></tr>"
            )
        parts.append("</table>")
        if pixel["difference"].get("status") != "measured":
            parts.append(
                '<p class="warn"><b>Pixel difference is inconclusive:</b> '
                "the event window does not contain enough quality-zero cadences."
                " Flagged cadences are not used as confirmation.</p>"
            )
        diff = pixel.get("difference", {}).get("difference_image")
        if diff:
            parts.append(_difference_svg(diff, pixel["target_pixel"], centroid.get("difference_centroid")))
        parts.append(
            "<p class=\"warn\"><b>Unresolved:</b> TESS pixels cannot resolve the 1–2 arcsec Gaia neighbours. "
            "High-resolution imaging remains required.</p>"
        )
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _difference_svg(
    image: Sequence[Sequence[float]],
    target: Sequence[float],
    centroid: Sequence[float] | None,
) -> str:
    import numpy as np

    values = np.asarray(image, dtype=float)
    scale = max(float(np.max(np.abs(values))), 1e-12)
    cell = 42
    parts = [f'<svg viewBox="0 0 {values.shape[1] * cell} {values.shape[0] * cell}" role="img">']
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x] / scale
            color = f"rgb({int(245 - max(value, 0) * 150)},{int(245 - abs(value) * 80)},{int(245 - max(-value, 0) * 150)})"
            parts.append(
                f'<rect x="{x * cell}" y="{y * cell}" width="{cell - 1}" height="{cell - 1}" fill="{color}"/>'
            )
    tx, ty = target
    parts.append(f'<circle cx="{(tx + .5) * cell:.1f}" cy="{(ty + .5) * cell:.1f}" r="8" fill="none" stroke="#159447" stroke-width="3"/>')
    if centroid is not None:
        cx, cy = centroid
        parts.append(f'<circle cx="{(cx + .5) * cell:.1f}" cy="{(cy + .5) * cell:.1f}" r="6" fill="none" stroke="#c43d4b" stroke-width="3"/>')
    parts.append("</svg>")
    return "".join(parts)


__all__ = [
    "aperture_depths",
    "audit_tesscut_file",
    "difference_centroid",
    "difference_image",
    "download_tesscut",
    "render_pixel_audit_html",
]
