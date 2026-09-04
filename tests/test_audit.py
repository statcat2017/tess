"""Single-event evidence measurements."""

import pytest

from tess_assoc.audit import (
    audit_flux_channels,
    build_single_event_audit,
    measure_centroid,
    measure_event_shape,
    measure_flux_channel,
    render_single_event_audit,
)


def _time():
    return [i * 0.01 for i in range(201)]


def test_flux_audit_requires_reduction_stability():
    time = _time()
    sap = [1.0 - (0.1 if abs(t - 1.0) <= 0.05 else 0.0) for t in time]
    pdc = [1.0 - (0.105 if abs(t - 1.0) <= 0.05 else 0.0) for t in time]
    result = audit_flux_channels(time, {"sap": sap, "pdcsap": pdc}, 1.0, 0.1)
    assert result["stable"] is True
    assert result["channels"]["pdcsap"]["depth"] == pytest.approx(0.105)


def test_flux_audit_marks_insufficient_windows():
    result = measure_flux_channel([0.0, 1.0, 2.0], [1.0, 0.9, 1.0], 1.0, 0.1)
    assert result["status"] == "insufficient-data"


def test_centroid_audit_reports_pixel_shift():
    time = _time()
    x = [0.0 + (0.5 if abs(t - 1.0) <= 0.05 else 0.0) for t in time]
    y = [0.0 for _ in time]
    result = measure_centroid(time, x, y, 1.0, 0.1)
    assert result["shift_status"] == "significant"
    assert result["shift_pixels"] == pytest.approx(0.5)


def test_centroid_audit_does_not_overstate_correlated_jitter():
    time = _time()
    x = [0.001 * ((i % 7) - 3) + (0.004 if abs(t - 1.0) <= 0.05 else 0.0) for i, t in enumerate(time)]
    y = [0.001 * ((i % 5) - 2) for i in range(len(time))]
    result = measure_centroid(time, x, y, 1.0, 0.1)
    assert result["shift_status"] == "not-significant"


def test_shape_audit_does_not_call_a_v_shape_a_planet():
    time = _time()
    flux = [1.0 - (0.1 if abs(t - 1.0) <= 0.05 else 0.0) for t in time]
    result = measure_event_shape(time, flux, 1.0, 0.1)
    assert result["shape"] == "flat-bottomed"
    assert result["edge_to_center_depth"] == pytest.approx(1.0)


def test_build_audit_preserves_unresolved_evidence(tmp_path):
    import numpy as np
    from astropy.io import fits

    time = np.asarray(_time(), dtype=float)
    flux = np.asarray([1.0 - (0.1 if abs(t - 1.0) <= 0.05 else 0.0) for t in time])
    columns = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="QUALITY", format="K", array=np.zeros(len(time), dtype=int)),
        fits.Column(name="SAP_FLUX", format="D", array=flux),
        fits.Column(name="PDCSAP_FLUX", format="D", array=flux),
        fits.Column(name="PSF_CENTR1", format="D", array=np.zeros(len(time))),
        fits.Column(name="PSF_CENTR2", format="D", array=np.zeros(len(time))),
    ]
    path = tmp_path / "event.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    report = build_single_event_audit(
        123, {"sector": 69, "t0": 1.0, "depth": 0.1, "duration_days": 0.1, "snr": 10.0},
        [{"sector": 69, "local_path": str(path)}],
        metadata={"tmag": 12.0, "available_sectors": [2, 29, 69]},
        catalogs={"cross_match": {"status": "clean"}},
        neighbors=[],
    )
    assert report["checks"]["sap_pdcsap"] == "pass"
    assert report["checks"]["period"] == "unconstrained"
    assert report["checks"]["pixel_difference_image"] == "required"
    assert "TIC 123" in render_single_event_audit([report])
