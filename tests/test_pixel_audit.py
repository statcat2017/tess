"""TESSCut pixel-level audit tests."""

import pytest

from tess_assoc.pixel_audit import aperture_depths, difference_centroid, difference_image


def _cube():
    import numpy as np

    time = [i * 0.01 for i in range(201)]
    cube = np.ones((len(time), 5, 5), dtype=float)
    for i, t in enumerate(time):
        if abs(t - 1.0) <= 0.05:
            cube[i, 2, 2] -= 0.2
            cube[i, 2, 3] -= 0.05
    return time, cube


def test_difference_image_localises_the_dimming_pixels():
    time, cube = _cube()
    result = difference_image(time, cube, 1.0, 0.1)
    assert result["status"] == "measured"
    centroid = difference_centroid(result["difference_image"], (2.0, 2.0))
    assert centroid["offset_pixels"] < 0.5


def test_aperture_depths_are_measured_at_multiple_scales():
    time, cube = _cube()
    results = aperture_depths(time, cube, (2.0, 2.0), 1.0, 0.1)
    assert [result["pixels"] for result in results] == [1, 9, 25]
    assert all(result["status"] == "measured" for result in results)
    assert results[0]["depth"] > results[-1]["depth"]


def test_difference_image_rejects_mismatched_cube():
    with pytest.raises(ValueError, match="shape"):
        difference_image([0.0], [[1.0]], 0.0, 0.1)


def test_difference_image_reports_flagged_event_cadences():
    time, cube = _cube()
    quality = [128 if abs(t - 1.0) <= 0.05 else 0 for t in time]
    result = difference_image(time, cube, 1.0, 0.1, quality=quality)
    assert result["status"] == "insufficient-data"
    assert result["n_inside"] == 0
    assert result["n_inside_all"] > 0
