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


def test_repeat_ranking_keeps_timing_failures_visible():
    from tess_assoc.candidate_followup import rank_repeat_events
    from tess_assoc.event import EventRecord

    def event(sector, t0):
        return EventRecord(
            tic_id=1,
            sector=sector,
            t0=t0,
            local_time=[t0 - 0.1, t0, t0 + 0.1],
            local_flux=[1.0, 0.99, 1.0],
            depth=0.01,
            duration_days=0.1,
            snr=10.0,
        )

    thresholds = {
        "max_rel_depth_diff": 0.25,
        "max_rel_duration_diff": 0.25,
        "min_morph_corr": 0.9,
    }
    ranked = rank_repeat_events(event(69, 100.0), [event(29, 105.0)], thresholds)
    assert len(ranked) == 1
    assert ranked[0]["timing_plausible"] is False
    assert ranked[0]["score"] is None


def test_followup_queue_deprioritizes_localisation_failure():
    from tess_assoc.candidate_followup import triage_followup

    repeat = {
        "tic_id": 137801807,
        "status": "complete",
        "ranked_repeats": [],
        "compatible_repeats": [],
    }
    pixel = {
        "tic_id": 137801807,
        "pixel": {
            "difference": {"status": "measured"},
            "centroid": {"offset_pixels": 1.3},
            "apertures": [{"status": "measured", "depth": -0.01}],
        },
    }
    result = triage_followup(repeat, pixel)
    assert result["decision"] == "source-localisation-failure"
    assert result["priority"] == 2
    assert result["planet_claim"] == "blocked"


def test_followup_queue_prioritizes_pixel_reprocessing():
    from tess_assoc.candidate_followup import build_followup_queue

    repeat = {
        "tic_id": 117549174,
        "status": "complete",
        "ranked_repeats": [],
        "compatible_repeats": [],
    }
    pixel = {
        "tic_id": 117549174,
        "pixel": {
            "difference": {"status": "insufficient-data"},
            "apertures": [{"status": "insufficient-data"}],
        },
    }
    queue = build_followup_queue([repeat], [pixel])
    assert queue["entries"][0]["decision"] == "pixel-inconclusive"
    assert queue["entries"][0]["priority"] == 1
