"""Blind proposer tests (issue #4). No period or ephemeris anywhere here."""

import pytest

from tess_assoc.extract import SkippedTransit, extract_at, refine_epoch
from tess_assoc.observability import CadenceEvidence
from tess_assoc.propose import (
    center_on_minimum,
    detrend,
    find_dips,
    propose_events,
    records_from_proposals,
)
from tess_assoc.replay import classify_pair


def _synthetic(seed_boxes=((5.0, 0.06, 0.02), (8.0, 0.10, 0.02), (12.5, 0.05, 0.015))):
    import random

    rng = random.Random(7)
    time = [i * 0.02 for i in range(1000)]
    flux = [1.0 + rng.gauss(0, 0.001) for _ in time]
    for center, width, depth in seed_boxes:
        flux = [
            f - depth if abs(t - center) <= width / 2 else f
            for t, f in zip(time, flux)
        ]
    return time, flux


def test_proposer_recalls_seeded_dips_without_period():
    time, flux = _synthetic()
    proposals = propose_events(time, flux)
    assert len(proposals) >= 3
    for center in (5.0, 8.0, 12.5):
        assert any(abs(p.t0_guess - center) < 0.05 for p in proposals), center


def test_proposer_silent_on_flat_lightcurve():
    import random

    rng = random.Random(3)
    time = [i * 0.02 for i in range(500)]
    flux = [1.0 + rng.gauss(0, 0.001) for _ in time]
    assert propose_events(time, flux) == []


def test_proposer_rejects_bad_inputs():
    with pytest.raises(ValueError):
        propose_events([], [])
    with pytest.raises(ValueError):
        find_dips([1.0, 2.0], [1.0], 0.001)
    with pytest.raises(ValueError):
        center_on_minimum([1.0], [1.0], 1.0, 0.0)


def test_detrend_centers_on_unity():
    time, flux = _synthetic()
    detrended, sigma = detrend(time, flux)
    assert abs(sum(detrended) / len(detrended) - 1.0) < 0.01
    assert sigma > 0


def test_proposer_does_not_merge_dips_across_gap():
    time = [0.0, 0.02, 0.04, 1.0, 1.02, 1.04]
    detrended = [0.98, 0.98, 1.0, 0.98, 0.98, 1.0]
    proposals = find_dips(
        time,
        detrended,
        0.001,
        snr_threshold=4.0,
        observing_windows=[(0.0, 0.04), (1.0, 1.04)],
    )
    assert len(proposals) == 2


def test_segment_noise_is_independent_between_observing_windows():
    import random

    rng = random.Random(11)
    quiet_time = [i * 0.02 for i in range(80)]
    noisy_time = [2.0 + i * 0.02 for i in range(80)]
    quiet_flux = [1.0 + rng.gauss(0.0, 0.0005) for _ in quiet_time]
    noisy_flux = [1.0 + rng.gauss(0.0, 0.05) for _ in noisy_time]
    for index, value in enumerate(quiet_time):
        if abs(value - 0.8) <= 0.04:
            quiet_flux[index] -= 0.01
    all_time = quiet_time + noisy_time
    all_flux = quiet_flux + noisy_flux
    evidence = CadenceEvidence(
        time=tuple(all_time),
        usable=(True,) * len(all_time),
        quality_flags=(0,) * len(all_time),
    )
    with_noisy = propose_events(all_time, all_flux, observability=evidence)
    quiet_only = propose_events(quiet_time, quiet_flux)
    quiet_with_noisy = [p for p in with_noisy if p.t0_guess < 1.5]
    assert quiet_with_noisy
    assert [p.t0_guess for p in quiet_with_noisy] == [p.t0_guess for p in quiet_only]


def test_epoch_refinement_stays_inside_requested_observing_window():
    time = [-0.2 + i * 0.04 for i in range(11)] + [5.0, 5.1, 5.2, 5.3, 5.4]
    flux = [1.0] * 11 + [0.5] * 5
    refined = refine_epoch(
        time,
        flux,
        period_days=40.0,
        t0_guess_btjd=0.0,
        duration_days=0.2,
        observing_window=(-0.2, 0.2),
    )
    assert -0.2 <= refined <= 0.2


def test_extraction_rejects_quality_gap_crossing():
    time = [-1.0 + i * 0.02 for i in range(50)] + [0.02 + i * 0.02 for i in range(50)]
    evidence = CadenceEvidence(
        time=tuple([-1.0 + i * 0.02 for i in range(101)]),
        usable=tuple([True] * 50 + [False] + [True] * 50),
        quality_flags=tuple([0] * 50 + [4] + [0] * 50),
    )
    result = extract_at(
        time,
        [1.0] * len(time),
        0.0,
        0.1,
        tic_id=1,
        sector=12,
        observability=evidence,
    )
    assert isinstance(result, SkippedTransit)
    assert result.reason == "quality-gap-crossing"


def test_extract_at_measures_and_skips():
    time, flux = _synthetic()
    record = extract_at(
        time, flux, 8.0, 0.10, tic_id=1, sector=12, quality={"role": "t"}
    )
    assert not isinstance(record, SkippedTransit)
    assert abs(record.depth - 0.02) < 0.005
    assert abs(record.t0 - 8.0) < 1e-9
    skipped = extract_at(
        time, flux, time[0], 0.10, tic_id=1, sector=12, quality={}
    )
    assert isinstance(skipped, SkippedTransit)


def test_records_from_proposals_share_contract():
    time, flux = _synthetic()
    proposals = propose_events(time, flux)
    records, skipped = records_from_proposals(
        time, flux, proposals, tic_id=9, sector=12, quality_base={"role": "x"}
    )
    assert records
    for rec in records.values():
        assert rec.validate() is None
        assert rec.quality["role"] == "blind-proposal"


def test_classify_pair_separates_failure_modes():
    from tess_assoc.event import EventRecord

    def rec(t0):
        return EventRecord(
            tic_id=1, sector=12, t0=t0,
            local_time=[t0 - 0.1, t0, t0 + 0.1],
            local_flux=[1.0, 0.99, 1.0],
            depth=0.01, duration_days=0.1, snr=8.0,
            stellar_meta={}, quality={},
        )

    records = {"a": rec(10.0), "b": rec(20.0)}
    pairs = [{"a": "a", "b": "b", "compatible": True}]
    assert classify_pair([10.0, 20.0], records, pairs) == "associated"
    assert classify_pair([10.0, 20.0], records, []) == "recalled-not-associated"
    assert classify_pair([10.0, 99.0], records, pairs) == "not-proposed"
