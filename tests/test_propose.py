"""Blind proposer tests (issue #4). No period or ephemeris anywhere here."""

import pytest

from tess_assoc.extract import SkippedTransit, extract_at
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
