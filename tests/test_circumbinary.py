"""Binary-aware Survey B pilot tests."""

import json
from pathlib import Path

import pytest

from tess_assoc.circumbinary import (
    BinaryPilotTarget,
    build_binary_pilot_manifest,
    eclipse_windows,
    load_binary_pilot_manifest,
    run_binary_pilot,
)
from tess_assoc.event import EventRecord
from tess_assoc.extract import BTJD_OFFSET


def _event(tic_id: int, sector: int, t0: float, *, snr: float = 12.0,
           truth: str = "") -> EventRecord:
    return EventRecord(
        tic_id=tic_id,
        sector=sector,
        t0=t0,
        local_time=[t0 - 0.1, t0, t0 + 0.1],
        local_flux=[1.0, 0.99, 1.0],
        depth=0.01,
        duration_days=0.1,
        snr=snr,
        quality={"pilot_truth": truth} if truth else {},
    )


def _target(target_class: str = "circumbinary-positive") -> BinaryPilotTarget:
    return BinaryPilotTarget(
        tic_id=101,
        name="pilot-positive",
        period_days=10.0,
        t0_bjd_tdb=BTJD_OFFSET + 2000.0,
        primary_duration_days=0.2,
        secondary_phase=0.5,
        sectors=(29, 69),
        target_class=target_class,
    )


def test_eclipse_windows_include_primary_and_secondary_events():
    target = _target()
    windows = eclipse_windows(
        target,
        {29: [(2004.0, 2006.0)], 69: [(2014.0, 2016.0)]},
        buffer_days=0.05,
    )
    assert windows[29] == [(2004.85, 2005.15)]
    assert windows[69] == [(2014.85, 2015.15)]


def test_eclipse_windows_mask_primary_eclipse():
    target = _target()
    windows = eclipse_windows(target, {29: [(1999.0, 2001.0)]})
    assert windows[29] == [(1999.85, 2000.15)]


def test_binary_pilot_masks_eclipses_but_keeps_nonperiodic_repeat():
    target = _target()
    coverage = {
        29: [(2004.0, 2006.0)],
        69: [(2014.0, 2016.0)],
    }
    events = [
        _event(101, 29, 2005.0, truth="same-circumbinary"),
        _event(101, 29, 2005.4, truth="same-circumbinary"),
        _event(101, 69, 2014.2, truth="same-circumbinary"),
    ]
    result = run_binary_pilot([target], events, coverage)
    system = result["systems"]["pilot-positive"]
    assert system["n_masked_events"] == 1
    assert system["n_retained_events"] == 2
    assert system["n_pairs_ranked"] == 1
    pair = system["pairs"][0]
    assert pair["timing_model"] == "binary-phase-free"
    assert pair["truth"] is True


def test_binary_pilot_does_not_reject_short_separation_as_long_period_alias():
    target = _target("eclipsing-binary-control")
    coverage = {29: [(2004.0, 2006.0)], 69: [(2014.0, 2016.0)]}
    result = run_binary_pilot(
        [target],
        [_event(101, 29, 2005.4), _event(101, 69, 2014.2)],
        coverage,
    )
    assert result["systems"]["pilot-positive"]["n_pairs_ranked"] == 1


def test_binary_manifest_round_trip_and_temporal_boundary():
    target = _target()
    manifest = build_binary_pilot_manifest(
        "pilot", [target], source_catalog="test-catalog"
    )
    assert manifest["version"] == "survey-b-v1"
    restored = load_binary_pilot_manifest(manifest)
    assert restored == (target,)
    json.dumps(manifest)
    with pytest.raises(ValueError, match="temporal leak"):
        BinaryPilotTarget(
            tic_id=102,
            name="sealed",
            period_days=10.0,
            t0_bjd_tdb=2000.0,
            primary_duration_days=0.2,
            secondary_phase=0.5,
            sectors=(82,),
            target_class="eclipsing-binary-control",
        )


def test_binary_manifest_rejects_duplicate_names():
    first = _target()
    second = BinaryPilotTarget(
        tic_id=102,
        name=first.name,
        period_days=11.0,
        t0_bjd_tdb=BTJD_OFFSET + 2000.0,
        primary_duration_days=0.2,
        secondary_phase=0.5,
        sectors=(29, 69),
        target_class="eclipsing-binary-control",
    )
    with pytest.raises(ValueError, match="names must be unique"):
        build_binary_pilot_manifest("pilot", [first, second], source_catalog="test")


def test_binary_pilot_rejects_event_outside_coverage():
    target = _target()
    result = run_binary_pilot(
        [target],
        [_event(101, 29, 2006.0), _event(101, 69, 2014.2)],
        {29: [(2004.0, 2005.0)], 69: [(2014.0, 2016.0)]},
    )
    system = result["systems"][target.name]
    assert system["n_out_of_cohort"] == 1
    assert system["n_pairs_ranked"] == 0


def test_binary_pilot_fixture_round_trip():
    fixture_path = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "circumbinary_pilot_v1.json"
    )
    fixture = json.loads(fixture_path.read_text())
    targets = load_binary_pilot_manifest(fixture)
    coverage = {
        int(sector): [tuple(window) for window in windows]
        for sector, windows in fixture["coverage_windows"].items()
    }
    result = run_binary_pilot(targets, fixture["events"], coverage)
    assert result["metrics"] == {
        "positive_targets": 1,
        "positive_targets_with_truth_pair": 1,
        "control_targets_with_ranked_pair": 1,
    }
