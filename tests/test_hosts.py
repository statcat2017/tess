import json
from pathlib import Path

import pytest

from tess_assoc.hosts import (
    KnownPlanet,
    deduplicate_planets,
    parse_confirmed_rows,
    parse_toi_rows,
    select_known_host_targets,
)
from tess_assoc.replay import mask_known_transits


def test_parse_confirmed_rows_accepts_tic_labels_and_hours():
    rows = [
        {
            "tic_id": "TIC 123",
            "pl_name": "Example b",
            "pl_orbper": 20.0,
            "pl_tranmid": 2458000.0,
            "pl_trandur": 4.8,
            "tran_flag": 1,
        }
    ]
    planet = parse_confirmed_rows(rows)[0]
    assert planet.tic_id == 123
    assert planet.duration_days == pytest.approx(0.2)


def test_parse_toi_rows_only_accepts_planetary_dispositions():
    rows = [
        {
            "tid": 123,
            "toi": "123.01",
            "pl_orbper": 10.0,
            "pl_tranmid": 2458000.0,
            "pl_trandurh": 2.4,
            "tfopwg_disp": "PC",
        },
        {
            "tid": 456,
            "toi": "456.01",
            "pl_orbper": 10.0,
            "pl_tranmid": 2458000.0,
            "pl_trandurh": 2.4,
            "tfopwg_disp": "FP",
        },
    ]
    planets = parse_toi_rows(rows)
    assert [p.tic_id for p in planets] == [123]
    assert planets[0].duration_days == pytest.approx(0.1)


def test_host_selection_is_deterministic_and_preserves_multiple_planets():
    planets = [
        KnownPlanet(2, "b", 5.0, 2458000.0, 0.1, "test"),
        KnownPlanet(1, "b", 3.0, 2458000.0, 0.1, "test"),
        KnownPlanet(1, "c", 8.0, 2458000.0, 0.1, "test"),
    ]
    targets = select_known_host_targets(
        planets,
        {1: [1, 2], 2: [1], 3: [1, 2]},
        min_sectors=2,
    )
    assert [target["tic_id"] for target in targets] == [1, 2]
    assert len(targets[0]["known_planets"]) == 2


def test_deduplicate_planets_keeps_distinct_ephemerides():
    planet = KnownPlanet(1, "b", 3.0, 2458000.0, 0.1, "test")
    duplicate = KnownPlanet(1, "catalogue-alias", 3.0, 2458006.0, 0.1, "toi")
    other = KnownPlanet(1, "c", 8.0, 2458000.0, 0.1, "test")
    assert len(deduplicate_planets([planet, duplicate, other])) == 2


def test_mask_known_transits_removes_only_known_windows():
    time = [float(i) / 10.0 for i in range(101)]
    flux = [1.0] * len(time)
    filtered_time, filtered_flux, masks = mask_known_transits(
        time,
        flux,
        [{"name": "b", "period_days": 5.0, "t0_bjd_tdb": 2457000.0, "duration_days": 0.2}],
    )
    assert masks
    assert len(filtered_time) < len(time)
    assert len(filtered_time) == len(filtered_flux)
    assert all(abs(t) > 0.05 for t in filtered_time)


def test_host_pilot_report_records_auditable_result():
    report = json.loads(
        (Path(__file__).resolve().parent.parent / "reports" / "known_host_survey_pilot.json").read_text()
    )
    assert report["status"] == "complete"
    assert report["downloaded"] == report["recordings"]
    assert report["known_transit_windows_masked"] > 0
    assert report["pair_candidates"] == 0
