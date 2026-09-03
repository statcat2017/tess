"""Injection study tests (issue #7). Synthetic cells run offline."""

import json

import pytest

from conftest import needs_archive
from tess_assoc.inject import (
    DEPTHS,
    SHAPES,
    feasible_pair,
    inject_transit,
    learned_comparison,
    run_grid,
    study_cell,
    summarize_grid,
)
from tess_assoc.event import EventRecord
from tess_assoc.manifest import ManifestSector, TracerManifest
from tess_assoc.window import filter_aliases

THRESHOLDS = {
    "max_rel_depth_diff": 0.25,
    "max_rel_duration_diff": 0.25,
    "min_morph_corr": 0.9,
}
SOURCE_PROVENANCE = {"source_a": {}, "source_b": {}}


def _flat(n=800, seed=5, sigma=0.0005, step=0.02):
    import random

    rng = random.Random(seed)
    time = [1000.0 + i * step for i in range(n)]
    flux = [1.0 + rng.gauss(0, sigma) for _ in time]
    return time, flux


def test_inject_transit_exact_and_provenance_free():
    time, flux = _flat()
    out = inject_transit(time, flux, 1005.0, 0.01, 0.2, "box")
    assert len(out) == len(flux)
    assert out[250] == pytest.approx(flux[250] * 0.99)
    assert out[0] == flux[0]
    with pytest.raises(ValueError, match="shape"):
        inject_transit(time, flux, 1005.0, 0.01, 0.2, "triangle")
    with pytest.raises(ValueError, match="equal length"):
        inject_transit([1.0], [], 1005.0, 0.01, 0.2)


def test_feasible_pair_geometry():
    assert feasible_pair((0.0, 10.0), (0.0, 10.0), 12.0, 0.6) is None
    t1 = feasible_pair((0.0, 30.0), (0.0, 30.0), 12.0, 0.6)
    assert t1 is not None
    assert t1 - 0.6 >= 0.0 and t1 + 12.0 + 0.6 <= 30.0
    with pytest.raises(ValueError):
        feasible_pair((0.0, 10.0), (0.0, 10.0), 0.0, 0.6)


def test_supported_pair_avoids_gaps():
    from tess_assoc.inject import supported_pair

    gappy = [1.0, 1.1, 1.2, 9.0, 9.1, 9.2]
    assert supported_pair(gappy, gappy, 12.0, 0.6) is None
    dense = [float(i) * 0.1 for i in range(200)]
    t1 = supported_pair(dense, dense, 12.0, 0.6, min_points=5)
    assert t1 is not None


def test_study_cell_detects_and_matches():
    ta, fa = _flat(n=3200, step=0.005)
    tb = [t + 50.0 for t in ta]
    row = study_cell(
        ta, fa, tb, list(fa), tic_id=1, sector_a=12, sector_b=39,
        t1=1005.0, delta_t_days=50.0, depth=0.01, duration_days=0.2,
        shape="box", thresholds=THRESHOLDS, provenance=SOURCE_PROVENANCE,
    )
    assert row["detected"] and row["compatible"]
    assert row["failure_mode"] == "recovered"
    assert row["alias_status"] == "evaluated"
    assert row["n_candidate_pairs"] >= 1
    assert row["target_rank"] is not None
    assert row["learn"] is not None and row["learn"]["label"] == 1
    assert row["aliases_before"] >= 1
    assert row["provenance"]["role"] == "injected"
    assert row["provenance"]["origin"] == "injection"
    json.dumps({k: v for k, v in row.items() if k != "learn"})


def test_study_cell_same_epoch_rejects_by_timing():
    ta, fa = _flat(n=3200, step=0.005)
    row = study_cell(
        ta, fa, ta, fa, tic_id=1, sector_a=12, sector_b=12,
        t1=1002.0, delta_t_days=12.0, depth=0.01, duration_days=0.2,
        shape="box", thresholds=THRESHOLDS, provenance=SOURCE_PROVENANCE,
    )
    assert row["detected"] and not row["compatible"]
    assert row["failure_mode"] == "recovered"
    assert row["alias_status"] == "not-evaluated"
    assert row["learn"]["label"] == 0


def test_study_cell_miss_reports_cleanly():
    ta, fa = _flat(n=3200, step=0.005)
    row = study_cell(
        ta, fa, ta, fa, tic_id=1, sector_a=12, sector_b=39,
        t1=1005.0, delta_t_days=50.0, depth=0.0001, duration_days=0.2,
        shape="box", thresholds=THRESHOLDS, provenance=SOURCE_PROVENANCE,
    )
    assert not row["detected"] and row["learn"] is None
    assert row["failure_mode"] == "detection-failure"
    with pytest.raises(ValueError, match="missing key"):
        study_cell(
            ta, fa, ta, fa, tic_id=1, sector_a=12, sector_b=39,
            t1=1005.0, delta_t_days=50.0, depth=0.01, duration_days=0.2,
            shape="box", thresholds={}, provenance=SOURCE_PROVENANCE,
        )


def test_run_grid_covers_separation_classes():
    ta, fa = _flat(n=1400, step=0.02)
    curves = {
        12: {"time": ta, "flux": fa},
        39: {"time": [t + 900.0 for t in ta], "flux": list(fa)},
    }
    rows = run_grid(
        curves, tic_id=1, thresholds=THRESHOLDS,
        depths=(0.01,), shapes=("box",),
    )
    seps = {r["sectors"][0] == r["sectors"][1] and "same" or "cross" for r in rows}
    assert seps == {"same", "cross"}
    assert all(r["detected"] for r in rows)
    assert all("source_a" in r["provenance"] for r in rows)
    assert all("source_b" in r["provenance"] for r in rows)
    summary = summarize_grid(rows)
    assert {s["separation"] for s in summary} == {"same-epoch", "cross-epoch"}


def test_run_grid_rejects_sealed_sectors():
    ta, fa = _flat(n=1400, step=0.02)
    with pytest.raises(ValueError, match="temporal leak"):
        run_grid(
            {80: {"time": ta, "flux": fa}},
            tic_id=1, thresholds=THRESHOLDS,
            depths=(0.01,), shapes=("box",),
        )
    with pytest.raises(ValueError, match="temporal leak"):
        study_cell(
            [1.0, 2.0], [1.0, 1.0], [1.0, 2.0], [1.0, 1.0],
            tic_id=1, sector_a=80, sector_b=80, t1=1.0, delta_t_days=12.0,
            depth=0.01, duration_days=0.2, shape="box", thresholds=THRESHOLDS,
            provenance=SOURCE_PROVENANCE,
        )


def test_alias_filter_uses_windows_after_the_pair():
    def event(sector, t0):
        return EventRecord(
            tic_id=1, sector=sector, t0=t0,
            local_time=[t0 - 0.1, t0, t0 + 0.1],
            local_flux=[1.0, 0.99, 1.0], depth=0.01,
            duration_days=0.2, snr=8.0, stellar_meta={}, quality={},
        )

    manifest = TracerManifest(
        name="after-pair", tic_id=1, epoch_match_tol_days=0.3,
        matcher_thresholds=THRESHOLDS,
        sectors=(
            ManifestSector(12, ((100.0, 101.0),)),
            ManifestSector(39, ((200.0, 201.0),)),
            ManifestSector(55, ((299.0, 301.0),)),
        ), events=(),
    )
    verdicts = filter_aliases(event(12, 100.0), event(39, 200.0), manifest,
                              [event(12, 100.0), event(39, 200.0)])
    assert not verdicts[0].retained
    assert verdicts[0].contradicting_epoch == pytest.approx(300.0)


def test_summarize_grid_separates_detection_from_association():
    rows = [
        {"depth": 0.01, "shape": "box", "sectors": [12, 39], "detected": True,
         "compatible": True, "aliases_before": 4, "aliases_after": 2},
        {"depth": 0.01, "shape": "box", "sectors": [12, 39], "detected": False,
         "compatible": False, "aliases_before": 0, "aliases_after": 0},
        {"depth": 0.01, "shape": "box", "sectors": [12, 12], "detected": True,
         "compatible": False, "aliases_before": 0, "aliases_after": 0},
    ]
    summary = summarize_grid(rows)
    by_sep = {s["separation"]: s for s in summary}
    assert by_sep["cross-epoch"]["detection_recall"] == 0.5
    assert by_sep["cross-epoch"]["deterministic_correct_rate"] == 0.5
    assert by_sep["cross-epoch"]["association_recall"] == 1.0
    assert by_sep["cross-epoch"]["association_evaluable_n"] == 1
    assert by_sep["cross-epoch"]["detection_failures"] == 1
    assert by_sep["cross-epoch"]["median_alias_reduction"] == 0.5
    assert by_sep["same-epoch"]["deterministic_correct_rate"] == 1.0
    assert by_sep["same-epoch"]["association_recall"] is None
    assert by_sep["same-epoch"]["negative_rejection_rate"] == 1.0


def test_learned_comparison_rotations():
    from tess_assoc.learn import LearnConfig

    ta, fa = _flat(n=3200, step=0.005)
    cells = []
    for tic in (1, 2, 3):
        tb = [t + 50.0 for t in ta]
        fb = list(fa)
        for dt in (50.0, 60.0):
            row = study_cell(
                ta, fa, tb, fb, tic_id=tic, sector_a=12, sector_b=39,
                t1=1005.0, delta_t_days=dt, depth=0.01, duration_days=0.2,
                shape="box", thresholds=THRESHOLDS, provenance=SOURCE_PROVENANCE,
            )
            assert row["learn"] is not None
            cells.append(row)
        negative = study_cell(
            ta, fa, ta, fa, tic_id=tic, sector_a=12, sector_b=12,
            t1=1002.0, delta_t_days=12.0, depth=0.01, duration_days=0.2,
            shape="box", thresholds=THRESHOLDS, provenance=SOURCE_PROVENANCE,
        )
        assert negative["learn"] is not None
        cells.append(negative)
        morphology_negative = study_cell(
            ta, fa, tb, fb, tic_id=tic, sector_a=12, sector_b=39,
            t1=1005.0, delta_t_days=50.0, depth=0.01, duration_days=0.2,
            shape="box", shape_b="v", thresholds=THRESHOLDS,
            provenance=SOURCE_PROVENANCE,
        )
        assert morphology_negative["learn"]["label"] == 0
        cells.append(morphology_negative)
    out = learned_comparison(
        cells, LearnConfig(seed=7, epochs=2, batch_size=8, embedding_dim=8)
    )
    assert out["ablation"] == "morphology+scalars"
    assert len(out["rotations"]) == 3
    assert all(0.0 <= r["ap"] <= 1.0 for r in out["rotations"])
    assert all(r["operating_regions"] for r in out["rotations"])
    assert out["config"]["seed"] == 7
    for rotation in out["rotations"]:
        for region in rotation["operating_regions"]:
            assert region["n_cells"] >= 1
            assert region["pairs_per_cell"] >= 1.0
            for key in ("learned_correct_rate", "deterministic_correct_rate"):
                assert region[key] is None or 0.0 <= region[key] <= 1.0
    assert {r["separation"] for r in out["rotations"][0]["operating_regions"]} == {
        "same-epoch", "cross-epoch"
    }
    assert any(
        r["shape"] != r["shape_b"]
        for r in out["rotations"][0]["operating_regions"]
    )


def _live_curves(tmp_path):
    from pathlib import Path

    from tess_assoc.archive import download_spoc_ffi
    from tess_assoc.extract import load_lightcurve
    from tess_assoc.replay import load_replay_manifest

    replay = load_replay_manifest(
        str(Path(__file__).resolve().parent.parent / "fixtures" / "replay_v1.json")
    )
    curves = {}
    for system in replay.systems:
        for sector in system.sectors:
            product = download_spoc_ffi(
                system.tic_id, sector, str(tmp_path)
            )
            time, flux = load_lightcurve(product)
            curves[(system.tic_id, sector)] = {
                "time": time,
                "flux": flux,
                "provenance": {
                    "data_uri": product.data_uri,
                    "local_path": product.local_path,
                    "retrieved_utc": product.retrieved_utc,
                },
                "system": system,
            }
    return replay, curves


@needs_archive
def test_live_injection_grid_and_learned(tmp_path):
    from tess_assoc.learn import LearnConfig

    replay, curves = _live_curves(tmp_path)
    frozen_thresholds = dict(replay.matcher_thresholds)
    assert all(sector < 80 for system in replay.systems for sector in system.sectors)
    all_rows = []
    for system in replay.systems:
        sector_curves = {
            s: curves[(system.tic_id, s)] for s in system.sectors
        }
        rows = run_grid(
            sector_curves, tic_id=system.tic_id,
            thresholds=frozen_thresholds,
        )
        assert rows, system.name
        all_rows.extend(rows)
    assert {r["depth"] for r in all_rows} == set(DEPTHS)
    assert {r["shape"] for r in all_rows} == set(SHAPES)
    assert {r["separation"] for r in all_rows} == {"same-epoch", "cross-epoch"}
    assert any(r["shape"] != r["shape_b"] for r in all_rows)
    deep = [
        r for r in all_rows
        if r["depth"] == 0.025 and r["separation"] == "cross-epoch"
    ]
    assert sum(1 for r in deep if r["detected"]) / len(deep) >= 0.8
    summary = summarize_grid(all_rows)
    assert summary and all(0.0 <= s["detection_recall"] <= 1.0 for s in summary)
    assert all(r["provenance"]["source_a"]["data_uri"] for r in all_rows)
    assert all(r["provenance"]["source_a"]["retrieved_utc"] for r in all_rows)
    out = learned_comparison(
        all_rows, LearnConfig(seed=7, epochs=3, batch_size=32, embedding_dim=16)
    )
    assert len(out["rotations"]) == 3
    assert frozen_thresholds == replay.matcher_thresholds
    json.dumps(out)
    json.dumps([{k: v for k, v in r.items() if k != "learn"} for r in all_rows])
