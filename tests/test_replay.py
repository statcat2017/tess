"""Known-system replay tests (issue #3 acceptance criteria).

Pure-unit tests always run. Archive/network tests skip cleanly when
dependencies or MAST are unreachable — and report unavailability clearly.
"""

import json
import os
from pathlib import Path

import pytest

from conftest import needs_archive
from tess_assoc.archive import ArchiveUnavailable, cache_dir, find_spoc_ffi_uri
from tess_assoc.extract import BTJD_OFFSET, coverage_windows, predicted_transits
from tess_assoc.replay import MISS_REASONS, load_replay_manifest, replay_all, replay_system

REPLAY = Path(__file__).resolve().parent.parent / "fixtures" / "replay_v1.json"
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_predicted_transits_enumeration():
    assert predicted_transits(2457000.0, 1.0, 0.0, 3.0) == [0.0, 1.0, 2.0, 3.0]
    assert predicted_transits(2457000.5, 1.0, 0.0, 3.0) == [0.5, 1.5, 2.5]
    assert BTJD_OFFSET == 2457000.0
    with pytest.raises(ValueError):
        predicted_transits(2457000.0, 0.0, 0.0, 3.0)


def test_coverage_windows_split_on_gaps():
    assert coverage_windows([]) == []
    assert coverage_windows([1.0, 1.1, 1.2, 5.0, 5.1]) == [(1.0, 1.2), (5.0, 5.1)]


def test_cache_dir_outside_repo_and_env_override(tmp_path):
    assert not os.path.realpath(cache_dir()).startswith(os.path.realpath(REPO_ROOT))
    os.environ["TESS_ASSOC_CACHE"] = str(tmp_path)
    try:
        assert cache_dir() == str(tmp_path)
    finally:
        del os.environ["TESS_ASSOC_CACHE"]


def test_replay_manifest_declares_spoc_product():
    replay = load_replay_manifest(str(REPLAY))
    assert replay.product == "TESS-SPOC FFI"
    assert 3 <= len(replay.systems) <= 5
    for system in replay.systems:
        assert len(system.sectors) >= 2


def test_replay_manifest_rejects_wrong_product(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x"}))
    with pytest.raises(ValueError, match="missing key"):
        load_replay_manifest(str(bad))


@needs_archive
def test_archive_reports_clearly_for_unknown_target():
    with pytest.raises(ArchiveUnavailable):
        find_spoc_ffi_uri(99999999999, 1)


@needs_archive
def test_live_blind_replay_measures_recall(tmp_path):
    from tess_assoc.replay import load_replay_manifest, replay_blind_system

    replay = load_replay_manifest(str(REPLAY))
    system = next(s for s in replay.systems if s.name == "WASP-121 b")
    res = replay_blind_system(replay, system, cache_dir=str(tmp_path))
    assert res["sealed_sectors_touched"] == []
    assert res["n_proposals"] > 0
    assert res["recall"]["rate"] >= 0.8, res["recall"]
    assert res["pair_outcome"] == "associated", res["pair_outcome"]
    missed = res["missed"]
    assert len(missed) == res["recall"]["known"] - res["recall"]["recalled"]
    for m in missed:
        assert set(m) == {"sector", "t0", "max_snr", "proposed", "reason"}
        assert m["reason"] in MISS_REASONS
        assert m["max_snr"] is None or isinstance(m["max_snr"], float)
    assert res["recall"]["coverable"] <= res["recall"]["known"]
    assert res["recall"]["recalled_coverable"] <= res["recall"]["coverable"]
    json.dumps(res)


@needs_archive
def test_live_replay_three_systems(tmp_path):
    from tess_assoc.orbit import generate_aliases

    results = replay_all(str(REPLAY), cache_dir=str(tmp_path))
    assert set(results) == {"WASP-43 b", "WASP-121 b", "KELT-9 b"}
    for name, res in results.items():
        # Known true pairs associate on real morphology.
        assert res["sealed_sectors_touched"] == []
        assert len(res["anchors"]) == 2
        pair_map = {
            tuple(sorted([p["a"], p["b"]])): p for p in res["pairs"]
        }
        assert pair_map[tuple(sorted(res["anchors"]))]["compatible"]
        # Deterministic alias/window machinery on the anchor pair.
        assoc = [a for a in res["associations"] if sorted(a["pair"]) == sorted(res["anchors"])]
        assert assoc, f"{name}: anchor pair has no association"
        assert assoc[0]["aliases_total"] == len(generate_aliases(assoc[0]["delta_t_days"]))
        assert len(assoc[0]["retained"]) + len(assoc[0]["rejected"]) == assoc[0]["aliases_total"]
        for prod in res["products"]:
            assert prod["data_uri"].endswith("_lc.fits")
            assert not os.path.realpath(prod["local_path"]).startswith(
                os.path.realpath(REPO_ROOT)
            )
        json.dumps(res)

    # Science outputs are deterministic across runs (provenance timestamps excluded).
    replay = load_replay_manifest(str(REPLAY))
    again = replay_system(replay, replay.systems[0], cache_dir=str(tmp_path))
    first = results[replay.systems[0].name]
    assert again["pairs"] == first["pairs"]
    assert again["associations"] == first["associations"]
