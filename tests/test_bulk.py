"""Bulk retrieval tests (issue #9 scale-up). Pattern checks run offline."""

import json
from pathlib import Path

import pytest

from conftest import needs_archive
from tess_assoc.archive import download_spoc_ffi, spoc_ffi_uri
from tess_assoc.bulk import bulk_fetch, direct_url, expected_filename, fetch_one


def test_parse_and_overlap_synthetic_scripts(tmp_path):
    from tess_assoc.bulk import overlap_tics, parse_sector_script

    def script(name, tics):
        path = tmp_path / name
        lines = [
            "curl -f --output 'x' "
            "'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/tess-spoc/"
            f"s0030/target/0000/0000/0000/0001/hlsp_tess-spoc_tess_phot_{t:016d}-s0030_tess_v1_lc.fits'"
            for t in tics
        ]
        path.write_text("\n".join(lines) + "\n# comment line without uri\n")
        return str(path)

    a = script("a.sh", [10, 20, 30])
    b = script("b.sh", [20, 30, 40])
    parsed = parse_sector_script(a)
    assert sorted(parsed) == [10, 20, 30]
    assert parsed[20].startswith("mast:HLSP/tess-spoc/")
    assert overlap_tics(a, b) == [20, 30]


@needs_archive
def test_live_fetch_sector_script(tmp_path):
    from tess_assoc.bulk import fetch_sector_script, parse_sector_script

    path = fetch_sector_script(30, str(tmp_path))
    parsed = parse_sector_script(path)
    assert len(parsed) > 100_000
    assert 42074448 in parsed


def test_spoc_ffi_uri_pattern():
    assert spoc_ffi_uri(22529346, 33) == (
        "mast:HLSP/tess-spoc/s0033/target/0000/0000/2252/9346/"
        "hlsp_tess-spoc_tess_phot_0000000022529346-s0033_tess_v1_lc.fits"
    )
    assert spoc_ffi_uri(16740101, 55) == (
        "mast:HLSP/tess-spoc/s0055/target/0000/0000/1674/0101/"
        "hlsp_tess-spoc_tess_phot_0000000016740101-s0055_tess_v1_lc.fits"
    )
    assert expected_filename(22529346, 33) == (
        "hlsp_tess-spoc_tess_phot_0000000022529346-s0033_tess_v1_lc.fits"
    )
    assert direct_url(1, 1).startswith("https://mast.stsci.edu/api/v0.1/Download/file?uri=")
    with pytest.raises(ValueError):
        spoc_ffi_uri(0, 33)


def test_download_prefers_cache_without_query(tmp_path, monkeypatch):
    import tess_assoc.archive as A

    Path(tmp_path, expected_filename(42074448, 30)).write_bytes(b"cached-bytes")

    def _boom(*args, **kwargs):
        raise AssertionError("MAST queried despite warm cache")

    monkeypatch.setattr(A, "find_spoc_ffi_uri", _boom)
    product = download_spoc_ffi(42074448, 30, str(tmp_path))
    assert product.cached is True
    assert product.local_path == str(tmp_path / expected_filename(42074448, 30))


@needs_archive
def test_live_direct_fetch_roundtrip(tmp_path):
    good = fetch_one(42074448, 30, str(tmp_path))
    assert good["status"] == "downloaded"
    assert Path(good["local_path"]).stat().st_size > 10_000
    again = fetch_one(42074448, 30, str(tmp_path))
    assert again["status"] == "cached"
    missing = fetch_one(99999999, 3, str(tmp_path))
    assert missing["status"] == "missing"
    assert not Path(missing["local_path"]).exists()


@needs_archive
def test_live_bulk_fetch_buckets(tmp_path):
    out = bulk_fetch(
        [(42074448, 30), (42073085, 3), (99999999, 3)],
        str(tmp_path),
        max_workers=3,
    )
    assert {r["tic_id"] for r in out["downloaded"]} == {42074448, 42073085}
    assert [r["tic_id"] for r in out["missing"]] == [99999999]
    assert out["failed"] == []


@needs_archive
def test_live_survey_runs_with_queries_blocked(tmp_path, monkeypatch):
    """Steady state needs zero MAST queries: cache + direct fetch suffice."""
    import tess_assoc.archive as A
    from tess_assoc import freeze as F
    from tess_assoc.learn import LearnConfig
    from tess_assoc.survey import build_survey_manifest, run_mining_survey

    manifest = build_survey_manifest(
        "blocked", [{"tic_id": 42074448, "sectors": [3, 30]}],
        thresholds={
            "max_rel_depth_diff": 0.25,
            "max_rel_duration_diff": 0.25,
            "min_morph_corr": 0.9,
        },
        purpose="mining",
        ephemeris_source="test",
    )
    manifest_path = str(tmp_path / "survey.json")
    Path(manifest_path).write_text(json.dumps(manifest))
    config = LearnConfig(seed=7, epochs=2, batch_size=8, embedding_dim=8)
    freeze_path = str(tmp_path / "freeze.json")
    F.create_freeze(
        str(Path(__file__).resolve().parent.parent / "fixtures" / "replay_v1.json"),
        manifest_path, config, output_path=freeze_path, cohort_key="discovery",
    )
    bulk = bulk_fetch([(42074448, 3), (42074448, 30)], str(tmp_path))
    assert len(bulk["downloaded"]) == 2

    def _boom(*args, **kwargs):
        raise AssertionError("MAST queried despite warm cache")

    from astroquery.mast import Observations

    monkeypatch.setattr(A, "find_spoc_ffi_uri", _boom)
    monkeypatch.setattr(Observations, "download_file", _boom)
    results = run_mining_survey(
        manifest_path, freeze_path=freeze_path, config=config,
        cache_dir=str(tmp_path), out_dir=str(tmp_path / "out"),
    )
    assert results["status"] == "complete"
    assert results["sealed_sectors_touched"] == []
    json.dumps(results)
