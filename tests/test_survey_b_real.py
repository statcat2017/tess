"""Real Survey B catalogue conversion tests."""

from pathlib import Path

from tools.run_survey_b_real import load_tebc_targets


def test_tebc_epochs_convert_from_btjd_and_filter_sealed_sectors(tmp_path: Path):
    path = tmp_path / "tebc.csv"
    path.write_text(
        "tess_id,period,bjd0,prim_width_2g,sec_width_2g,sec_pos_2g,sectors\n"
        "101,10,1553,0.02,0.03,0.5,\"29,80\"\n"
        "101,10,1553,0.02,0.03,0.5,29\n"
    )
    targets, excluded = load_tebc_targets(path)

    assert len(targets) == 1
    assert targets[0].sectors == (29,)
    assert targets[0].t0_bjd_tdb == 2458553.0
    assert targets[0].primary_duration_days == 0.2
    assert excluded["duplicate-tic"] == 1
