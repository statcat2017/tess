"""Event-record contract tests (Phase 0)."""

import pytest

from tess_assoc.event import EventRecord


def _valid_kwargs():
    n = 11
    t0 = 2000.0
    times = [t0 - 0.5 + i * 0.1 for i in range(n)]
    fluxes = [1.0] * n
    fluxes[n // 2] = 0.99
    return {
        "tic_id": 123456,
        "sector": 32,
        "t0": t0,
        "local_time": times,
        "local_flux": fluxes,
        "depth": 0.01,
        "duration_days": 0.2,
        "snr": 9.5,
        "stellar_meta": {"r_star": 1.0},
        "quality": {"flags": 0},
    }


def test_valid_record_and_roundtrip():
    rec = EventRecord(**_valid_kwargs())
    rec.validate()
    d = rec.to_dict()
    rec2 = EventRecord.from_dict(d)
    assert rec2.to_dict() == d


def test_rejects_nonfinite_and_bad_ordering():
    kw = _valid_kwargs()
    kw["local_flux"] = list(kw["local_flux"])
    kw["local_flux"][0] = float("nan")
    with pytest.raises(ValueError):
        EventRecord(**kw).validate()

    kw = _valid_kwargs()
    kw["local_time"] = list(reversed(kw["local_time"]))
    with pytest.raises(ValueError):
        EventRecord(**kw).validate()


def test_rejects_mismatched_and_nonpositive():
    kw = _valid_kwargs()
    kw["local_flux"] = kw["local_flux"][:-1]
    with pytest.raises(ValueError):
        EventRecord(**kw).validate()

    for bad in ({"depth": 0.0}, {"duration_days": -1.0}, {"snr": 0.0}):
        kw = _valid_kwargs()
        kw.update(bad)
        with pytest.raises(ValueError):
            EventRecord(**kw).validate()
