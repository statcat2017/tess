"""Event-record contract tests (Phase 0)."""

import dataclasses

import pytest

from tess_assoc import protocol as P
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
    d = rec.to_dict()
    assert d["local_time"] == list(_valid_kwargs()["local_time"])
    rec2 = EventRecord.from_dict(d)
    assert rec2.to_dict() == d
    assert rec2 == rec


def test_dataclass_fields_match_protocol():
    field_names = {f.name for f in dataclasses.fields(EventRecord)}
    assert set(P.EVENT_REQUIRED_FIELDS) == field_names
    assert set(EventRecord(**_valid_kwargs()).to_dict()) == field_names


def test_construction_validates_immediately():
    kw = _valid_kwargs()
    kw["depth"] = 0.0
    with pytest.raises(ValueError):
        EventRecord(**kw)


def test_rejects_nonfinite_and_bad_ordering():
    kw = _valid_kwargs()
    kw["local_flux"] = list(kw["local_flux"])
    kw["local_flux"][0] = float("nan")
    with pytest.raises(ValueError):
        EventRecord(**kw)

    kw = _valid_kwargs()
    kw["local_time"] = list(reversed(kw["local_time"]))
    with pytest.raises(ValueError):
        EventRecord(**kw)


def test_rejects_mismatched_and_nonpositive():
    kw = _valid_kwargs()
    kw["local_flux"] = kw["local_flux"][:-1]
    with pytest.raises(ValueError):
        EventRecord(**kw)

    for bad in ({"depth": 0.0}, {"duration_days": -1.0}, {"snr": 0.0}):
        kw = _valid_kwargs()
        kw.update(bad)
        with pytest.raises(ValueError):
            EventRecord(**kw)


def test_rejects_bool_ids_and_unknown_sector():
    kw = _valid_kwargs()
    kw["tic_id"] = True
    with pytest.raises(ValueError):
        EventRecord(**kw)

    kw = _valid_kwargs()
    kw["sector"] = 999
    with pytest.raises(ValueError, match="known TESS sector"):
        EventRecord(**kw)


def test_record_is_immutable():
    rec = EventRecord(**_valid_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.depth = 0.5  # type: ignore[misc]


def test_from_dict_missing_and_malformed():
    kw = _valid_kwargs()
    del kw["depth"]
    with pytest.raises(ValueError, match="missing event fields"):
        EventRecord.from_dict(kw)
    with pytest.raises(ValueError, match="must be a dict"):
        EventRecord.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown event fields"):
        EventRecord.from_dict({**_valid_kwargs(), "foo": 1})


def test_from_dict_rejects_coercible_shapes():
    base = _valid_kwargs()
    for bad in (
        {**base, "local_time": "abc"},
        {**base, "local_flux": 1.0},
        {**base, "stellar_meta": [("r_star", 1.0)]},
        {**base, "quality": "flags=0"},
    ):
        with pytest.raises(ValueError):
            EventRecord.from_dict(bad)
