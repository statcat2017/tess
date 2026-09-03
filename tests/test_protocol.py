"""Phase 0 protocol-freeze tests (executable subset only)."""

import pytest

from tess_assoc import protocol as P
from tess_assoc.protocol import SectorRole


def test_sector_boundaries_disjoint_and_complete():
    assert P.DEV_SECTORS == frozenset(range(1, 80))
    assert P.SEALED_SECTORS == frozenset(range(80, 106))
    assert P.DISCOVERY_SECTORS == frozenset({106})
    assert not (P.DEV_SECTORS & P.SEALED_SECTORS)
    assert not (P.DEV_SECTORS & P.DISCOVERY_SECTORS)
    assert not (P.SEALED_SECTORS & P.DISCOVERY_SECTORS)
    assert P.ALL_KNOWN_SECTORS == frozenset(range(1, 107))


def test_sector_role_single_dispatch():
    assert P.sector_role(1) is SectorRole.DEV
    assert P.sector_role(79) is SectorRole.DEV
    assert P.sector_role(80) is SectorRole.SEALED
    assert P.sector_role(105) is SectorRole.SEALED
    assert P.sector_role(106) is SectorRole.DISCOVERY
    assert P.sector_role(999) is SectorRole.UNKNOWN
    assert P.sector_role(True) is SectorRole.UNKNOWN
    assert P.sector_role("32") is SectorRole.UNKNOWN  # type: ignore[arg-type]


def test_orbital_freeze_values():
    assert P.LONG_PERIOD_LOWER_BOUND_DAYS == 27.0
    assert P.MAX_ALIAS_N == 10_000


def test_temporal_leak_guard():
    P.validate_no_temporal_leak({1, 32, 79})
    with pytest.raises(ValueError, match="temporal leak"):
        P.validate_no_temporal_leak({79, 80})
    with pytest.raises(ValueError, match="temporal leak"):
        P.validate_no_temporal_leak({106})


def test_tic_partition_guard():
    P.validate_tic_partition({1, 2}, {3, 4}, {5})
    with pytest.raises(ValueError, match="TIC partition leak"):
        P.validate_tic_partition({1, 2}, {2, 3})
