"""Phase 0 protocol-freeze tests."""

from tess_assoc import protocol as P


def test_sector_boundaries_disjoint_and_complete():
    assert P.DEV_SECTORS == frozenset(range(1, 80))
    assert P.SEALED_SECTORS == frozenset(range(80, 106))
    assert P.DISCOVERY_SECTORS == frozenset({106})
    assert not (P.DEV_SECTORS & P.SEALED_SECTORS)
    assert not (P.DEV_SECTORS & P.DISCOVERY_SECTORS)
    assert P.ALL_KNOWN_SECTORS == frozenset(range(1, 107))


def test_primary_product_frozen():
    assert P.PRIMARY_PHOTOMETRIC_PRODUCT == "TESS-SPOC FFI"
    assert set(P.RESERVED_FOR_VETTING_ONLY) == {"SPOC", "QLP", "TGLC"}


def test_period_and_alias_formula():
    assert P.LONG_PERIOD_LOWER_BOUND_DAYS == 27.0
    assert P.ALIAS_FORMULA == "P_n = DeltaT / n"


def test_tic_never_a_feature():
    assert P.TIC_AS_FEATURE_FORBIDDEN is True
    assert "tic_id" in P.MODEL_FEATURE_BLACKLIST


def test_metrics_and_semantics():
    assert P.PRIMARY_METRIC == "true-repeat retrieval at fixed candidate burden"
    assert "mean reciprocal rank" in P.SUPPORTING_METRICS
    assert P.LEARNED_OUTPUT_SEMANTICS == "P(same transit-producing object)"
    assert "orbital-period prediction" in P.LEARNED_OUTPUT_IS_NOT


def test_temporal_leak_guard():
    P.validate_no_temporal_leak({1, 32, 79})
    try:
        P.validate_no_temporal_leak({79, 80})
    except ValueError:
        pass
    else:
        raise AssertionError("expected temporal leak to raise")


def test_tic_partition_guard():
    P.validate_tic_partition({1, 2}, {3, 4}, {5})
    try:
        P.validate_tic_partition({1, 2}, {2, 3})
    except ValueError:
        pass
    else:
        raise AssertionError("expected TIC leak to raise")
