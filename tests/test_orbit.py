"""Alias-arithmetic tests (Phase 0, Proposal §19)."""

import pytest

from tess_assoc.orbit import generate_aliases, predict_epochs


def test_alias_example_900d():
    aliases = generate_aliases(900.0)
    assert aliases[:5] == [900.0, 450.0, 300.0, 225.0, 180.0]
    assert all(p >= 27.0 for p in aliases)
    # n increases => periods strictly decrease
    assert all(b < a for a, b in zip(aliases, aliases[1:]))


def test_alias_boundaries_and_errors():
    with pytest.raises(ValueError):
        generate_aliases(10.0)  # below 27d lower bound
    with pytest.raises(ValueError):
        generate_aliases(0.0)
    with pytest.raises(ValueError):
        generate_aliases(-5.0)


def test_predict_epochs_invariants():
    t1, period = 2000.0, 300.0
    epochs = predict_epochs(t1, period, 3)
    assert epochs == [2000.0, 2300.0, 2600.0, 2900.0]
    assert epochs[0] == t1  # preserves observed event time
    assert all(b > a for a, b in zip(epochs, epochs[1:]))
    # alias preserves DeltaT: t1 + n*P_n == t2
    t2 = 2900.0
    for n, p in enumerate(generate_aliases(t2 - t1), start=1):
        assert abs((t1 + n * p) - t2) < 1e-9
