"""Alternative-reduction helper tests."""

import pytest

from tess_assoc.alternative import choose_flux_column, measure_event


def test_choose_flux_column_prefers_detrended_provider_flux():
    assert choose_flux_column(["TIME", "SAP_FLUX", "DET_FLUX", "QUALITY"]) == "DET_FLUX"


def test_choose_flux_column_rejects_unknown_columns():
    with pytest.raises(ValueError, match="no supported flux column"):
        choose_flux_column(["TIME", "QUALITY"])


def test_measure_event_preserves_quality_provenance():
    time = [i * 0.01 for i in range(201)]
    flux = [0.9 if abs(t - 1.0) <= 0.05 else 1.0 for t in time]
    result = measure_event(
        {
            "time": time,
            "flux": flux,
            "flux_column": "DET_FLUX",
            "cadences_total": 201,
            "cadences_good": 201,
            "quality_flagged": 0,
        },
        t0=1.0,
        duration_days=0.1,
    )
    assert result["flux_column"] == "DET_FLUX"
    assert result["event"]["status"] == "measured"
    assert result["event"]["depth"] > 0
