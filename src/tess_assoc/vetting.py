"""Candidate vetting for discovery (issue #9).

Automated checks computable from light curves and catalogs: secondary-
eclipse search per alias period, TIC contamination, TOI cross-match, and
the promotion rule that admits only clean pairs to the shortlist.
Anything requiring pixels, alternate reductions, or external photometry
is an explicit manual step — recorded, never silently skipped.
"""

from __future__ import annotations

import math
from typing import Any

from tess_assoc._validate import require_finite, require_positive_finite
from tess_assoc.propose import dip_snr_at

SECONDARY_SNR_THRESHOLD = 4.0
CONTAMINATION_LIMIT = 0.1

MANUAL_CHECKLIST: tuple[str, ...] = (
    "centroid movement during event (pixel-level)",
    "difference imaging (in/out of transit)",
    "alternate TESS reduction (QLC/QLP/TGLC) on this candidate only",
    "external photometry on this candidate and its aliases only",
    "odd/even depth comparison (needs 3+ events)",
)


def secondary_search(
    time: list[float],
    detrended: list[float],
    sigma: float,
    t_ref: float,
    alias_periods: list[float],
    half_width_days: float,
    snr_threshold: float = SECONDARY_SNR_THRESHOLD,
) -> dict[str, Any]:
    """Secondary-eclipse hunt at phase 0.5 for each alias period.

    A retained alias predicting a confident secondary where the light
    curve is flat is fine; one landing on a real dip smells like an
    eclipsing binary. Returns per-alias flags plus the worst offender.
    """
    require_finite("t_ref", t_ref)
    require_positive_finite("half_width_days", half_width_days)
    if not sigma > 0:
        raise ValueError("sigma must be > 0")
    t_min, t_max = time[0], time[-1]
    aliases = []
    worst: dict[str, Any] | None = None
    for period in alias_periods:
        require_positive_finite("alias period", period)
        n_max = math.ceil((t_max - t_min) / period) + 1
        found: float | None = None
        for k in range(-n_max, n_max + 1):
            epoch = t_ref + (k + 0.5) * period
            if epoch < t_min or epoch > t_max:
                continue
            if not any(abs(t - epoch) <= half_width_days for t in time):
                continue
            snr = dip_snr_at(time, detrended, sigma, epoch, half_width_days)
            if snr >= snr_threshold and (found is None or snr > found):
                found = snr
        flag = found is not None
        aliases.append(
            {"period_days": period, "secondary_found": flag, "max_snr": found}
        )
        if flag and (worst is None or found > worst["max_snr"]):
            worst = {"period_days": period, "max_snr": found}
    return {
        "n_aliases": len(aliases),
        "n_flagged": sum(1 for a in aliases if a["secondary_found"]),
        "aliases": aliases,
        "worst": worst,
    }


def check_contamination(
    tic_id: int, *, contratio: float | None = None
) -> dict[str, Any]:
    """TIC contamination ratio (passed value or live catalog query)."""
    if contratio is None:
        from astroquery.mast import Catalogs

        try:
            tab = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
            contratio = float(tab["contratio"][0])
        except Exception as e:
            return {"status": "unknown", "reason": f"TIC query failed: {e}"}
    require_finite("contratio", contratio)
    return {
        "status": "high" if contratio > CONTAMINATION_LIMIT else "low",
        "contratio": contratio,
        "limit": CONTAMINATION_LIMIT,
        "reason": (
            "needs pixel-level vetting" if contratio > CONTAMINATION_LIMIT else None
        ),
    }


def cross_match_toi(
    tic_id: int, *, toi_rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """TOI cross-match (passed rows or live Exoplanet Archive TAP query).

    Promotion requires status "clean": the query succeeded AND no TOI
    matched. Anything else blocks auto-promotion (manual review instead).
    """
    if toi_rows is None:
        try:
            from astroquery.ipac.nexsci.nasa_exoplanet_archive import (
                NasaExoplanetArchive,
            )

            tab = NasaExoplanetArchive.query_criteria(
                table="toi",
                where=f"tid={int(tic_id)}",
                select="top 5 tid,toi,tfopwg_disp",
            )
            toi_rows = [
                {"toi": str(r["toi"]), "disposition": str(r["tfopwg_disp"])}
                for r in tab
            ]
        except Exception as e:
            return {"status": "unknown", "matches": [], "reason": f"TOI query failed: {e}"}
    if toi_rows:
        return {"status": "known-toi", "matches": list(toi_rows), "reason": None}
    return {"status": "clean", "matches": [], "reason": None}


def promote_candidate(
    *,
    compatible: bool,
    aliases_retained: int,
    cross_match: dict[str, Any],
    contamination: dict[str, Any],
    secondary: dict[str, Any],
) -> dict[str, Any]:
    """Admission rule: clean on every automated check, or not a candidate."""
    reasons: list[str] = []
    if not compatible:
        reasons.append("pair incompatible under frozen thresholds")
    if not aliases_retained:
        reasons.append("no viable period alias survives window filtering")
    if cross_match.get("status") != "clean":
        reasons.append(f"TOI cross-match: {cross_match.get('status')}")
    if contamination.get("status") != "low":
        reasons.append(f"contamination: {contamination.get('status')}")
    if secondary.get("n_flagged"):
        reasons.append(
            f"secondary eclipse at {secondary['n_flagged']} alias period(s)"
        )
    return {
        "candidate": not reasons,
        "reasons": reasons,
        "manual_checklist": list(MANUAL_CHECKLIST),
    }


__all__ = [
    "CONTAMINATION_LIMIT",
    "MANUAL_CHECKLIST",
    "SECONDARY_SNR_THRESHOLD",
    "check_contamination",
    "cross_match_toi",
    "promote_candidate",
    "secondary_search",
]
