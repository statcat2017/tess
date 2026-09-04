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
COMPANION_RADIUS_LIMIT_RSUN = 0.25

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


def combine_secondary_searches(
    retained_periods: list[float], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Union secondary flags across event sectors (either window can convict)."""
    flagged: set[float] = set()
    worst: dict[str, Any] | None = None
    for result in results:
        for alias in result["aliases"]:
            if alias["secondary_found"]:
                flagged.add(alias["period_days"])
                if worst is None or alias["max_snr"] > worst["max_snr"]:
                    worst = {
                        "period_days": alias["period_days"],
                        "max_snr": alias["max_snr"],
                    }
    return {
        "n_aliases": len(retained_periods),
        "n_flagged": len(flagged),
        "flagged_periods": sorted(flagged),
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


def _query_toi_rows(tic_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import (
        NasaExoplanetArchive,
    )

    matches: dict[int, list[dict[str, Any]]] = {t: [] for t in tic_ids}
    for i in range(0, len(tic_ids), 500):
        chunk = tic_ids[i : i + 500]
        tab = NasaExoplanetArchive.query_criteria(
            table="toi",
            where="tid in (%s)" % ",".join(str(t) for t in chunk),
            select="top 2500 tid,toi,tfopwg_disp",
        )
        for r in tab:
            matches.setdefault(int(r["tid"]), []).append(
                {"toi": str(r["toi"]), "disposition": str(r["tfopwg_disp"])}
            )
    return matches


def cross_match_tois(
    tic_ids: list[int],
) -> dict[str, Any]:
    """Batched TOI cross-match: one TAP call per 500 TICs.

    Returns {"matches": {tic: rows}, "ok": bool}. ok=False (query failed)
    must block promotion exactly like status "unknown" does.
    """
    try:
        matches = _query_toi_rows(list(dict.fromkeys(tic_ids)))
    except Exception as e:
        return {"matches": {}, "ok": False, "reason": f"TOI query failed: {e}"}
    return {"matches": matches, "ok": True, "reason": None}


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


def stellar_radius(tic_id: int) -> float | None:
    """Stellar radius from the TIC catalog (None when unavailable)."""
    from astroquery.mast import Catalogs

    try:
        return float(Catalogs.query_criteria(catalog="Tic", ID=tic_id)["rad"][0])
    except Exception:
        return None


def check_companion_radius(
    tic_id: int, depth: float, *, rad: float | None = None
) -> dict[str, Any]:
    """Companion-size lower bound from transit depth and stellar radius.

    sqrt(depth) * R* lower-bounds the companion radius (central-transit
    assumption; grazing or dilution only make the truth larger). Above
    0.25 Rsun (~2.7 Rjup, beyond every confirmed planet) the pair cannot
    be planetary around a single star — the typical EB interloper signature.
    """
    require_positive_finite("depth", depth)
    if rad is None:
        from astroquery.mast import Catalogs

        try:
            tab = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
            rad = float(tab["rad"][0])
        except Exception as e:
            return {"status": "unknown", "reason": f"TIC radius query failed: {e}"}
    require_positive_finite("rad", rad)
    companion = depth**0.5 * rad
    stellar = companion > COMPANION_RADIUS_LIMIT_RSUN
    return {
        "status": "stellar" if stellar else "planetary-range",
        "companion_r_sun": companion,
        "stellar_r_sun": rad,
        "limit_r_sun": COMPANION_RADIUS_LIMIT_RSUN,
        "reason": (
            f"companion >={companion:.2f} Rsun: stellar"
            if stellar else None
        ),
    }


def promote_candidate(
    *,
    compatible: bool,
    aliases_retained: int,
    cross_match: dict[str, Any],
    contamination: dict[str, Any],
    secondary: dict[str, Any],
    companion: dict[str, Any] | None = None,
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
    if companion is not None and companion.get("status") != "planetary-range":
        reasons.append(
            f"companion radius: {companion.get('status')}"
            + (
                f" {companion['companion_r_sun']:.2f} Rsun"
                if companion.get("companion_r_sun") is not None else ""
            )
        )
    return {
        "candidate": not reasons,
        "reasons": reasons,
        "manual_checklist": list(MANUAL_CHECKLIST),
    }


__all__ = [
    "COMPANION_RADIUS_LIMIT_RSUN",
    "CONTAMINATION_LIMIT",
    "MANUAL_CHECKLIST",
    "SECONDARY_SNR_THRESHOLD",
    "check_companion_radius",
    "check_contamination",
    "combine_secondary_searches",
    "cross_match_toi",
    "cross_match_tois",
    "promote_candidate",
    "secondary_search",
    "stellar_radius",
]
