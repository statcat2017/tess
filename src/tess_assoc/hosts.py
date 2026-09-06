"""Known planet-host selection for the outer-companion survey.

The host survey is deliberately separate from ordinary TOI vetting. Known
planet ephemerides are inputs to the blind search: their transit windows are
masked, while any additional events remain subject to the normal association
and vetting rules.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from tess_assoc._validate import require_finite, require_positive_finite, require_strict_int


@dataclass(frozen=True)
class KnownPlanet:
    """One known transiting planet used to mask a host's light curves."""

    tic_id: int
    name: str
    period_days: float
    t0_bjd_tdb: float
    duration_days: float
    source: str
    disposition: str = ""

    def __post_init__(self) -> None:
        require_strict_int("tic_id", self.tic_id, minimum=1)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("known planet name must be non-empty")
        require_positive_finite("period_days", self.period_days)
        require_finite("t0_bjd_tdb", self.t0_bjd_tdb)
        require_positive_finite("duration_days", self.duration_days)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("known planet source must be non-empty")
        if not isinstance(self.disposition, str):
            raise ValueError("known planet disposition must be a str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "period_days": self.period_days,
            "t0_bjd_tdb": self.t0_bjd_tdb,
            "duration_days": self.duration_days,
            "source": self.source,
            "disposition": self.disposition,
        }

    @classmethod
    def from_host_dict(cls, tic_id: int, value: Mapping[str, Any]) -> "KnownPlanet":
        """Construct the typed record stored inside a host manifest."""
        required = ("name", "period_days", "t0_bjd_tdb", "duration_days")
        if not isinstance(value, Mapping):
            raise ValueError("known planet must be a mapping")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"known planet missing keys: {missing}")
        return cls(
            tic_id=tic_id,
            name=value["name"],
            period_days=value["period_days"],
            t0_bjd_tdb=value["t0_bjd_tdb"],
            duration_days=value["duration_days"],
            source=value.get("source", "manifest"),
            disposition=value.get("disposition", ""),
        )


def _get(row: Mapping[str, Any] | Any, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        return None


def _number(value: Any) -> float | None:
    if hasattr(value, "value"):
        value = value.value
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _tic(row: Mapping[str, Any] | Any, key: str) -> int | None:
    value = _get(row, key)
    if isinstance(value, str):
        value = value.strip().upper().removeprefix("TIC ")
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def parse_confirmed_rows(rows: Iterable[Mapping[str, Any] | Any]) -> list[KnownPlanet]:
    """Parse transit-capable rows from the Exoplanet Archive ``pscomppars`` table."""
    planets: list[KnownPlanet] = []
    for row in rows:
        tic_id = _tic(row, "tic_id")
        period = _number(_get(row, "pl_orbper"))
        epoch = _number(_get(row, "pl_tranmid"))
        duration_hours = _number(_get(row, "pl_trandur"))
        duration = duration_hours / 24.0 if duration_hours is not None else None
        if tic_id is None or period is None or epoch is None or duration is None:
            continue
        tran_flag = _get(row, "tran_flag")
        if tran_flag is not None and str(tran_flag).strip().lower() in {"0", "false", "no"}:
            continue
        planets.append(
            KnownPlanet(
                tic_id=tic_id,
                name=str(_get(row, "pl_name") or f"TIC {tic_id} planet"),
                period_days=period,
                t0_bjd_tdb=epoch,
                duration_days=duration,
                source="pscomppars",
            )
        )
    return planets


def parse_toi_rows(rows: Iterable[Mapping[str, Any] | Any]) -> list[KnownPlanet]:
    """Parse confirmed and planet-candidate rows from the Exoplanet Archive TOI table."""
    planets: list[KnownPlanet] = []
    for row in rows:
        tic_id = _tic(row, "tid")
        period = _number(_get(row, "pl_orbper"))
        epoch = _number(_get(row, "pl_tranmid"))
        duration_hours = _number(_get(row, "pl_trandurh"))
        disposition = str(_get(row, "tfopwg_disp") or "")
        if (
            tic_id is None
            or period is None
            or epoch is None
            or duration_hours is None
            or duration_hours <= 0
            or disposition not in {"CP", "PC"}
        ):
            continue
        toi = str(_get(row, "toi") or f"TIC {tic_id}")
        planets.append(
            KnownPlanet(
                tic_id=tic_id,
                name=toi,
                period_days=period,
                t0_bjd_tdb=epoch,
                duration_days=duration_hours / 24.0,
                source="toi",
                disposition=disposition,
            )
        )
    return planets


def deduplicate_planets(planets: Iterable[KnownPlanet]) -> list[KnownPlanet]:
    """Merge repeated archive rows while retaining distinct planets on a host."""
    out: list[KnownPlanet] = []
    for planet in planets:
        duplicate = next(
            (
                existing
                for existing in out
                if existing.tic_id == planet.tic_id
                and abs(existing.period_days - planet.period_days)
                <= max(existing.period_days, planet.period_days) * 1e-4
                and _phase_distance(existing, planet)
                <= max(existing.duration_days, planet.duration_days, 0.05)
            ),
            None,
        )
        if duplicate is None:
            out.append(planet)
        elif duplicate.source == "toi" and planet.source == "pscomppars":
            out[out.index(duplicate)] = planet
    return sorted(out, key=lambda p: (p.tic_id, p.period_days, p.name))


def _phase_distance(left: KnownPlanet, right: KnownPlanet) -> float:
    """Smallest epoch separation modulo the shared candidate period."""
    period = (left.period_days + right.period_days) / 2.0
    delta = abs(left.t0_bjd_tdb - right.t0_bjd_tdb) % period
    return min(delta, period - delta)


def query_known_planets(*, include_toi_candidates: bool = True) -> list[KnownPlanet]:
    """Fetch confirmed transiting planets and optionally high-confidence TOIs."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    confirmed = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="tic_id,pl_name,pl_orbper,pl_tranmid,pl_trandur,tran_flag",
        where="tic_id is not null and tran_flag=1",
    )
    rows = parse_confirmed_rows(confirmed)
    if include_toi_candidates:
        toi = NasaExoplanetArchive.query_criteria(
            table="toi",
            select="tid,toi,pl_orbper,pl_tranmid,pl_trandurh,tfopwg_disp",
            where="tfopwg_disp in ('CP','PC')",
        )
        rows.extend(parse_toi_rows(toi))
    return deduplicate_planets(rows)


def select_known_host_targets(
    planets: Iterable[KnownPlanet],
    sector_tics: Mapping[int, Iterable[int]],
    *,
    min_sectors: int = 2,
    max_targets: int | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic host targets from known planets and SPOC coverage."""
    require_strict_int("min_sectors", min_sectors, minimum=1)
    if max_targets is not None:
        require_strict_int("max_targets", max_targets, minimum=1)
    by_tic: dict[int, list[KnownPlanet]] = defaultdict(list)
    for planet in deduplicate_planets(planets):
        by_tic[planet.tic_id].append(planet)
    coverage = {
        int(sector): {int(tic) for tic in tics}
        for sector, tics in sector_tics.items()
    }
    targets: list[dict[str, Any]] = []
    for tic_id in sorted(by_tic):
        sectors = sorted(sector for sector, tics in coverage.items() if tic_id in tics)
        if len(sectors) < min_sectors:
            continue
        targets.append(
            {
                "tic_id": tic_id,
                "name": f"TIC {tic_id}",
                "sectors": sectors,
                "known_planets": [p.to_dict() for p in by_tic[tic_id]],
            }
        )
        if max_targets is not None and len(targets) >= max_targets:
            break
    return targets


__all__ = [
    "KnownPlanet",
    "deduplicate_planets",
    "parse_confirmed_rows",
    "parse_toi_rows",
    "query_known_planets",
    "select_known_host_targets",
]
