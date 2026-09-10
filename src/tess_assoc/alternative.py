"""Alternative MAST light-curve reductions for candidate-specific checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tess_assoc.audit import measure_event_shape, measure_flux_channel
from tess_assoc.event import EventRecord
from tess_assoc.observability import CadenceEvidence, SourceProduct


DEFAULT_PROVIDERS = ("QLP", "TARS", "TGLC")
FLUX_COLUMNS = (
    "DET_FLUX",
    "PDCSAP_FLUX",
    "FLUX",
    "TGLC_FLUX",
    "SAP_FLUX",
    "LC_FLUX",
)


@dataclass(frozen=True)
class AlternativeProduct:
    provider: str
    sector: int
    data_uri: str
    obs_id: str


def choose_flux_column(columns: Iterable[str]) -> str:
    """Choose the least processed available light-curve column by priority."""
    names = set(columns)
    for column in FLUX_COLUMNS:
        if column in names:
            return column
    raise ValueError(f"no supported flux column in {sorted(names)}")


def query_alternative_products(
    tic_id: int,
    sectors: Sequence[int],
    *,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
) -> list[AlternativeProduct]:
    """List one downloadable FITS light curve per provider and sector."""
    from astroquery.mast import Observations

    wanted = set(providers)
    wanted_sectors = set(int(sector) for sector in sectors)
    rows = Observations.query_criteria(
        target_name=str(tic_id),
        obs_collection="HLSP",
        dataproduct_type="timeseries",
    )
    products: list[AlternativeProduct] = []
    for row in rows:
        provider = str(row["provenance_name"])
        sector = int(row["sequence_number"])
        if provider not in wanted or sector not in wanted_sectors:
            continue
        obs_id = str(row["obs_id"])
        for product in Observations.get_product_list(row):
            uri = str(product["dataURI"])
            if uri.endswith(".fits"):
                products.append(AlternativeProduct(provider, sector, uri, obs_id))
                break
    return sorted(products, key=lambda product: (product.provider, product.sector))


def download_product(product: AlternativeProduct, cache_dir: str | Path) -> Path:
    """Download or reuse an alternative product outside the repository."""
    from astroquery.mast import Observations

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    filename = f"{product.provider.lower()}_s{product.sector:04d}_" + product.data_uri.rsplit("/", 1)[-1]
    path = cache / filename
    if path.exists():
        return path
    result = Observations.download_file(product.data_uri, local_path=str(path))
    if isinstance(result, tuple) and result and result[0] != "COMPLETE":
        raise RuntimeError(f"MAST download failed for {product.data_uri}: {result}")
    if not path.exists():
        raise RuntimeError(f"MAST did not create {path}")
    return path


def read_lightcurve(
    path: str | Path,
    *,
    source_product: SourceProduct | None = None,
) -> dict[str, Any]:
    """Read a provider-neutral time, flux, quality payload from a FITS file."""
    try:
        import numpy as np
        from astropy.io import fits
    except ImportError as error:
        raise RuntimeError("alternative reduction checks need the replay extra") from error
    with fits.open(path) as handle:
        data = handle[1].data
        columns = list(data.columns.names)
        flux_column = choose_flux_column(columns)
        raw_time = np.asarray(data["TIME"])
        raw_flux = np.asarray(data[flux_column])
        quality = (
            np.asarray(data["QUALITY"], dtype=int)
            if "QUALITY" in columns
            else np.zeros(len(time), dtype=int)
        )
    if raw_time.dtype.kind not in "fiu" or raw_flux.dtype.kind not in "fiu":
        raise ValueError("alternative TIME and flux columns must be numeric")
    if quality.dtype.kind not in "iu":
        raise ValueError("alternative QUALITY column must contain integer flags")
    if len(raw_time) != len(raw_flux) or len(raw_time) != len(quality):
        raise ValueError("alternative light-curve columns must have equal length")
    time = raw_time.astype(float, copy=False)
    flux = raw_flux.astype(float, copy=False)
    finite_time = np.isfinite(time)
    invalid_time_count = int((~finite_time).sum())
    time = time[finite_time]
    flux = flux[finite_time]
    quality = quality[finite_time]
    usable = np.isfinite(flux) & (quality == 0)
    quality_values = quality.tolist()
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in quality_values
    ):
        raise ValueError("alternative quality flags must be non-negative ints")
    evidence = CadenceEvidence(
        time=tuple(float(value) for value in time),
        usable=tuple(bool(value) for value in usable),
        quality_flags=tuple(quality_values),
        source_product=source_product,
        invalid_time_count=invalid_time_count,
    )
    return {
        "time": list(evidence.usable_time),
        "flux": [float(value) for value in flux[usable]],
        "flux_column": flux_column,
        "cadences_total": int(len(time)),
        "cadences_good": int(usable.sum()),
        "quality_flagged": int((~(quality == 0)).sum()),
        "observability": evidence,
    }


def measure_event(
    curve: Mapping[str, Any],
    *,
    t0: float,
    duration_days: float,
    half_span_days: float = 0.6,
) -> dict[str, Any]:
    """Measure the known event and shape in one alternative reduction."""
    time = curve["time"]
    flux = curve["flux"]
    result = measure_flux_channel(time, flux, t0, duration_days, half_span_days=half_span_days)
    return {
        "flux_column": curve["flux_column"],
        "cadences_total": curve["cadences_total"],
        "cadences_good": curve["cadences_good"],
        "quality_flagged": curve["quality_flagged"],
        "event": result,
        "shape": measure_event_shape(time, flux, t0, duration_days, half_span_days=half_span_days),
    }


def event_record_from_curve(
    curve: Mapping[str, Any],
    *,
    tic_id: int,
    sector: int,
    t0: float,
    duration_days: float,
    role: str,
) -> EventRecord | None:
    """Extract one fixed event window for repeat ranking."""
    from tess_assoc.extract import SkippedTransit, extract_at

    observability = curve.get("observability")
    if isinstance(observability, dict):
        observability = CadenceEvidence.from_dict(observability)
    result = extract_at(
        curve["time"],
        curve["flux"],
        t0,
        duration_days,
        tic_id=tic_id,
        sector=sector,
        quality={"role": role, "provider_flux_column": curve["flux_column"]},
        observability=observability,
    )
    return None if isinstance(result, SkippedTransit) else result


__all__ = [
    "AlternativeProduct",
    "DEFAULT_PROVIDERS",
    "choose_flux_column",
    "download_product",
    "event_record_from_curve",
    "measure_event",
    "query_alternative_products",
    "read_lightcurve",
]
