"""TESS-SPOC FFI archive access (issue #3).

Retrieves light curves from MAST into a cache directory that always lives
outside the repository (acceptance criterion: raw archive data never enter
Git). Every retrieval records product URI + retrieval date provenance.
Without archive dependencies or network, retrieval raises
ArchiveUnavailable with a clear message instead of failing obscurely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone


class ArchiveUnavailable(ValueError):
    """Raised when archive data cannot be retrieved (no deps, no network)."""


def cache_dir() -> str:
    base = os.environ.get(
        "TESS_ASSOC_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "tess-assoc")
    )
    os.makedirs(base, exist_ok=True)
    return base


@dataclass(frozen=True)
class ArchiveProduct:
    tic_id: int
    sector: int
    local_path: str
    data_uri: str
    retrieved_utc: str
    cached: bool


def _require_deps() -> None:
    try:
        import astropy  # noqa: F401
        import astroquery  # noqa: F401
    except ImportError as e:
        raise ArchiveUnavailable(
            "replay needs the 'replay' extra: pip install tess-assoc[replay] "
            f"({e})"
        ) from e


def find_spoc_ffi_uri(tic_id: int, sector: int) -> str:
    """Return the MAST dataURI of the TESS-SPOC FFI light curve."""
    _require_deps()
    from astroquery.mast import Observations

    try:
        rows = Observations.query_criteria(
            target_name=str(tic_id),
            obs_collection="HLSP",
            dataproduct_type="timeseries",
        )
    except Exception as e:
        raise ArchiveUnavailable(f"MAST query failed for TIC {tic_id}: {e}") from e
    for r in rows:
        if (
            str(r["provenance_name"]) == "TESS-SPOC"
            and int(r["sequence_number"]) == sector
        ):
            try:
                products = Observations.get_product_list(r)
            except Exception as e:
                raise ArchiveUnavailable(
                    f"MAST product list failed for TIC {tic_id} sector {sector}: {e}"
                ) from e
            for p in products:
                uri = str(p["dataURI"])
                if uri.endswith("_lc.fits"):
                    return uri
    raise ArchiveUnavailable(
        f"no TESS-SPOC FFI light curve for TIC {tic_id} sector {sector}"
    )


def download_spoc_ffi(tic_id: int, sector: int, directory: str | None = None) -> ArchiveProduct:
    """Download (or reuse cached) TESS-SPOC FFI light curve. Never in-repo."""
    _require_deps()
    from astroquery.mast import Observations

    directory = directory or cache_dir()
    os.makedirs(directory, exist_ok=True)
    uri = find_spoc_ffi_uri(tic_id, sector)
    filename = uri.rsplit("/", 1)[-1].replace("mast:HLSP/", "").replace(":", "_")
    local_path = os.path.join(directory, filename)
    if os.path.exists(local_path):
        cached = True
    else:
        try:
            downloaded = Observations.download_file(uri, local_path=local_path)
        except Exception as e:
            raise ArchiveUnavailable(f"MAST download failed for {uri}: {e}") from e
        if isinstance(downloaded, str) and os.path.exists(downloaded):
            local_path = downloaded
        cached = False
    return ArchiveProduct(
        tic_id=tic_id,
        sector=sector,
        local_path=local_path,
        data_uri=uri,
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        cached=cached,
    )
