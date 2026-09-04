"""Bulk SPOC FFI retrieval without per-star queries (issue #9 scale-up).

MAST dataURIs are deterministic in (TIC, sector), so a cohort is fetched
by attempting direct downloads: hits land on disk, 404s mean no product.
One catalog/cone query enumerates hundreds of TICs; afterwards nothing
touches MAST except the downloads themselves (plus batched catalog
cross-matches). No astroquery needed for the fetch path at all.
"""

from __future__ import annotations

import concurrent.futures
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tess_assoc.archive import cache_dir, spoc_ffi_uri

_MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file"


def direct_url(tic_id: int, sector: int) -> str:
    """Public HTTPS URL for a SPOC FFI light curve (existence unchecked)."""
    return _MAST_DOWNLOAD + "?uri=" + urllib.parse.quote(spoc_ffi_uri(tic_id, sector), safe="")


def expected_filename(tic_id: int, sector: int) -> str:
    """Cache filename for a (TIC, sector) pair (matches download_spoc_ffi)."""
    uri = spoc_ffi_uri(tic_id, sector)
    return uri.rsplit("/", 1)[-1].replace("mast:HLSP/", "").replace(":", "_")


def fetch_one(
    tic_id: int, sector: int, directory: str | None = None, timeout: int = 120
) -> dict[str, Any]:
    """Direct download; 404 becomes missing (never an exception for 404)."""
    directory = directory or cache_dir()
    os.makedirs(directory, exist_ok=True)
    local_path = os.path.join(directory, expected_filename(tic_id, sector))
    if os.path.exists(local_path):
        return {
            "tic_id": tic_id, "sector": sector, "local_path": local_path,
            "status": "cached",
        }
    request = urllib.request.Request(
        direct_url(tic_id, sector), headers={"User-Agent": "tess-assoc/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(local_path, "wb") as f:
                f.write(response.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {
                "tic_id": tic_id, "sector": sector, "local_path": local_path,
                "status": "missing",
            }
        return {
            "tic_id": tic_id, "sector": sector, "local_path": local_path,
            "status": "failed", "reason": f"HTTP {e.code}",
        }
    except Exception as e:  # noqa: BLE001 — per-file faults stay per-file
        return {
            "tic_id": tic_id, "sector": sector, "local_path": local_path,
            "status": "failed", "reason": str(e)[:200],
        }
    return {
        "tic_id": tic_id, "sector": sector, "local_path": local_path,
        "status": "downloaded",
    }


def bulk_fetch(
    pairs: list[tuple[int, int]],
    directory: str | None = None,
    *,
    max_workers: int = 8,
    timeout: int = 120,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch many (TIC, sector) files in parallel; faults never propagate."""
    directory = directory or cache_dir()
    jobs = list(dict.fromkeys((int(t), int(s)) for t, s in pairs))
    buckets: dict[str, list[dict[str, Any]]] = {
        "downloaded": [], "cached": [], "missing": [], "failed": [],
    }

    def job(pair: tuple[int, int]) -> dict[str, Any]:
        return fetch_one(pair[0], pair[1], directory, timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(job, jobs):
            buckets[result["status"]].append(result)
    return buckets


__all__ = [
    "bulk_fetch",
    "direct_url",
    "expected_filename",
    "fetch_one",
]
