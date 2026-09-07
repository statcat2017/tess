"""Run the offline Survey B circumbinary tracer pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Keep this script runnable from a clean checkout without requiring installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tess_assoc.circumbinary import (
    load_binary_pilot_manifest,
    render_binary_pilot_report,
    run_binary_pilot,
)


def run(fixture_path: str, output_dir: str) -> dict:
    fixture = json.loads(Path(fixture_path).read_text())
    targets = load_binary_pilot_manifest(fixture)
    coverage = {
        int(sector): [tuple(window) for window in windows]
        for sector, windows in fixture["coverage_windows"].items()
    }
    results = run_binary_pilot(targets, fixture["events"], coverage)
    results["manifest"] = {
        "name": fixture["name"],
        "version": fixture["version"],
        "source_catalog": fixture["source_catalog"],
        "fixture_sha256": hashlib.sha256(
            json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "report.md").write_text(render_binary_pilot_report(results))
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default="fixtures/circumbinary_pilot_v1.json",
    )
    parser.add_argument("--output-dir", default="/tmp/circumbinary_pilot")
    args = parser.parse_args()
    run(args.fixture, args.output_dir)


if __name__ == "__main__":
    main()
