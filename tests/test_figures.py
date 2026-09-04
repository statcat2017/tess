"""Published-figure smoke test (ADR-0001 rule 6).

The STATUS page presents these SVGs as proof; regenerating them here pins
the artifact pipeline instead of trusting checked-in files.
"""

import os
import subprocess
from pathlib import Path

from conftest import needs_archive

ASSETS = [
    "fixture_match.svg",
    "fixture_distractor.svg",
    "replay_wasp-43b.svg",
    "replay_wasp-121b.svg",
    "replay_kelt-9b.svg",
]
TIMELINES = ["blind_recall.svg"]
SINGLES = [
    "blink_wasp-43b_s9.svg",
    "blink_wasp-43b_s35.svg",
    "blink_wasp-121b_s7.svg",
    "blink_wasp-121b_s33.svg",
    "blink_kelt-9b_s14.svg",
    "blink_kelt-9b_s55.svg",
]
CURVES = [
    "sector_wasp-121b_s7.svg",
    "sector_wasp-121b_s33.svg",
]
MINING_CURVES = [
    "sector_mining_toi190_s30.svg",
    "sector_mining_toi190_s69.svg",
    "sector_mining_quiet_42015014_s69.svg",
    "sector_mining_noisy_42055368_s69.svg",
    "sector_mining_injected_42015014_s69.svg",
]


@needs_archive
def test_demo_plots_regenerate():
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    proc = subprocess.run(
        ["python3", "tools/make_demo_plots.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ASSETS:
        path = root / "assets" / name
        assert path.exists(), name
        text = path.read_text()
        assert text.startswith("<svg")
        assert text.count("<polyline") == 2, name
    for name in TIMELINES:
        path = root / "assets" / name
        assert path.exists(), name
        text = path.read_text()
        assert text.startswith("<svg")
        assert "<line" in text and "<circle" in text, name
    for name in SINGLES:
        path = root / "assets" / name
        assert path.exists(), name
        text = path.read_text()
        assert text.startswith("<svg")
        assert text.count("<polyline") == 1, name
    for name in CURVES:
        path = root / "assets" / name
        assert path.exists(), name
        text = path.read_text()
        assert text.startswith("<svg")
        assert text.count("<polyline") == 1, name
        assert "<circle" in text and "<line" in text, name
    for name in MINING_CURVES:
        path = root / "assets" / name
        assert path.exists(), name
        text = path.read_text()
        assert text.startswith("<svg")
        assert text.count("<polyline") == 1, name
        assert "<circle" in text, name  # legend dots; quiet curves carry no markers
