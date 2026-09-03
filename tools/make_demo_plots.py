"""Generate SVG demo plots from real pipeline outputs (stdlib only).

Reads fixture + replay event windows through the package and writes
small dark-theme SVG line charts into assets/. Re-run to refresh:
  PYTHONPATH=src python3 tools/make_demo_plots.py
"""

from __future__ import annotations

import os

W, H = 680, 300
ML, MR, MT, MB = 58, 16, 34, 42
BG = "#0d1730"
GRID = "rgba(183,199,239,.14)"
INK = "#edf2ff"
DIM = "#7e8baa"


def _svg_two_windows(
    path: str,
    title: str,
    subtitle: str,
    s1: tuple[list[float], list[float], str, str],
    s2: tuple[list[float], list[float], str, str],
    note: str,
) -> None:
    (x1, y1, label1, color1), (x2, y2, label2, color2) = s1, s2
    x0 = min(min(x1), min(x2))
    x1max = max(max(x1), max(x2))
    y0 = min(min(y1), min(y2))
    y1max = max(max(y1), max(y2))
    pad = max((y1max - y0) * 0.15, 1e-6)
    y0, y1max = y0 - pad, y1max + pad
    if x1max == x0:
        x1max = x0 + 1.0

    def px(x: float) -> float:
        return ML + (x - x0) / (x1max - x0) * (W - ML - MR)

    def py(y: float) -> float:
        return MT + (1.0 - (y - y0) / (y1max - y0)) * (H - MT - MB)

    def poly(x: list[float], y: list[float], color: str, width: int) -> str:
        pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, y))
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" opacity="0.9"/>'
        )

    grid_lines = ""
    for frac in (0.0, 0.5, 1.0):
        gy = MT + frac * (H - MT - MB)
        val = y1max - frac * (y1max - y0)
        grid_lines += (
            f'<line x1="{ML}" y1="{gy:.1f}" x2="{W - MR}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
            f'<text x="{ML - 8}" y="{gy + 4:.1f}" fill="{DIM}" font-size="11" '
            f'text-anchor="end">{val:.4f}</text>'
        )
    for xv in (x0, (x0 + x1max) / 2.0, x1max):
        grid_lines += (
            f'<text x="{px(xv):.1f}" y="{H - MB + 20}" fill="{DIM}" '
            f'font-size="11" text-anchor="middle">{xv:+.2f}d</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img">
<rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="{BG}" stroke="{GRID}"/>
<text x="{ML}" y="22" fill="{INK}" font-size="14" font-weight="bold">{title}</text>
<text x="{W - MR}" y="22" fill="{DIM}" font-size="11" text-anchor="end">{subtitle}</text>
{grid_lines}
{poly(x1, y1, color1, 2)}
{poly(x2, y2, color2, 2)}
<circle cx="{ML + 8}" cy="{H - 12}" r="4" fill="{color1}"/><text x="{ML + 18}" y="{H - 8}" fill="{INK}" font-size="11">{label1}</text>
<circle cx="{ML + 180}" cy="{H - 12}" r="4" fill="{color2}"/><text x="{ML + 190}" y="{H - 8}" fill="{INK}" font-size="11">{label2}</text>
<text x="{W - MR}" y="{H - 8}" fill="{DIM}" font-size="11" text-anchor="end">{note}</text>
</svg>"""
    with open(path, "w") as f:
        f.write(svg + "\n")


def main() -> None:
    from tess_assoc.manifest import load_manifest_file
    from tess_assoc.provider import provide_events
    from tess_assoc.replay import load_replay_manifest, replay_system

    os.makedirs("assets", exist_ok=True)
    fix = provide_events(load_manifest_file("fixtures/tracer_v1.json"))

    def centered(rec) -> tuple[list[float], list[float]]:
        return [t - rec.t0 for t in rec.local_time], list(rec.local_flux)

    xa, ya = centered(fix["A"])
    xb, yb = centered(fix["B"])
    xc, yc = centered(fix["C"])
    _svg_two_windows(
        "assets/fixture_match.svg",
        "Fixture: matched pair A–B (corr 1.000)",
        "tracer_v1 · TIC 1234567",
        (xa, ya, "A · sector 12", "#f7c982"),
        (xb, yb, "B · sector 39", "#92b7ff"),
        "identical by construction",
    )
    _svg_two_windows(
        "assets/fixture_distractor.svg",
        "Fixture: distractor C vs A (rejected)",
        "tracer_v1 · TIC 1234567",
        (xa, ya, "A · ephemeris", "#92b7ff"),
        (xc, yc, "C · distractor", "#f2a3a3"),
        "deeper, longer, V-shaped",
    )

    replay = load_replay_manifest("fixtures/replay_v1.json")
    for system in replay.systems:
        res = replay_system(replay, system)
        by_id = {}
        for e in res["events"]:
            by_id.setdefault(e["sector"], []).append(e)
        secs = sorted(by_id)
        ea = sorted(by_id[secs[0]], key=lambda e: e["t0"])[0]
        eb = sorted(by_id[secs[1]], key=lambda e: e["t0"])[0]
        pa = [t - ea["t0"] for t in ea["local_time"]]
        pb = [t - eb["t0"] for t in eb["local_time"]]
        slug = system.name.lower().replace(" ", "")
        pair = next(
            p for p in res["pairs"]
            if sorted([p["a"], p["b"]]) == sorted(res["anchors"])
        )
        _svg_two_windows(
            f"assets/replay_{slug}.svg",
            f"Real data: {system.name} anchor pair (corr {pair['morph_corr']:.3f})",
            f"TIC {system.tic_id} · S{secs[0]}+S{secs[1]}",
            (pa, list(ea["local_flux"]), f"S{secs[0]} anchor", "#f7c982"),
            (pb, list(eb["local_flux"]), f"S{secs[1]} anchor", "#92b7ff"),
            "TESS-SPOC FFI windows",
        )
    print("wrote assets/*.svg")


if __name__ == "__main__":
    main()
