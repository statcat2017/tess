"""Generate SVG demo plots from real pipeline outputs (stdlib only).

Reads fixture + replay event windows through the package and writes
small dark-theme SVG line charts into assets/. Re-run to refresh:
  PYTHONPATH=src python3 tools/make_demo_plots.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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


def _svg_single_window(
    path: str,
    title: str,
    subtitle: str,
    x: list[float],
    y: list[float],
    color: str,
    stats: str,
) -> None:
    x0, x1max = min(x), max(x)
    y0, y1max = min(y), max(y)
    pad = max((y1max - y0) * 0.15, 1e-6)
    y0, y1max = y0 - pad, y1max + pad
    if x1max == x0:
        x1max = x0 + 1.0

    def px(v: float) -> float:
        return ML + (v - x0) / (x1max - x0) * (W - ML - MR)

    def py(v: float) -> float:
        return MT + (1.0 - (v - y0) / (y1max - y0)) * (H - MT - MB)

    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, y))
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
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img">
<rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="{BG}" stroke="{GRID}"/>
<text x="{ML}" y="22" fill="{INK}" font-size="14" font-weight="bold">{title}</text>
<text x="{W - MR}" y="22" fill="{DIM}" font-size="11" text-anchor="end">{subtitle}</text>
{grid_lines}
<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" opacity="0.9"/>
<text x="{ML}" y="{H - 10}" fill="{DIM}" font-size="11">{stats}</text>
</svg>"""
    with open(path, "w") as f:
        f.write(svg + "\n")


def _blink_stats(ev: dict) -> str:
    return (
        f"depth {ev['depth']:.4f} · duration {ev['duration_days'] * 24:.2f}h · "
        f"SNR {ev['snr']:.1f} · {len(ev['local_flux'])} samples"
    )


def _svg_sector_curve(
    path: str,
    title: str,
    subtitle: str,
    time: list[float],
    flux: list[float],
    known: list[float],
    found: list[float],
    missed: list[float],
) -> None:
    """Full detrended sector curve: green ticks (known), gold dots (found)."""
    step = max(1, len(time) // 1200)
    tx, fx = time[::step], flux[::step]
    x0, x1max = tx[0], tx[-1]
    y0, y1max = min(fx), max(fx)
    pad = max((y1max - y0) * 0.15, 1e-6)
    y0, y1max = y0 - pad, y1max + pad

    def px(v: float) -> float:
        return ML + (v - x0) / (x1max - x0) * (W - ML - MR)

    def py(v: float) -> float:
        return MT + (1.0 - (v - y0) / (y1max - y0)) * (H - MT - MB)

    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(tx, fx))
    fmap = {round(t, 6): f for t, f in zip(tx, fx)}

    def at(t: float) -> float:
        near = min(fmap, key=lambda k: abs(k - t))
        return fmap[near]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img">',
        f'<rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="{BG}" stroke="{GRID}"/>',
        f'<text x="{ML}" y="22" fill="{INK}" font-size="14" font-weight="bold">{title}</text>',
        f'<text x="{W - MR}" y="22" fill="{DIM}" font-size="11" text-anchor="end">{subtitle}</text>',
        f'<polyline points="{pts}" fill="none" stroke="#92b7ff" stroke-width="1" opacity="0.55"/>',
    ]
    for t in known:
        parts.append(
            f'<line x1="{px(t):.1f}" y1="{MT}" x2="{px(t):.1f}" y2="{MT + 12}" '
            f'stroke="#9fe8b3" stroke-width="2" opacity="0.9"/>'
        )
    for t in found:
        parts.append(
            f'<circle cx="{px(t):.1f}" cy="{py(at(t)):.1f}" r="4" '
            f'fill="#f7c982" opacity="0.95"/>'
        )
    for t in missed:
        cx, cy = px(t), py(at(t))
        parts.append(
            f'<line x1="{cx - 5:.1f}" y1="{cy - 5:.1f}" x2="{cx + 5:.1f}" y2="{cy + 5:.1f}" '
            f'stroke="#f2a3a3" stroke-width="2"/>'
            f'<line x1="{cx - 5:.1f}" y1="{cy + 5:.1f}" x2="{cx + 5:.1f}" y2="{cy - 5:.1f}" '
            f'stroke="#f2a3a3" stroke-width="2"/>'
        )
    parts.append(
        f'<circle cx="{ML}" cy="{H - 12}" r="4" fill="#9fe8b3"/>'
        f'<text x="{ML + 12}" y="{H - 8}" fill="{INK}" font-size="11">known</text>'
        f'<circle cx="{ML + 110}" cy="{H - 12}" r="4" fill="#f7c982"/>'
        f'<text x="{ML + 122}" y="{H - 8}" fill="{INK}" font-size="11">found</text>'
        f'<text x="{ML + 200}" y="{H - 8}" fill="#f2a3a3" font-size="14">×</text>'
        f'<text x="{ML + 216}" y="{H - 8}" fill="{INK}" font-size="11">missed</text>'
    )
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")


def _svg_blind_timeline(
    path: str,
    title: str,
    lanes: list[tuple[int, float, float, list[float], list[float]]],
) -> None:
    """Two lanes (sectors): green ticks = known transits, gold dots = proposals."""
    lane_h, gap = 64, 26
    top = 44
    H2 = top + len(lanes) * (lane_h + gap) + 34
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H2}" width="100%" role="img">',
        f'<rect x="0" y="0" width="{W}" height="{H2}" rx="12" fill="{BG}" stroke="{GRID}"/>',
        f'<text x="{ML}" y="24" fill="{INK}" font-size="14" font-weight="bold">{title}</text>',
    ]
    y = top
    for sector, t0, t1, known, found in lanes:
        def px(t: float) -> float:
            return ML + (t - t0) / (t1 - t0) * (W - ML - MR)

        parts.append(
            f'<text x="{ML}" y="{y + 14}" fill="{DIM}" font-size="11">S{sector}</text>'
        )
        for t in known:
            parts.append(
                f'<line x1="{px(t):.1f}" y1="{y}" x2="{px(t):.1f}" y2="{y + lane_h}" '
                f'stroke="#9fe8b3" stroke-width="2" opacity="0.85"/>'
            )
        for t in found:
            parts.append(
                f'<circle cx="{px(t):.1f}" cy="{y + lane_h / 2:.1f}" r="4" '
                f'fill="#f7c982" opacity="0.9"/>'
            )
        parts.append(
            f'<text x="{ML}" y="{y + lane_h + 16}" fill="{DIM}" font-size="10">'
            f'{t0:.1f} – {t1:.1f} (BTJD)</text>'
        )
        y += lane_h + gap
    parts.append(
        f'<circle cx="{ML}" cy="{H2 - 12}" r="4" fill="#9fe8b3"/>'
        f'<text x="{ML + 12}" y="{H2 - 8}" fill="{INK}" font-size="11">known transit</text>'
        f'<circle cx="{ML + 150}" cy="{H2 - 12}" r="4" fill="#f7c982"/>'
        f'<text x="{ML + 162}" y="{H2 - 8}" fill="{INK}" font-size="11">blind proposal</text>'
    )
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")


def main() -> None:
    from tess_assoc.archive import download_spoc_ffi
    from tess_assoc.extract import load_lightcurve, predicted_transits
    from tess_assoc.manifest import load_manifest_file
    from tess_assoc.provider import provide_events
    from tess_assoc.replay import (
        load_replay_manifest,
        replay_blind_system,
        replay_system,
    )

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
        for ev_dict, sec in ((ea, secs[0]), (eb, secs[1])):
            t0 = ev_dict["t0"]
            _svg_single_window(
                f"assets/blink_{slug}_s{sec}.svg",
                f"{system.name} · sector {sec} blink at BTJD {t0:.3f}",
                f"TIC {system.tic_id} · TESS-SPOC FFI",
                [t - t0 for t in ev_dict["local_time"]],
                list(ev_dict["local_flux"]),
                "#92b7ff",
                _blink_stats(ev_dict),
            )

    system = next(s for s in replay.systems if s.name == "WASP-121 b")
    blind = replay_blind_system(replay, system)
    from tess_assoc.propose import detrend

    lanes = []
    missed_by_sector: dict[int, list[float]] = {}
    for m in blind["missed"]:
        missed_by_sector.setdefault(m["sector"], []).append(m["t0"])
    for sector in system.sectors:
        time, flux = load_lightcurve(download_spoc_ffi(system.tic_id, sector))
        detrended, _ = detrend(time, flux)
        known = predicted_transits(
            system.t0_bjd_tdb, system.period_days, time[0], time[-1]
        )
        found = sorted(
            e["t0"] for e in blind["events"] if e["sector"] == sector
        )
        lanes.append((sector, time[0], time[-1], known, found))
        _svg_sector_curve(
            f"assets/sector_wasp-121b_s{sector}.svg",
            f"WASP-121 b · sector {sector} light curve (detrended)",
            f"{len(found)} found · {len(missed_by_sector.get(sector, []))} missed",
            time,
            detrended,
            known,
            found,
            missed_by_sector.get(sector, []),
        )
    _svg_blind_timeline(
        "assets/blind_recall.svg",
        f"Blind recall: WASP-121 b ({blind['recall']['recalled']}/{blind['recall']['known']} known transits found, no period given)",
        lanes,
    )

    _mining_figures()
    _findings_figures()
    _candidate_figures()
    _long_period_figures()
    print("wrote assets/*.svg")


def _mining_figures() -> None:
    """Phase 8: TOI-190 validation redetection, quiet/noisy hunt curves."""
    from tess_assoc.archive import download_spoc_ffi
    from tess_assoc.extract import load_lightcurve, predicted_transits
    from tess_assoc.inject import inject_transit
    from tess_assoc.propose import detrend, propose_events

    t0_190, period_190 = 2460982.967362, 10.0205932
    for tic, sector, tag in (
        (166739520, 30, "mining_toi190_s30"),
        (166739520, 69, "mining_toi190_s69"),
        (42015014, 69, "mining_quiet_42015014_s69"),
        (42055368, 69, "mining_noisy_42055368_s69"),
    ):
        time, flux = load_lightcurve(download_spoc_ffi(tic, sector))
        detrended, sigma = detrend(time, flux)
        found = sorted(p.t0_guess for p in propose_events(time, flux))
        if tic == 166739520:
            known = predicted_transits(t0_190, period_190, time[0], time[-1])
            sub = f"TOI-190.01 P=10.02d · {len(found)} proposals"
        else:
            known = []
            sub = f"TIC {tic} · no known ephemeris · {len(found)} proposals"
        missed = [
            t for t in known
            if not any(abs(t - f) <= 0.15 for f in found)
        ]
        _svg_sector_curve(
            f"assets/sector_{tag}.svg",
            f"Mining: TIC {tic} sector {sector} (detrended, sigma {sigma * 100:.3f}%)",
            sub,
            time,
            detrended,
            known,
            found,
            missed,
        )

    time, flux = load_lightcurve(download_spoc_ffi(42015014, 69))
    t_inj = time[len(time) // 2]
    mod = inject_transit(time, flux, t_inj, 0.01, 0.12, "box")
    detrended, _ = detrend(time, mod)
    found = sorted(p.t0_guess for p in propose_events(time, mod))
    _svg_sector_curve(
        "assets/sector_mining_injected_42015014_s69.svg",
        "Mining sensitivity: planted 1% dip recovered blind",
        f"TIC 42015014 s69 · {len(found)} proposal(s) near BTJD {t_inj:.2f}",
        time,
        detrended,
        [],
        [t for t in found if abs(t - t_inj) <= 0.15],
        [],
    )


@dataclass(frozen=True)
class FoldSpec:
    sector: int
    t0: float
    period: float
    note: str


@dataclass(frozen=True)
class FindingCase:
    tic: int
    sectors: tuple[int, ...]
    slug: str
    title: str
    pair: tuple[int, float, int, float]
    curves: tuple[int, ...]
    fold: FoldSpec | None = None


def _findings_figures() -> None:
    """Phase 8 findings: the two eclipsing binaries, as the pipeline saw them."""
    from tess_assoc.archive import download_spoc_ffi
    from tess_assoc.extract import extract_at, load_lightcurve
    from tess_assoc.propose import detrend, propose_events

    cases = [
        FindingCase(
            tic=224224413, sectors=(2, 29, 69), slug="eb224224413",
            title="TIC 224224413",
            pair=(29, 2096.300, 69, 3198.904),
            curves=(2,),
            fold=FoldSpec(
                sector=2, t0=1359.328, period=5.71296,
                note="primary + secondary at phase ~0.49",
            ),
        ),
        FindingCase(
            tic=197931848, sectors=(29, 69), slug="eb197931848",
            title="TIC 197931848",
            pair=(29, 2103.130, 69, 3197.023),
            curves=(29, 69),
        ),
    ]
    for case in cases:
        tic = case.tic
        curves = {}
        for sector in case.sectors:
            time, flux = load_lightcurve(download_spoc_ffi(tic, sector))
            curves[sector] = (time, flux)
        for sector in case.curves:
            time, flux = curves[sector]
            detrended, _ = detrend(time, flux)
            found = sorted(p.t0_guess for p in propose_events(time, flux))
            _svg_sector_curve(
                f"assets/sector_{case.slug}_s{sector}.svg",
                f"Finding: {case.title} sector {sector} (detrended)",
                f"{len(found)} blind proposals",
                time,
                detrended,
                [],
                found,
                [],
            )
        sec_a, t_a, sec_b, t_b = case.pair
        ta, fa = curves[sec_a]
        tb, fb = curves[sec_b]
        ra = extract_at(ta, fa, t_a, 0.2, tic_id=tic, sector=sec_a, quality={})
        rb = extract_at(tb, fb, t_b, 0.2, tic_id=tic, sector=sec_b, quality={})
        xa = [t - ra.t0 for t in ra.local_time]
        xb = [t - rb.t0 for t in rb.local_time]
        _svg_two_windows(
            f"assets/{case.slug}_pair.svg",
            f"Finding: {case.title} associated pair "
            f"(depth {ra.depth:.3f} vs {rb.depth:.3f})",
            f"S{sec_a}@{t_a:.1f} + S{sec_b}@{t_b:.1f} BTJD",
            (xa, list(ra.local_flux), f"S{sec_a}", "#f7c982"),
            (xb, list(rb.local_flux), f"S{sec_b}", "#92b7ff"),
            "same dip, years apart",
        )
        if case.fold is not None:
            fold = case.fold
            time, flux = curves[fold.sector]
            detrended, _ = detrend(time, flux)
            phased = sorted(
                ((t - fold.t0) % fold.period, f)
                for t, f in zip(time, detrended)
            )
            step = max(1, len(phased) // 1500)
            xs = [p for p, _ in phased[::step]]
            ys = [f for _, f in phased[::step]]
            _svg_single_window(
                f"assets/{case.slug}_fold.svg",
                f"Finding: {case.title} folded at P={fold.period:.5f}d",
                f"sector {fold.sector} · {fold.note}",
                xs,
                ys,
                "#92b7ff",
                "period fit to the minute over 322 orbits",
            )


def _candidate_figures() -> None:
    """Phase 8 planet candidate: TIC 231279823 multi-epoch fold + curves."""
    from tess_assoc.archive import download_spoc_ffi
    from tess_assoc.extract import load_lightcurve
    from tess_assoc.propose import detrend, propose_events

    tic, period, t0 = 231279823, 5.96848, 1355.67458
    render_sectors = (29, 30, 69)
    folded: list[tuple[float, float]] = []
    for sector in (2, 29, 30, 69):
        time, flux = load_lightcurve(download_spoc_ffi(tic, sector))
        detrended, _ = detrend(time, flux)
        if sector in render_sectors:
            found = sorted(p.t0_guess for p in propose_events(time, flux))
            _svg_sector_curve(
                f"assets/sector_cand231279823_s{sector}.svg",
                f"Candidate: TIC {tic} sector {sector} (detrended)",
                f"{len(found)} blind proposals",
                time,
                detrended,
                [],
                found,
                [],
            )
        folded.extend(
            (((t - t0) % period, f) for t, f in zip(time, detrended))
        )
    folded.sort()
    step = max(1, len(folded) // 1500)
    _svg_single_window(
        f"assets/cand231279823_fold.svg",
        f"Candidate: TIC {tic} folded at P={period:.5f}d",
        "sectors 2+29+30+69 · 13 transits, cycles 0-309",
        [p for p, _ in folded[::step]],
        [f for _, f in folded[::step]],
        "#92b7ff",
        "joint fit rms 16.7 min over 5 years",
    )


def _long_period_figures() -> None:
    """Isolated-event follow-up queue: no period or fold is assumed."""
    from tess_assoc.archive import download_spoc_ffi
    from tess_assoc.extract import load_lightcurve
    from tess_assoc.propose import detrend

    cases = (
        (137801807, 3204.41835, "depth 5.20% · duration 3.06h · SNR 30.8"),
        (117549174, 3183.13348, "depth 3.18% · duration 1.67h · SNR 11.7"),
    )
    for tic, t0, stats in cases:
        product = download_spoc_ffi(tic, 69, directory="/tmp/long_followup_cache")
        time, flux = load_lightcurve(product)
        detrended, _ = detrend(time, flux)
        selected = [
            (t, f) for t, f in zip(time, detrended) if abs(t - t0) <= 0.6
        ]
        _svg_single_window(
            f"assets/single_{tic}_s69.svg",
            f"Isolated event: TIC {tic} sector 69",
            "single-transit queue · period unconstrained",
            [t - t0 for t, _ in selected],
            [f for _, f in selected],
            "#f7c982",
            stats,
        )


if __name__ == "__main__":
    main()
