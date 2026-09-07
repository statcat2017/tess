# Candidate extension findings
Run: 2026-09-07T10:29:12.222428+00:00

This report covers neighbour attribution, quality-flag diagnostics, injection sensitivity, and public MAST product inventory. It does not run the real Survey B cohort.

## Neighbour attribution
### TIC 137801807
- Gaia 2399367919244466432 at 2.15"; TIC lookup: no-tic-match .
### TIC 117549174
- Gaia 2420819528542292224 at 1.21"; TIC lookup: offset-match-review 610432593.
- Gaia 2420819528542292096 at 3.38"; TIC lookup: matched 610432593.

## TIC 117549174 flagged-cadence diagnostic
- 30/30 event cadences are flagged with QUALITY values [128]; valid-cadence audit remains insufficient.
- All-cadence target-centred 1/3/5-pixel depth values: -0.0057, -0.0116, -0.0308; negative values are brightenings, not transit-like dimmings.
- The same all-cadence aperture values are returned at both Gaia comparison positions, so this diagnostic is spatially broad and not usable localisation evidence.

## TIC 117549174 sensitivity
- Sector 2 at 5 placements: 0.0050:0/5, 0.0100:0/5, 0.0200:0/5, 0.0318:0/5, 0.0500:0/5
- Sector 29 at 5 placements: 0.0050:0/5, 0.0100:0/5, 0.0200:0/5, 0.0318:0/5, 0.0500:3/5
- Sector 70 at 5 placements: 0.0050:0/5, 0.0100:0/5, 0.0200:0/5, 0.0318:0/5, 0.0500:5/5

## MAST provider inventory
- TIC 137801807: providers GSFC-ELEANOR-LITE, QLP, T16, TARS, TASOC, TESS-SPOC, TGLC; anchor-sector providers QLP, TARS, TESS-SPOC.
- TIC 117549174: providers GSFC-ELEANOR-LITE, QLP, T16, TARS, TASOC, TESS-SPOC, TGLC; anchor-sector providers QLP, TARS, TESS-SPOC.
