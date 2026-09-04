# Single-event follow-up requests

These are observing requests, not completed observations. The two candidates
have one TESS event each, so no orbital period is available for scheduling a
predicted repeat.

## Priority 1: pixel localisation and high-resolution imaging

Request a difference image and centroid measurement from the Sector 69 TESSCut
data, then resolve the Gaia neighbours with speckle imaging.

| TIC | RA (deg) | Dec (deg) | Event | Depth | SNR |
| --- | ---: | ---: | --- | ---: | ---: |
| 137801807 | 344.099834 | -20.279416 | BTJD 3204.41835 | 5.20% | 30.8 |
| 117549174 | 0.142169 | -13.415334 | BTJD 3183.13348 | 3.18% | 11.7 |

Suggested facilities: SOAR/HRCam or Gemini South/Zorro. Request contrast curves
that cover companions at 1--5 arcsec and report whether the TESS difference
image is centred on the TIC source or a Gaia neighbour.

Coordination route: [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/).

## Priority 2: reconnaissance spectroscopy

Obtain 3--6 spectra over nights or weeks, not only during the TESS event. The
goal is to detect double lines, line-profile changes, or km/s radial velocity
motion characteristic of an eclipsing binary.

Suggested facilities: CHIRON/SMARTS, Magellan/MIKE, HARPS, ESPRESSO, SALT/HRS,
or MAROON-X. Record exposure time, resolving power, barycentric timestamps,
RV uncertainties, bisectors, and any secondary spectrum.

## Priority 3: multicolour photometry

Use LCO 1m, MuSCAT2/MuSCAT3, SPECULOOS, or an AAVSO campaign. Since the period
is unconstrained, schedule a continuous nightly baseline or use a period prior
from the transit-duration/stellar-density fit. Simultaneous filters should
test whether the event is chromatic, which would favour a blended eclipsing
binary.

## Current evidence boundary

The TESSCut audit can test source localisation and aperture dependence. It
cannot confirm a planet from one event. A clean difference image, no resolved
companion, stable single-lined spectrum, and achromatic shape would strengthen
the hypothesis; none alone would establish an orbit.
