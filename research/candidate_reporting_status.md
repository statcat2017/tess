# Candidate reporting status

**Targets:** TIC 137801807 and TIC 117549174
**Queries run:** 2026-09-07 (UTC date)

## Summary

Both identifiers are real TIC v8 records. I found no record that either target has been reported as a TOI, confirmed exoplanet host, ExoFOP candidate, TESS eclipsing binary, ASAS-SN variable/binary, or Gaia DR3 eclipsing binary. This is a catalogue non-detection, not a statement that the stars lack variability: MAST contains public TESS light-curve products for both.

## Sources checked

| Source | Query and result |
|---|---|
| [MAST TIC service](https://mast.stsci.edu/api/v0/pages.html) | `Mast.Catalogs.Filtered.Tic`, filtering `ID=137801807` and `ID=117549174`: one row for each, TIC version `20190415`; the returned rows include Gaia DR3 IDs `2399367919244466432` and `2420819528542292224`, respectively. |
| [NASA Exoplanet Archive TAP](https://exoplanetarchive.ipac.caltech.edu/TAP/sync) | `ps` and `stellarhosts`, filtering `tic_id='137801807'` or `'117549174'`: empty result sets. `toi` has no TIC-ID column, so I searched within 0.01 degrees of each TIC coordinate: empty result set. The Archive TAP schema query `select table_name ... like '%tic%'` exposed no TIC table in this service. |
| [ExoFOP TIC 137801807](https://exofop.ipac.caltech.edu/tess/target.php?id=137801807) and [TIC 117549174](https://exofop.ipac.caltech.edu/tess/target.php?id=117549174) | Both pages show `TOIs 0`, `CTOIs 0`, `Community Planet Candidates 0`, `Files 0`, and `Observing Notes 0`; both show no confirmed planet. |
| [TESS EB catalogue](https://tessebs.villanova.edu/) | Direct target pages [137801807](https://tessebs.villanova.edu/137801807) and [117549174](https://tessebs.villanova.edu/117549174) both say: “This TESS ID is not in the catalog.” |
| [ASAS-SN Variable Stars Database](https://asas-sn.osu.edu/variables) | Coordinate searches at `(344.0998, -20.2800)` and `(0.1421, -13.4156)`, radius 1 arcmin, returned `Total Light Curves Found: 0` for each. Name searches for both TIC IDs also returned zero. The site says its variable database was updated 2021-08-10. |
| [ASAS-SN Binary Stars Database](https://asas-sn.osu.edu/binaries/main) | Same two 1-arcmin coordinate searches returned `Total Light Curves Found: 0` / “No Binarys found.” The site says this database was updated 2022-06-01. |
| [Gaia DR3 archive TAP](https://gea.esac.esa.int/tap-server/tap/sync) | ADQL: `SELECT source_id, solution_id, frequency, frequency_error, num_samples FROM gaiadr3.vari_eclipsing_binary WHERE source_id IN (2399367919244466432,2420819528542292224)`. Result contained only the header, with no rows. |
| [MAST CAOM/TESS-SPOC metadata](https://mast.stsci.edu/api/v0/pages.html) | Cone searches (`Mast.Caom.Cone`) at each TIC position with radius 0.001 degrees returned TESS/SPOC metadata. TIC 137801807 has a SPOC light curve in Sector 2; TIC 117549174 has SPOC light curves in Sectors 2, 29, 69, and 70. These are observation/processing records, not candidate dispositions. The same cones also returned QLP, TESS-SPOC, and other HLSP products. |

## Per-target conclusions

### TIC 137801807

- **Catalogue identity:** confirmed as a TIC record; Gaia DR3 counterpart `2399367919244466432`.
- **Reporting status:** no NASA Exoplanet Archive `ps`/`stellarhosts` record, no nearby TOI, no ExoFOP TOI/CTOI/community-candidate record, no TESS EB catalogue entry, no ASAS-SN variable or binary entry, and no Gaia DR3 eclipsing-binary row.
- **Data availability:** public TESS/SPOC Sector 2 light-curve metadata exists in MAST, along with several HLSP light curves. Therefore this target has been observed and processed, but I found no catalogue report identifying it as a candidate or eclipsing binary.

### TIC 117549174

- **Catalogue identity:** confirmed as a TIC record; Gaia DR3 counterpart `2420819528542292224`.
- **Reporting status:** no NASA Exoplanet Archive `ps`/`stellarhosts` record, no nearby TOI, no ExoFOP TOI/CTOI/community-candidate record, no TESS EB catalogue entry, no ASAS-SN variable or binary entry, and no Gaia DR3 eclipsing-binary row.
- **Data availability:** public TESS/SPOC light-curve metadata exists in MAST for Sectors 2, 29, 69, and 70, plus other HLSP products. As with TIC 137801807, the target is observed but not identified in the checked reporting catalogues.

## Limitations and access notes

- The NASA Exoplanet Archive TAP service currently exposes `toi`, `ps`, and `stellarhosts` but no TIC table through `TAP_SCHEMA`; therefore TIC identity was verified through MAST and ExoFOP, while candidate/planet status was checked through the Archive tables that do expose `tic_id` or coordinates.
- The Gaia ADQL GET form returned HTTP 400 during one request; the identical ADQL submitted as a TAP POST succeeded and returned zero rows.
- The MAST API endpoint documented as `/api/v0.1/invoke` returned HTTP 404; the current `/api/v0/invoke` endpoint worked. A direct `Mast.Caom.Filtered` target-name query was inconsistent for TIC 117549174 (empty), so the position-based CAOM cone result was used. The cone result independently contains the exact TIC 117549174 TESS-SPOC records.
- ASAS-SN’s displayed catalogue update dates (2021 and 2022) mean those non-detections do not exclude a later or unpublished variable-star classification. The TESS EB site is a live catalogue, but its target pages explicitly report both IDs as absent at query time.
- A TESS/SPOC or HLSP metadata record proves archival data processing, not that a target was announced as a candidate or assigned a positive astrophysical classification.
