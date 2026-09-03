# Protocol Freeze v1 — Phase 0

Source docs: `RESEARCH_PIPELINE_PRD.md`, `RESEARCH_PROPOSAL.md`
Code source of truth (executable subset): `src/tess_assoc/protocol.py` (`PROTOCOL_VERSION = "v1"`).
Prose freeze (product, metrics, semantics, baselines) lives in this document.

## Frozen definitions

- Primary photometric product: **TESS-SPOC FFI** (fixed for core benchmark).
  Reserved for vetting only: SPOC, QLP, TGLC.
- Development: Sectors **1–79**. Sealed holdout: **80–105**. Conditional discovery: **106**.
- Long-period lower boundary: **27.0 days**. Alias formula: `P_n = DeltaT / n`.
  Executable definition: `tess_assoc.orbit.generate_aliases`. Alias cap: **MAX_ALIAS_N = 10000**.
- Event record required: tic_id, sector, t0, local_time, local_flux,
  depth, duration_days, snr, stellar_meta, quality.
- TIC is grouping/partition key only — never a model feature.
- Learned output: `P(same transit-producing object)` — not planet probability,
  not period prediction. Primary input: normalized local transit morphology.
- Deterministic baseline: relative depth diff + relative duration diff +
  morphology correlation + timing plausibility.
- Primary metric: true-repeat retrieval at fixed candidate burden.

## Reproducibility

Record per run: TIC list, sector manifest, source-product version, catalogue
version + download date, preprocessing params, event-selection rules, pair
construction, TIC partition assignments, random seed, model checkpoint,
thresholds, injection params, holdout-unblinding date.
Holdout audit helpers: `validate_no_temporal_leak`, `validate_tic_partition`.

## Change rule

v1 is immutable. Any change requires `PROTOCOL_VERSION = "v2"` plus a new
freeze record. Sealed sectors must not enter dev inputs before the
freeze/seal audit (Phase 6).
