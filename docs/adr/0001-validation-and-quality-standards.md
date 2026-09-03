# ADR-0001: Construction, Validation, and Quality Standards

Date: 2026-09-03 · Status: accepted · Scope: all `src/tess_assoc` code and tests.

## Context

Five rounds of strict code review kept surfacing the same defect classes in new
code, including in code written *after* earlier rounds had banned them:
silent type coercion at loader boundaries, `TypeError` leaking where the
contract promises `ValueError`, dead helpers and re-exports, unclosed file
handles, untested published artifacts, placeholder literals, and duplicated
validation dialects. Each instance was small; the pattern was expensive.
This ADR records the standards so new phases inherit them instead of
re-discovering them under review.

## Decision

1. **Strict payloads, `ValueError` everywhere.** Loaders validate with
   `isinstance` / `require_*` checks and raise `ValueError` on any violation.
   Never coerce (`float(x)`, `str(x)`, `dict(x)`) to make bad input fit, and
   never let `TypeError`/`KeyError` escape a public boundary — check presence
   first (`missing …` / `unknown …`), then delegate.
2. **One validation dialect.** All scalar checks go through
   `tess_assoc/_validate.py` (`is_*` predicates for branch conditions,
   `require_*` raising helpers where failure always raises). New modules must
   reuse them, not invent local equivalents.
3. **Manifests validate themselves.** Every manifest type enforces its own
   invariants in `__post_init__` (field types, ranges, uniqueness, defensive
   copies of mutable containers). Loaders keep only *relational* checks
   (cross-references between records). No `dict[str, Any]` where a record
   type exists.
4. **No dead code lands.** Every public function is called by something tested;
   every import is used. No re-exports from non-canonical homes — import from
   the module that owns the concept.
5. **Resources are closed.** File handles, network sessions, and caches use
   context managers or explicit close paths. `fits.open(...)` always in
   a `with` block.
6. **Published artifacts are tested.** Anything the project presents as proof
   (reports, plots, STATUS figures) has a test that regenerates or verifies
   it. An untested figure generator is a defect.
7. **Literals stay honest.** Enum/literal fields describe what the data is
   (`"real"` for real transits, not `"box"`). Never launder one vocabulary
   through another to satisfy a type.
8. **Comparisons are boring.** Prefer `x <= 0` over `not x > 0`, direct
   iteration over `while True` + epsilon, and named helpers over long clever
   expressions. If a reader must simulate float behavior to trust a line,
   rewrite the line.

## Consequences

- New loaders copy the `load_manifest` pattern (strict, `ValueError`), not
  ad-hoc coercion. Reviewers reject `float(`/`str(` at boundaries on sight.
- New helpers go in `_validate.py` or justify their existence in review.
- New public functions arrive with a caller and a test, or they don't land.
- These rules are enforced by review, not tooling — until a lint pass encodes
  the checkable subset (unused imports, unclosed handles).
