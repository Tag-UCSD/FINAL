# Project Constitution: Image Tagger v3 (v3.4.12)

This document encodes the core rules for how the Image Tagger v3 repository should evolve.
It is meant to be read by humans (students, collaborators, AI agents) before making changes.

## 1. Versioning and releases

- The v3.x line is versioned as `3.M.N` where `N` increments for each release.
- Each release must be shippable as:
  - A ZIP of the full directory tree.
  - A single concatenated TXT file with a `deconcat.py` header that can reconstruct the tree.
- A later version should be a **superset** of earlier releases in terms of file paths.
  If code is removed from active use, it should be archived rather than deleted.

## 2. No-deletion rule (with archives)

- Do not delete files that have shipped in a prior release.
- When replacing or majorly refactoring a file, move the old version into an `archive/`
  subtree such as:
  - `archive/v3_2_38_old_science/…`
- New files can be added freely, as long as they pass syntax checks and do not break imports.

## 3. Governance and Guardian

- The file `v3_governance.yml` and `scripts/guardian.py` define a “drift shield” over
  critical parts of the system (science modules, API, deploy scripts, governance config).
- Contributors should run `python scripts/guardian.py verify` before proposing a new release.
- If intentional changes are made to protected scopes, the baseline can be updated with
  `python scripts/guardian.py freeze` after review.

## 4. Code quality guarantees

For active (non-archived) code:

- No `...` placeholders in Python modules.
- No syntax errors in `.py` files under `backend/`, `scripts/` or `tests/`.
- No obvious stubs that would cause runtime failures when a documented feature is used.
- Every API router and endpoint imported in `backend/main.py` must exist and be importable.

## 5. Science and UX honesty

- Science modules are heuristic and deterministic; they should be implemented in a way that
  allows students to:
  - Read the code.
  - Reproduce results.
  - Modify thresholds and see predictable changes.
- UX should be honest about system state:
  - When there is no data yet, screens should say so explicitly.
  - When the backend is unreachable or access is denied, error messages should be clear.

## 6. Documentation guarantees

Each release should keep student-facing documentation reasonably aligned with reality, e.g.:

- `README_v3.md` for high-level overview and quickstart.
- `docs/science_overview.md` for the science pipeline.
- `docs/devops_quickstart.md` for setup and basic operations.
- `docs/governance_guide.md` for governance and Guardian.
- `docs/AI_COLLAB_WORKFLOW.md` for the multi-agent AI collaboration rules specific to this project.

If the behaviour of a core subsystem changes (science pipeline, routers, governance), the
corresponding document should be updated in the same release.

## 7. Contributions

- Prefer small, well-structured changes with clear commit messages or release notes.
- When making structural changes (for example new routers, new science analyzers), update:
  - Relevant docs.
  - Tests where applicable.
  - The index catalog if new indices are introduced.
- When working with AI tools, follow `docs/AI_COLLAB_WORKFLOW.md` so that all AIs respect
  the same governance and packaging rules.

By following this constitution, the Image Tagger v3 line can evolve rapidly while remaining
readable, teachable and robust enough for real experiments.
