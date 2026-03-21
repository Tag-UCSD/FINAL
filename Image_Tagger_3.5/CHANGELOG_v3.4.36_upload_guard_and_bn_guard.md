# v3.4.36 – Admin upload hardening + BN naming guard

This version is strictly additive relative to v3.4.35 and focuses on
closing the specific concerns raised by the five-panel Ruthless
review.

## Admin bulk upload

- Replaced the truncated `/api/v1/admin/upload` implementation with a
  hardened endpoint that:
  - restricts uploads to `.jpg`, `.jpeg`, `.png`, `.webp`;
  - enforces a 10 MiB per-file size limit;
  - validates the entire batch before writing anything;
  - stores files under `IMAGE_STORAGE_ROOT` with UUID filenames; and
  - records `upload_batch_id` for all new `Image` rows.

- Extended `tests/test_admin_upload.py` with:
  - rejection of unsupported extensions;
  - rejection of oversized uploads; and
  - an atomicity test for mixed valid/invalid batches.

## BN naming and glossary

- Added `backend/science/bn_naming_guard.py` as an advisory naming
  guard for `candidate_bn_input` keys.
- Added `scripts/export_bn_glossary.py` to export a JSON glossary to
  `docs/BN_GLOSSARY_AUTO.json`.
- Added `docs/BN_NAMING_GUIDE.md` and
  `docs/INTERPRETING_RUTHLESS_REPORTS.md`.
- Added `reports/Ruthless_3.4.35_five_panel_summary.md` to capture
  the five-panel verdict and how v3.4.36 responds.

