# What's New in Image Tagger v3.3.x

This note summarizes the main changes introduced in the 3.3.x series,
relative to the earlier 3.2.x line.

## 1. Four dedicated GUIs

The old single-HTML / iframe shell has been replaced by four separate apps:

- **Tagger Workbench** (`frontend/apps/workbench`): high-throughput labeling UI.
- **Supervisor Monitor** (`frontend/apps/monitor`): team metrics, IRR summaries, Tag Inspector.
- **Admin Cockpit** (`frontend/apps/admin`): model + cost settings, tool configs, bulk image upload.
- **Research Explorer** (`frontend/apps/explorer`): faceted search over validated images, export hooks.

Each app has its own `ApiClient` with a clear base path and role header.

## 2. Admin bulk image upload

Admins can now upload multiple images in one operation via the Admin Cockpit.
The backend endpoint is:

- `POST /api/v1/admin/upload`

The endpoint returns `created_count`, `image_ids`, and `storage_paths`. It is
restricted to users with `X-User-Role: admin`.

## 3. Explorer attribute descriptions

The Explorer now surfaces attribute descriptions (or notes) as hover tooltips
on attribute chips, when available in the attribute catalog.

## 4. Frontend smoketest

A new script, `scripts/smoke_frontend.py`, checks that the portal is reachable
and that at least one of the expected key phrases appears in the rendered HTML
(or in a Playwright-driven browser session if Playwright is installed). This
is wired into `install.sh` as a lightweight end-to-end sanity check.

## 5. Governance and packaging

The 3.3.x line standardizes governance and packaging:

- `v3_governance.yml` + `scripts/guardian.py` enforce no-deletion and other rules.
- Releases are shipped both as a ZIP and as a concatenated TXT with a `deconcat.py` helper.

## 6. v3.3.7 refinements

In v3.3.7, we additionally:

- Unified version strings across `VERSION`, `backend/main.py`, `install.sh`, and key docs.
- Made `scripts/smoke_science.py` tolerant of a zero-image database (it now skips with guidance
  instead of failing hard).
- Added pytest coverage for the Admin bulk upload endpoint and the Monitor Tag Inspector.
- Improved empty-state messaging in the Monitor and Explorer apps.
- Added this `WHATS_NEW_v3_3_x.md` summary and small documentation polish.

## 7. v3.3.10 debug layers and deployment guide

- Added a `/v1/debug/images/{{image_id}}/edges` endpoint and wired it into Explorer
  so that students can toggle between the original image and its edge-map view.
- Introduced `docs/PRODUCTION_DEPLOYMENT.md` describing how to rotate API secrets,
  mount persistent volumes, and front the system with HTTPS.

