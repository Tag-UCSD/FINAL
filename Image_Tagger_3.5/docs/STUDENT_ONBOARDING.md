# Student Onboarding Guide (Image Tagger v3.3.7)

This guide is for students who are joining the Image Tagger v3 project. It gives you a
practical sequence of steps so you can get productive quickly.

## Phase 0 – Read the ground rules (30–60 minutes)

Before you touch code, skim:

- `PROJECT_CONSTITUTION.md` – how this repo is expected to evolve.
- `docs/governance_guide.md` – what Guardian is and why it matters.
- `docs/AI_COLLAB_WORKFLOW.md` – how we work with AI tools on this project.

You do *not* need to memorise everything; just get a sense of:

- Why we avoid deleting files.
- Why each release ships as a ZIP + concatenated TXT.
- How Guardian is used to detect drift.

## Phase 1 – Get the system running (1–2 sessions)

1. Make sure Docker Desktop is installed and running.
2. From the repo root:

   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. When `install.sh` finishes, check:

   - The API health endpoint: `http://localhost:8000/health`
   - The role portal (index page) – you should see links to:
     - Tagger Workbench
     - Supervisor Monitor
     - Admin Cockpit
     - Research Explorer

If something fails, see `docs/devops_quickstart.md` for common issues.

## Phase 2 – Explore the GUIs (1–2 sessions)

With the stack running:

1. **Tagger Workbench**
   - Open the Tagger app from the portal.
   - Load a small batch of images (if available) and try tagging a few.
   - Verify that your tags appear in the database by refreshing the Workbench and, later,
     by looking at Monitor.

2. **Supervisor Monitor**
   - Open the Monitor app.
   - You should see velocity and IRR summaries.
   - If there is no data yet, Monitor may show empty tables; once taggers have worked,
     it should display real stats.
   - Use the Tag Inspector to inspect disagreements for individual images.

3. **Admin Cockpit**
   - Inspect model and budget settings.
   - You do not need to change anything at first; just understand what controls exist.

4. **Research Explorer**
   - Open the Explorer app.
   - Load attributes and try simple searches or filters.
   - Notice how attribute keys are grouped; these map to the underlying attribute registry.

## Phase 3 – Understand the science pipeline (2–3 sessions)

Read:

- `docs/science_overview.md` – how the science pipeline is structured.
- Skim the code under `backend/science/`, especially:
  - `pipeline.py`
  - `summary.py`
  - A couple of analyzers (e.g. colour, texture, fractals).

Your goal in this phase:

- Understand what `AnalysisFrame` does.
- Know which attributes the pipeline writes into the `Validation` table.
- See how composite indices (for example `science.visual_richness`) and their bins are built.

If you have time, run or adapt `scripts/smoke_science.py` to confirm that the pipeline can
successfully write science attributes for at least one image.

## Phase 4 – BN export and data for modelling (2–3 sessions)

Once you are comfortable with the science outputs:

1. Read `backend/api/v1_bn_export.py` and `backend/schemas/bn_export.py`.
2. Inspect `backend/science/index_catalog.py` to see which indices are considered for BN.
3. Ensure there are science `Validation` rows in the database (either by running the
   pipeline or by using a seeding script).

Then, either:

- Call the `/v1/export/bn-snapshot` endpoint from a notebook or script, or
- Use the `export_bn_snapshot` function directly inside a Python session.

Your goal is to produce a small CSV or JSONL of BN-ready rows that can be used in external
tools (for example PyMC, pgmpy, or a custom BN visualiser).

## Phase 5 – Contribute safely (ongoing)

When you are ready to make changes:

1. Decide which area you are touching:
   - Science (new analyzers or indices).
   - UX (Workbench, Monitor, Explorer).
   - DevOps / tests.
2. Update the relevant code and docs *together*.
3. Run at least:
   - `python scripts/guardian.py verify`
   - `pytest -q`

When you are ready to make changes, run at least:

- `pytest tests/test_v3_api.py` – API health and RBAC sanity.
- `pytest tests/test_guardian.py` – governance (Guardian) sanity.
- `pytest tests/test_bn_export_smoke.py` – BN export coupled to Validation.
- `pytest tests/test_workbench_smoke.py` – Tagger Workbench basic flow.
- `pytest tests/test_explorer_smoke.py` – Research Explorer basic flow.
- (Optional) `pytest -m slow` – includes `test_science_pipeline_smoke.py`, which
  runs the full science pipeline on a synthetic image.

If Guardian reports drift and the changes are intentional, talk to the project lead about
updating the baseline (`freeze`) as part of the next release.

## When in doubt

- Ask a TA or project lead to sanity-check your plan.
- Use AI tools as helpers, but remember that this repo has governance rules – changes
  should keep tests, docs and contracts aligned.

## Example Lab 1 – End-to-end walkthrough (60–90 minutes)

This lab is designed as a first-session exercise:

1. Run `./install.sh` and confirm the role portal and `/health` endpoint work.
2. Open the Tagger Workbench and tag at least 5 images.
3. Open the Supervisor Monitor and observe:
   - Velocity table (see whether your user ID appears).
   - IRR table (if multiple taggers have worked on the same images).
4. Open the Research Explorer and:
   - Load attributes.
   - Filter images (if available) by a science attribute such as
     `science.visual_richness` or by a tag you just created.
5. (Optional) Run `pytest tests/test_bn_export_smoke.py` and inspect the output
   of `/v1/export/bn-snapshot` in a notebook to see how science attributes and
   bins are packaged for BN tools.


## A note on using AI tools

AI assistants (ChatGPT, Claude, Gemini, etc.) are welcome, but please:

- Do **not** ask an AI to regenerate this repository from scratch.
- Always start from a current ZIP and request incremental, governance-respecting
  changes (no deletions, keep Guardian and tests intact).
- Treat AI as a collaborator that proposes diffs you can review, rather than as
  an opaque code generator.

