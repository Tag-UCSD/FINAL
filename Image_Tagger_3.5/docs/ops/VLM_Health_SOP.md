# VLM Health & Turing SOP

This SOP defines the standard checks we run on the Vision-Language Model (VLM) tagging pipeline and its integration with the BN for CNfA experiments.

It has two main components:

1. A **variance audit**, to detect mode-collapsed or low-variance attributes in VLM outputs.
2. A **Turing-style panel**, where human judges compare anonymised VLM vs human labels.

If you just want the short, operational version of this procedure,
see **`docs/ops/VLM_Health_Quickstart.md`**.

---

## 1. When to run this SOP

Run the VLM Health SOP in each of these situations:

1. **New VLM weights or prompt.**
2. **New BN version that changes observables or binning.**
3. **New major dataset or domain shift.**
4. **Pre-release check** before a public or classroom deployment.

We identify each run by: `YYYY-MM-DD_<repo_version>_<vlm_profile>` (the RUN_ID).

---

## 2. Required scripts

The SOP assumes these scripts exist under `scripts/`:

- `audit_vlm_variance.py`
- `vlm_turing_test_prep.py`
- `vlm_turing_test_score.py`

Each script has its own `--help` describing CLI options.

---

## 3. Folder structure

For each run, create:

```text
reports/
  vlm_health/
    RUN_ID/
      raw/
      derived/
      log.md
```

where `RUN_ID = YYYY-MM-DD_<repo_version>_<vlm_profile>`.

- `raw/`   holds input CSVs and the Turing panel files.
- `derived/` holds audit outputs and Turing summaries.
- `log.md` summarises decisions and follow-ups.

---

## 4. Inputs

The SOP expects three CSVs:

1. `bn_validations_flat.csv`  
2. `vlm_validations.csv`  
3. `human_validations.csv`  

The actual filenames may differ, but they should provide:

- `image_id`
- `attribute_key`
- `source` (for the flat BN export)
- `value` (numeric or ordinal label)

---

## 5. Operational steps

Assume you are at repo root and have already produced the three CSVs.

### 5.1 Initialise the run folder

```bash
RUN_ID="YYYY-MM-DD_vX.Y.Z_profile"

mkdir -p "reports/vlm_health/${RUN_ID}/raw"
mkdir -p "reports/vlm_health/${RUN_ID}/derived"

cp reports/bn_validations_flat.csv "reports/vlm_health/${RUN_ID}/raw/"
cp reports/vlm_validations.csv    "reports/vlm_health/${RUN_ID}/raw/"
cp reports/human_validations.csv  "reports/vlm_health/${RUN_ID}/raw/"
```

### 5.2 Variance audit

```bash
python scripts/audit_vlm_variance.py       "reports/vlm_health/${RUN_ID}/raw/bn_validations_flat.csv"       --out "reports/vlm_health/${RUN_ID}/derived/vlm_variance_audit.csv"       --source-column source       --attribute-column attribute_key       --value-column value       --source-prefix science_pipeline.vlm
```

Review the resulting CSV and note:

- Attributes with high `dominant_ratio` and low variance.
- Whether each flagged attribute is:
  - Acceptable (true domain skew),
  - Needs investigation,
  - Blocking for release.

Document your judgment in `log.md`.

### 5.3 Build the Turing panel

```bash
python scripts/vlm_turing_test_prep.py       --vlm   "reports/vlm_health/${RUN_ID}/raw/vlm_validations.csv"       --human "reports/vlm_health/${RUN_ID}/raw/human_validations.csv"       --out   "reports/vlm_health/${RUN_ID}/raw/vlm_turing_panel.csv"       --max-trials 400       --seed 42
```

Give `vlm_turing_panel.csv` to human judges (e.g., via Google Sheets or a small UI) and ask them to complete:

- `rater_id`
- `rating_A`, `rating_B` (if used)
- `guess_is_ai` (A or B)
- optional `notes`

Save the completed file as:

```text
reports/vlm_health/${RUN_ID}/raw/vlm_turing_panel_completed.csv
```

### 5.4 Score the Turing panel

```bash
python scripts/vlm_turing_test_score.py       --panel "reports/vlm_health/${RUN_ID}/raw/vlm_turing_panel_completed.csv"       > "reports/vlm_health/${RUN_ID}/derived/vlm_turing_summary.txt"
```

The summary includes:

- Overall guess accuracy vs chance.
- Per-rater accuracy.
- Mean ratings of AI vs human labels (if rating fields are present).

Decide whether the VLM is acceptable for the intended use, and document in `log.md`.

---

## 6. Logging

For each run, fill in `reports/vlm_health/${RUN_ID}/log.md` with:

- Date, repo version, VLM profile.
- Pointer to:
  - `derived/vlm_variance_audit.csv`
  - `derived/vlm_turing_summary.txt`
- Short narrative of:
  - Which attributes, if any, are problematic.
  - Whether human judges can reliably distinguish AI vs human labels.
  - Whether apparent quality is acceptable.
- A small checklist of follow-up actions.

Example sections:

- `## 1. Variance audit`
- `## 2. VLM Turing Test`
- `## 3. Actions / follow-ups`

---

## 7. Makefile targets (optional)

For convenience, you can add Makefile targets:

- `vlm-health-init`
- `vlm-health-audit`
- `vlm-health-panel`
- `vlm-health-score`

so that operators can run each stage with a single command.

This SOP should be kept under version control and updated as the VLM, BN,
or evaluation protocol evolves.
