# VLM Health Quickstart

This is the short version of the VLM health procedure for TAs and students.

Use this when you just need to run the standard checks and hand the results
to the PI, without reading the full SOP.

---

## Prerequisites

- You can run the normal Image Tagger pipeline end-to-end.
- The following CSVs have already been generated for the run you care about:
  - `reports/bn_validations_flat.csv`
  - `reports/vlm_validations.csv`
  - `reports/human_validations.csv`

If you are not sure whether these exist, ask your supervisor before running
the health checks.

---

## Step 1 – Initialise a health run

From the repo root:

```bash
make vlm-health-init
```

This will:

- Create a new folder under `reports/vlm_health/` named with today’s date,
  the current VERSION file, and a default VLM profile (e.g. `main_vlm`).
- Copy the three input CSVs into `raw/` inside that folder.

You should see a message telling you which `RUN_ID` it used.

---

## Step 2 – Run the variance audit

```bash
make vlm-health-audit
```

This will run `scripts/audit_vlm_variance.py` on the flat BN validations
export and write a CSV of potentially problematic attributes to:

```text
reports/vlm_health/<RUN_ID>/derived/vlm_variance_audit.csv
```

At minimum:

- Open this CSV.
- Sort by `dominant_ratio` and `n`.
- Note any attributes that look suspicious (e.g. almost always the same value).

You do not have to fix anything; just flag them in the log.

---

## Step 3 – Prepare the Turing panel

```bash
make vlm-health-panel
```

This will create:

```text
reports/vlm_health/<RUN_ID>/raw/vlm_turing_panel.csv
```

Send this file (or a Google Sheet based on it) to the human judges who will
compare VLM vs human labels. They should fill in at least:

- `rater_id`
- `guess_is_ai` (A or B)
- optionally `rating_A` and `rating_B` if you are using numeric ratings.

When they are done, save the completed CSV as:

```text
reports/vlm_health/<RUN_ID>/raw/vlm_turing_panel_completed.csv
```

---

## Step 4 – Score the Turing panel

```bash
make vlm-health-score
```

This will run `scripts/vlm_turing_test_score.py` on the completed panel and
write a text summary to:

```text
reports/vlm_health/<RUN_ID>/derived/vlm_turing_summary.txt
```

Open this file and check that it contains:

- Total number of judgments.
- Overall accuracy of `guess_is_ai` vs the hidden ground truth.
- Per-rater accuracy.
- Mean ratings for AI vs human labels (if ratings were provided).

---

## Step 5 – Fill in the log

In the same folder, open or create:

```text
reports/vlm_health/<RUN_ID>/log.md
```

Add a short note with:

- Date, repo version, VLM profile (copy the RUN_ID at the top).
- Where the audit and Turing files live.
- Any attributes you think look suspicious in the variance audit.
- Whether the Turing summary “looks okay” (no need for deep interpretation).

Example sketch:

```markdown
# VLM health log – 2025-11-27_v3.4.69_main_vlm

- Audit CSV: derived/vlm_variance_audit.csv
- Turing summary: derived/vlm_turing_summary.txt

Noted attributes:
- `ceiling_height` has very high dominance; flagged for PI review.

Turing headline:
- Overall accuracy ~0.53; nothing obviously broken.

```

Once this is done, send the RUN_ID and the log file to the PI or lead
researcher as instructed.
