# Case Study: Comparing Human vs AI Ratings (VLM Cognitive / Affective Analysis)

This case study is designed for a small "scientist role" exercise.
Students compare **human ratings** of architectural images with the
**AI-generated cognitive and affective ratings** produced by Image Tagger's
Visual Language Model (VLM) pipeline.

## 1. Goal

Given a small set of images (e.g., 20–40 interiors), students will:

- Collect **human ratings** on:
  - coherence
  - complexity
  - legibility
  - mystery
  - restoration
  - cozy, welcoming, tranquil, scary, jarring

- Run the **science pipeline** with a real VLM provider configured
  (Gemini / OpenAI / Anthropic).

- Export a **BN-ready dataset** from Image Tagger.

- Merge the two datasets in a notebook and compute:
  - correlations (Pearson / Spearman) between human and AI ratings
  - simple scatter plots and inspection of outliers.

## 2. Setup

1. Configure the VLM provider in the **Admin Cockpit → VLM Engine** panel:

   - Choose a provider (Gemini, OpenAI, Anthropic, or Stub for dry runs).
   - Optionally set a *cognitive & affective prompt override* to customize
     the wording for this class or experiment.
   - Set a **max recommended batch size** (e.g., 50 images).
   - Set an approximate **VLM cost per 1000 images** (USD). This is used for
     rough cost estimates only.

2. Add a small dataset of images via:

   - **Admin Bulk Upload** (recommended), or
   - the existing seeding scripts.

3. Confirm that the science pipeline runs successfully on at least one image
   (e.g., using `scripts/run_science_on_sample.py` or the science harness).

## 3. Collecting Human Ratings

Create a simple survey (Qualtrics, Google Forms, or a custom experiment) that:

- Shows each image.
- Asks participants to rate each dimension from 0.0 (very low) to 1.0 (very high),
  or on a Likert scale that can be rescaled to [0, 1].

Export the human ratings as a CSV with at least:

- `image_id` or `image_filename`
- the ten rating columns:
  - `human.coherence`, `human.complexity`, `human.legibility`,
    `human.mystery`, `human.restoration`,
  - `human.cozy`, `human.welcoming`, `human.tranquil`,
    `human.scary`, `human.jarring`.

## 4. Exporting AI / VLM Ratings from Image Tagger

1. Run the science pipeline on the selected images so that the VLM-based
   **cognitive.* and affect.* attributes** are written to the database
   (or CSV, if using the science harness).

2. Use the BN export script to obtain a machine-readable dataset, e.g.:

```bash
python -m scripts.export_bn_ready_dataset --output bn_dataset.csv
```

3. Inspect `bn_dataset.csv` and confirm that it contains columns such as:

- `cognitive.coherence`, `cognitive.complexity`,
  `cognitive.legibility`, `cognitive.mystery`, `cognitive.restoration`
- `affect.cozy`, `affect.welcoming`, `affect.tranquil`,
  `affect.scary`, `affect.jarring`

## 5. Merging the Datasets

In a Jupyter notebook (Python), students can:

1. Load both CSV files:

```python
import pandas as pd

ai_df = pd.read_csv("bn_dataset.csv")
human_df = pd.read_csv("human_ratings.csv")

# Align on image identifier (image_id or filename)
merged = ai_df.merge(human_df, on="image_id", how="inner")
print("Merged shape:", merged.shape)
```

2. Compute correlations:

```python
cog_cols = [
    "coherence", "complexity", "legibility", "mystery", "restoration",
]
affect_cols = ["cozy", "welcoming", "tranquil", "scary", "jarring"]

for dim in cog_cols + affect_cols:
    ai_col = f"cognitive.{dim}" if dim in cog_cols else f"affect.{dim}"
    human_col = f"human.{dim}"
    corr = merged[[ai_col, human_col]].corr(method="pearson").iloc[0, 1]
    print(f"{dim:12s} Pearson r = {corr: .3f}")
```

3. Plot scatter plots for a few dimensions:

```python
import matplotlib.pyplot as plt

for dim in ["coherence", "complexity", "restoration"]:
    ai_col = f"cognitive.{dim}"
    human_col = f"human.{dim}"
    plt.figure()
    plt.scatter(merged[human_col], merged[ai_col])
    plt.xlabel(f"Human {dim}")
    plt.ylabel(f"AI {dim}")
    plt.title(f"Human vs AI: {dim}")
    plt.grid(True)
    plt.show()
```

## 6. Discussion Questions

- On which dimensions do human and AI ratings agree most strongly?
- Are there systematic biases (e.g., AI consistently overestimates "cozy"
  compared to human ratings)?
- Do disagreements cluster in particular kinds of images (e.g., high clutter,
  unusual lighting, ambiguous spaces)?
- How sensitive are the AI ratings to changes in the **prompt override**
  configured in the Admin Cockpit?

## 7. Variations

- Compare different VLM providers (Gemini vs OpenAI vs Anthropic) on the same
  set of images.
- Run a second cohort of students with a slightly different cognitive prompt
  and explore how the wording affects AI–human agreement.
- Use the BN tooling to discretize the AI ratings into bins and compare
  agreement at the categorical level rather than raw scores.
