# Revision Response

We thank the reviewer for their thorough and constructive critique. Below we address each point raised and record the concrete revisions made to the manuscript and analysis artifacts.

## Major Weaknesses

### C.1 — Ground Truth Validity
**Reviewer concern:** The original paper conflated VLM score approximation with true affordance prediction and did not clearly state that the labels were proxy annotations rather than human judgments.

**Action taken:** We reframed the paper throughout around `VLM-derived affordance scores` rather than absolute affordances. This change appears in the title, abstract, introduction, discussion, conclusion, and the new Limitations section in [main.tex](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/report/main.tex). We also added explicit prose that Qwen2-VL-7B is a scalable proxy annotator, not ground truth, and cited prior work on model-assisted annotation (`Gilardi2023`) in Related Work and the Introduction.

**Rationale:** This makes the paper scientifically honest about what is being measured. We did not fabricate a human-validation study; instead, we state directly that human-rated ground truth is the highest-priority next step.

### C.2 — Statistical Power and Sample Size
**Reviewer concern:** The original paper reported point estimates without confidence intervals and used language stronger than the sample size supported.

**Action taken:** We added bootstrap 95% confidence intervals for RMSE, MAE, and Pearson `r` in [metric_confidence_intervals.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/metric_confidence_intervals.csv), regenerated the main comparison figure with CI bars, and added paired Wilcoxon tests plus Cohen’s `d` in [statistical_tests.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/statistical_tests.csv). The Experiments and Discussion sections now state explicitly that only four of five Model B vs. Model D improvements are statistically significant and that the L141 effect remains uncertain with `n = 84` test samples.

**Rationale:** This addresses the uncertainty issue directly and prevents overclaiming.

### C.3 — Synthetic-Only Evaluation
**Reviewer concern:** The original draft never acknowledged that all evaluation used synthetic Hypersim renderings and made no statement about domain gap.

**Action taken:** We now identify the images as synthetic in the title-adjacent framing, abstract, method, dataset summary, discussion, and Limitations section in [main.tex](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/report/main.tex).

**Rationale:** This constrains the paper’s claims to the actual benchmark used.

### C.4 — Novelty Assessment
**Reviewer concern:** The original “Novelty Arguments” section overclaimed novelty for components that were already standard.

**Action taken:** We removed the standalone “Novelty Arguments (Bonus)” section and folded a narrower novelty claim into the Introduction and Related Work: the contribution is the end-to-end combination and application of structured VLM distillation to scene-level score prediction, not any individual model component.

**Rationale:** The revised framing is more defensible and less defensive.

### C.5 — Experimental Methodology Concerns
**Reviewer concern:** The split strategy was unspecified; LightGBM search settings were underspecified; the CNN comparison used a very small validation set; and some table values were not reproducible from retained artifacts.

**Action taken:** We reconstructed and saved the missing assembled dataset and split summary in [pilot_dataset.parquet](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/data/assembled_dataset/pilot_dataset.parquet) and [split_summary.json](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/data/assembled_dataset/split_summary.json). The paper now states explicitly that the split is scene-level and leak-free. We also added the LightGBM search ranges and final per-affordance hyperparameters to the Method and a new hyperparameter table. We retained only the reproducible best-CNN aggregate results in the main comparison and avoided unsupported per-architecture correlation claims.

**Rationale:** This resolves the missing-methods issues and closes the split-description gap the reviewer identified.

### C.6 — Missing Baselines and Comparisons
**Reviewer concern:** The paper lacked a mean-prediction baseline and a simple linear regression baseline.

**Action taken:** We computed and saved both baselines in [baseline_mean_prediction.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/baseline_mean_prediction.csv) and [baseline_linear_regression.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/baseline_linear_regression.csv), and integrated them into the main results tables and discussion. The revision now states explicitly that Model B is worse than the mean baseline on L091.

**Rationale:** These baselines are essential for interpreting whether the students learn anything beyond simple priors.

### C.7 — CNN vs. LightGBM Comparison Confound
**Reviewer concern:** The draft described this as a CNN-versus-LightGBM architecture comparison even though the tabular models received strong segmentation-derived features.

**Action taken:** We rewrote the comparison language in the Experiments and Discussion sections to describe it as a feature-pipeline comparison in a low-data regime, not a pure architecture comparison.

**Rationale:** This preserves the useful empirical result while avoiding a misleading causal interpretation.

### C.8 — Indicator Feature Dimensionality Confound
**Reviewer concern:** The gain from Model D could come from extra binary dimensions rather than indicator semantics.

**Action taken:** We ran the requested random-indicator permutation control and saved the results in [indicator_permutation_test.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/indicator_permutation_test.csv). We added the result to the paper and now state the mixed conclusion directly: semantics dominate on L059/L079/L091, while dimensionality also contributes on L130/L141.

**Rationale:** This directly tests the confound rather than speculating about it.

### C.9 — SHAP Interpretation Overreach
**Reviewer concern:** The matching between SHAP features and VLM indicators was unspecified and the draft overstated what positive rank correlations implied.

**Action taken:** We inspected the original experiment code and documented the actual matching procedure: token overlap with a threshold of `0.3`, implemented automatically but still subjective in what constitutes a meaningful overlap. We also replaced causal language with correlational language in the SHAP discussion.

**Rationale:** This addresses both the missing-method description and the causal overreach.

### C.10 — Writing and Presentation
**Reviewer concern:** The draft under-described the synthetic setting, used unsupported language, had a thin Related Work section, included an informal novelty section, lacked a Limitations section, and contained a bad proxy citation.

**Action taken:** We expanded Related Work substantially, added a dedicated Limitations section, removed the novelty section, replaced unsupported superlatives with precise statements, fixed p-value formatting and units, and removed the incorrect `Li2023` proxy citation from [references.bib](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/report/references.bib).

**Rationale:** These revisions improve clarity, scientific tone, and bibliographic integrity.

## Minor Weaknesses

### D.1 — Inconsistent Decimal Places
**Action taken:** We standardized reported metrics to three decimal places in the paper and used threshold notation for very small p-values.

### D.2 — `0.0000` p-values
**Action taken:** We now report values below `0.0001` as `< 0.0001`.

### D.3 — Affordance Codes Used Without Definition in Tables
**Action taken:** Table entries and text now pair affordance IDs with descriptive names.

### D.4 — Validation Split is 9%, Not 10%
**Action taken:** The scene-level split description now states the exact split counts: 224/32/64 scenes and 298/38/84 images per affordance.

### D.5 — No Error Bars on Figures
**Action taken:** We regenerated the main comparison and indicator-distillation figures with bootstrap confidence intervals where per-sample predictions were available.

### D.6 — Hyperparameter Search Had No Quantitative Table
**Action taken:** We added the search ranges in Method and a final-hyperparameter table in the paper.

### D.7 — Missing LightGBM Training Details
**Action taken:** We added the objective and all requested hyperparameter families to the Method section and reported final values in-table.

### D.8 — Unused Bibliography Entries
**Action taken:** We removed the misleading proxy citation and updated the bibliography to match the revised text.

### D.9 — No Pipeline Diagram
**Action taken:** We created [pipeline_overview.pdf](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/figures/pipeline_overview.pdf) and inserted it into the Method section.

### D.10 — Missing Units
**Action taken:** The RMSE/MAE table caption now states that errors are measured in score points on the 1–7 VLM scale.

### D.11 — “Finalized” Process Language
**Action taken:** We removed process-language phrasing and rewrote the paper as a self-contained study.

### D.12 — ResNet-50 Correlations Not Verifiable
**Action taken:** We removed the unverifiable per-architecture correlation breakdown from the paper and retained only reproducible best-CNN aggregate metrics in the main comparison.

## Questions for the Authors

### Q1 — Evidence that Qwen2-VL-7B Scores Correlate with Human Judgments
**Answer:** We could not answer this with the available data because no human-rated subset exists in the workspace. Rather than fabricate an informal validation, we now state this explicitly as the primary limitation and highest-priority future work in the paper’s Limitations section.

### Q2 — RMSE of a Mean-Prediction Baseline
**Answer:** The mean-prediction baseline is now computed in [baseline_mean_prediction.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/baseline_mean_prediction.csv). Per-affordance RMSEs are: L059 `1.099`, L079 `1.287`, L091 `0.980`, L130 `1.180`, L141 `1.226`; macro RMSE is `1.154`. This confirms the reviewer’s key point that Model B is worse than the mean baseline on L091 (`1.140 > 0.980`).

### Q3 — Indicator Permutation Robustness
**Answer:** We ran the requested control and saved it in [indicator_permutation_test.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/indicator_permutation_test.csv). The control largely removes the gain on L059/L079/L091 but preserves part of it on L130/L141, indicating that semantics matter substantially, though extra sparse dimensions also help on some tasks.

### Q4 — Splitting Strategy
**Answer:** The split is scene-level and stratified by room cluster. We reconstructed and saved the summary in [split_summary.json](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/data/assembled_dataset/split_summary.json): 224 train scenes, 32 validation scenes, and 64 test scenes; zero scenes appear in multiple splits.

### Q5 — SHAP-to-VLM Matching Procedure
**Answer:** We inspected the original experiment code and documented the exact procedure in the paper. Feature names were normalized by removing prefixes and underscores, then matched to VLM indicator strings using token-overlap score with a threshold of `0.3`. This is automatic, but still an imperfect proxy for semantic alignment, and the revised discussion now states that explicitly.

## Revision Checklist Status

| # | Priority | Item | Status |
|---|----------|------|--------|
| 1 | MUST | Reframe claims to distinguish VLM distillation from affordance prediction | ✅ Done |
| 2 | MUST | Add mean-prediction baseline | ✅ Done |
| 3 | MUST | Add explicit Limitations section | ✅ Done |
| 4 | MUST | State train/val/test splitting strategy | ✅ Done |
| 5 | SHOULD | Add linear regression baseline | ✅ Done |
| 6 | SHOULD | Add bootstrap confidence intervals | ✅ Done |
| 7 | SHOULD | Run random-indicator control | ✅ Done |
| 8 | SHOULD | Acknowledge CNN vs. LightGBM confound | ✅ Done |
| 9 | SHOULD | Expand Related Work | ✅ Done |
| 10 | SHOULD | Report final LightGBM hyperparameters | ✅ Done |
| 11 | SHOULD | Describe SHAP-to-VLM matching procedure | ✅ Done |
| 12 | NICE | Remove “Novelty Arguments (Bonus)” section | ✅ Done |
| 13 | NICE | Add a pipeline diagram | ✅ Done |
| 14 | NICE | Format tiny p-values as `< 0.0001` and standardize decimals | ✅ Done |
| 15 | NICE | Remove process language | ✅ Done |

## Additional Output Artifacts

- Revised manuscript: [main.tex](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/report/main.tex)
- Compiled PDF: [main.pdf](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/report/main.pdf)
- Rebuilt dataset split summary: [split_summary.json](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/data/assembled_dataset/split_summary.json)
- Baselines and tests: [baseline_mean_prediction.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/baseline_mean_prediction.csv), [baseline_linear_regression.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/baseline_linear_regression.csv), [statistical_tests.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/statistical_tests.csv), [indicator_permutation_test.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/indicator_permutation_test.csv), [metric_confidence_intervals.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/metric_confidence_intervals.csv), [vlm_score_diagnostics.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/vlm_score_diagnostics.csv)
- Thresholded auxiliary analysis: [classification_f1_results.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/classification_f1_results.csv), [classification_f1_macro.csv](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/results/classification_f1_macro.csv)
- Qualitative example figure: [qualitative_l091_examples.pdf](/Users/taggertsmith/Desktop/COGS%20185/FINAL/project/outputs/figures/qualitative_l091_examples.pdf)
