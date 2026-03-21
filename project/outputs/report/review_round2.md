# Follow-Up Review (Round 2)

**Paper:** "Distilling VLM-Derived Indoor Affordance Scores from Synthetic Scene Images"
**Venue standard:** NeurIPS / ICML (course-project calibration)
**Reviewer:** Reviewer 2 (adversarial)

---

## SECTION A: Assessment of the Revision

This is an unusually thorough and honest revision. The authors addressed all 15 items on the original checklist — four [MUST], seven [SHOULD], and four [NICE] — and in nearly every case the fix is substantive rather than cosmetic. The title, abstract, introduction, discussion, and conclusion have been genuinely rewritten to frame the work as VLM distillation rather than affordance prediction. Two new baselines (mean prediction, linear regression), bootstrap confidence intervals, effect sizes, and a permutation control for the indicator dimensionality confound have all been computed correctly and integrated into the narrative. A dedicated Limitations section addresses the four major structural concerns I raised (proxy labels, synthetic domain, sample size, comparison confound) with enough specificity that a reader encountering them cold would understand the issues and their implications. The Related Work has been expanded from three sentences to three substantive paragraphs with real engagement. The revision also exposed and honestly disclosed an asymmetry in artifact retention (per-image predictions for tabular models but only aggregate summaries for CNNs), which the original draft had silently papered over. I verified every new number against the underlying CSVs. With one trivial exception (L079 Best CNN rounds to 1.244, not 1.245), all values match.

---

## SECTION B: Point-by-Point Evaluation

### C.1 — Ground truth validity

**Original concern:** The paper conflated VLM score approximation with true affordance prediction throughout.
**Authors' response:** Reframed the entire paper — title, abstract, introduction, discussion, conclusion, and new Limitations section — around "VLM-derived affordance scores" rather than absolute affordances.
**Verdict:** ✅ RESOLVED
**Assessment:** The reframing is pervasive and genuine, not a patch. The new title ("Distilling VLM-Derived Indoor Affordance Scores from Synthetic Scene Images") makes the scope immediately clear. The abstract states "we frame these results as evidence about *distillation fidelity*" (line 52) and concludes "human-rated affordance validity remains untested" (line 54–55). The Introduction says "Qwen2-VL-7B is a scalable *proxy annotator*, not ground truth" (line 73) and "our evaluation measures how well lightweight students reproduce the teacher's outputs, not whether those outputs match human affordance judgments" (lines 74–76). The Limitations section devotes its first paragraph to this issue and calls human validation "the first priority for future work" (line 451). The Discussion explicitly lists claims that are "no longer defensible" (line 433–438). I checked for residual overclaiming throughout and found none.

### C.2 — Statistical power and sample size

**Original concern:** No confidence intervals, no effect sizes; language stronger than sample size warranted.
**Authors' response:** Added bootstrap 95% CIs for RMSE/MAE/Pearson r, Wilcoxon p-values with Cohen's d, and hedged language for L141.
**Verdict:** ✅ RESOLVED
**Assessment:** The new metric_confidence_intervals.csv contains CIs for all four reproducible models across all five affordances. Table 6 (tab:indicator) now reports delta RMSE, 95% CI, Wilcoxon p, and Cohen's d for each affordance. The text states "the L141 Yoga/Stretching improvement remains numerically meaningful... but is not statistically significant at α = 0.05 (p = 0.0619)" (lines 312–314) and "With only 84 test samples per affordance, power to detect small effects is limited" (line 314). I verified the CIs against the CSV: all match to 3 decimal places. The CIs for Model D L091 RMSE (0.812, CI [0.661, 0.953]) vs. Model B L091 RMSE (1.140, CI [0.920, 1.337]) show clear separation, supporting the claim. The CIs for Model D L141 RMSE (1.002, CI [0.835, 1.142]) vs. Model B L141 RMSE (1.175, CI [0.980, 1.348]) show overlap, consistent with the non-significant p-value. This is exactly the kind of transparent reporting I asked for.

### C.3 — Synthetic-only evaluation

**Original concern:** No mention of synthetic domain, domain gap, or transfer limitations.
**Authors' response:** Identified images as synthetic throughout; added domain-gap discussion to Limitations.
**Verdict:** ✅ RESOLVED
**Assessment:** The word "synthetic" now appears in the title, abstract (line 38, "synthetic indoor scenes"), dataset table caption ("420-image synthetic Hypersim benchmark"), method section (line 137, "*synthetic* Hypersim images"), and Limitations (lines 455–460). The Limitations paragraph on domain says "The segmentation-based feature pipeline may also behave differently on synthetic renders than on real photographs. We therefore make no empirical claim about transfer beyond this synthetic benchmark" (lines 458–460). This is substantive and sufficient.

### C.4 — Novelty assessment

**Original concern:** The "Novelty Arguments (Bonus)" section overclaimed; contributions were standard.
**Authors' response:** Removed the section; folded narrower claims into Introduction and Related Work.
**Verdict:** ✅ RESOLVED
**Assessment:** Section 7 is gone. The Related Work now says "The novelty is not any single component... The contribution is the combination of these components into a reproducible pipeline for scene-level VLM score distillation, together with analyses of when structured indicator supervision helps" (lines 116–121). This is an honest and defensible framing.

### C.5 — Experimental methodology concerns

**Original concern:** (a) Split strategy unspecified; (b) CNN validation set tiny; (c) Optuna ranges unreported; (d) per-architecture CNN correlations unverifiable.
**Authors' response:** (a) Scene-level split described and saved in split_summary.json; (b) acknowledged as limitation; (c) full search space reported; (d) unverifiable per-architecture breakdowns removed.
**Verdict:** ✅ RESOLVED
**Assessment:** (a) Section 3.1 now states "The split is *scene-level* and stratified by room cluster" (line 162) with exact counts: "224 train scenes, 32 validation scenes, and 64 test scenes" (line 165). I verified split_summary.json: 224+32+64 = 320 scenes, 298+38+84 = 420 images, and the cluster-stratified breakdown is provided. (b) The CNN artifact asymmetry is honestly disclosed: "only aggregate checkpoint summaries were retained for the CNNs" (line 237). (c) The full Optuna search space is reported in lines 207–214 with parameter names, ranges, and scale types. (d) Tables 5–6 now show only "Best CNN" (a single reproducible value per affordance) rather than separate ResNet-18/50 correlations. The hyperparameter table (Table 7) reports final values for all 10 LightGBM models.

### C.6 — Missing baselines and comparisons

**Original concern:** No mean-prediction baseline, no linear regression baseline, no comparison to prior methods.
**Authors' response:** Added both baselines to tables and discussion. Stated that Model B is worse than the mean baseline on L091.
**Verdict:** ✅ RESOLVED
**Assessment:** Both baselines appear in Tables 5 and 6. I verified the mean baseline against baseline_mean_prediction.csv and the linear baseline against baseline_linear_regression.csv — all numbers match. The paper explicitly states "Model B's L091 RMSE (1.140) is worse than predicting the training-set mean (0.980)" (lines 226–227). This is the single most important sentence added in the revision, because it contextualizes Model B's L091 failure. The linear regression baseline also reveals a noteworthy result: it achieves r = 0.656 on L059 Sleep with RMSE only slightly below the mean baseline, suggesting the relationship is roughly linear for that affordance. The macro RMSE for linear regression (1.592) confirms that LightGBM's non-linearity matters substantially. Both baselines are computed correctly (training means applied to test set; linear regression trained on training set, evaluated on test).

### C.7 — Unfair CNN vs. LightGBM comparison

**Original concern:** The comparison confounded architecture choice with feature pipeline quality.
**Authors' response:** Rewrote comparison language to describe it as a feature-pipeline comparison.
**Verdict:** ✅ RESOLVED
**Assessment:** Lines 296–304 now say: "This is not a pure architecture comparison between CNNs and gradient boosting. The CNNs receive raw pixels and must learn the functional representation from roughly 298 training images per affordance, whereas the tabular models receive a strong pretrained segmentation-derived representation. The scientific claim we can support is therefore that *structured feature pipelines outperform image-only fine-tuning in this low-data synthetic regime*, not that boosted trees are inherently superior to CNNs for affordance modeling in general." This is exactly the caveat I asked for, and it is substantive rather than perfunctory.

### C.8 — Indicator feature dimensionality confound

**Original concern:** Model D's improvement could reflect dimensionality rather than indicator semantics; no control experiment.
**Authors' response:** Ran a random-indicator permutation control; reported mixed results.
**Verdict:** ✅ RESOLVED
**Assessment:** The permutation control is saved in indicator_permutation_test.csv and reported in Table 6 and lines 350–356. On L059/L079/L091, the permuted model's RMSE is nearly identical to Model B's (e.g., L059: permuted = 0.815 vs. Model B = 0.814), confirming that the gain is semantic. On L130/L141, the permuted model retains partial improvement (L130: permuted = 0.905 vs. Model B = 0.978 vs. Model D = 0.813), indicating that dimensionality contributes. The paper concludes "indicator semantics explain a substantial fraction of [the gain]" (line 356) rather than claiming full explanation. This nuanced reporting is exactly what I requested.

### C.9 — SHAP interpretation overreach

**Original concern:** Matching procedure undescribed; causal language used without justification.
**Authors' response:** Documented the matching procedure (token overlap with threshold 0.3); replaced causal language with correlational language.
**Verdict:** ✅ RESOLVED
**Assessment:** Lines 396–405 now describe the procedure: "tokenized feature names after removing prefixes such as presence\_ and count\_, and matched them to VLM indicator strings using word-overlap score |A ∩ B| / |A ∪ B| with a threshold of 0.3." The paragraph ends: "We therefore interpret the positive SHAP–VLM rank correlations (0.305–0.474) as suggestive alignment of salient cues, not as evidence that the student has recovered the teacher's causal reasoning" (lines 402–405). Both concerns are addressed.

### C.10 — Writing and presentation

**Original concern:** (a) Abstract didn't say "synthetic"; (b) Related Work was three sentences; (c) informal "Novelty Arguments (Bonus)" section; (d) no Limitations section; (e) table formatting issues; (f) unsupported superlatives; (g) proxy citation.
**Authors' response:** All subpoints addressed.
**Verdict:** ✅ RESOLVED
**Assessment:** (a) "Synthetic" appears in the title and abstract. (b) Related Work is now three substantive paragraphs citing AffordanceNet, Chuang 2018, Li 2019, Ego-Topo, Gilardi 2023, and the standard distillation/segmentation references—a genuine engagement with prior work. (c) Section 7 removed. (d) A four-paragraph Limitations section (Section 6) covers proxy labels, synthetic domain, statistical power, and artifact asymmetry, each in enough detail to stand alone. (e) Tables are reformatted with units stated in captions ("score points on the 1–7 VLM scale," line 244). (f) The word "substantially" appears only once in a context supported by the data (Discussion, line 424, "Model B already improves substantially over the best CNN and the mean baseline in four of five affordances"—supported by Tables 5–6 with CIs). (g) The Li2023 proxy citation is gone. The bibliography still contains unused entries for Caesar2018 and Breiman2001, but these are harmless (they don't appear in the compiled text).

### D.1 — Inconsistent decimal places
**Verdict:** ✅ RESOLVED. Tables use 3 decimal places uniformly for metrics.

### D.2 — "0.0000" p-values
**Verdict:** ✅ RESOLVED. Table 6 now uses "< 0.0001" for L079 and L130.

### D.3 — Affordance codes without definition
**Verdict:** ✅ RESOLVED. Table rows now include full affordance names (e.g., "L059 Sleep," "L079 Cook").

### D.4 — Validation split size
**Verdict:** ✅ RESOLVED. Section 3.1 states exact scene and image counts for each split.

### D.5 — No error bars
**Verdict:** ✅ RESOLVED. Figure 1 caption states "Error bars show 95% bootstrap confidence intervals" (line 233–234). Figure 3 caption describes CI bars on delta RMSE.

### D.6 — No quantitative hyperparameter table
**Verdict:** ✅ RESOLVED. Table 7 (tab:hparams) reports n_estimators, learning rate, leaves, depth, and min_child_samples for all 10 models.

### D.7 — Missing LightGBM training details
**Verdict:** ✅ RESOLVED. Lines 207–214 report the full Optuna search space with parameter names, ranges, and scale types. The objective (L2 regression) is stated at line 207.

### D.8 — Unused bibliography entries
**Verdict:** ⚠️ PARTIALLY RESOLVED. Li2023 proxy citation removed. Caesar2018 and Breiman2001 remain in the .bib but are not cited. Harmless — they don't appear in the compiled paper — but the .bib should be cleaned.

### D.9 — No pipeline diagram
**Verdict:** ✅ RESOLVED. Figure 1 (pipeline_overview.pdf) is included at line 155 with a descriptive caption.

### D.10 — Missing units
**Verdict:** ✅ RESOLVED. Table 5 caption states "RMSE and MAE are measured in score points on the 1–7 VLM scale" (lines 244–245).

### D.11 — Process language
**Verdict:** ✅ RESOLVED. No instances of "finalized," "final version of the pilot study," or similar process language remain. One instance of "The revised evidence supports" (line 422) and "The original feature-ablation result remains" (line 360) are revision-aware phrasing rather than process language; these are acceptable in a revision.

### D.12 — Unverifiable ResNet-50 correlations
**Verdict:** ✅ RESOLVED. The paper now reports only "Best CNN" (a single row per affordance) rather than separate per-architecture correlation breakdowns.

---

## SECTION C: New Issues Introduced by the Revision

### N.1 — Minor rounding discrepancy in Best CNN RMSE for L079

Table 5 (line 258) reports L079 Best CNN RMSE as 1.245. The underlying data (experiment1_model_comparison.csv and experiment3_cnn_ablation.csv) gives 1.2437, which rounds to 1.244, not 1.245. This is a single-digit error in the third decimal place and has no impact on any conclusion.

### N.2 — Retrained vs. original model numbers

The paper silently uses retrained-on-train+val results for Model B and Model D (e.g., Model D L059 RMSE = 0.676 in the revised paper vs. 0.666 in the original). This is methodologically correct — retraining on the full non-test data before final evaluation is standard practice — but the switch from the original experiment1 numbers is not explicitly flagged. The differences are small (max ~0.01 RMSE) and the retrained values are internally consistent with the CIs and statistical tests, so this does not undermine any conclusion.

### N.3 — Indicator vocabulary count discrepancy

The original paper stated "755 canonical indicator types." The revision states "624 canonical indicator types, expanded into 1,248 affordance-specific positive/negative binary columns" (lines 150–151). The change from 755 to 624 is unexplained. The per-affordance indicator column counts (288+140+268+280+272 = 1,248) are consistent with the total. This may reflect a different vocabulary consolidation during the revision pipeline, but the discrepancy should be noted.

### N.4 — Revision-aware language

Lines 360 ("The original feature-ablation result remains qualitatively unchanged"), 422 ("The revised evidence supports three claims"), and 433 ("several stronger claims from the original draft are no longer defensible") are meta-commentary on the revision process. While acceptable in a revision letter, these would need to be rewritten for a camera-ready submission. For a course project, this is a very minor issue.

None of these new issues are individually or collectively severe enough to affect the decision.

---

## SECTION D: Decision

**Decision: ACCEPT**

The revision addresses all four [MUST] items substantively: (1) the reframing from affordance prediction to VLM distillation is pervasive and genuine; (2) the mean-prediction and linear regression baselines are added and correctly computed; (3) the Limitations section is four paragraphs of specific, honest self-assessment; (4) the scene-level split is confirmed and documented. All seven [SHOULD] items are also resolved, including the permutation control that produced a genuinely informative mixed result. The paper's claims are now appropriately scoped to what the evidence supports. Every number I verified matches the underlying data. The writing is at a professional standard. A reader encountering this paper without knowledge of the review history would find a competent pilot study that is transparent about its scope and limitations — which is exactly the standard for acceptance.

---

## SECTION F: Remaining Suggestions (Optional)

1. **Clean unused bibliography entries.** Caesar2018 and Breiman2001 remain in references.bib but are not cited. Remove them.

2. **Fix the L079 Best CNN rounding.** Table 5, L079 Best CNN RMSE should be 1.244, not 1.245.

3. **Remove revision-aware language for camera-ready.** Lines 360, 422, and 433 reference "the original draft" or "the revised evidence." Rewrite these as self-contained statements (e.g., "The evidence supports three claims" rather than "The revised evidence supports three claims").

4. **Consider adding a brief qualitative example.** One or two example images showing a high-scoring and low-scoring scene for a single affordance (e.g., L079 Cook), alongside the VLM's indicator checklist and Model D's prediction, would make the pipeline concrete for readers unfamiliar with Hypersim.

5. **Report the number of bootstrap resamples.** The CIs are reported but the number of bootstrap iterations is not stated. Adding "(10,000 resamples)" or similar to the relevant caption would close this small reproducibility gap.
