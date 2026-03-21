# Adversarial Peer Review

**Paper:** "Predicting Environmental Affordances from Indoor Scene Images via Vision-Language Model Knowledge Distillation"
**Venue standard:** NeurIPS / ICML
**Reviewer:** Reviewer 2 (adversarial)

---

## SECTION A: Summary

This paper proposes a two-phase pipeline for predicting room-level affordance scores (Sleep, Cook, Computer Work, Conversation, Yoga/Stretching) from indoor scene images. In Phase 1, a VLM (Qwen2-VL-7B) scores 420 synthetic Hypersim images on each affordance and emits structured indicator checklists. In Phase 2, lightweight student models—CNN baselines (ResNet-18/50), a LightGBM regressor on 310 Mask2Former-derived scene features (Model B), and an indicator-augmented LightGBM (Model D)—are trained to reproduce the VLM's scalar scores. The main finding is that Model D achieves macro RMSE 0.763 versus 0.977 for Model B and ~1.13 for the CNNs, with indicator distillation improving every affordance. SHAP analysis and feature ablation provide interpretability. The paper frames this as a contribution to affordance prediction, but what it actually measures is VLM score approximation.

---

## SECTION B: Strengths

1. **Well-structured experimental design.** The paper runs six clearly delineated experiments (model comparison, hyperparameter sensitivity, CNN ablation, feature ablation, indicator distillation value, SHAP–VLM comparison), each answering a specific question. This makes the paper easy to follow and the claims easy to evaluate.

2. **Honest reporting of non-monotonic results.** Section 4.4 (Experiment 4) candidly notes that adding features is not monotonically beneficial—presence-only features win for L059 and L091, and the full feature set helps mainly on L141. This is a mark of intellectual honesty that many papers lack.

3. **Statistical testing for the key claim.** Experiment 5 reports Wilcoxon signed-rank tests for the Model B vs. Model D comparison, including the one non-significant result (L141, p = 0.0812). This transparency is commendable.

4. **Interpretability as a first-class concern.** The SHAP–VLM rank comparison (Experiment 6) and the feature ablation (Experiment 4) go beyond accuracy reporting and attempt to explain *what* the models learn. The consistently positive Spearman correlations (0.305–0.474) between SHAP rankings and VLM indicator rankings are a genuinely interesting finding.

5. **Numerical accuracy.** I verified every number in the paper against the underlying CSV and JSON files. All reported values are correct to the stated precision (3 decimal places), including macro averages, deltas, and p-values. This is a basic requirement, but one that many papers fail.

6. **Creative problem formulation.** Room-level continuous affordance scoring is a less-explored formulation compared to object-level binary affordance classification. The idea of distilling not just the VLM's scores but also its structured rationale (indicators) is a thoughtful extension of standard knowledge distillation.

---

## SECTION C: Major Weaknesses

### C.1 — The ground truth is a VLM, not human judgment

This is the paper's most fundamental problem. The entire evaluation—every RMSE, every Pearson r, every Wilcoxon test—measures how well lightweight models approximate Qwen2-VL-7B's outputs. The paper never establishes that Qwen2-VL-7B's affordance scores correspond to human affordance judgments. Yet the title says "Predicting Environmental Affordances," the abstract says the models predict affordances, and the Introduction frames the problem as perceiving "actionable possibilities" (Section 1, line 58–65).

This conflation is pervasive. The paper *does not* contain the word "proxy" or any explicit acknowledgment that VLM scores are an unvalidated surrogate for actual affordances. The Discussion (Section 5) refers to "the VLM's internal notion of a workable personal workspace" (line 387) without flagging that this internal notion may be wrong. The Conclusion (Section 6) says "VLM-based affordance scoring can be distilled effectively" (line 502–503), which is true but is a much weaker claim than "affordance prediction works."

**Severity:** This alone could warrant rejection at a top venue. The paper should either (a) validate VLM scores against human ratings on a subset, or (b) reframe the entire contribution as VLM distillation and remove the affordance prediction framing from the title and claims.

### C.2 — Statistical power and sample size

The dataset contains 420 images with an approximately 71/9/20 train/val/test split, yielding 84 test images per affordance. While the Wilcoxon tests for Experiment 5 are appropriate for paired comparisons, the paper reports no confidence intervals or standard errors for any of the primary metrics (RMSE, MAE, Pearson r, Spearman ρ) in Tables 1–3. With n = 84, the sampling variability of Pearson r is non-trivial. For example, the difference between Model B's r = 0.403 and Model D's r = 0.582 on L141 (Table 3) looks substantial, but without a confidence interval or a Fisher z-test, the reader cannot assess whether this is reliable.

More concerning: the Optuna search uses 100 trials with 5-fold CV on ~298 training images per affordance (not per fold—each fold has ~238 training images). With 100 hyperparameter configurations evaluated on 5 folds of ~60 validation images each, there is a meaningful risk of validation-set overfitting, especially for the indicator-augmented model with 310 + ~270 features and only ~238 training samples per fold.

The validation set for CNN early stopping contains only 38 images per affordance. Patience of 5 over 30 epochs on 38 validation images is likely to produce noisy stopping decisions, which may partially explain why CNN performance is unstable across learning rates (Experiment 3).

### C.3 — Synthetic-only evaluation

All 420 images come from Hypersim, a photorealistic but synthetic dataset with perfect geometry, no motion blur, no real-world clutter, and algorithmically generated lighting. The paper makes no mention of this limitation. It does not use the words "synthetic," "domain gap," or "transfer" anywhere in the Discussion or Conclusion. The implicit generalization to real indoor scenes is unsupported.

This matters especially because the Mask2Former features (object presence, counts, spatial relationships) are extracted from a model trained on real images but applied to synthetic renders. Segmentation quality may be systematically different on synthetic data—either better (cleaner scenes) or worse (unusual textures)—and this confound is never discussed.

### C.4 — Novelty assessment

The paper claims three contributions in Section 7 ("Novelty Arguments"):

1. **The distillation pipeline** is a straightforward composition of existing components: Qwen2-VL-7B generates labels, Mask2Former extracts features, LightGBM fits a regressor. No component is novel, and the composition is the obvious thing to try. The pipeline contribution is an *application*, not a methodological advance.

2. **The structured dataset** consists of VLM-generated labels on an existing synthetic image set. This is fundamentally different from a human-annotated dataset. The dataset's value is contingent on the VLM labels being correct (see C.1), which is unestablished. A dataset of machine-generated labels whose quality is unvalidated against humans is of limited value to the community.

3. **The comparative analysis** (Model B vs. Model D) is a single ablation: does adding indicator features help? This is one experimental condition, not a "comparative feature analysis" contribution. The feature ablation (Experiment 4) provides additional insight but is standard practice.

None of these individually meet the novelty bar of a top venue. In combination, they constitute a reasonable pilot study or workshop paper, but the paper's framing in NeurIPS format implies a higher standard than the evidence supports.

### C.5 — Experimental methodology concerns

**a) Split strategy.** The paper states a 298/38/84 image split but does not describe the splitting criterion. It does not say "scene-level split" or "random split." If images from the same Hypersim scene (different camera poses) appear in both train and test, there is data leakage through shared geometry, textures, and objects. This is critical information that is simply missing.

**b) CNN training regime.** Early stopping with patience = 5 on 38 validation images per affordance is likely unreliable. The best_epoch values in experiment3_cnn_ablation.csv range from 3 to 28, suggesting high variance in when training stops. With such a small validation set, early stopping may be selecting for noise.

**c) Optuna hyperparameter ranges.** The paper mentions that "best Model B configurations ranged from 55 to 486 estimators" (Section 4.2, line 263) but does not report the search ranges for any hyperparameter. The reader cannot assess whether the search was reasonable or whether the best configurations are at the boundary of the search space (which would indicate the range was too narrow).

**d) Per-affordance CNN correlations.** Table 3 reports Pearson r and Spearman ρ separately for ResNet-18 and ResNet-50 across all five affordances. However, experiment1_model_comparison.csv contains only a single "CNN" row per affordance (the best architecture). The per-architecture correlation values cannot be verified from the provided result files. This is a reproducibility gap.

### C.6 — Missing baselines and comparisons

The paper compares four model variants against each other but includes no external baselines:

- **No mean-prediction baseline.** A model that predicts the training-set mean VLM score for each affordance would establish the floor. Without this, the reader cannot determine whether an RMSE of 0.763 is good or bad. For context: L091 has mean 3.848 and std 0.999. A mean-prediction baseline would achieve RMSE ≈ std ≈ 1.0 on test. Model B's RMSE of 1.140 on L091 is *worse* than mean prediction—meaning Model B has learned nothing useful for L091. This critical fact is obscured by the paper's focus on Model D.

- **No linear regression baseline.** A simple linear regression on the 310 features would establish whether the non-linearity of LightGBM is actually needed.

- **No comparison to prior affordance prediction methods.** The Related Work cites Ego-Topo but does not compare against it or any other method, even conceptually in terms of task setup or accuracy ranges.

### C.7 — Unfair CNN vs. LightGBM comparison

The paper frames the model comparison as "CNN vs. LightGBM" (Section 4.3, Table 1), but the comparison is confounded by the feature pipeline:

- The CNNs receive 224×224 RGB images and must learn everything from pixels with ~298 training images—a regime where fine-tuning large pretrained models is known to be unreliable.
- LightGBM receives 310 hand-engineered features extracted by Mask2Former, a state-of-the-art panoptic segmentation model pretrained on vastly more data.

This is not a comparison of *architectures*; it is a comparison of *feature representations*. The conclusion "structured features outperform CNNs" (Section 5, line 481–482) is misleading because the structured features encode the output of a model far more powerful than ResNet-18/50 when applied to this task. A fairer comparison would either (a) give the CNN access to segmentation maps as additional input channels, or (b) use CNN-extracted features as input to LightGBM. The paper does not acknowledge this confound.

### C.8 — Indicator feature dimensionality confound

Model D adds 140–288 binary indicator columns (per experiment5_indicator_value.csv) to the 310 structured features. The improvement could reflect either (a) genuine semantic information in the indicators or (b) the benefit of a much richer, sparser feature space that LightGBM can exploit. No control experiment was run—for example, adding the same number of *random* binary features to test whether the improvement is specific to indicator semantics versus dimensionality increase. Without this control, the "indicator distillation provides measurable value" claim (Section 7, paragraph 2) is not fully supported.

### C.9 — SHAP interpretation overreach

Experiment 6 computes Spearman correlations between SHAP feature importance rankings and VLM indicator frequency rankings. There are two problems:

1. **Incommensurability.** SHAP ranks are over *machine features* (e.g., "vertical_spread," "scene_depth_median_m") while VLM ranks are over *natural-language indicator types* (e.g., "large open space," "hardwood flooring"). The paper computes a rank correlation over n_matched_pairs (57–99 per affordance), but the matching procedure is not described. How is "scene_depth_range_m" matched to "large open space"? If the matching is subjective, the correlation is subjective.

2. **Conflation with causality.** The paper says "the student is not learning an arbitrary shortcut unrelated to the teacher's rationale" (line 474–475). SHAP values reflect the model's decision process, not causal relationships. A positive rank correlation between SHAP importance and VLM indicator frequency could arise from both responding to the same data distribution, not from the student learning the teacher's *reasoning*.

### C.10 — Writing and presentation

**a) Abstract accuracy.** The abstract is generally accurate and self-contained, but it says "420 Hypersim images" while the title says "Indoor Scene Images" without qualification that they are synthetic. A reader of the abstract would not know the images are synthetic renders.

**b) Related Work is thin.** Section 2 is three sentences long. It cites six papers and engages substantively with none. It does not discuss: object-level affordance detection (e.g., AffordanceNet, LOCATE), functional scene understanding, VLM-based labeling pipelines, or tabular vs. image model comparisons in low-data regimes. For a venue like NeurIPS, this is insufficient.

**c) Section 7 ("Novelty Arguments (Bonus)").** This section is unusual and reads as defensive rather than letting the work speak for itself. At a top venue, novelty should be evident from the Introduction and Results, not argued in a separate section. The "(Bonus)" label is informal.

**d) Missing Limitations section.** There is no explicit Limitations section, which NeurIPS requires. The Discussion touches on some limitations implicitly but does not address synthetic-only evaluation, unvalidated ground truth, or sample size.

**e) Table formatting.** Tables 1–2 are full-width (\texttt{table*}) which is appropriate, but Table 6 (Experiment 6) has very long text entries in the Top-5 columns that will be difficult to read at NeurIPS single-column width. The SHAP beeswarm plots in Figure 6 span the full page width, which is fine, but the subplot labels may be too small.

**f) Unsupported language.** The abstract says Model D "substantially outperform[s]" (implied by "macro-averaged RMSE of 0.763") without a corresponding statistical test on the macro average. The word "substantially" appears in Section 5, line 483 ("substantially outperform the CNN baselines") without any test comparing LightGBM to CNN. The Wilcoxon tests only cover Model B vs. Model D.

**g) Proxy citation.** The bibliography entry for Li2023 contains the note "(proxy citation for Li et al. 2023 on affordance prediction)" but actually points to a paper on image inpainting. If this citation appears in the compiled PDF, it misrepresents the cited work.

---

## SECTION D: Minor Weaknesses

1. **Inconsistent decimal places.** Table 1 uses 3 decimal places for RMSE/MAE; Table 5 uses 4 decimal places for p-values (0.0005, 0.0000, 0.0036); the abstract rounds to 3. A uniform convention should be adopted.

2. **"0.0000" for p-values.** Table 5 reports p = 0.0000 for L079 and L130. This should be reported as p < 0.0001 rather than implying an exact zero probability.

3. **Affordance codes used without definition in tables.** Tables use "L059," "L079," etc. The mapping is defined in Section 3.1, but table captions should be self-contained enough that a reader scanning tables first can understand them. A brief legend or full names would help.

4. **The split is approximately 71/9/20, not 70/10/20.** The paper doesn't claim 70/10/20, but the 38-image validation set (9% of data) is unusually small and warrants explicit justification.

5. **No error bars on any figure.** Figures 1, 3, 4, and 5 show point estimates without confidence intervals or error bars.

6. **Experiment 2 provides no quantitative results in the paper.** Section 4.2 summarizes the Optuna search qualitatively ("ranged from 55 to 486 estimators") but does not include a table of best hyperparameters. The reader cannot reproduce the models.

7. **No model training details for LightGBM.** The paper does not report: objective function, number of leaves, learning rate, regularization parameters, or other LightGBM hyperparameters for the final models.

8. **Unused bibliography entries.** The .bib file contains entries for Caesar2018, Breiman2001, and Li2023 that do not appear to be cited in the text.

9. **The paper lacks a figure showing the pipeline.** A system diagram showing VLM → indicators/scores → feature extraction → student training would help readers quickly grasp the method.

10. **Missing units.** Table 1 headers say "RMSE" and "MAE" but do not specify units. Since VLM scores are on a 1–7 integer scale, the units are "score points," which should be stated.

11. **"Finalized" language.** The paper uses "finalized pilot dataset" (line 106), "finalized experiments" (line 76), "final version of the pilot study" (line 502). This process language is appropriate for an internal report but not for a venue submission.

12. **ResNet-50 correlation values in Table 3 are not verifiable from provided CSVs.** The experiment1_model_comparison.csv contains only the best CNN per affordance, and experiment3_cnn_ablation.csv does not include correlation metrics. This is a reproducibility concern.

---

## SECTION E: Questions for the Authors

1. **Can you provide evidence that Qwen2-VL-7B affordance scores correlate with human judgments, even on a small subset (e.g., 30–50 images)?** Without this validation, the entire evaluation measures distillation quality, not affordance prediction quality. Even an informal annotation by the authors would meaningfully strengthen the paper.

2. **What is the RMSE of a mean-prediction baseline (predicting the per-affordance training-set mean) on your test set?** This is essential context for interpreting all reported numbers. If Model B's L091 RMSE (1.140) is worse than the mean baseline, that should be stated.

3. **Have you tested whether Model D's improvement is robust to permutation of the indicator feature labels?** Specifically: if you randomly shuffle the indicator column assignments (breaking the semantic mapping), does Model D still outperform Model B? This would distinguish semantic information from dimensionality effects.

4. **What is the splitting strategy—random by image, or by Hypersim scene?** If the same Hypersim scene contributes images to both train and test via different camera poses, the reported metrics may be inflated by data leakage.

5. **How were SHAP features matched to VLM indicators for computing the Spearman correlation in Experiment 6?** The paper says n = 57–99 matched pairs per affordance, but the matching procedure is not described. Was it automated (string matching) or manual?

---

## SECTION F: Recommendation

**Recommendation: Weak Reject**

The paper presents a well-executed pilot study with honest reporting and an interesting problem formulation. However, the fundamental conflation of VLM score approximation with affordance prediction (C.1), the absence of any human validation, the missing baselines (C.6), the synthetic-only evaluation without acknowledgment (C.3), and the unfair architecture comparison (C.7) are collectively too severe for acceptance at a top venue. The experimental execution is competent—the numbers are correct, the statistical tests are present where they matter most—but the claims consistently outreach the evidence. As a course project, this is strong work that demonstrates real methodological care; as a NeurIPS submission, it needs the revisions listed below to be credible.

---

## SECTION G: Prioritized Revision Checklist

1. **[MUST]** Reframe the paper's claims to distinguish VLM distillation from affordance prediction. Either validate VLM scores against human judgments on a subset, or change the title, abstract, and discussion to make explicit that the target is VLM score approximation, not ground-truth affordance prediction. (Affects: title, abstract, Sections 1, 5, 6, 7.)

2. **[MUST]** Add a mean-prediction baseline (per-affordance training-set mean) to Tables 1–3. This is a one-line computation per affordance and is essential context. Report the RMSE, MAE, and correlation (which will be 0) for this baseline.

3. **[MUST]** Add a Limitations section (after Discussion) that explicitly addresses: (a) VLM scores as unvalidated proxy labels, (b) synthetic-only evaluation and domain gap, (c) small sample size, (d) unfair CNN vs. tabular comparison.

4. **[MUST]** State the train/val/test splitting strategy (random vs. scene-level) in Section 3.2. If not scene-level, acknowledge the potential for leakage.

5. **[SHOULD]** Add a linear regression baseline on the 310 features to Table 1. This establishes whether LightGBM's non-linearity is needed.

6. **[SHOULD]** Add bootstrap confidence intervals (95%) for RMSE and Pearson r in Tables 1–3, at least for Model B and Model D. With n = 84, this is computationally trivial.

7. **[SHOULD]** Run a random-indicator control for Experiment 5: add the same number of random binary features to Model B and report the RMSE. This distinguishes indicator semantics from dimensionality effects.

8. **[SHOULD]** Acknowledge the CNN vs. LightGBM comparison confound explicitly in Section 4.3 or Discussion. Note that LightGBM receives Mask2Former features while CNNs receive raw pixels, and that this is a feature-pipeline comparison, not an architecture comparison.

9. **[SHOULD]** Expand Related Work (Section 2) to at least one full column. Engage with: object-level affordance detection methods, functional scene understanding, VLM-as-annotator pipelines, and low-data transfer learning.

10. **[SHOULD]** Report the best hyperparameters for each final LightGBM model (at minimum: n_estimators, learning_rate, num_leaves, max_depth, reg_alpha, reg_lambda) in a supplementary table or appendix. This is necessary for reproducibility.

11. **[SHOULD]** Describe the SHAP-to-VLM matching procedure for Experiment 6 (how machine feature names were aligned with natural-language indicator types). If manual, acknowledge subjectivity.

12. **[NICE]** Remove Section 7 ("Novelty Arguments (Bonus)") and integrate the novelty claims into the Introduction and Conclusion where they belong.

13. **[NICE]** Add a pipeline diagram figure showing the two-phase workflow (VLM annotation → feature extraction → student training → evaluation).

14. **[NICE]** Report p-values < 0.0001 as "< 0.0001" rather than "0.0000" in Table 5. Standardize decimal places across all tables to 3.

15. **[NICE]** Remove process language ("finalized pilot dataset," "final version of the pilot study") and present the work as a self-contained study rather than the endpoint of an iterative process.
