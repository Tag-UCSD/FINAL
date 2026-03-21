# How to Read Ruthless Audit Reports

The Ruthless audit prompts produce multi-panel reports (e.g., systems
architect, governance officer, chief scientist). This file explains how
to use those reports when working with the Image Tagger repository.

## Typical Sections

- **Executive Verdict (GO / NO-GO / CONDITIONAL GO)**  
  High-level readiness judgement for teaching, research, or deployment.
- **Kill List / Blockers**  
  Items that must be fixed before the system is considered runnable.
- **High-Priority Warnings**  
  Issues that do not break the build but affect scientific validity,
  governance, or student experience.
- **Role-Specific Notes**  
  Recommendations from different perspectives (DevOps, UI/HITL, etc.).

## How to act on a report

1. Start with the **Kill List** and confirm that all blockers are fixed.
2. Treat high-priority warnings as the next sprint planning input.
3. Capture concrete actions in CHANGELOG and `reports/` for traceability.
4. When a new version is prepared, re-run the Ruthless prompts and store
   the new report alongside the old one to show progress over time.

For v3.4.36, the main themes are:
- Hardening the Admin upload pipeline.
- Improving scientific naming hygiene for BN and restorativeness models.
- Keeping GO/NO-GO checks transparent and reproducible for students.
