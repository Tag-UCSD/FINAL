# BN Naming Guide (v3.4.36)

This guide explains the naming conventions for Bayesian Network nodes
and variables used in the Image Tagger science stack.

## Principles

- **One construct → one canonical name.**
  Use a single, stable identifier for each psychological or physical
  construct (e.g., `RESTORATIVENESS_H1`, `VISUAL_COMPLEXITY_L1`).
- **No whitespace.**
  Use `UPPER_SNAKE_CASE` with underscores instead of spaces.
- **Explicit modality / level tags.**
  When relevant, include a short suffix indicating modality or level,
  e.g. `STRESS_SUBJECTIVE`, `STRESS_PHYSIO`, `RESTORATIVENESS_H1`.
- **Match BN names to docs.**
  Each node name should be defined in the BN documentation or glossary
  with a short description, measurement notes, and key references.

## How to use this in practice

- When adding a new node to a BN config, choose an identifier that:
  - is concise but descriptive,
  - has no spaces or punctuation other than `_`,
  - matches the construct name used in papers and docs.
- Run `python -m backend.science.bn_naming_guard` locally to inspect
  node names and spot obvious problems.
- When in doubt, prefer *clarity* over brevity: a slightly longer,
  unambiguous name is better than a short, cryptic one.

The goal is to keep the BN layer readable and scientifically auditable
for students, TAs, and external collaborators.
