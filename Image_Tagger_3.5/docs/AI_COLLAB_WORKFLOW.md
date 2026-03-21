# AI Collaboration Workflow (Image Tagger v3)

This repository is often developed in collaboration with large language models
(LLMs) such as ChatGPT, Claude, and Gemini. To keep the history sane and the
system reproducible, we follow these conventions:

## 1. Deliverables from AI

When asking an AI to modify this repo, do not request patches or partial diffs.
Instead, always request:

1. A full ZIP of the repository at the new version.
2. A single concatenated TXT that:
   - Contains all repo files (excluding `__pycache__`),
   - Uses clear file markers (e.g. `----- FILE PATH: ...`),
   - Includes a small `deconcat.py` helper at the top that can reconstruct the
     directory tree from the TXT alone.

These artifacts should be versioned, e.g.:

- `Image_Tagger_v3.2.xx_*.zip`
- `Image_Tagger_v3.2.xx_*.txt`

## 2. Non-destructive updates

AI tools must not delete files. If something needs to be replaced:

- Archive the previous version under `archive/<version_phase>/...`
- Write the new version in-place.

## 3. Guardian + Governance

Any AI-driven code change should respect:

- `v3_governance.yml` for:
  - `protected_scopes`
  - `critical_files`
  - `constraints`
- `scripts/guardian.py` for:
  - `freeze` → updating `governance.lock`
  - `verify` → blocking installs when invariants are broken

## 4. "Vibe Coding" preferences (David)

The primary human collaborator prefers:

- Single-shot phases: each phase should bundle all code edits + packaging into
  one run (no long back-and-forth of tiny patches).
- Clear version bumps (v3.2.15 → v3.2.16, etc.).
- Explicit stats with every artifact:
  - Total file count,
  - Number of `# STUB:` markers,
  - ZIP + TXT sizes and SHA256 hashes.

When prompting an AI, you can paraphrase this section as:

> "Please apply all requested changes in a single run, then give me a full ZIP
> and a concatenated TXT with a deconcat script, plus file counts and SHA256
> hashes. Do not delete any files; archive old versions under `archive/`."