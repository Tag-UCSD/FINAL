# Governance & Guardian Guide (v3.2.39)

This document explains how repository governance is enforced in the v3 line and how to work
with the `Guardian` tool without getting stuck.

## 1. Governance goals

The v3 line is designed to be:

- Inspectable and teachable.
- Resistant to unintentional drift and file loss.
- Friendly to multi-agent AI development (multiple AIs + humans editing the repo).

Key principles:

1. **No silent deletions**  
   Once a file path has existed in a release, it should not disappear in later versions.
   If code is replaced, the old version is moved under `archive/` with a version tag.

2. **Minimal viable stubs**  
   Active modules should not contain `...` placeholders or pseudo-code. If a feature is not
   yet implemented, it should be clearly marked with comments, but the module must still
   import and run without crashing.

3. **Reproducible releases**  
   Each release should be shippable as:
   - A ZIP of the full tree, and
   - A single concatenated TXT with a `deconcat.py` header so the tree can be reconstructed.

## 2. v3_governance.yml

The file `v3_governance.yml` describes:

- **protected_scopes** – directories and patterns that are considered critical
  (for example `backend/science/`, `backend/api/`, `deploy/`, `scripts/guardian.py`).
- **critical_files** – specific files that must not be deleted or modified lightly
  (for example main app entrypoints and governance config).

You can read this YAML file to see which areas of the repo are most tightly controlled.

## 3. Guardian commands

The `scripts/guardian.py` tool provides two main operations:

- `freeze` (or `snapshot`) – capture the current state of protected files into a lock file.
- `verify` – compare the current tree against the lock file and report differences.

Typical workflow:

```bash
# First-time setup after a clean release
python scripts/guardian.py freeze

# Before proposing a new release
python scripts/guardian.py verify
```

If `verify` reports drift, you can either:

- Fix the changes (for example restore accidentally deleted files), or
- Intentionally update the baseline by running `freeze` again *after review*.

## 4. Using Guardian with install.sh and CI

- The `install.sh` script may invoke Guardian to detect drift in a local checkout.
- The CI workflow (`.github/workflows/ci_v3.yml`) is expected to run:

  ```bash
  python scripts/guardian.py verify
  ```

When Guardian fails in CI:

- Treat it as a signal that the repo's critical surfaces have changed.
- Review the diffs carefully before updating the baseline (`freeze`).

## 5. Guidelines for students

If you are a student or collaborator, please:

- **Do**:
  - Add new modules, scripts and docs as needed.
  - Modify existing files to improve science, UX or tests.
  - Use `archive/` to keep old versions when making major rewrites.

- **Do not**:
  - Delete files that shipped in a prior version without archiving them.
  - Remove governance-related files (for example `v3_governance.yml`, `scripts/guardian.py`).
  - Disable Guardian checks in CI without discussing it with the project lead.

When in doubt, ask a TA or project lead before modifying anything under a protected scope.

## 6. Drift Shield in practice

The intent is not to block experimentation but to make it explicit when the public surface
of the system changes. Guardian helps answer questions such as:

- “Did we lose any science module files between versions?”
- “Have API contracts changed without updating tests and docs?”

By respecting these governance rules, teams can safely iterate on the system while keeping
its history and public surface stable.


## Guardian in everyday work

A few common scenarios and what to do:

1. **You have made local changes and want to cut a new release**

   - Run `scripts/guardian.py verify` to confirm that you have not broken
     any governance rules (e.g., file deletions in protected scopes).
   - If you are satisfied with the state, bump `VERSION` and the visible
     version strings in the README and backend banner.
   - Run `scripts/guardian.py freeze` to mint a new `governance.lock`
     baseline for this release.

2. **Guardian verification fails after some edits**

   - Read the error message; it will typically say which file or scope
     violated the rules.
   - If you removed a protected file by mistake, restore it from version
     control or from the last release ZIP / concatenated TXT.
   - Re-run `scripts/guardian.py verify` until it passes.

3. **You are collaborating with an AI assistant**

   - Instruct the assistant not to delete files and to keep historical
     directories under `archive/` rather than removing them.
   - Ask it to provide both a ZIP and a concatenated TXT with a `deconcat.py`
     script so that the full repository can be reconstructed elsewhere.
   - Use `scripts/guardian.py verify` on the result before merging it into
     your main branch or release line.

Guardian is deliberately conservative: if it complains, treat that as a
signal to inspect the change rather than as an obstacle.
