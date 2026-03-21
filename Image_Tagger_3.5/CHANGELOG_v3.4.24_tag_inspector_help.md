# Image Tagger v3.4.24 — Tag Inspector inline help

This version builds on v3.4.23_tag_inspector_full and adds:

- Inline help to the Tag Inspector drawer in the Supervisor GUI:
  * A HelpCircle icon in the drawer header toggles a help panel.
  * The help panel explains what Tag Inspector shows:
    - science attributes,
    - high-level indices (BN inputs),
    - human validations,
    - how to interpret the snapshot counts and IRR.
- No backend changes; this is a pure UI help enhancement on top of the existing inspector endpoint.

All existing files from v3.4.23 are preserved; this version is strictly additive.
