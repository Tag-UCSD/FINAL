# Image Tagger v3.4.27 — Priority GUI help (Monitor + Admin)

This version builds on v3.4.26_governance_todo and implements the highest-priority role-based help strips:

- Supervisor / Monitor app:
  * Added an inline "How to use the Supervisor dashboard" help box beneath the controls row,
    explaining how supervisors should interpret throughput, IRR, and error metrics,
    and when to drill into Tag Inspector vs escalate to engineering.

- Admin app:
  * VLMConfigPanel: Added a "VLM configuration: handle with care" help strip under the VLM Engine header,
    clarifying that provider/API settings are global and should be smoke-tested with a known image before use.
  * Training Export card: Added a "What this export is for" help strip explaining how the JSON export maps
    to image/case rows, how it is used for BN/regression/ML training, and why canon/schema versioning matters.

- Governance:
  * Updated `governance/TODO_running_list.md` with a status section for v3.4.27,
    marking which GUI help tasks are now implemented and which remain pending (schema/index catalog UI, pipeline health panel).

All existing files from v3.4.26 are preserved; this version is strictly additive.
