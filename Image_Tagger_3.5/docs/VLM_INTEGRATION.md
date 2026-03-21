# VLM Integration Guide (v3.4.0)

Image Tagger v3.4.0 introduces a unified Visual Language Model (VLM) service –
"the Brain" – that can plug in different multimodal providers while keeping the
science pipeline code stable.

## 1. Supported providers

The backend currently knows about four logical providers:

- **Gemini** (Google): e.g. `gemini-1.5-flash`, `gemini-1.5-pro`.
- **OpenAI**: e.g. `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`.
- **Anthropic**: e.g. `claude-3.5-sonnet`.
- **Stub**: local neutral placeholders; no network calls.

The Admin Cockpit exposes a **VLM Engine** card where you can:

- See which API keys are visible inside the container.
- Choose a preferred provider (`auto`, `gemini`, `openai`, `anthropic`, `stub`).
- Test the current configuration on a specific image ID.

## 2. Configuration: API keys

VLM providers are activated via standard environment variables when you start
the backend container:

- Gemini: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`

If none of these are set, the system falls back to the `StubEngine`. The
**VLM Engine** panel will show that only the stub is available and cognitive
attributes will be filled with neutral 0.5 scores.

## 3. Configuration: Provider preference

At runtime, the preferred provider is resolved in this order:

1. The provider saved from the Admin Cockpit (`VLM Engine` card).
2. The `VLM_PROVIDER` environment variable, if set.
3. Automatic detection based on available keys, with priority:

   `Gemini → OpenAI → Anthropic → Stub`.

The Admin Cockpit writes a small JSON file at:

- `backend/data/vlm_config.json`

This file is safe to commit if you want a default in a lab deployment, or you
can keep it in a local volume in production.

## 4. How the science pipeline uses the VLM

The `SciencePipeline` constructs an `AnalysisFrame` for each image. The
`CognitiveStateAnalyzer` then calls the unified VLM service:

- Module: `backend/science/context/cognitive.py`
- Service: `backend/services/vlm.py`

The VLM is asked to rate the scene on five dimensions derived from
environmental psychology (Kaplan & Kaplan + restoration):

- `coherence`
- `complexity`
- `legibility`
- `mystery`
- `restoration`

Scores are expected in `[0.0, 1.0]`. They are stored as attributes:

- `cognitive.coherence`
- `cognitive.complexity`
- `cognitive.legibility`
- `cognitive.mystery`
- `cognitive.restoration`

If the engine is a stub (no keys configured), the analyzer writes 0.5 for all
five attributes with confidence 0.0 so downstream tools can still assume the
keys exist.

## 5. Cost and safety notes

- VLM calls are network-bound and relatively expensive; start by testing on a
  single image from the Admin Cockpit before running large batches.
- For student or classroom use, you can run in **Stub** mode (no keys) and
  still exercise the full pipeline shape.
- When providing real keys, prefer the most cost-effective model (e.g.
  Gemini 1.5 Flash or OpenAI gpt-4o-mini) for bulk tagging; reserve more
  expensive models for research-grade runs.

## 6. Quick sanity test

1. Ensure DB is seeded and at least one image exists.
2. In the Admin Cockpit → **VLM Engine**, set a provider and save.
3. Enter a small image ID (e.g. `1`) and click **Test VLM**.
4. Confirm that:
   - `engine` reports the expected backend (Gemini / OpenAI / Anthropic / Stub).
   - The response JSON includes the five keys above.

If that works, you can now treat the cognitive metrics as part of the standard
attribute set for export, BN construction, and dashboards.


## 7. Affective / experiential dimensions

In addition to the five core cognitive/environmental metrics, the VLM prompt
now asks for five affective tone dimensions:

- `cozy`      – how cozy / snug / intimate the space feels
- `welcoming` – how welcoming / socially inviting it feels
- `tranquil`  – how calm / tranquil it feels
- `scary`     – how scary / threatening it feels
- `jarring`   – how visually or affectively jarring it feels

These are written into the attribute store as:

- `affect.cozy`
- `affect.welcoming`
- `affect.tranquil`
- `affect.scary`
- `affect.jarring`

In stub mode they are set to 0.5 with confidence 0.0; with a live VLM they are
clamped to [0.0, 1.0] with confidence 0.9 by default.
