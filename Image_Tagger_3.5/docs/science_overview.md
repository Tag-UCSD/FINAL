# Science Overview (v3.2.39)

This document explains the heuristics-based science pipeline implemented in `backend/science/`
and how its outputs are used by the Explorer, Monitor, and downstream analysis tools.

## 1. Pipeline sketch

1. Images are stored in the database via the `Image` model (`backend/models/assets.py`), with a
   `storage_path` or similar file reference.
2. The science pipeline orchestrator (`backend/science/pipeline.py`) loads pixels into an
   `AnalysisFrame` abstraction.
3. A set of analyzers run over this frame, each responsible for a family of measures:

   - Color and luminance (`backend/science/color.py`)
   - Texture via gray-level co-occurrence matrix (GLCM) (`backend/science/texture.py`)
   - Fractal dimension (`backend/science/fractals.py`)
   - Visual complexity (`backend/science/complexity.py`)
   - Perceptual / depth cues (`backend/science/perception.py`)
   - Social disposition (rule-based) (`backend/science/context_social.py`)
   - Cognitive/restorative disposition (rule-based) (`backend/science/context_cognitive.py`)
   - Summary / composite indices (`backend/science/summary.py`)

4. Each analyzer adds attributes to the frame using a simple API such as:

   ```python
   frame.add_attribute("science.edge_density", value, confidence=0.9)
   ```

5. At the end of the pipeline, `_save_results` (in `pipeline.py`) persists these attributes
   into the `Validation` table (`backend/models/annotation.py`) as science-sourced rows
   (e.g. `source='science_pipeline_v3.2'`).

6. These persisted attributes can be:

   - Visualised and filtered in the Explorer GUI.
   - Used as moderators in the Monitor (Supervisor) GUI.
   - Exported as BN-ready rows via the `/v1/export/bn-snapshot` endpoint.

## 2. Primitive measures

The current v3.2.39 line implements a modest but coherent set of primitives, including:

- **Color primitives** (in `color.py`):

  - Mean luminance and saturation in a perceptually uniform colour space.
  - Warm/cool ratio based on hue ranges.
  - Simple colour entropy over a coarse histogram.

- **Texture / GLCM primitives** (in `texture.py`):

  - GLCM contrast, homogeneity, energy, correlation on a downsampled grayscale channel.

- **Fractal measures** (in `fractals.py`):

  - Global fractal dimension `D` estimated via box-counting over a binarised edge map.

- **Complexity primitives** (in `complexity.py`):

  - Edge density: proportion of edge pixels relative to total image pixels.
  - Organisation ratio: heuristic ratio of structured vs noisy regions, derived from local
    variance and edge continuity.

These primitives are deterministic for a given image and configuration and are intended as
transparent, inspectable building blocks rather than final psychological truths.

## 3. Composite indices and bins

`backend/science/summary.py` combines several primitives into composite indices that are easier
to interpret in teaching and exploratory analysis:

- **Visual richness** (`science.visual_richness`)
  - Combines colour entropy, edge density and texture variation into a 0–1 composite.
- **Organised complexity** (`science.organized_complexity`)
  - Combines fractal dimension and organisation ratio into a 0–1 composite.

Both composites are also discretised into bins for BN and UX:

- `science.visual_richness_bin`
- `science.organized_complexity_bin`

The binning rule is:

- `0` → low
- `1` → mid
- `2` → high

The BN export layer (`backend/api/v1_bn_export.py`) maps these numeric codes to string labels
(`"low"`, `"mid"`, `"high"`) so that downstream BN tools work purely with categorical
values.

## 4. Index catalog

`backend/science/index_catalog.py` defines a canonical index catalog. It exposes:

- `INDEX_CATALOG`: a dict mapping attribute keys to metadata (label, description, type, bins).
- `get_candidate_bn_keys()`: returns attribute keys recommended as BN inputs.
- `get_index_metadata()`: returns the full catalog for metadata endpoints.

The catalog keeps the BN export layer, Explorer, Monitor and notebooks in sync about which
indices exist and how they should be interpreted.

## 5. Relation to Explorer and Monitor

- The **Explorer** GUI can show science attributes as columns and allow filtering/sorting
  by indices and bins. In v3.2.39 it primarily uses attribute keys; a future iteration may
  fetch labels/descriptions from the index catalog for richer tooltips.

- The **Monitor** GUI can use science indices (especially composites and bins) as moderators
  in its analyses of tagging velocity and inter-rater reliability. For example, you can
  ask whether difficult images (high organised complexity) systematically slow taggers or
  increase disagreement.

## 6. BN export

The BN export endpoint (`/v1/export/bn-snapshot`) produces a list of `BNRow` objects (`backend/schemas/bn_export.py`):

- `image_id`: primary key of the image.
- `source`: software version string (e.g. `"image_tagger_v3.2.39"`).
- `indices`: a dict of {attribute_key → float or null} for continuous indices.
- `bins`: a dict of {bin_attribute_key → label or null} for categorical bins.

The underlying implementation reads from the `Validation` table, restricted to science
pipeline sources, and uses the numeric→label mapping described above.

## 7. Extending the pipeline

To add a new index:

1. Implement an analyzer in `backend/science/` with a function such as
   `analyze(frame: AnalysisFrame)` that calls `frame.add_attribute(...)`.
2. Add a corresponding entry to `backend/science/index_catalog.py` (with optional bins).
3. Register the analyzer in `backend/science/pipeline.py` so it runs for each image.
4. If the index should be part of BN export, tag it as a `"candidate_bn_input"` in the
   catalog and, if needed, define a binning strategy.

The goal of this v3.2.39 line is to be modest, explicit and inspectable: science code is
short enough to read in a seminar, and all heuristics are visible and adjustable.
