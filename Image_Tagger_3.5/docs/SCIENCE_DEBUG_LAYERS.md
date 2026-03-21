# Science Debug Layers: Edges, Overlays, and Parameters

This document explains the "debug view" features in the Explorer app and how
they connect to the underlying computer vision pipeline.

The goal is to make the **science visible** to students: when the pipeline
computes complexity or edge-based measures, they should be able to see which
edges and structures the algorithm is operating on.

## 1. Where to find the debug controls

In the **Explorer** GUI, each image card has a **Debug** toggle in the toolbar
above the image:

- Click once: `Debug: Edges`
- Click twice: `Debug: Overlay`
- Click a third time: `Debug: Off`

When `Debug` is set to `Edges` or `Overlay`, a small control panel appears
with sliders for:

- **Edges**: low and high thresholds
- **Overlay**: opacity (only when in overlay mode)

These controls are lightweight and safe to adjust during exploration.

## 2. What the backend is doing

The debug views are powered by the `v1_debug` API in the backend:

- Endpoint: `GET /api/v1/debug/images/{image_id}/edges`
- Query parameters:
  - `t1`: low Canny threshold (default 50)
  - `t2`: high Canny threshold (default 150)
  - `l2`: whether to use the L2 gradient option (default `true`)

Under the hood, the backend:

1. Resolves the image path based on the `Image.storage_path` field and
   `IMAGE_STORAGE_ROOT`.
2. Loads the image and converts it to grayscale.
3. Runs OpenCV's Canny edge detector:

   ```python
   edges = cv2.Canny(gray, t1, t2, L2gradient=l2)
   ```

4. Converts the edge map to a PNG image and returns it as `image/png`.

### 2.1 Cache behaviour

Because Canny is CPU-intensive for repeated requests, the debug endpoint keeps
a small **on-disk cache** keyed by:

- `image_id`
- `t1`
- `t2`
- `l2`

If a PNG edge map already exists for that combination, it is served directly.
If not, it is computed once and then cached.

This makes classroom usage (many students experimenting with the same images)
fast and predictable.

## 3. How the frontend uses these parameters

In the Explorer React app, the debug state is:

```js
const [debugMode, setDebugMode] = useState('none');    // 'none' | 'edges' | 'overlay'
const [overlayOpacity, setOverlayOpacity] = useState(0.5);
const [edgeThresholds, setEdgeThresholds] = useState({ low: 50, high: 150 });
```

When `debugMode` is not `'none'`, the app:

- Builds a URL of the form:

  ```text
  /api/v1/debug/images/{image_id}/edges?t1={low}&t2={high}
  ```

- Either:
  - uses that URL directly as the `<img>` source in **Edges** mode, or
  - draws the edges as a second `<img>` on top of the original image in
    **Overlay** mode with `opacity = overlayOpacity`.

The sliders in the UI update `edgeThresholds.low`, `edgeThresholds.high`, and
`overlayOpacity`, so you can see in real time how the parameters affect the
edge map.

## 4. How to use this pedagogically

Some example classroom exercises:

### 4.1 Parameter sensitivity

- Choose a single image (e.g., a cluttered interior).
- Ask students to vary `t1` and `t2` and observe:
  - When do edges become too dense to be meaningful?
  - When do edges become too sparse (missing important structure)?
- Discuss how threshold choice influences downstream measures of
  **visual complexity**.

### 4.2 Overlay vs. full replacement

- In **Edges** mode, only the edges are visible.
- In **Overlay** mode, edges are drawn on top of the original image with
  adjustable opacity.

Ask students which mode makes it easier to understand *why* the algorithm is
detecting certain edges.

### 4.3 Compare different image types

- Low clutter vs high clutter.
- High contrast vs low contrast.
- Natural vs highly geometric spaces.

How do edge maps differ across these, and what does that imply for complexity
and legibility?

### 4.4 Linking to BN factors

When complexity is high (based on edges), how might this influence BN nodes
related to:

- cognitive load,
- preference,
- legibility?

Encourage students to think about which types of edges and patterns might
matter most for **human** perception, not just the algorithm.

## 5. Interpreting Canny thresholds

A few guiding intuitions you can give students:

- The **low threshold (`t1`)** controls how easy it is for a gradient to
  start an edge. Lower values → more edges, including weak gradients.
- The **high threshold (`t2`)** controls how strong a gradient must be to be
  accepted as a definite edge. Higher values → only the strongest contrast
  boundaries survive.
- The **ratio between `t1` and `t2`** often matters more than absolute values.

In architectural images:

- Very low thresholds can produce noisy edge maps where textures or sensor
  noise dominate.
- Very high thresholds can drop important structural lines (e.g., corners,
  door frames) and make the space look simpler than it really is.

Part of the scientific judgement is choosing a regime where the edge map
captures **structurally meaningful** lines without drowning in noise.

## 6. Takeaways

- The debug layers are there to make the pipeline **transparent**, not just
  pretty.
- Students should use them to connect:
  - raw pixels → edges → quantitative measures → BN factors → psychological
    interpretations.
- Experimenting with the thresholds and overlay is encouraged; it is a safe,
  reversible way to build intuition about the relationship between visual
  structure and computed complexity.
