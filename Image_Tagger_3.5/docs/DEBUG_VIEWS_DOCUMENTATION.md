# Debug Views Documentation

## Overview

The Explorer application includes **debug visualization modes** that allow users to see how the image analysis algorithms "see" the images. These views are intended for teaching and debugging purposes, helping users understand the underlying computer vision techniques.

This document covers two debug modes:
1. **Debug: Edges** - Canny edge detection visualization
2. **Debug: Complexity** - Regionalized edge density heatmap

---

## 1. Debug: Edges (Canny Edge Detection)

### What It Is

The **Debug: Edges** mode displays a Canny edge detection visualization of images. Instead of showing the original photograph, it shows a black-and-white image where white pixels represent detected edges (boundaries between regions) and black pixels represent smooth areas.

### What It Does

This mode helps users understand:
- Where the algorithm detects boundaries and transitions in the image
- How edge detection forms the foundation for complexity measurement
- How different threshold settings affect edge sensitivity

### How It Works - The Algorithm

The Canny edge detection algorithm is a multi-stage process:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CANNY EDGE DETECTION PIPELINE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INPUT: Color image (RGB)                                        │
│                    ↓                                                │
│  2. GRAYSCALE CONVERSION                                            │
│     • Convert RGB to single-channel grayscale                       │
│     • gray = 0.299*R + 0.587*G + 0.114*B                           │
│                    ↓                                                │
│  3. CANNY EDGE DETECTOR (cv2.Canny)                                │
│     • Gaussian blur to reduce noise                                 │
│     • Compute intensity gradients (Sobel operators)                 │
│     • Non-maximum suppression (thin edges)                          │
│     • Hysteresis thresholding with t1 (low) and t2 (high)          │
│                    ↓                                                │
│  4. OUTPUT: Binary edge map (white = edge, black = no edge)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `t1` (Low threshold) | 50 | Pixels with gradient below this are discarded |
| `t2` (High threshold) | 150 | Pixels with gradient above this are definite edges |
| `L2gradient` | True | Use L2 norm for gradient magnitude (more accurate) |

Pixels between t1 and t2 are kept only if connected to a definite edge.

### Step-Through: Input to Output

**Step 1: User Interaction**
```
User clicks "Debug: None" button repeatedly until it shows "Debug: Edges"
```

**Step 2: Frontend Request**
```
For each image displayed, the frontend changes the image src from:
  img.url (e.g., "https://example.com/kitchen.jpg")
to:
  /api/v1/debug/images/{image_id}/edges?t1=50&t2=150
```

**Step 3: Backend Processing**
```python
# 1. Look up image in database by ID
image = db.query(Image).filter(Image.id == image_id).first()

# 2. Load image from URL or local path
img_bgr = _load_image_from_url_or_path(image.storage_path)
# Example: Downloads from "https://example.com/kitchen.jpg"

# 3. Convert to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 4. Apply Canny edge detection
edges = cv2.Canny(gray, t1=50, t2=150, L2gradient=True)

# 5. Encode as PNG and return
_, buf = cv2.imencode(".png", edges)
return Response(content=buf.tobytes(), media_type="image/png")
```

**Step 4: Output Display**
```
The browser displays a black and white edge map:
- WHITE pixels = detected edges (boundaries, lines, textures)
- BLACK pixels = smooth regions (walls, sky, solid surfaces)
```

### GUI Interaction

| Action | Result |
|--------|--------|
| Click "Debug: None" → "Debug: Edges" | Switches all images to edge view |
| Adjust "Low" slider (0-255) | Changes t1 threshold - lower = more edges |
| Adjust "High" slider (0-255) | Changes t2 threshold - higher = fewer edges |
| Click debug button again | Cycles to next mode (Overlay → Depth → Complexity → None) |

---

## 2. Debug: Complexity (Regionalized Edge Density Heatmap)

### What It Is

The **Debug: Complexity** mode displays a color-coded heatmap overlaid on the original image, showing how visually complex different regions of the image are. "Complexity" here is measured as **edge density** - the proportion of edge pixels in each region.

### What It Does

This mode helps users understand:
- Which areas of an image are visually "busy" (high edge density)
- Which areas are visually "calm" or smooth (low edge density)
- How complexity varies spatially across the image
- The relationship between edges and perceived visual complexity

### How It Works - The Algorithm

The complexity heatmap uses a **sliding window approach** to compute local edge density:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   COMPLEXITY HEATMAP PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INPUT: Color image (RGB)                                        │
│                    ↓                                                │
│  2. GRAYSCALE CONVERSION                                            │
│     • gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                 │
│                    ↓                                                │
│  3. SLIDING WINDOW ANALYSIS                                         │
│     • For each patch (64×64 pixels, stride 32):                    │
│       ┌───────────────────────────────────────┐                    │
│       │  a. Extract patch from grayscale      │                    │
│       │  b. Apply Canny edge detection        │                    │
│       │  c. Count edge pixels (white pixels)  │                    │
│       │  d. Compute: edge_pixels/total_pixels │                    │
│       └───────────────────────────────────────┘                    │
│                    ↓                                                │
│  4. RESIZE HEATMAP                                                  │
│     • Upscale complexity_map to original image size                │
│     • Uses bilinear interpolation for smooth gradients             │
│                    ↓                                                │
│  5. APPLY COLORMAP                                                  │
│     • Map 0.0-1.0 values to HOT colormap:                          │
│       Black → Dark Red → Red → Orange → Yellow → White             │
│                    ↓                                                │
│  6. BLEND WITH ORIGINAL                                             │
│     • 50% original image + 50% heatmap                             │
│                    ↓                                                │
│  7. OUTPUT: Blended visualization PNG                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**The Core Formula:**
```
complexity_score = edge_pixels / total_pixels
```

Where:
- `edge_pixels` = count of white pixels after Canny edge detection
- `total_pixels` = patch_size × patch_size (e.g., 64 × 64 = 4,096)

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 64 | Size of each analysis region (64×64 pixels) |
| `stride` | 32 | Step between regions (32px = 50% overlap) |
| `t1` (canny_low) | 50 | Canny low threshold |
| `t2` (canny_high) | 150 | Canny high threshold |

### Step-Through: Input to Output

**Step 1: User Interaction**
```
User clicks debug button until it shows "Debug: Complexity"
```

**Step 2: Frontend Request**
```
For each image displayed, the frontend changes the image src to:
  /api/v1/debug/images/{image_id}/complexity?t1=50&t2=150
```

**Step 3: Backend Processing (Detailed)**

```python
# 1. Look up image and load it
img_bgr = _load_image_from_url_or_path(storage_path)
# Result: NumPy array, shape (1280, 1920, 3), dtype uint8

# 2. Convert to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# Result: NumPy array, shape (1280, 1920), dtype uint8

# 3. Calculate output dimensions for the complexity map
h, w = gray.shape  # h=1280, w=1920
patch_size = 64
stride = 32
out_h = (h - patch_size) // stride + 1  # (1280-64)/32+1 = 39
out_w = (w - patch_size) // stride + 1  # (1920-64)/32+1 = 59

# 4. Create empty complexity map
complexity_map = np.zeros((39, 59), dtype=np.float32)

# 5. Sliding window loop
for i in range(39):      # rows
    for j in range(59):  # columns
        y_start = i * 32  # e.g., 0, 32, 64, 96, ...
        x_start = j * 32
        
        # Extract 64×64 patch
        patch = gray[y_start:y_start+64, x_start:x_start+64]
        
        # Apply Canny edge detection to this patch
        edges = cv2.Canny(patch, 50, 150)
        
        # Count edge pixels and compute density
        edge_pixels = np.count_nonzero(edges)  # e.g., 512
        total_pixels = 64 * 64                  # = 4096
        complexity_map[i, j] = 512 / 4096       # = 0.125

# 6. Resize heatmap to original image size
heatmap_resized = cv2.resize(complexity_map, (1920, 1280))
# Result: Shape (1280, 1920) with smooth interpolated values

# 7. Normalize to 0-255 and apply colormap
heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)
# Result: BGR image with HOT colormap applied

# 8. Blend with original (50% each)
blended = cv2.addWeighted(img_bgr, 0.5, heatmap_colored, 0.5, 0)
# Result: Original image with semi-transparent heatmap overlay

# 9. Encode and return
_, buf = cv2.imencode(".png", blended)
return Response(content=buf.tobytes(), media_type="image/png")
```

**Step 4: Output Display**
```
The browser displays the original image with a colored overlay:

COLOR LEGEND (COLORMAP_HOT):
┌────────────────────────────────────────────────────────────┐
│  Black/Dark → Low complexity (smooth walls, sky, floors)  │
│  Dark Red   → Slight complexity                           │
│  Red        → Moderate complexity                         │
│  Orange     → High complexity                             │
│  Yellow     → Very high complexity (detailed textures)    │
│  White      → Maximum complexity (dense edges everywhere) │
└────────────────────────────────────────────────────────────┘
```

### GUI Interaction

| Action | Result |
|--------|--------|
| Click to "Debug: Complexity" | Shows heatmap overlay on all images |
| Adjust "Low" slider | Changes Canny t1 - lower = detects more subtle edges |
| Adjust "High" slider | Changes Canny t2 - affects edge sensitivity |
| Hover over image | See which regions are high/low complexity |

### Interpreting the Results

**High Complexity Regions (Red/Yellow):**
- Detailed textures (brick walls, foliage, patterned fabrics)
- Areas with many objects or furniture
- Regions with high contrast or intricate details

**Low Complexity Regions (Black/Dark Red):**
- Plain walls, ceilings, floors
- Sky or solid-colored backgrounds
- Smooth surfaces with minimal detail

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FRONTEND (React)                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  App.jsx                                                     │   │
│  │  • debugMode state: 'none' | 'edges' | 'complexity' | ...   │   │
│  │  • edgeThresholds state: { low: 50, high: 150 }             │   │
│  │  • Image src switches based on debugMode                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓ HTTP GET                             │
│  BACKEND API (FastAPI)                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  v1_debug.py                                                 │   │
│  │  • GET /api/v1/debug/images/{id}/edges?t1=&t2=              │   │
│  │  • GET /api/v1/debug/images/{id}/complexity?t1=&t2=         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  IMAGE PROCESSING (OpenCV)                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  _compute_edge_map_bytes()      → Canny edge detection      │   │
│  │  _compute_complexity_heatmap_bytes() → Sliding window +     │   │
│  │                                        edge density + HOT    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  CACHING (Filesystem)                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  backend/data/debug_edges/      → Cached edge PNGs          │   │
│  │  backend/data/debug_complexity/ → Cached heatmap PNGs       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Locations

| Component | File Path |
|-----------|-----------|
| Frontend UI | `frontend/apps/explorer/src/App.jsx` |
| Backend API | `backend/api/v1_debug.py` |
| Demo Script | `complexity_regions_demo.py` |
| Edge Cache | `backend/data/debug_edges/` |
| Complexity Cache | `backend/data/debug_complexity/` |

---

## API Reference

### GET `/api/v1/debug/images/{image_id}/edges`

Returns a PNG edge map for the specified image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_id` | int | required | Database ID of the image |
| `t1` | int | 50 | Canny low threshold (0-255) |
| `t2` | int | 150 | Canny high threshold (0-255) |
| `l2` | bool | true | Use L2 gradient norm |

**Response:** `image/png`

### GET `/api/v1/debug/images/{image_id}/complexity`

Returns a PNG complexity heatmap for the specified image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_id` | int | required | Database ID of the image |
| `patch_size` | int | 64 | Size of analysis regions |
| `stride` | int | 32 | Step between regions |
| `t1` | int | 50 | Canny low threshold |
| `t2` | int | 150 | Canny high threshold |

**Response:** `image/png`

---

## Dependencies

- **OpenCV (cv2):** Core image processing library
- **NumPy:** Array operations and numerical computing
- **Requests:** Downloading images from URLs
- **FastAPI:** Web framework for API endpoints

