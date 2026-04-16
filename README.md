# minicv

A lightweight image-processing library that emulates a well-defined subset of OpenCV — built entirely from scratch using **NumPy, Matplotlib, Pandas, and the Python standard library**. No OpenCV or Pillow required for core operations.

---

## Package Layout

```
minicv/
├── __init__.py
├── io/
│   ├── __init__.py
│   ├── readwrite.py       # read_image, write_image
│   └── color.py           # rgb_to_gray, gray_to_rgb
├── utils/
│   ├── __init__.py
│   ├── validation.py      # validate_image, validate_kernel
│   ├── dtype.py           # to_float64, to_uint8
│   ├── normalize.py       # normalize (minmax / zscore / fixed)
│   ├── clip.py            # clip_pixels
│   ├── padding.py         # pad_image (reflect / constant / replicate)
│   └── convolution.py     # convolve2d, spatial_filter
├── filtering/
│   ├── __init__.py
│   ├── blur.py            # mean_filter, gaussian_kernel, gaussian_filter, median_filter
│   ├── threshold.py       # threshold_global, threshold_otsu, threshold_adaptive
│   ├── edges.py           # sobel_gradients, canny
│   ├── bitplane.py        # bit_plane_slice
│   ├── histogram.py       # histogram, histogram_equalization
│   └── segmentation.py    # kmeans_segment
├── transforms/
│   ├── __init__.py
│   └── geometric.py       # resize, rotate, translate
├── features/
│   ├── __init__.py
│   ├── global_descriptors.py    # color_histogram, pixel_statistics
│   └── gradient_descriptors.py  # hog_descriptor, lbp_descriptor
└── drawing/
    ├── __init__.py
    ├── primitives.py      # draw_point, draw_line, draw_rectangle, draw_polygon
    └── text.py            # draw_text
```

---

## Installation

```bash
pip install -e .
```

---

## Quick Start

```python
import minicv.io         as io
import minicv.filtering  as filt
import minicv.transforms as transforms
import minicv.drawing    as drawing

# Load an image
img  = io.read_image("photo.png")
gray = io.rgb_to_gray(img)

# Blur + edge detect
blurred = filt.gaussian_filter(gray, kernel_size=5, sigma=1.4)
edges   = filt.canny(blurred, low_threshold=30, high_threshold=90)

# Resize and rotate
small   = transforms.resize(img, 128, 128, interpolation="bilinear")
rotated = transforms.rotate(img, 45)

# Draw on image
canvas = img.copy()
drawing.draw_rectangle(canvas, 10, 10, 100, 80, (255, 0, 0), thickness=2)
drawing.draw_text(canvas, "Hello!", x=15, y=15, font_scale=1.2, color=(255,255,255))

# Save result
io.write_image(canvas, "output.png")
```

---

## API Reference

### `minicv.utils`

| Function | Description |
|---|---|
| `validate_image(image, ...)` | Assert valid ndarray shape/dtype |
| `validate_kernel(kernel)` | Assert odd-size numeric 2-D kernel |
| `to_float64(image)` | Cast to float64 in [0, 1] |
| `to_uint8(image)` | Cast to uint8 in [0, 255] |
| `normalize(image, mode, ...)` | `'minmax'` / `'zscore'` / `'fixed'` |
| `clip_pixels(image, low, high)` | Clamp values to [low, high] |
| `pad_image(image, pad_h, pad_w, mode)` | `'reflect'` / `'constant'` / `'replicate'` |
| `convolve2d(image, kernel, pad_mode)` | True 2-D convolution (grayscale) |
| `spatial_filter(image, kernel, ...)` | Convolution for grayscale or RGB |

### `minicv.io`

| Function | Description |
|---|---|
| `read_image(path, as_gray)` | Load PNG/JPG → uint8 ndarray |
| `write_image(image, path, quality)` | Save ndarray → PNG/JPG |
| `rgb_to_gray(image)` | BT.601 luma: 0.299R + 0.587G + 0.114B |
| `gray_to_rgb(image)` | Stack grayscale into (H,W,3) |

### `minicv.filtering`

| Function | Description |
|---|---|
| `mean_filter(image, kernel_size)` | Box blur |
| `gaussian_kernel(size, sigma)` | Generate normalised 2-D Gaussian kernel |
| `gaussian_filter(image, kernel_size, sigma)` | Gaussian blur |
| `median_filter(image, kernel_size)` | Median blur (stride-tricks, no pixel loop) |
| `threshold_global(image, thresh, ...)` | Fixed threshold |
| `threshold_otsu(image)` | Automatic Otsu threshold → (binary, t) |
| `threshold_adaptive(image, block_size, method, C)` | Local mean / Gaussian adaptive |
| `sobel_gradients(image)` | Returns dict: Gx, Gy, magnitude, angle |
| `canny(image, low, high, ...)` | Gaussian → Sobel → NMS → hysteresis |
| `bit_plane_slice(image, plane)` | Extract bit plane 0–7 |
| `histogram(image, bins)` | Returns (counts, edges) |
| `histogram_equalization(image)` | CDF-based contrast stretching |
| `kmeans_segment(image, k, ...)` | K-means++ segmentation → (seg, labels) |

### `minicv.transforms`

| Function | Description |
|---|---|
| `resize(image, H, W, interpolation)` | `'nearest'` or `'bilinear'` |
| `rotate(image, angle, ...)` | Rotate about centre, inverse mapping |
| `translate(image, tx, ty, ...)` | Shift by (tx, ty) pixels |

### `minicv.features`

| Function | Description |
|---|---|
| `color_histogram(image, bins, normalize)` | Global intensity/colour histogram |
| `pixel_statistics(image)` | Dict of mean, std, skewness, kurtosis, entropy per channel |
| `hog_descriptor(image, cell_size, n_bins, block_size)` | Dense HOG with L2-Hys normalisation |
| `lbp_descriptor(image, n_bins, normalize)` | Local Binary Pattern histogram |

### `minicv.drawing`

| Function | Description |
|---|---|
| `draw_point(image, x, y, color, radius)` | Filled circle |
| `draw_line(image, x0,y0,x1,y1, color, thickness)` | Bresenham line |
| `draw_rectangle(image, x,y,w,h, color, thickness, filled)` | Outline or filled rect |
| `draw_polygon(image, points, color, thickness, filled)` | Polygon outline or filled (scanline) |
| `draw_text(image, text, x, y, font_scale, color, ...)` | Matplotlib-rasterised text |

---

## Design Principles

- **No OpenCV / Pillow** for core algorithms — only NumPy, Matplotlib, standard library.
- **Vectorised by default** — `numpy.lib.stride_tricks`, broadcasting, and advanced indexing instead of pixel loops.
- **Justified loops** — Median filter uses stride-tricks windows; K-means iterates Lloyd's algorithm; Bresenham's line requires sequential stepping. All loops are documented.
- **Centralised utilities** — Validation, padding, dtype conversion, and convolution live in `minicv.utils` and are reused by all other modules.
- **Consistent dtypes** — `uint8` in → `uint8` out; float in → float out.
- **Full docstrings** — Every public function documents Parameters, Returns, Raises, and Notes.

---

## Running Tests

```bash
python tests/test_minicv.py
# 101/101 passed
```
