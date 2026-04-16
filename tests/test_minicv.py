"""
tests/test_minicv.py
====================
Comprehensive unit tests for the minicv library.
Covers every public function across all six modules.
"""

import sys
import traceback
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
#  Minimal test harness
# ──────────────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✓ {name}")
        PASS += 1
    except Exception as e:
        print(f"  ✗ {name}")
        traceback.print_exc()
        FAIL += 1

def expect_raises(exc_type, fn):
    try:
        fn()
        raise AssertionError(f"Expected {exc_type.__name__} but no exception was raised.")
    except exc_type:
        pass

# ──────────────────────────────────────────────────────────────────────────
#  Synthetic images
# ──────────────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)

def make_rgb(h=64, w=64):
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

GRAY = make_gray()
RGB  = make_rgb()

# ══════════════════════════════════════════════════════════════════════════
#  1. minicv.utils
# ══════════════════════════════════════════════════════════════════════════

import minicv.utils as utils

print("\n─── utils ───")

test("validate_image: accepts grayscale", lambda: utils.validate_image(GRAY))
test("validate_image: accepts RGB",       lambda: utils.validate_image(RGB))
test("validate_image: rejects list",      lambda: expect_raises(TypeError,  lambda: utils.validate_image([1,2,3])))
test("validate_image: rejects 4-ch",     lambda: expect_raises(ValueError,  lambda: utils.validate_image(np.zeros((4,4,4), dtype=np.uint8))))

test("validate_kernel: valid 3×3",        lambda: utils.validate_kernel(np.ones((3,3))))
test("validate_kernel: rejects even",     lambda: expect_raises(ValueError,  lambda: utils.validate_kernel(np.ones((4,4)))))
test("validate_kernel: rejects 1-D",     lambda: expect_raises(ValueError,  lambda: utils.validate_kernel(np.ones((3,)))))
test("validate_kernel: rejects empty",   lambda: expect_raises(ValueError,  lambda: utils.validate_kernel(np.zeros((0,0)))))

def assert_(cond, msg="Assertion failed"):
    if not cond: raise AssertionError(msg)

test("to_float64: uint8→float", lambda: (
    lambda out: (assert_(out.dtype == np.float64), assert_(out.max() <= 1.0))
)(utils.to_float64(GRAY)))

test("to_uint8: float→uint8", lambda: (
    lambda out: (assert_(out.dtype == np.uint8), assert_(out.max() <= 255))
)(utils.to_uint8(GRAY.astype(np.float64) / 255.0)))

test("normalize: minmax",  lambda: assert_(utils.normalize(GRAY, "minmax").max() == 255))
test("normalize: zscore",  lambda: assert_(abs(utils.normalize(GRAY, "zscore").mean()) < 0.1))
test("normalize: fixed",   lambda: assert_(utils.normalize(GRAY, "fixed", high=255).max() <= 1.0))
test("normalize: bad mode",lambda: expect_raises(ValueError, lambda: utils.normalize(GRAY, "unknown")))

test("clip_pixels: clamps values", lambda: (
    lambda out: assert_(out.max() <= 100)
)(utils.clip_pixels(GRAY, 0, 100)))
test("clip_pixels: low>high raises", lambda: expect_raises(ValueError, lambda: utils.clip_pixels(GRAY, 200, 100)))

test("pad_image: reflect",   lambda: assert_(utils.pad_image(GRAY, 2, 3, "reflect").shape == (68, 70)))
test("pad_image: constant",  lambda: assert_(utils.pad_image(GRAY, 1, 1, "constant").shape == (66, 66)))
test("pad_image: replicate", lambda: assert_(utils.pad_image(GRAY, 4, 4, "replicate").shape == (72, 72)))
test("pad_image: bad mode",  lambda: expect_raises(ValueError, lambda: utils.pad_image(GRAY, 1, 1, "wrap")))
test("pad_image: negative",  lambda: expect_raises(ValueError, lambda: utils.pad_image(GRAY, -1, 0)))

test("convolve2d: identity kernel preserves image",
     lambda: assert_(np.allclose(
         utils.convolve2d(GRAY.astype(np.float64), np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)),
         GRAY.astype(np.float64)
     )))
test("convolve2d: output shape unchanged", lambda: assert_(
    utils.convolve2d(GRAY.astype(np.float64), np.ones((3,3))/9).shape == GRAY.shape))
test("convolve2d: rejects RGB",    lambda: expect_raises(ValueError, lambda: utils.convolve2d(RGB, np.ones((3,3)))))
test("convolve2d: rejects even kernel", lambda: expect_raises(ValueError, lambda: utils.convolve2d(GRAY.astype(np.float64), np.ones((4,4)))))

test("spatial_filter: grayscale", lambda: assert_(utils.spatial_filter(GRAY, np.ones((3,3))/9).shape == GRAY.shape))
test("spatial_filter: RGB",       lambda: assert_(utils.spatial_filter(RGB,  np.ones((3,3))/9).shape == RGB.shape))

# ══════════════════════════════════════════════════════════════════════════
#  2. minicv.io
# ══════════════════════════════════════════════════════════════════════════

import minicv.io as io
import tempfile, os

print("\n─── io ───")

test("rgb_to_gray: shape (H,W)", lambda: assert_(io.rgb_to_gray(RGB).shape == (64, 64)))
test("rgb_to_gray: dtype uint8", lambda: assert_(io.rgb_to_gray(RGB).dtype == np.uint8))
test("rgb_to_gray: rejects gray",lambda: expect_raises(ValueError, lambda: io.rgb_to_gray(GRAY)))

test("gray_to_rgb: shape (H,W,3)", lambda: assert_(io.gray_to_rgb(GRAY).shape == (64, 64, 3)))
test("gray_to_rgb: rejects RGB",   lambda: expect_raises(ValueError, lambda: io.gray_to_rgb(RGB)))

def test_write_read_png():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "test.png")
        io.write_image(RGB, p)
        loaded = io.read_image(p)
        assert loaded.shape == RGB.shape, f"Shape mismatch: {loaded.shape} vs {RGB.shape}"
        assert loaded.dtype == np.uint8

test("write/read PNG roundtrip (RGB)", test_write_read_png)

def test_write_read_gray_png():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gray.png")
        io.write_image(GRAY, p)
        loaded = io.read_image(p, as_gray=True)
        assert loaded.ndim == 2

test("write/read PNG roundtrip (gray)", test_write_read_gray_png)
test("write_image: bad extension", lambda: expect_raises(ValueError, lambda: io.write_image(GRAY, "/tmp/x.bmp")))
test("read_image: missing file",   lambda: expect_raises(ValueError, lambda: io.read_image("/nonexistent/path.png")))

# ══════════════════════════════════════════════════════════════════════════
#  3. minicv.filtering
# ══════════════════════════════════════════════════════════════════════════

import minicv.filtering as filt

print("\n─── filtering ───")

test("mean_filter: gray shape", lambda: assert_(filt.mean_filter(GRAY, 3).shape == GRAY.shape))
test("mean_filter: rgb shape",  lambda: assert_(filt.mean_filter(RGB,  3).shape == RGB.shape))
test("mean_filter: even size rejects", lambda: expect_raises(ValueError, lambda: filt.mean_filter(GRAY, 4)))

test("gaussian_kernel: sums to 1",    lambda: assert_(abs(filt.gaussian_kernel(5, 1.0).sum() - 1.0) < 1e-10))
test("gaussian_kernel: shape (5,5)",  lambda: assert_(filt.gaussian_kernel(5, 1.0).shape == (5,5)))
test("gaussian_kernel: even rejects", lambda: expect_raises(ValueError, lambda: filt.gaussian_kernel(4, 1.0)))
test("gaussian_kernel: neg sigma",    lambda: expect_raises(ValueError, lambda: filt.gaussian_kernel(5, -1.0)))

test("gaussian_filter: gray shape", lambda: assert_(filt.gaussian_filter(GRAY).shape == GRAY.shape))
test("gaussian_filter: rgb shape",  lambda: assert_(filt.gaussian_filter(RGB).shape == RGB.shape))

test("median_filter: gray shape", lambda: assert_(filt.median_filter(GRAY, 3).shape == GRAY.shape))
test("median_filter: rgb shape",  lambda: assert_(filt.median_filter(RGB,  3).shape == RGB.shape))

test("threshold_global: binary output",
     lambda: assert_(set(np.unique(filt.threshold_global(GRAY, 128))) <= {0, 255}))
test("threshold_global: inverted",
     lambda: assert_(set(np.unique(filt.threshold_global(GRAY, 128, invert=True))) <= {0, 255}))

def test_otsu():
    out, t = filt.threshold_otsu(GRAY)
    assert set(np.unique(out)) <= {0, 255}
    assert 0 <= t <= 255

test("threshold_otsu: binary + threshold range", test_otsu)
test("threshold_otsu: non-uint8 rejects", lambda: expect_raises(ValueError, lambda: filt.threshold_otsu(GRAY.astype(np.float64))))

test("threshold_adaptive: mean", lambda: assert_(filt.threshold_adaptive(GRAY, 11, "mean").shape == GRAY.shape))
test("threshold_adaptive: gaussian", lambda: assert_(filt.threshold_adaptive(GRAY, 11, "gaussian").shape == GRAY.shape))
test("threshold_adaptive: even block rejects", lambda: expect_raises(ValueError, lambda: filt.threshold_adaptive(GRAY, 10)))

def test_sobel():
    g = filt.sobel_gradients(GRAY)
    for k in ("Gx", "Gy", "magnitude", "angle"):
        assert k in g
        assert g[k].shape == GRAY.shape
    assert g["angle"].min() >= 0 and g["angle"].max() < 180.1

test("sobel_gradients: keys and shape", test_sobel)
test("sobel_gradients: rejects RGB", lambda: expect_raises(ValueError, lambda: filt.sobel_gradients(RGB)))

test("bit_plane_slice: shape preserved", lambda: assert_(filt.bit_plane_slice(GRAY, 7).shape == GRAY.shape))
test("bit_plane_slice: binary output",   lambda: assert_(set(np.unique(filt.bit_plane_slice(GRAY, 0))) <= {0, 255}))
test("bit_plane_slice: plane out of range", lambda: expect_raises(ValueError, lambda: filt.bit_plane_slice(GRAY, 8)))
test("bit_plane_slice: rejects float",  lambda: expect_raises(ValueError, lambda: filt.bit_plane_slice(GRAY.astype(np.float64), 3)))

def test_hist():
    counts, edges = filt.histogram(GRAY)
    assert len(counts) == 256
    assert counts.sum() == GRAY.size

test("histogram: counts sum", test_hist)

def test_heq():
    eq = filt.histogram_equalization(GRAY)
    assert eq.shape == GRAY.shape
    assert eq.dtype == np.uint8

test("histogram_equalization: shape and dtype", test_heq)

def test_canny():
    edges = filt.canny(GRAY, 30, 90)
    assert edges.shape == GRAY.shape
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)) <= {0, 255}

test("canny: shape, dtype, binary", test_canny)
test("canny: low>=high rejects", lambda: expect_raises(ValueError, lambda: filt.canny(GRAY, 100, 50)))

def test_kmeans():
    seg, labels = filt.kmeans_segment(GRAY, k=3, max_iter=20)
    assert seg.shape == GRAY.shape
    assert labels.shape == GRAY.shape
    assert set(np.unique(labels)) <= {0, 1, 2}

test("kmeans_segment: gray k=3", test_kmeans)

def test_kmeans_rgb():
    seg, labels = filt.kmeans_segment(RGB, k=2, max_iter=10)
    assert seg.shape == RGB.shape

test("kmeans_segment: rgb k=2", test_kmeans_rgb)
test("kmeans_segment: k<1 rejects", lambda: expect_raises(ValueError, lambda: filt.kmeans_segment(GRAY, k=0)))

# ══════════════════════════════════════════════════════════════════════════
#  4. minicv.transforms
# ══════════════════════════════════════════════════════════════════════════

import minicv.transforms as transforms

print("\n─── transforms ───")

test("resize: nearest gray",    lambda: assert_(transforms.resize(GRAY, 32, 32, "nearest").shape == (32,32)))
test("resize: bilinear gray",   lambda: assert_(transforms.resize(GRAY, 128, 128, "bilinear").shape == (128,128)))
test("resize: bilinear rgb",    lambda: assert_(transforms.resize(RGB,  32, 32, "bilinear").shape == (32,32,3)))
test("resize: dtype preserved", lambda: assert_(transforms.resize(GRAY, 32, 32).dtype == np.uint8))
test("resize: zero size rejects", lambda: expect_raises(ValueError, lambda: transforms.resize(GRAY, 0, 32)))
test("resize: bad interp rejects", lambda: expect_raises(ValueError, lambda: transforms.resize(GRAY, 32, 32, "cubic")))

test("rotate: shape preserved", lambda: assert_(transforms.rotate(GRAY, 45).shape == GRAY.shape))
test("rotate: rgb preserved",   lambda: assert_(transforms.rotate(RGB, 90).shape == RGB.shape))
test("rotate: dtype preserved", lambda: assert_(transforms.rotate(GRAY, 30).dtype == np.uint8))
test("rotate: 0 deg = identity",lambda: assert_(np.array_equal(transforms.rotate(GRAY, 0), GRAY)))

test("translate: shape preserved", lambda: assert_(transforms.translate(GRAY, 10, 5).shape == GRAY.shape))
test("translate: rgb preserved",   lambda: assert_(transforms.translate(RGB, -5, 3).shape == RGB.shape))
test("translate: zero = identity", lambda: assert_(np.array_equal(transforms.translate(GRAY, 0, 0), GRAY)))

# ══════════════════════════════════════════════════════════════════════════
#  5. minicv.features
# ══════════════════════════════════════════════════════════════════════════

import minicv.features as features

print("\n─── features ───")

def test_color_hist_gray():
    h = features.color_histogram(GRAY, bins=32)
    assert h.shape == (32,)
    assert abs(h.sum() - 1.0) < 1e-9

def test_color_hist_rgb():
    h = features.color_histogram(RGB, bins=16)
    assert h.shape == (48,)   # 3 × 16

test("color_histogram: gray", test_color_hist_gray)
test("color_histogram: rgb",  test_color_hist_rgb)
test("color_histogram: bins<1 rejects", lambda: expect_raises(ValueError, lambda: features.color_histogram(GRAY, bins=0)))

def test_pixel_stats():
    s = features.pixel_statistics(GRAY)
    for key in ("mean_0", "std_0", "skewness_0", "kurtosis_0", "entropy_0"):
        assert key in s, f"Missing key: {key}"

def test_pixel_stats_rgb():
    s = features.pixel_statistics(RGB)
    assert "mean_2" in s  # 3 channels

test("pixel_statistics: gray keys", test_pixel_stats)
test("pixel_statistics: rgb keys",  test_pixel_stats_rgb)

def test_hog():
    gray_sm = GRAY[:32, :32]   # ensure enough cells
    h = features.hog_descriptor(gray_sm, cell_size=4, n_bins=9, block_size=2)
    assert h.ndim == 1
    assert len(h) > 0

test("hog_descriptor: 1-D output", test_hog)
test("hog_descriptor: rejects RGB", lambda: expect_raises(ValueError, lambda: features.hog_descriptor(RGB)))

def test_lbp():
    h = features.lbp_descriptor(GRAY, n_bins=256)
    assert h.shape == (256,)
    assert abs(h.sum() - 1.0) < 1e-9

test("lbp_descriptor: normalised histogram", test_lbp)
test("lbp_descriptor: rejects RGB", lambda: expect_raises(ValueError, lambda: features.lbp_descriptor(RGB)))

# ══════════════════════════════════════════════════════════════════════════
#  6. minicv.drawing
# ══════════════════════════════════════════════════════════════════════════

import minicv.drawing as drawing

print("\n─── drawing ───")

def make_canvas_rgb(): return np.zeros((64, 64, 3), dtype=np.uint8)
def make_canvas_gray(): return np.zeros((64, 64), dtype=np.uint8)

def test_draw_point_rgb():
    c = make_canvas_rgb()
    drawing.draw_point(c, 32, 32, (255, 0, 0), radius=3)
    assert c[32, 32, 0] == 255

def test_draw_point_gray():
    c = make_canvas_gray()
    drawing.draw_point(c, 32, 32, 200, radius=2)
    assert c[32, 32] == 200

test("draw_point: RGB",  test_draw_point_rgb)
test("draw_point: gray", test_draw_point_gray)
test("draw_point: OOB coords (clipped silently)",
     lambda: drawing.draw_point(make_canvas_rgb(), -10, -10, (255,0,0)))

def test_draw_line():
    c = make_canvas_rgb()
    drawing.draw_line(c, 0, 0, 63, 63, (0, 255, 0), thickness=1)
    assert c[0, 0, 1] == 255
    assert c[63, 63, 1] == 255

test("draw_line: endpoints painted", test_draw_line)

def test_draw_rect_outline():
    c = make_canvas_gray()
    drawing.draw_rectangle(c, 10, 10, 20, 20, 128, thickness=1, filled=False)
    assert c[10, 10] == 128   # corner pixel

def test_draw_rect_filled():
    c = make_canvas_gray()
    drawing.draw_rectangle(c, 5, 5, 10, 10, 200, filled=True)
    assert c[10, 10] == 200   # interior pixel

test("draw_rectangle: outline", test_draw_rect_outline)
test("draw_rectangle: filled",  test_draw_rect_filled)

def test_draw_polygon_outline():
    c = make_canvas_rgb()
    pts = [(10,10),(50,10),(30,50)]
    drawing.draw_polygon(c, pts, (255,255,0), thickness=1)
    assert c[10, 10, 0] == 255  # vertex pixel

def test_draw_polygon_filled():
    c = make_canvas_gray()
    pts = [(5,5),(55,5),(55,55),(5,55)]
    drawing.draw_polygon(c, pts, 150, filled=True)
    assert c[30, 30] == 150  # interior pixel

test("draw_polygon: outline", test_draw_polygon_outline)
test("draw_polygon: filled",  test_draw_polygon_filled)
test("draw_polygon: <2 pts rejects", lambda: expect_raises(ValueError, lambda: drawing.draw_polygon(make_canvas_rgb(), [(1,1)], (255,0,0))))

def test_draw_text():
    c = make_canvas_rgb()
    result = drawing.draw_text(c, "Hi", x=5, y=5, font_scale=1.0, color=(255,255,255))
    assert result is c   # in-place
    assert result.dtype == np.uint8

test("draw_text: returns same array", test_draw_text)
test("draw_text: non-str rejects", lambda: expect_raises(TypeError, lambda: drawing.draw_text(make_canvas_rgb(), 42, 0, 0)))
test("draw_text: neg scale rejects", lambda: expect_raises(ValueError, lambda: drawing.draw_text(make_canvas_rgb(), "x", 0, 0, font_scale=-1.0)))

# ══════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════

total = PASS + FAIL
print(f"\n{'='*50}")
print(f"  Results: {PASS}/{total} passed  |  {FAIL} failed")
print(f"{'='*50}")
if FAIL > 0:
    sys.exit(1)
