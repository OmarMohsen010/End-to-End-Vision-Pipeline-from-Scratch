"""
minicv.transforms.geometric
============================
Resize, rotate, and translate operations on NumPy image arrays.

All operations are implemented via inverse-mapping (destination → source),
which avoids holes in the output.  Two interpolation strategies are
provided:

* ``'nearest'``  – Nearest-neighbour (fast, blocky at low resolutions).
* ``'bilinear'`` – Bilinear interpolation (smoother, recommended).

Both are fully vectorised using NumPy advanced indexing — no Python loops
over individual pixels.
"""

import numpy as np
from minicv.utils.validation import validate_image


# ──────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────

def _sample_nearest(src: np.ndarray,
                    src_r: np.ndarray,
                    src_c: np.ndarray,
                    fill: float = 0.0) -> np.ndarray:
    """Sample *src* at floating-point (src_r, src_c) using nearest-neighbour.

    Parameters
    ----------
    src : numpy.ndarray
        Source image, shape (H, W) or (H, W, 3), float64.
    src_r, src_c : numpy.ndarray
        Arrays of the same shape giving row / column coordinates in the
        source image.
    fill : float
        Value used for out-of-bounds coordinates.

    Returns
    -------
    numpy.ndarray
        Sampled values, shape == src_r.shape [× C for RGB].
    """
    H, W = src.shape[:2]
    ri = np.round(src_r).astype(np.int64)
    ci = np.round(src_c).astype(np.int64)

    valid = (ri >= 0) & (ri < H) & (ci >= 0) & (ci < W)
    ri_c  = np.clip(ri, 0, H - 1)
    ci_c  = np.clip(ci, 0, W - 1)

    if src.ndim == 2:
        out = src[ri_c, ci_c].astype(np.float64)
        out[~valid] = fill
    else:
        out = src[ri_c, ci_c, :].astype(np.float64)
        out[~valid] = fill

    return out


def _sample_bilinear(src: np.ndarray,
                     src_r: np.ndarray,
                     src_c: np.ndarray,
                     fill: float = 0.0) -> np.ndarray:
    """Sample *src* at floating-point (src_r, src_c) using bilinear interpolation.

    Parameters
    ----------
    src : numpy.ndarray
        Source image, shape (H, W) or (H, W, 3), float64.
    src_r, src_c : numpy.ndarray
        Arrays of the same shape giving sub-pixel row / column coordinates.
    fill : float
        Value for out-of-bounds coordinates.

    Returns
    -------
    numpy.ndarray
        Bilinearly interpolated values with the same spatial shape as
        *src_r* (and a trailing channel dimension for RGB).
    """
    H, W = src.shape[:2]

    r0 = np.floor(src_r).astype(np.int64)
    c0 = np.floor(src_c).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1

    wr1 = src_r - r0.astype(np.float64)   # fractional row weight
    wc1 = src_c - c0.astype(np.float64)   # fractional col weight
    wr0 = 1.0 - wr1
    wc0 = 1.0 - wc1

    # Clamp indices for gather; we'll zero out OOB after
    r0c = np.clip(r0, 0, H - 1)
    r1c = np.clip(r1, 0, H - 1)
    c0c = np.clip(c0, 0, W - 1)
    c1c = np.clip(c1, 0, W - 1)

    valid = (src_r >= 0) & (src_r < H) & (src_c >= 0) & (src_c < W)

    if src.ndim == 2:
        v00 = src[r0c, c0c].astype(np.float64)
        v01 = src[r0c, c1c].astype(np.float64)
        v10 = src[r1c, c0c].astype(np.float64)
        v11 = src[r1c, c1c].astype(np.float64)
        out = wr0 * wc0 * v00 + wr0 * wc1 * v01 + wr1 * wc0 * v10 + wr1 * wc1 * v11
        out[~valid] = fill
    else:
        wr0 = wr0[..., np.newaxis]
        wc0 = wc0[..., np.newaxis]
        wr1 = wr1[..., np.newaxis]
        wc1 = wc1[..., np.newaxis]
        v00 = src[r0c, c0c, :].astype(np.float64)
        v01 = src[r0c, c1c, :].astype(np.float64)
        v10 = src[r1c, c0c, :].astype(np.float64)
        v11 = src[r1c, c1c, :].astype(np.float64)
        out = wr0 * wc0 * v00 + wr0 * wc1 * v01 + wr1 * wc0 * v10 + wr1 * wc1 * v11
        out[~valid] = fill

    return out


def _interp(src, src_r, src_c, method, fill=0.0):
    if method == "nearest":
        return _sample_nearest(src, src_r, src_c, fill)
    return _sample_bilinear(src, src_r, src_c, fill)


# ──────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────

def resize(image: np.ndarray,
           out_height: int,
           out_width: int,
           interpolation: str = "bilinear") -> np.ndarray:
    """Resize *image* to (*out_height*, *out_width*) pixels.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    out_height : int
        Target height in pixels.  Must be ≥ 1.
    out_width : int
        Target width in pixels.  Must be ≥ 1.
    interpolation : {'nearest', 'bilinear'}, optional
        Interpolation method.  Default ``'bilinear'``.

    Returns
    -------
    numpy.ndarray
        Resized image of shape (*out_height*, *out_width*) or
        (*out_height*, *out_width*, 3).  dtype matches *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If dimensions are non-positive or *interpolation* is unknown.

    Notes
    -----
    Uses inverse mapping: each destination pixel maps back to source
    coordinates via the scale ratio, then samples the source with the
    chosen interpolation method.  Fully vectorised — no Python pixel loops.
    """
    validate_image(image)
    if out_height < 1 or out_width < 1:
        raise ValueError(
            f"out_height and out_width must be ≥ 1, "
            f"got {out_height} × {out_width}."
        )
    interp = interpolation.lower()
    if interp not in {"nearest", "bilinear"}:
        raise ValueError(
            f"interpolation must be 'nearest' or 'bilinear', got {interpolation!r}."
        )

    H, W    = image.shape[:2]
    scale_r = H / out_height
    scale_c = W / out_width

    # Destination pixel grid
    dst_r, dst_c = np.meshgrid(
        np.arange(out_height, dtype=np.float64),
        np.arange(out_width,  dtype=np.float64),
        indexing="ij",
    )

    # Map to source coordinates (centre-aligned)
    src_r = (dst_r + 0.5) * scale_r - 0.5
    src_c = (dst_c + 0.5) * scale_c - 0.5

    src_f = image.astype(np.float64)
    out   = _interp(src_f, src_r, src_c, interp)

    if image.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out.astype(image.dtype)


def rotate(image: np.ndarray,
           angle: float,
           interpolation: str = "bilinear",
           fill: float = 0.0) -> np.ndarray:
    """Rotate *image* by *angle* degrees counter-clockwise about its centre.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    angle : float
        Rotation angle in **degrees**.  Positive = counter-clockwise.
    interpolation : {'nearest', 'bilinear'}, optional
        Interpolation method.  Default ``'bilinear'``.
    fill : float, optional
        Value used for pixels that map outside the source image boundary.
        Default ``0.0`` (black).

    Returns
    -------
    numpy.ndarray
        Rotated image with the **same shape** as *image*.  Regions outside
        the original extent are filled with *fill*.  dtype matches *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *interpolation* is not recognised.

    Notes
    -----
    Inverse mapping:

    1. Build a destination coordinate grid centred at (cx, cy).
    2. Apply the **inverse** rotation matrix (rotate by −*angle*) to obtain
       source coordinates.
    3. Sample the source with the chosen interpolator.

    Fully vectorised — no Python pixel loops.
    """
    validate_image(image)
    interp = interpolation.lower()
    if interp not in {"nearest", "bilinear"}:
        raise ValueError(
            f"interpolation must be 'nearest' or 'bilinear', got {interpolation!r}."
        )

    H, W  = image.shape[:2]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    theta = np.deg2rad(-angle)   # inverse rotation
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    dst_r, dst_c = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )

    # Centre-relative coordinates
    dr = dst_r - cy
    dc = dst_c - cx

    # Inverse rotation
    src_r = cos_t * dr - sin_t * dc + cy
    src_c = sin_t * dr + cos_t * dc + cx

    src_f = image.astype(np.float64)
    out   = _interp(src_f, src_r, src_c, interp, fill=fill)

    if image.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out.astype(image.dtype)


def translate(image: np.ndarray,
              tx: float,
              ty: float,
              interpolation: str = "bilinear",
              fill: float = 0.0) -> np.ndarray:
    """Shift *image* by (*tx*, *ty*) pixels.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    tx : float
        Horizontal shift in pixels.  Positive = right.
    ty : float
        Vertical shift in pixels.  Positive = down.
    interpolation : {'nearest', 'bilinear'}, optional
        Interpolation method.  Default ``'bilinear'``.
    fill : float, optional
        Fill value for border regions exposed by the shift.  Default ``0.0``.

    Returns
    -------
    numpy.ndarray
        Translated image with the same shape and dtype as *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *interpolation* is not recognised.

    Notes
    -----
    Inverse mapping: each destination pixel (r, c) samples the source at
    (r − ty, c − tx).  Fully vectorised — no Python pixel loops.
    """
    validate_image(image)
    interp = interpolation.lower()
    if interp not in {"nearest", "bilinear"}:
        raise ValueError(
            f"interpolation must be 'nearest' or 'bilinear', got {interpolation!r}."
        )

    H, W = image.shape[:2]

    dst_r, dst_c = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )

    src_r = dst_r - float(ty)
    src_c = dst_c - float(tx)

    src_f = image.astype(np.float64)
    out   = _interp(src_f, src_r, src_c, interp, fill=fill)

    if image.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out.astype(image.dtype)
