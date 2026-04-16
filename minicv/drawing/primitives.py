"""
minicv.drawing.primitives
==========================
Drawing primitives — point, line (Bresenham), rectangle, polygon —
all operating in-place on NumPy arrays.

Color conventions
-----------------
* **Grayscale** images (H×W, uint8): color is a scalar int/float in [0, 255].
* **RGB** images (H×W×3, uint8):     color is a tuple/list of 3 ints in [0, 255].

Thickness
---------
``thickness=1`` draws a single-pixel-wide primitive.  Larger values dilate
the strokes by expanding each stroke pixel into a square of side *thickness*
(clamped to canvas boundaries).

Clipping
--------
All drawing functions clip coordinates to the valid canvas region before
writing, so out-of-bounds drawing calls are safe and silently clipped.
"""

import numpy as np
from minicv.utils.validation import validate_image


# ──────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────

def _validate_color(image: np.ndarray, color) -> np.ndarray:
    """Return *color* as a numpy array compatible with *image*."""
    if image.ndim == 2:
        # Accept pre-validated numpy scalar arrays from internal callers
        if isinstance(color, np.ndarray) and color.ndim == 0:
            return color.astype(image.dtype)
        if not np.isscalar(color):
            raise TypeError(
                "color must be a scalar for a grayscale image, "
                f"got {type(color).__name__}."
            )
        return np.array(color, dtype=image.dtype)
    else:
        color = np.asarray(color, dtype=image.dtype)
        if color.shape != (3,):
            raise ValueError(
                f"color must be an RGB tuple of length 3, got shape {color.shape}."
            )
        return color


def _paint(image: np.ndarray, rows: np.ndarray, cols: np.ndarray,
           color, thickness: int) -> None:
    """Paint pixels at *(rows, cols)* with *color*, expanding by *thickness*."""
    H, W = image.shape[:2]
    half = thickness // 2

    # Expand each point by thickness (vectorised bounding-box dilation)
    r_min = np.clip(rows[:, np.newaxis] - half,       0, H - 1)
    r_max = np.clip(rows[:, np.newaxis] + thickness - 1 - half, 0, H - 1)
    c_min = np.clip(cols[:, np.newaxis] - half,       0, W - 1)
    c_max = np.clip(cols[:, np.newaxis] + thickness - 1 - half, 0, W - 1)

    # For each point, fill the square block
    for i in range(len(rows)):
        r0, r1 = int(r_min[i, 0]), int(r_max[i, 0]) + 1
        c0, c1 = int(c_min[i, 0]), int(c_max[i, 0]) + 1
        image[r0:r1, c0:c1] = color


def _validate_thickness(thickness: int) -> None:
    if not isinstance(thickness, int) or thickness < 1:
        raise ValueError(f"thickness must be a positive integer, got {thickness}.")


# ──────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────

def draw_point(image: np.ndarray, x: int, y: int,
               color, radius: int = 1) -> np.ndarray:
    """Draw a filled circle (point) centred at (*x*, *y*).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) uint8 array.  Modified **in place**
        and also returned for convenience.
    x : int
        Column coordinate (0 = left edge).
    y : int
        Row coordinate (0 = top edge).
    color : scalar or tuple
        Grayscale scalar or RGB tuple of ints in [0, 255].
    radius : int, optional
        Radius of the point in pixels.  Default ``1``.

    Returns
    -------
    numpy.ndarray
        The modified *image* (same object).

    Raises
    ------
    TypeError
        If *image* is not a numpy array, or *color* has the wrong type.
    ValueError
        If *radius* < 1 or *image* has an unsupported shape.

    Notes
    -----
    Uses a vectorised disk mask: builds all (row, col) pairs within the
    bounding square and keeps those whose Euclidean distance from the centre
    is ≤ *radius*.  Coordinates outside the canvas are silently clipped.
    """
    validate_image(image)
    color_arr = _validate_color(image, color)
    if not isinstance(radius, int) or radius < 1:
        raise ValueError(f"radius must be a positive integer, got {radius}.")

    H, W = image.shape[:2]
    r_lo = max(y - radius, 0)
    r_hi = min(y + radius, H - 1)
    c_lo = max(x - radius, 0)
    c_hi = min(x + radius, W - 1)

    # Fully out-of-bounds: nothing to paint
    if r_lo > r_hi or c_lo > c_hi:
        return image

    rr, cc = np.mgrid[r_lo:r_hi + 1, c_lo:c_hi + 1]
    dist   = np.hypot(rr - y, cc - x)
    mask   = dist <= radius

    if image.ndim == 2:
        image[rr[mask], cc[mask]] = color_arr
    else:
        image[rr[mask], cc[mask], :] = color_arr

    return image


def draw_line(image: np.ndarray,
              x0: int, y0: int, x1: int, y1: int,
              color, thickness: int = 1) -> np.ndarray:
    """Draw a line segment from (*x0*, *y0*) to (*x1*, *y1*) using Bresenham's algorithm.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) uint8 array.  Modified in place.
    x0, y0 : int
        Start point (column, row).
    x1, y1 : int
        End point (column, row).
    color : scalar or tuple
        Grayscale scalar or RGB 3-tuple in [0, 255].
    thickness : int, optional
        Stroke width in pixels.  Default ``1``.

    Returns
    -------
    numpy.ndarray
        The modified *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *thickness* < 1 or *image* shape is invalid.

    Notes
    -----
    Bresenham's line algorithm generates pixel coordinates without
    floating-point division.  The implementation loops over the number of
    steps (|dx| or |dy| — whichever is larger), which is inherent to the
    rasterisation algorithm and cannot be eliminated; however each step is
    O(1) arithmetic and the loop body is kept minimal.
    """
    validate_image(image)
    color_arr = _validate_color(image, color)
    _validate_thickness(thickness)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    # Collect all line pixels, then paint in one call
    xs, ys = [], []
    cx, cy = int(x0), int(y0)
    while True:
        xs.append(cx)
        ys.append(cy)
        if cx == int(x1) and cy == int(y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx  += sx
        if e2 < dx:
            err += dx
            cy  += sy

    rows = np.array(ys, dtype=np.int64)
    cols = np.array(xs, dtype=np.int64)
    _paint(image, rows, cols, color_arr, thickness)
    return image


def draw_rectangle(image: np.ndarray,
                   x: int, y: int, width: int, height: int,
                   color, thickness: int = 1,
                   filled: bool = False) -> np.ndarray:
    """Draw a rectangle with top-left corner at (*x*, *y*).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) uint8 array.  Modified in place.
    x : int
        Left edge column coordinate.
    y : int
        Top edge row coordinate.
    width : int
        Rectangle width in pixels (must be ≥ 1).
    height : int
        Rectangle height in pixels (must be ≥ 1).
    color : scalar or tuple
        Grayscale scalar or RGB 3-tuple in [0, 255].
    thickness : int, optional
        Outline stroke width.  Ignored when *filled* is ``True``.
        Default ``1``.
    filled : bool, optional
        If ``True``, flood-fill the interior with *color*.  Default ``False``.

    Returns
    -------
    numpy.ndarray
        The modified *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *width* or *height* < 1.
    """
    validate_image(image)
    color_arr = _validate_color(image, color)
    _validate_thickness(thickness)
    if width < 1 or height < 1:
        raise ValueError(f"width and height must be ≥ 1, got {width}×{height}.")

    H, W = image.shape[:2]
    x0, y0 = int(x),           int(y)
    x1, y1 = int(x + width - 1), int(y + height - 1)

    if filled:
        r0 = np.clip(y0, 0, H - 1)
        r1 = np.clip(y1, 0, H - 1)
        c0 = np.clip(x0, 0, W - 1)
        c1 = np.clip(x1, 0, W - 1)
        if image.ndim == 2:
            image[r0:r1 + 1, c0:c1 + 1] = color_arr
        else:
            image[r0:r1 + 1, c0:c1 + 1, :] = color_arr
    else:
        # Four sides as lines
        draw_line(image, x0, y0, x1, y0, color_arr, thickness)  # top
        draw_line(image, x0, y1, x1, y1, color_arr, thickness)  # bottom
        draw_line(image, x0, y0, x0, y1, color_arr, thickness)  # left
        draw_line(image, x1, y0, x1, y1, color_arr, thickness)  # right

    return image


def draw_polygon(image: np.ndarray,
                 points: list,
                 color, thickness: int = 1,
                 filled: bool = False) -> np.ndarray:
    """Draw a polygon defined by a list of (x, y) vertex coordinates.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) uint8 array.  Modified in place.
    points : list of (int, int)
        Vertex list as ``[(x0, y0), (x1, y1), …]``.  At least 2 vertices
        are required; fewer than 3 produce a polyline, not a closed shape.
    color : scalar or tuple
        Grayscale scalar or RGB 3-tuple in [0, 255].
    thickness : int, optional
        Outline stroke width.  Ignored when *filled* is ``True``.
        Default ``1``.
    filled : bool, optional
        If ``True``, fill the polygon interior using scanline rasterisation.
        Default ``False``.

    Returns
    -------
    numpy.ndarray
        The modified *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If fewer than 2 points are supplied.

    Notes
    -----
    Filled polygon uses a scanline algorithm:

    1. For each scanline row, compute intersections with polygon edges.
    2. Sort intersections and fill between pairs.

    The scanline loop (over image rows inside the bounding box) is
    justified because scanline rasterisation is inherently row-sequential;
    each row's computation is vectorised.
    """
    validate_image(image)
    color_arr = _validate_color(image, color)
    _validate_thickness(thickness)

    pts = [(int(px), int(py)) for px, py in points]
    if len(pts) < 2:
        raise ValueError(f"draw_polygon requires at least 2 points, got {len(pts)}.")

    # Close the polygon
    closed = pts + [pts[0]]

    if not filled:
        for i in range(len(pts)):
            x0, y0 = closed[i]
            x1, y1 = closed[i + 1]
            draw_line(image, x0, y0, x1, y1, color_arr, thickness)
        return image

    # Filled: scanline rasterisation
    H, W = image.shape[:2]
    ys   = [p[1] for p in pts]
    y_lo = max(int(min(ys)), 0)
    y_hi = min(int(max(ys)), H - 1)

    for scan_y in range(y_lo, y_hi + 1):
        intersections = []
        for i in range(len(pts)):
            x0, y0 = closed[i]
            x1, y1 = closed[i + 1]
            if (y0 <= scan_y < y1) or (y1 <= scan_y < y0):
                # x-coordinate where this edge intersects the scanline
                t  = (scan_y - y0) / (y1 - y0)
                xi = x0 + t * (x1 - x0)
                intersections.append(xi)
        intersections.sort()
        for j in range(0, len(intersections) - 1, 2):
            c0 = max(int(np.floor(intersections[j])),     0)
            c1 = min(int(np.ceil(intersections[j + 1])),  W - 1)
            if image.ndim == 2:
                image[scan_y, c0:c1 + 1] = color_arr
            else:
                image[scan_y, c0:c1 + 1, :] = color_arr

    return image
