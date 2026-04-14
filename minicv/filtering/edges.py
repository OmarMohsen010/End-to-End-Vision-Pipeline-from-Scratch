"""
minicv.filtering.edges
======================
Sobel gradient computation and Canny edge detection.
"""

import numpy as np
from minicv.utils.validation import validate_image
from minicv.utils.convolution import convolve2d


# ─────────────────────────── Sobel Kernels ────────────────────────────────

_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)

_SOBEL_Y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float64)


def sobel_gradients(image: np.ndarray,
                    pad_mode: str = "reflect") -> dict:
    """Compute Sobel gradient magnitude and orientation for a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).  uint8 or float.
    pad_mode : {'reflect', 'constant', 'replicate'}, optional
        Padding strategy for convolution.  Default ``'reflect'``.

    Returns
    -------
    dict with keys:

    ``'Gx'`` : numpy.ndarray
        Horizontal gradient, shape (H, W), float64.
    ``'Gy'`` : numpy.ndarray
        Vertical gradient, shape (H, W), float64.
    ``'magnitude'`` : numpy.ndarray
        Gradient magnitude ``√(Gx² + Gy²)``, shape (H, W), float64.
    ``'angle'`` : numpy.ndarray
        Gradient orientation in **degrees** in [0°, 180°) (unsigned),
        shape (H, W), float64.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not grayscale.

    Notes
    -----
    The Sobel operator approximates the image gradient by convolving with
    the 3×3 kernels Kx and Ky shown below::

        Kx = [[-1, 0, 1],    Ky = [[-1, -2, -1],
               [-2, 0, 2],          [ 0,  0,  0],
               [-1, 0, 1]]          [ 1,  2,  1]]

    Angles are returned modulo 180° (unsigned) for use in NMS.
    """
    validate_image(image, allow_rgb=False)

    img = image.astype(np.float64)
    Gx  = convolve2d(img, _SOBEL_X, pad_mode=pad_mode)
    Gy  = convolve2d(img, _SOBEL_Y, pad_mode=pad_mode)
    mag = np.hypot(Gx, Gy)
    ang = np.degrees(np.arctan2(Gy, Gx)) % 180.0

    return {"Gx": Gx, "Gy": Gy, "magnitude": mag, "angle": ang}


# ──────────────────────────── Canny Edge Detector ─────────────────────────

def canny(image: np.ndarray,
          low_threshold: float = 50.0,
          high_threshold: float = 150.0,
          kernel_size: int = 5,
          sigma: float = 1.4,
          pad_mode: str = "reflect") -> np.ndarray:
    """Detect edges using the Canny algorithm.

    The pipeline is:

    1. Gaussian smoothing.
    2. Sobel gradient magnitude and orientation.
    3. Non-maximum suppression (NMS).
    4. Double-threshold hysteresis.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).  uint8 or float.
    low_threshold : float, optional
        Weak-edge threshold.  Default ``50.0``.
    high_threshold : float, optional
        Strong-edge threshold.  Default ``150.0``.
    kernel_size : int, optional
        Gaussian smoothing kernel size (odd).  Default ``5``.
    sigma : float, optional
        Gaussian sigma in pixels.  Default ``1.4``.
    pad_mode : str, optional
        Padding mode for convolutions.  Default ``'reflect'``.

    Returns
    -------
    numpy.ndarray
        Binary uint8 edge map of shape (H, W); 255 = edge, 0 = non-edge.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not grayscale, or thresholds are invalid.

    Notes
    -----
    Non-maximum suppression and hysteresis are implemented with NumPy
    advanced indexing — no explicit pixel loops.
    """
    validate_image(image, allow_rgb=False)
    if low_threshold >= high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) must be < high_threshold "
            f"({high_threshold})."
        )
    if low_threshold < 0:
        raise ValueError(f"low_threshold must be ≥ 0, got {low_threshold}.")

    # 1. Gaussian smoothing
    from minicv.filtering.blur import gaussian_filter
    smoothed = gaussian_filter(image, kernel_size=kernel_size, sigma=sigma,
                               pad_mode=pad_mode)
    if smoothed.dtype == np.uint8:
        smoothed = smoothed.astype(np.float64)

    # 2. Sobel gradients
    grads = sobel_gradients(smoothed, pad_mode=pad_mode)
    mag   = grads["magnitude"]
    ang   = grads["angle"]  # [0, 180)

    # 3. Non-maximum suppression
    H, W  = mag.shape
    nms   = np.zeros_like(mag)

    # Quantise angles to 4 directions: 0°, 45°, 90°, 135°
    ang_q = (np.floor((ang + 22.5) / 45.0) % 4).astype(int)

    # Neighbour offsets for each quantised direction
    offsets = {
        0: (0, 1),   # 0°  → horizontal
        1: (1, 1),   # 45° → diagonal
        2: (1, 0),   # 90° → vertical
        3: (1, -1),  # 135°→ anti-diagonal
    }

    padded_mag = np.pad(mag, 1, mode="edge")  # 1-pixel border for indexing

    # Vectorised NMS: build shifted arrays for forward/backward neighbours
    for d, (dr, dc) in offsets.items():
        mask   = ang_q == d
        fore   = padded_mag[1 + dr : H + 1 + dr, 1 + dc : W + 1 + dc]
        back   = padded_mag[1 - dr : H + 1 - dr, 1 - dc : W + 1 - dc]
        local_max = (mag >= fore) & (mag >= back) & mask
        nms[local_max] = mag[local_max]

    # 4. Double-threshold hysteresis
    strong = nms >= high_threshold
    weak   = (nms >= low_threshold) & ~strong

    # Propagate strong edges to connected weak edges via label dilation
    out = np.zeros((H, W), dtype=np.uint8)
    out[strong] = 255

    # Connect weak pixels adjacent (8-connected) to strong pixels
    # Implemented via iterative dilation (typically converges in 1-2 passes)
    kernel_connect = np.ones((3, 3), dtype=bool)
    prev_count = -1
    while True:
        padded = np.pad(out, 1, mode="constant")
        # 8-neighbour sum
        neighbours = sum(
            padded[1 + dr : H + 1 + dr, 1 + dc : W + 1 + dc]
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )
        newly_strong = weak & (neighbours > 0)
        if not newly_strong.any():
            break
        out[newly_strong] = 255
        weak[newly_strong] = False

    return out
