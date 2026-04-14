"""
minicv.filtering.threshold
==========================
Global thresholding, Otsu's method, and adaptive thresholding.
"""

import numpy as np
from minicv.utils.validation import validate_image
from minicv.utils.padding    import pad_image


def threshold_global(image: np.ndarray, thresh: float,
                     max_val: float = 255.0,
                     invert: bool = False) -> np.ndarray:
    """Apply a fixed intensity threshold to a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).  uint8 or float.
    thresh : float
        Threshold level.  Pixels with intensity **above** *thresh* are set
        to *max_val*; all others are set to 0.
    max_val : float, optional
        Value assigned to foreground pixels.  Default ``255.0``.
    invert : bool, optional
        If ``True``, foreground and background are swapped.  Default ``False``.

    Returns
    -------
    numpy.ndarray
        Binary image of the same shape and dtype as *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not grayscale, or *thresh* / *max_val* are invalid.
    """
    validate_image(image, allow_rgb=False)
    if max_val <= 0:
        raise ValueError(f"max_val must be > 0, got {max_val}.")

    img = image.astype(np.float64)
    if not invert:
        out = np.where(img > thresh, max_val, 0.0)
    else:
        out = np.where(img <= thresh, max_val, 0.0)

    if image.dtype == np.uint8:
        return out.astype(np.uint8)
    return out


def threshold_otsu(image: np.ndarray,
                   max_val: float = 255.0) -> tuple[np.ndarray, float]:
    """Threshold a grayscale image using Otsu's method.

    Otsu's algorithm finds the threshold that maximises the inter-class
    variance (equivalently, minimises the intra-class variance) of the
    pixel histogram.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W) with dtype ``uint8`` (values in
        [0, 255]).
    max_val : float, optional
        Value assigned to foreground pixels.  Default ``255.0``.

    Returns
    -------
    thresholded : numpy.ndarray
        Binary image with the same shape as *image*.  dtype ``uint8``.
    optimal_thresh : float
        The threshold value chosen by Otsu's criterion.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not 2-D or not uint8.

    Notes
    -----
    Implemented fully via NumPy vectorisation over the 256-bin histogram —
    no Python loops.
    """
    validate_image(image, allow_rgb=False)
    if image.dtype != np.uint8:
        raise ValueError(
            "threshold_otsu requires a uint8 grayscale image "
            f"(values in [0, 255]), got dtype={image.dtype}."
        )

    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return image.copy(), 0.0

    prob = hist / total
    levels = np.arange(256, dtype=np.float64)

    # Cumulative sums for fast inter-class variance computation
    cum_prob  = np.cumsum(prob)           # ω₀(t)
    cum_mean  = np.cumsum(prob * levels)  # ω₀(t) · μ₀(t) component

    global_mean = cum_mean[-1]

    # Avoid divide-by-zero at edges
    w0 = cum_prob
    w1 = 1.0 - cum_prob
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_sq = np.where(
            (w0 > 0) & (w1 > 0),
            (global_mean * w0 - cum_mean) ** 2 / (w0 * w1),
            0.0,
        )

    optimal = float(np.argmax(sigma_b_sq))
    out = np.where(image.astype(np.float64) > optimal, max_val, 0.0)
    return out.astype(np.uint8), optimal


def threshold_adaptive(image: np.ndarray,
                       block_size: int = 11,
                       method: str = "mean",
                       C: float = 2.0,
                       max_val: float = 255.0) -> np.ndarray:
    """Apply local adaptive thresholding to a grayscale image.

    Each pixel is compared against a locally computed threshold derived
    from its neighbourhood.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).  uint8 or float.
    block_size : int, optional
        Side length of the local neighbourhood window.  Must be a positive
        odd integer ≥ 3.  Default ``11``.
    method : {'mean', 'gaussian'}, optional
        How the local threshold is computed (default ``'mean'``):

        * ``'mean'``     – Threshold = mean of the block − *C*.
        * ``'gaussian'`` – Threshold = Gaussian-weighted mean of the block − *C*.
    C : float, optional
        Constant subtracted from the local mean.  Default ``2.0``.
    max_val : float, optional
        Value assigned to foreground pixels.  Default ``255.0``.

    Returns
    -------
    numpy.ndarray
        Binary image with the same shape as *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not grayscale, *block_size* is invalid, or *method*
        is not recognised.
    """
    validate_image(image, allow_rgb=False)

    if not isinstance(block_size, int) or block_size < 3 or block_size % 2 == 0:
        raise ValueError(
            f"block_size must be an odd integer ≥ 3, got {block_size}."
        )
    method = method.lower()
    if method not in {"mean", "gaussian"}:
        raise ValueError(f"method must be 'mean' or 'gaussian', got {method!r}.")

    img = image.astype(np.float64)

    if method == "mean":
        kernel = (np.ones((block_size, block_size), dtype=np.float64)
                  / block_size ** 2)
    else:  # gaussian
        from minicv.filtering.blur import gaussian_kernel
        sigma   = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8  # OpenCV heuristic
        kernel  = gaussian_kernel(block_size, sigma)

    from minicv.utils.convolution import convolve2d
    local_mean = convolve2d(img, kernel, pad_mode="reflect")

    thresh_map = local_mean - C
    out = np.where(img > thresh_map, max_val, 0.0)

    if image.dtype == np.uint8:
        return out.astype(np.uint8)
    return out
