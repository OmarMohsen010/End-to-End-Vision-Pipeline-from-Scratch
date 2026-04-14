"""
minicv.filtering.blur
=====================
Mean/box filter, Gaussian filter, and median filter.
"""

import numpy as np
from minicv.utils.validation import validate_image
from minicv.utils.convolution import spatial_filter
from minicv.utils.padding     import pad_image


# ─────────────────────────── Mean / Box Filter ────────────────────────────

def mean_filter(image: np.ndarray, kernel_size: int = 3,
                pad_mode: str = "reflect") -> np.ndarray:
    """Smooth *image* with a uniform box (mean) filter.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    kernel_size : int, optional
        Side length of the square kernel.  Must be a positive odd integer.
        Default ``3``.
    pad_mode : {'reflect', 'constant', 'replicate'}, optional
        Padding strategy.  Default ``'reflect'``.

    Returns
    -------
    numpy.ndarray
        Blurred image with the same shape and dtype as *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *kernel_size* is even, non-positive, or *image* has bad shape.
    """
    validate_image(image)
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}."
        )
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / kernel_size ** 2
    return spatial_filter(image, kernel, pad_mode=pad_mode)


# ──────────────────────────── Gaussian Filter ─────────────────────────────

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generate a normalised 2-D Gaussian kernel.

    Parameters
    ----------
    size : int
        Side length of the square kernel.  Must be a positive odd integer.
    sigma : float
        Standard deviation of the Gaussian in pixels.  Must be > 0.

    Returns
    -------
    numpy.ndarray
        float64 kernel of shape (size, size) that sums to 1.

    Raises
    ------
    ValueError
        If *size* is not a positive odd integer, or *sigma* ≤ 0.

    Notes
    -----
    The kernel is computed as the outer product of a 1-D Gaussian vector
    with itself — a separable approach that is numerically equivalent to
    the 2-D formula ``G(x,y) = exp(-(x²+y²) / (2σ²))``.
    """
    if not isinstance(size, int) or size < 1 or size % 2 == 0:
        raise ValueError(
            f"size must be a positive odd integer, got {size}."
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    g1d = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    g2d = np.outer(g1d, g1d)
    return g2d / g2d.sum()


def gaussian_filter(image: np.ndarray, kernel_size: int = 5,
                    sigma: float = 1.0,
                    pad_mode: str = "reflect") -> np.ndarray:
    """Smooth *image* with a Gaussian kernel.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    kernel_size : int, optional
        Side length of the square kernel (must be odd).  Default ``5``.
    sigma : float, optional
        Gaussian standard deviation in pixels.  Default ``1.0``.
    pad_mode : {'reflect', 'constant', 'replicate'}, optional
        Padding strategy.  Default ``'reflect'``.

    Returns
    -------
    numpy.ndarray
        Blurred image with the same shape and dtype as *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *kernel_size* or *sigma* are invalid.
    """
    validate_image(image)
    kernel = gaussian_kernel(kernel_size, sigma)
    return spatial_filter(image, kernel, pad_mode=pad_mode)


# ──────────────────────────── Median Filter ───────────────────────────────

def median_filter(image: np.ndarray, kernel_size: int = 3,
                  pad_mode: str = "reflect") -> np.ndarray:
    """Denoise *image* with a median filter.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    kernel_size : int, optional
        Side length of the square window.  Must be a positive odd integer.
        Default ``3``.
    pad_mode : {'reflect', 'constant', 'replicate'}, optional
        Padding strategy.  Default ``'reflect'``.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same shape and dtype as *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *kernel_size* is invalid.

    Notes
    -----
    **Loop justification**: The median is a non-linear operation and cannot
    be expressed as a linear convolution.  A loop over all (H×W) spatial
    positions is therefore required.  The implementation uses
    ``numpy.lib.stride_tricks.as_strided`` to extract all windows in one
    vectorised call, and then ``numpy.median`` over a reshaped view — no
    explicit Python loop over pixels.  Only a single loop over the channels
    is retained for RGB images (3 iterations at most).
    """
    validate_image(image)
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}."
        )

    def _median_gray(gray: np.ndarray) -> np.ndarray:
        """Apply median filter to a single-channel image."""
        k = kernel_size
        pad_h = pad_w = k // 2
        padded = pad_image(gray.astype(np.float64), pad_h, pad_w, mode=pad_mode)
        H, W = gray.shape

        # Build (H, W, k, k) window view — fully vectorised
        shape   = (H, W, k, k)
        strides = (padded.strides[0], padded.strides[1],
                   padded.strides[0], padded.strides[1])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        # windows.reshape(H, W, k*k) then median along last axis
        return np.median(windows.reshape(H, W, k * k), axis=2)

    is_uint8 = image.dtype == np.uint8

    if image.ndim == 2:
        out = _median_gray(image)
    else:
        # Loop over 3 channels — justified: only 3 iterations, no pixel loop
        channels = [_median_gray(image[:, :, c]) for c in range(3)]
        out = np.stack(channels, axis=2)

    if is_uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out
