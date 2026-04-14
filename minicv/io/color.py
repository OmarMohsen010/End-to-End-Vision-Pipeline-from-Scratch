"""
minicv.io.color
===============
Color-space conversion between RGB and grayscale.
"""

import numpy as np
from minicv.utils.validation import validate_image


# ITU-R BT.601 luma coefficients (same as OpenCV cvtColor default)
_LUMA_R = 0.299
_LUMA_G = 0.587
_LUMA_B = 0.114


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using the BT.601 luma formula.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image of shape (H, W, 3).  Accepts ``uint8`` or float dtypes.

    Returns
    -------
    numpy.ndarray
        2-D grayscale array of shape (H, W).  dtype matches *image*:
        ``uint8`` in and ``uint8`` out; float in and float out.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not 3-D with exactly 3 channels.

    Notes
    -----
    Formula: ``Y = 0.299·R + 0.587·G + 0.114·B``

    This is a fully vectorised operation — no Python loops.
    """
    validate_image(image, allow_grayscale=False)

    weights = np.array([_LUMA_R, _LUMA_G, _LUMA_B], dtype=np.float64)
    gray = (image.astype(np.float64) * weights).sum(axis=2)

    if image.dtype == np.uint8:
        return np.clip(np.round(gray), 0, 255).astype(np.uint8)
    return gray.astype(image.dtype)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """Broadcast a grayscale image to a 3-channel RGB array.

    Each output channel is an identical copy of the grayscale channel,
    which produces a neutral grey when rendered as RGB.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).

    Returns
    -------
    numpy.ndarray
        RGB array of shape (H, W, 3).  dtype matches *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not 2-D (i.e. not grayscale).

    Notes
    -----
    The conversion is ``np.stack([image, image, image], axis=2)`` —
    O(1) memory layout with three views onto the same data (before the
    stack forces a copy).
    """
    validate_image(image, allow_rgb=False)
    return np.stack([image, image, image], axis=2)
