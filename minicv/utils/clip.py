"""
minicv.utils.clip
=================
Pixel-value clipping utilities.
"""

import numpy as np
from minicv.utils.validation import validate_image


def clip_pixels(image: np.ndarray,
                low: float = 0.0,
                high: float = 255.0) -> np.ndarray:
    """Clamp every pixel in *image* to the closed interval [*low*, *high*].

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.  Any numeric dtype is
        accepted; the output dtype matches the input.
    low : float, optional
        Minimum allowed value.  Default ``0.0``.
    high : float, optional
        Maximum allowed value.  Default ``255.0``.

    Returns
    -------
    numpy.ndarray
        Clipped array with the same dtype and shape as *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *low* > *high*, or if *image* has an unsupported shape.

    Notes
    -----
    For ``uint8`` images the canonical call is
    ``clip_pixels(img, 0, 255)``.  For float images normalised to [0, 1]
    use ``clip_pixels(img, 0.0, 1.0)``.
    """
    validate_image(image)

    if low > high:
        raise ValueError(
            f"low ({low}) must be ≤ high ({high})."
        )

    return np.clip(image, low, high).astype(image.dtype)
