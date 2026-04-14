"""
minicv.utils.dtype
==================
Dtype-conversion helpers: float64 ↔ uint8.
"""

import numpy as np


def to_float64(image: np.ndarray) -> np.ndarray:
    """Return a float64 copy of *image* normalised to [0.0, 1.0].

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array with dtype ``uint8`` or float.

    Returns
    -------
    numpy.ndarray
        float64 array of the same shape with values in [0.0, 1.0].

    Raises
    ------
    TypeError
        If *image* is not a numpy array.

    Notes
    -----
    * ``uint8`` input is divided by 255.
    * Float input already in [0, 1] is cast to float64 without scaling.
    * Float input outside [0, 1] is clipped to that range before returning.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}.")

    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0

    out = image.astype(np.float64)
    if out.min() < 0.0 or out.max() > 1.0:
        out = np.clip(out, 0.0, 1.0)
    return out


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Return a uint8 copy of *image* scaled to [0, 255].

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array with dtype float or ``uint8``.
        Float arrays are assumed to be in [0.0, 1.0].

    Returns
    -------
    numpy.ndarray
        uint8 array of the same shape.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.

    Notes
    -----
    * Float values are multiplied by 255 and rounded before casting.
    * Values outside [0, 255] are clipped to that range.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}.")

    if image.dtype == np.uint8:
        return image.copy()

    scaled = image.astype(np.float64)
    if scaled.max() <= 1.0 and scaled.min() >= 0.0:
        scaled = scaled * 255.0

    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)
