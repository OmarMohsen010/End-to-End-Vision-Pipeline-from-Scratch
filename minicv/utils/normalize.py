"""
minicv.utils.normalize
======================
Image normalisation utilities supporting multiple modes.
"""

import numpy as np
from minicv.utils.validation import validate_image


_MODES = ("minmax", "zscore", "fixed")


def normalize(image: np.ndarray, mode: str = "minmax",
              low: float = 0.0, high: float = 255.0) -> np.ndarray:
    """Normalize pixel intensities using one of three strategies.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    mode : {'minmax', 'zscore', 'fixed'}, optional
        Normalization strategy (default ``'minmax'``):

        * ``'minmax'``  – Rescale so the minimum maps to *low* and the
          maximum maps to *high*.  Output dtype matches the input when
          *low* = 0 and *high* = 255; otherwise float64.
        * ``'zscore'``  – Subtract the mean and divide by the standard
          deviation (zero-centred, unit variance).  Returns float64.
        * ``'fixed'``   – Divide by *high* so values are in [0, 1].
          Useful for scaling ``uint8`` images to float.
    low : float, optional
        Lower bound for ``'minmax'`` rescaling.  Default ``0.0``.
    high : float, optional
        Upper bound for ``'minmax'`` rescaling / divisor for ``'fixed'``.
        Default ``255.0``.

    Returns
    -------
    numpy.ndarray
        Normalised array.  Shape is unchanged.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *mode* is not one of the recognised strings, or if *high* == *low*
        for ``'minmax'`` mode.

    Notes
    -----
    The function operates globally (across all channels simultaneously) for
    ``'minmax'`` and ``'zscore'`` modes, which preserves inter-channel
    relationships in RGB images.
    """
    validate_image(image)

    mode = mode.lower()
    if mode not in _MODES:
        raise ValueError(
            f"mode must be one of {_MODES}, got {mode!r}."
        )

    img = image.astype(np.float64)

    if mode == "minmax":
        img_min, img_max = img.min(), img.max()
        if img_max == img_min:
            raise ValueError(
                "Cannot apply 'minmax' normalisation: image has constant "
                f"intensity ({img_min}).  All pixels have the same value."
            )
        img = (img - img_min) / (img_max - img_min) * (high - low) + low
        # Round-trip to uint8 when caller uses the standard 0-255 range
        if low == 0.0 and high == 255.0 and image.dtype == np.uint8:
            return np.clip(np.round(img), 0, 255).astype(np.uint8)
        return img

    if mode == "zscore":
        mean, std = img.mean(), img.std()
        if std == 0.0:
            raise ValueError(
                "Cannot apply 'zscore' normalisation: image standard "
                "deviation is zero (constant image)."
            )
        return (img - mean) / std

    # mode == "fixed"
    if high == 0:
        raise ValueError("'fixed' mode requires high != 0.")
    return img / float(high)
