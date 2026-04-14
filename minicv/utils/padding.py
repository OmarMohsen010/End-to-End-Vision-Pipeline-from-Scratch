"""
minicv.utils.padding
====================
Multi-mode image padding for use before convolution and filtering.
"""

import numpy as np
from minicv.utils.validation import validate_image


_MODES = ("reflect", "constant", "replicate")


def pad_image(image: np.ndarray,
              pad_h: int,
              pad_w: int,
              mode: str = "reflect",
              constant_value: float = 0.0) -> np.ndarray:
    """Add a border of *pad_h* rows and *pad_w* columns around *image*.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    pad_h : int
        Number of rows to add on **each** side (top and bottom).
        Must be ≥ 0.
    pad_w : int
        Number of columns to add on **each** side (left and right).
        Must be ≥ 0.
    mode : {'reflect', 'constant', 'replicate'}, optional
        Border strategy (default ``'reflect'``):

        * ``'reflect'``   – Mirror the image content at the border
          (e.g. ``dcb|abcde|dcb``).
        * ``'constant'``  – Fill the border with *constant_value*
          (default 0).  Equivalent to zero-padding when the value is 0.
        * ``'replicate'`` – Repeat the edge pixel (e.g.
          ``aaa|abcde|eee``).
    constant_value : float, optional
        Fill value used when *mode* == ``'constant'``.  Default ``0.0``.

    Returns
    -------
    numpy.ndarray
        Padded array of shape (H + 2·pad_h) × (W + 2·pad_w) [× C].
        dtype is preserved from *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *mode* is not recognised, or if *pad_h*/*pad_w* are negative.

    Notes
    -----
    ``'reflect'`` padding is the default because it avoids the artificial
    edge discontinuities introduced by zero-padding, leading to less
    ringing after convolution.
    """
    validate_image(image)

    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"pad_h and pad_w must be ≥ 0, got pad_h={pad_h}, pad_w={pad_w}."
        )

    mode = mode.lower()
    if mode not in _MODES:
        raise ValueError(
            f"mode must be one of {_MODES}, got {mode!r}."
        )

    if pad_h == 0 and pad_w == 0:
        return image.copy()

    is_rgb = image.ndim == 3
    pad_width = ((pad_h, pad_h), (pad_w, pad_w))
    if is_rgb:
        pad_width = pad_width + ((0, 0),)

    if mode == "reflect":
        numpy_mode = "reflect"
    elif mode == "constant":
        numpy_mode = "constant"
    elif mode == "replicate":
        numpy_mode = "edge"

    kwargs = {}
    if mode == "constant":
        kwargs["constant_values"] = constant_value

    return np.pad(image, pad_width, mode=numpy_mode, **kwargs)
