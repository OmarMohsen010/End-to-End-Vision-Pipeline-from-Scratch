"""
minicv.filtering.bitplane
=========================
Bit-plane slicing for uint8 grayscale images.
"""

import numpy as np
from minicv.utils.validation import validate_image


def bit_plane_slice(image: np.ndarray, plane: int) -> np.ndarray:
    """Extract a single bit plane from a grayscale uint8 image.

    Each pixel value is an 8-bit integer.  Bit plane *n* isolates the
    contribution of the 2ⁿ position (LSB = plane 0, MSB = plane 7).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W) with dtype ``uint8``.
    plane : int
        Bit plane index in [0, 7].  0 is the least significant bit;
        7 is the most significant bit.

    Returns
    -------
    numpy.ndarray
        Binary uint8 image of shape (H, W).  Pixels where the selected
        bit is set are 255; others are 0.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not 2-D, not uint8, or *plane* is out of [0, 7].

    Notes
    -----
    Implemented via a single vectorised bitwise AND — no loops.

    Example
    -------
    >>> import numpy as np
    >>> img = np.array([[128, 64], [32, 16]], dtype=np.uint8)
    >>> bit_plane_slice(img, 7)  # MSB: only 128 has it set
    array([[255,   0],
           [  0,   0]], dtype=uint8)
    """
    validate_image(image, allow_rgb=False)
    if image.dtype != np.uint8:
        raise ValueError(
            f"bit_plane_slice requires a uint8 image, got dtype={image.dtype}."
        )
    if not isinstance(plane, int) or plane < 0 or plane > 7:
        raise ValueError(
            f"plane must be an integer in [0, 7], got {plane}."
        )

    bit_mask = np.uint8(1 << plane)
    return np.where((image & bit_mask) > 0, np.uint8(255), np.uint8(0))
