"""
minicv.filtering.histogram
==========================
Grayscale histogram computation and histogram equalisation.
"""

import numpy as np
from minicv.utils.validation import validate_image


def histogram(image: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Compute the intensity histogram of a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).  Must be uint8 (values in
        [0, 255]).
    bins : int, optional
        Number of histogram bins.  Default ``256``.

    Returns
    -------
    counts : numpy.ndarray
        1-D int64 array of length *bins* with pixel counts per bin.
    bin_edges : numpy.ndarray
        1-D float64 array of length *bins* + 1 with bin boundary values.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not 2-D, not uint8, or *bins* < 1.

    Notes
    -----
    Thin wrapper around ``numpy.histogram`` with range fixed to [0, 256)
    for ``uint8`` images.
    """
    validate_image(image, allow_rgb=False)
    if image.dtype != np.uint8:
        raise ValueError(
            f"histogram requires a uint8 grayscale image, got dtype={image.dtype}."
        )
    if bins < 1:
        raise ValueError(f"bins must be ≥ 1, got {bins}.")

    counts, edges = np.histogram(image.ravel(), bins=bins, range=(0, 256))
    return counts.astype(np.int64), edges


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Enhance contrast by equalising the histogram of a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W) with dtype ``uint8``.

    Returns
    -------
    numpy.ndarray
        Contrast-enhanced uint8 image of the same shape.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not 2-D or not uint8.

    Notes
    -----
    Algorithm:

    1. Compute the 256-bin histogram.
    2. Compute the normalised CDF.
    3. Build a lookup table: ``T(k) = round((L-1) · CDF(k))``.
    4. Apply the lookup table via ``image`` as an index array.

    Fully vectorised — no Python loops.
    """
    validate_image(image, allow_rgb=False)
    if image.dtype != np.uint8:
        raise ValueError(
            f"histogram_equalization requires a uint8 image, got dtype={image.dtype}."
        )

    counts, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total = counts.sum()
    if total == 0:
        return image.copy()

    cdf = counts.cumsum().astype(np.float64) / total
    # Mask out zero-count bins (CDF skip)
    cdf_min = cdf[counts > 0].min()
    lut = np.round(255.0 * (cdf - cdf_min) / max(1.0 - cdf_min, 1e-9))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[image]
