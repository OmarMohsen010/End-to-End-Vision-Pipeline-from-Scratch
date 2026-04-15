"""
minicv.features.global_descriptors
====================================
Two global image descriptors:

1. **Color histogram** – captures the distribution of intensity / colour
   values across the whole image.
2. **Pixel statistics** – compact summary statistics per channel.
"""

import numpy as np
from minicv.utils.validation import validate_image


def color_histogram(image: np.ndarray,
                    bins: int = 32,
                    normalize: bool = True) -> np.ndarray:
    """Compute a global colour (or intensity) histogram descriptor.

    For grayscale images a single 1-D histogram is returned.  For RGB
    images, per-channel histograms are computed and concatenated into one
    feature vector.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.  uint8 or float.
        Float arrays are assumed to be in [0, 1]; uint8 in [0, 255].
    bins : int, optional
        Number of histogram bins per channel.  Default ``32``.
    normalize : bool, optional
        If ``True``, each per-channel histogram is L1-normalised so that
        it sums to 1.  Default ``True``.

    Returns
    -------
    numpy.ndarray
        1-D float64 feature vector of length *bins* (grayscale) or
        3·*bins* (RGB).

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *bins* < 1 or *image* has an unsupported shape.

    Notes
    -----
    The value range is determined automatically from the dtype:
    ``[0, 256)`` for uint8 and ``[0.0, 1.0]`` for float images.
    """
    validate_image(image)
    if bins < 1:
        raise ValueError(f"bins must be ≥ 1, got {bins}.")

    is_uint8  = image.dtype == np.uint8
    val_range = (0, 256) if is_uint8 else (0.0, 1.0)

    def _hist_channel(channel: np.ndarray) -> np.ndarray:
        h, _ = np.histogram(channel.ravel(), bins=bins, range=val_range)
        h = h.astype(np.float64)
        if normalize:
            total = h.sum()
            if total > 0:
                h /= total
        return h

    if image.ndim == 2:
        return _hist_channel(image)

    return np.concatenate([_hist_channel(image[:, :, c]) for c in range(3)])


def pixel_statistics(image: np.ndarray) -> dict:
    """Compute per-channel descriptive statistics as a feature dictionary.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.  uint8 or float.

    Returns
    -------
    dict
        Keys are stat names with a channel suffix, e.g.
        ``'mean_0'``, ``'std_0'``, ``'skewness_0'``, ``'kurtosis_0'``
        for grayscale, and additionally ``*_1``, ``*_2`` for RGB.

        Returned statistics per channel:

        * ``mean``     – arithmetic mean.
        * ``std``      – standard deviation.
        * ``skewness`` – Fisher's skewness (3rd standardised moment).
        * ``kurtosis`` – excess kurtosis (4th standardised moment − 3).
        * ``min``      – minimum pixel value.
        * ``max``      – maximum pixel value.
        * ``energy``   – sum of squared pixel values (L2² norm).
        * ``entropy``  – Shannon entropy (bits).

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* has an unsupported shape.

    Notes
    -----
    All computations are fully vectorised.  Shannon entropy is estimated
    from a 256-bin histogram for uint8 images, or a 256-bin histogram over
    [0, 1] for float images.
    """
    validate_image(image)

    channels = [image] if image.ndim == 2 else [image[:, :, c] for c in range(3)]
    result   = {}

    for i, ch in enumerate(channels):
        flat = ch.astype(np.float64).ravel()
        mean = flat.mean()
        std  = flat.std()

        result[f"mean_{i}"] = mean
        result[f"std_{i}"]  = std
        result[f"min_{i}"]  = flat.min()
        result[f"max_{i}"]  = flat.max()
        result[f"energy_{i}"] = float(np.sum(flat ** 2))

        if std > 0:
            z = (flat - mean) / std
            result[f"skewness_{i}"] = float(np.mean(z ** 3))
            result[f"kurtosis_{i}"] = float(np.mean(z ** 4) - 3.0)
        else:
            result[f"skewness_{i}"] = 0.0
            result[f"kurtosis_{i}"] = 0.0

        # Shannon entropy from histogram
        val_range = (0, 256) if ch.dtype == np.uint8 else (0.0, 1.0)
        h, _ = np.histogram(flat, bins=256, range=val_range)
        p = h[h > 0].astype(np.float64) / flat.size
        result[f"entropy_{i}"] = float(-np.sum(p * np.log2(p)))

    return result
