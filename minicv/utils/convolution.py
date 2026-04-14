"""
minicv.utils.convolution
========================
True 2-D convolution and spatial filtering for grayscale and RGB images.

Design notes
------------
* The convolution is implemented via NumPy stride tricks to build a
  sliding-window view of the padded image, then a single vectorised
  dot-product computes all output pixels simultaneously.
* No Python pixel-level loops are used; only the window accumulation is
  vectorised through NumPy broadcasting.
* Boundary handling is delegated to :func:`minicv.utils.padding.pad_image`.
"""

import numpy as np
from minicv.utils.validation import validate_image, validate_kernel
from minicv.utils.padding import pad_image


def convolve2d(image: np.ndarray,
               kernel: np.ndarray,
               pad_mode: str = "reflect") -> np.ndarray:
    """Apply a 2-D convolution kernel to a single-channel (grayscale) image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).  Any numeric dtype; computations
        are carried out in float64.
    kernel : numpy.ndarray
        2-D convolution kernel of shape (kH, kW).  Both dimensions must be
        odd and ≥ 1.
    pad_mode : {'reflect', 'constant', 'replicate'}, optional
        Padding strategy applied before convolution.  Default ``'reflect'``.
        See :func:`minicv.utils.padding.pad_image` for details.

    Returns
    -------
    numpy.ndarray
        float64 array of shape (H, W) — same spatial size as the input
        (full-padding / "same" convolution).

    Raises
    ------
    TypeError
        If *image* or *kernel* are not numpy arrays.
    ValueError
        If *image* is not 2-D, or if *kernel* fails validation.

    Notes
    -----
    The kernel is **flipped** (180° rotation) before sliding, which is the
    definition of convolution.  For symmetric kernels (e.g. Gaussian, box)
    the result is identical to correlation.

    Implementation uses ``numpy.lib.stride_tricks.as_strided`` to create a
    zero-copy sliding-window view of the padded image, followed by a single
    ``tensordot`` call — fully vectorised, no Python loops.
    """
    validate_image(image, allow_rgb=False)
    validate_kernel(kernel)

    img = image.astype(np.float64)
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2

    padded = pad_image(img, pH, pW, mode=pad_mode)
    H, W = img.shape

    # Build a (H, W, kH, kW) sliding-window view — zero copy
    shape   = (H, W, kH, kW)
    strides = (padded.strides[0], padded.strides[1],
               padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape,
                                               strides=strides)

    # Flip kernel for true convolution (not cross-correlation)
    k_flipped = kernel[::-1, ::-1].astype(np.float64)

    # Single vectorised multiply-accumulate: (H, W, kH, kW) · (kH, kW)
    return (windows * k_flipped).sum(axis=(2, 3))


def spatial_filter(image: np.ndarray,
                   kernel: np.ndarray,
                   pad_mode: str = "reflect") -> np.ndarray:
    """Apply *kernel* to a grayscale **or** RGB image.

    For RGB images the kernel is applied independently to each channel
    (per-channel strategy).  The output is clipped to the valid uint8
    range [0, 255] when the input dtype is ``uint8``.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.
    kernel : numpy.ndarray
        2-D kernel; see :func:`convolve2d` for requirements.
    pad_mode : {'reflect', 'constant', 'replicate'}, optional
        Padding strategy.  Default ``'reflect'``.

    Returns
    -------
    numpy.ndarray
        Filtered image.  Shape matches *image*.  dtype is ``float64`` for
        float input and ``uint8`` for ``uint8`` input.

    Raises
    ------
    TypeError
        If *image* or *kernel* are not numpy arrays.
    ValueError
        If *image* has an unsupported shape, or *kernel* fails validation.
    """
    validate_image(image)
    validate_kernel(kernel)

    is_uint8 = image.dtype == np.uint8

    if image.ndim == 2:
        out = convolve2d(image, kernel, pad_mode=pad_mode)
    else:
        channels = [convolve2d(image[:, :, c], kernel, pad_mode=pad_mode)
                    for c in range(3)]
        out = np.stack(channels, axis=2)

    if is_uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out
