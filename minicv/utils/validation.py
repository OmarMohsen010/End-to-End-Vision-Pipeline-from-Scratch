"""
minicv.utils.validation
=======================
Input-validation helpers used across all minicv modules.
"""

import numpy as np


def validate_image(image, *, name: str = "image", allow_grayscale: bool = True,
                   allow_rgb: bool = True) -> None:
    """Raise an informative exception if *image* is not a valid minicv image.

    Parameters
    ----------
    image : object
        The value to validate.
    name : str, optional
        Variable name used in error messages.  Default ``"image"``.
    allow_grayscale : bool, optional
        Accept 2-D arrays (H × W).  Default ``True``.
    allow_rgb : bool, optional
        Accept 3-D arrays with 3 channels (H × W × 3).  Default ``True``.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *image* is not a :class:`numpy.ndarray`.
    ValueError
        If the array has an unsupported shape or no enabled shape is satisfied.

    Notes
    -----
    Accepted dtypes are ``uint8`` and any floating-point dtype.
    The function does **not** enforce value ranges; use :func:`clip_pixels`
    or :func:`normalize` for that purpose.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray, got {type(image).__name__}."
        )

    ndim = image.ndim
    if ndim == 2:
        if not allow_grayscale:
            raise ValueError(
                f"{name} must be an RGB image (H×W×3), got a 2-D grayscale array."
            )
    elif ndim == 3:
        if image.shape[2] != 3:
            raise ValueError(
                f"{name} must have exactly 3 channels (H×W×3), "
                f"got shape {image.shape}."
            )
        if not allow_rgb:
            raise ValueError(
                f"{name} must be a grayscale image (H×W), got a 3-D array."
            )
    else:
        raise ValueError(
            f"{name} must be 2-D (H×W) or 3-D (H×W×3), got {ndim}-D array "
            f"with shape {image.shape}."
        )

    if image.dtype == object:
        raise TypeError(
            f"{name} has dtype=object; expected uint8 or a float dtype."
        )


def validate_kernel(kernel, *, name: str = "kernel") -> None:
    """Raise an informative exception if *kernel* is not valid for convolution.

    Parameters
    ----------
    kernel : object
        The value to validate.
    name : str, optional
        Variable name used in error messages.  Default ``"kernel"``.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *kernel* is not a :class:`numpy.ndarray` or contains non-numeric data.
    ValueError
        If *kernel* is not 2-D, is empty, or has an even-sized dimension.

    Notes
    -----
    Both dimensions of the kernel must be odd and ≥ 1.  A 1×1 kernel is
    technically valid (identity operation).
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray, got {type(kernel).__name__}."
        )
    if kernel.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (rows × cols), got {kernel.ndim}-D array."
        )
    if kernel.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(kernel.dtype, np.number):
        raise TypeError(
            f"{name} must have a numeric dtype, got {kernel.dtype}."
        )
    for axis, size in enumerate(kernel.shape):
        if size % 2 == 0:
            raise ValueError(
                f"{name} dimension {axis} has even size {size}; "
                "kernel dimensions must be odd."
            )
