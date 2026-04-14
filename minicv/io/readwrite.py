"""
minicv.io.readwrite
===================
Read images from disk and write NumPy arrays back to disk.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def read_image(path: str, as_gray: bool = False) -> np.ndarray:
    """Load an image from disk into a NumPy array.

    Supported formats depend on the Matplotlib backend; PNG is always
    supported; JPEG requires Pillow or a compatible backend.

    Parameters
    ----------
    path : str
        Path to the image file.
    as_gray : bool, optional
        If ``True``, return a 2-D grayscale array (H×W).  Conversion
        follows the ITU-R BT.601 luma formula.  Default ``False``.

    Returns
    -------
    numpy.ndarray
        * ``as_gray=False`` : uint8 array of shape (H, W, 3) for colour
          images, or (H, W) for already-grayscale images.
        * ``as_gray=True``  : uint8 array of shape (H, W).

    Raises
    ------
    TypeError
        If *path* is not a string.
    ValueError
        If the file does not exist or cannot be decoded as an image.

    Notes
    -----
    Matplotlib reads PNG files as float32 in [0, 1].  This function
    converts them to uint8 for a consistent interface.  JPEG files are
    already returned as uint8 by Matplotlib.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be a str, got {type(path).__name__}.")
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path!r}.")

    try:
        img = mpimg.imread(path)
    except Exception as exc:
        raise ValueError(f"Cannot read image from {path!r}: {exc}") from exc

    # Matplotlib returns PNG as float32 [0, 1]; normalise to uint8
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Drop alpha channel if present
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if as_gray:
        from minicv.io.color import rgb_to_gray
        if img.ndim == 3:
            img = rgb_to_gray(img)

    return img


def write_image(image: np.ndarray, path: str, quality: int = 95) -> None:
    """Save a NumPy array to disk as PNG or JPEG.

    The output format is inferred from the file extension of *path*.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array with dtype ``uint8`` or
        float in [0, 1].
    path : str
        Destination file path.  Extension must be ``.png`` or
        ``.jpg``/``.jpeg`` (case-insensitive).
    quality : int, optional
        JPEG compression quality [1, 95].  Ignored for PNG.  Default 95.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *image* is not a numpy array, or *path* is not a string.
    ValueError
        If the file extension is not supported, or *quality* is out of
        range.

    Notes
    -----
    PNG export uses Matplotlib's ``imsave``, which is lossless.  The
    ``cmap`` parameter is set to ``'gray'`` automatically for 2-D arrays
    so that grayscale images are stored correctly (not colourised).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}.")
    if not isinstance(path, str):
        raise TypeError(f"path must be a str, got {type(path).__name__}.")
    if not (1 <= quality <= 95):
        raise ValueError(f"quality must be in [1, 95], got {quality}.")

    ext = os.path.splitext(path)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(
            f"Unsupported file extension {ext!r}. Use '.png', '.jpg', or '.jpeg'."
        )

    # Ensure uint8
    if image.dtype != np.uint8:
        img = np.clip(
            image * 255.0 if image.max() <= 1.0 else image,
            0, 255
        ).astype(np.uint8)
    else:
        img = image

    # Ensure output directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    cmap = "gray" if img.ndim == 2 else None

    if ext == ".png":
        plt.imsave(path, img, cmap=cmap)
    else:
        # JPEG via Matplotlib (requires Pillow backend or compatible)
        plt.imsave(path, img, cmap=cmap,
                   pil_kwargs={"quality": quality, "optimize": True})
