"""
minicv.features.gradient_descriptors
=====================================
Two gradient-based feature descriptors:

1. **HOG** – Histogram of Oriented Gradients (simplified, dense).
2. **LBP** – Local Binary Patterns histogram.
"""

import numpy as np
from minicv.utils.validation import validate_image


# ─────────────────────────── HOG Descriptor ───────────────────────────────

def hog_descriptor(image: np.ndarray,
                   cell_size: int = 8,
                   n_bins: int = 9,
                   block_size: int = 2,
                   signed: bool = False) -> np.ndarray:
    """Compute a dense Histogram of Oriented Gradients (HOG) descriptor.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).  uint8 or float.
    cell_size : int, optional
        Side length of each HOG cell in pixels.  Default ``8``.
    n_bins : int, optional
        Number of orientation bins per cell.  Default ``9``.
    block_size : int, optional
        Number of cells per block side (blocks are *block_size* × *block_size*
        cells).  Default ``2``.
    signed : bool, optional
        If ``False`` (default), use unsigned gradients in [0°, 180°).
        If ``True``, use signed gradients in [0°, 360°).

    Returns
    -------
    numpy.ndarray
        1-D float64 HOG feature vector.  Length =
        n_blocks_y × n_blocks_x × block_size² × n_bins, where
        n_blocks_y = (n_cells_y − block_size + 1) and similarly for x.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not grayscale, or parameters are invalid.

    Notes
    -----
    Pipeline:

    1. Compute Sobel gradients Gx and Gy.
    2. Compute magnitude and (optionally signed) angle per pixel.
    3. Soft-bin each pixel's gradient into the two nearest orientation
       bins within its cell (bilinear angle interpolation).
    4. Normalise each block with L2-Hys (L2 norm → clip to 0.2 → renorm).
    5. Concatenate all block descriptors into one vector.

    Steps 1-4 are fully vectorised.
    """
    validate_image(image, allow_rgb=False)
    if cell_size < 1:
        raise ValueError(f"cell_size must be ≥ 1, got {cell_size}.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be ≥ 1, got {n_bins}.")
    if block_size < 1:
        raise ValueError(f"block_size must be ≥ 1, got {block_size}.")

    from minicv.filtering.edges import sobel_gradients

    img_f  = image.astype(np.float64)
    grads  = sobel_gradients(img_f)
    mag    = grads["magnitude"]
    ang    = grads["angle"] if not signed else np.degrees(
        np.arctan2(
            np.sin(np.deg2rad(grads["angle"])),
            np.cos(np.deg2rad(grads["angle"]))
        )
    ) % 360.0

    max_angle = 180.0 if not signed else 360.0
    H, W     = image.shape

    n_cells_y = H // cell_size
    n_cells_x = W // cell_size

    # Trim image to exact cell grid
    H_trim = n_cells_y * cell_size
    W_trim = n_cells_x * cell_size
    mag    = mag[:H_trim, :W_trim]
    ang    = ang[:H_trim, :W_trim]

    # Build cell histograms via soft binning
    bin_width   = max_angle / n_bins
    bin_centres = np.arange(n_bins) * bin_width + bin_width / 2.0

    # Normalised angle position within bins (continuous)
    ang_norm = ang / bin_width  # continuous bin index

    lower_bin  = np.floor(ang_norm).astype(np.int64) % n_bins
    upper_bin  = (lower_bin + 1) % n_bins
    upper_w    = ang_norm - np.floor(ang_norm)
    lower_w    = 1.0 - upper_w

    # cell_hist: (n_cells_y, n_cells_x, n_bins)
    cell_hist = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float64)

    for b in range(n_bins):
        contrib_lower = np.where(lower_bin == b, lower_w * mag, 0.0)
        contrib_upper = np.where(upper_bin == b, upper_w * mag, 0.0)
        combined = contrib_lower + contrib_upper
        # Reshape to (n_cells_y, cell_size, n_cells_x, cell_size) then sum
        reshaped = combined.reshape(n_cells_y, cell_size, n_cells_x, cell_size)
        cell_hist[:, :, b] = reshaped.sum(axis=(1, 3))

    # Block normalisation (L2-Hys)
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1

    if n_blocks_y < 1 or n_blocks_x < 1:
        raise ValueError(
            f"Image too small for block_size={block_size} with "
            f"cell_size={cell_size}.  Increase image size or decrease parameters."
        )

    features = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cell_hist[by:by + block_size, bx:bx + block_size, :]
            vec   = block.ravel()
            norm  = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vec = np.clip(vec, 0.0, 0.2)
            norm2 = np.linalg.norm(vec)
            if norm2 > 0:
                vec = vec / norm2
            features.append(vec)

    return np.concatenate(features)


# ─────────────────────────── LBP Descriptor ───────────────────────────────

def lbp_descriptor(image: np.ndarray,
                   n_bins: int = 256,
                   normalize: bool = True) -> np.ndarray:
    """Compute a Local Binary Pattern (LBP) histogram descriptor.

    The basic 3×3 LBP: for each pixel, compare its intensity with its
    8 neighbours.  Each comparison produces a bit (1 if neighbour ≥ centre,
    else 0).  The 8 bits form a byte (0-255) — the LBP code.  The
    descriptor is the histogram of all LBP codes in the image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image of shape (H, W).  uint8 or float.
    n_bins : int, optional
        Number of histogram bins.  Use 256 for full LBP codes; smaller
        values group codes together.  Default ``256``.
    normalize : bool, optional
        L1-normalise the histogram.  Default ``True``.

    Returns
    -------
    numpy.ndarray
        1-D float64 feature vector of length *n_bins*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *image* is not grayscale or *n_bins* < 1.

    Notes
    -----
    LBP codes are computed via 8 vectorised shifted comparisons — one per
    neighbour position — and then combined with bit-shift operations.
    No Python pixel loop is used.

    The 8 neighbours are sampled at offsets::

        (-1,-1) (-1,0) (-1,+1)
        ( 0,-1)  [c]  ( 0,+1)
        (+1,-1) (+1,0) (+1,+1)

    Bit ordering: neighbour 0 (top-left) → LSB, neighbour 7 (left) → MSB.
    """
    validate_image(image, allow_rgb=False)
    if n_bins < 1:
        raise ValueError(f"n_bins must be ≥ 1, got {n_bins}.")

    img = image.astype(np.float64)
    H, W = img.shape

    # Pad with replicate (edge) to handle image borders
    padded = np.pad(img, 1, mode="edge")
    centre = padded[1:H + 1, 1:W + 1]

    # 8 neighbours in clockwise order starting from top-left
    neighbour_offsets = [
        (0, 0), (0, 1), (0, 2),   # top row:  (-1,-1), (-1,0), (-1,+1)
        (1, 2),                    # right:    ( 0,+1)
        (2, 2), (2, 1), (2, 0),   # bottom:   (+1,+1), (+1,0), (+1,-1)
        (1, 0),                    # left:     ( 0,-1)
    ]

    lbp_code = np.zeros((H, W), dtype=np.uint8)
    for bit, (dr, dc) in enumerate(neighbour_offsets):
        neighbour = padded[dr:dr + H, dc:dc + W]
        lbp_code |= ((neighbour >= centre).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp_code.ravel(), bins=n_bins, range=(0, 256))
    hist = hist.astype(np.float64)
    if normalize:
        total = hist.sum()
        if total > 0:
            hist /= total
    return hist
