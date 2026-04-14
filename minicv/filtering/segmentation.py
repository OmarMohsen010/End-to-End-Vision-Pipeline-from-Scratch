"""
minicv.filtering.segmentation
==============================
K-means clustering for image segmentation.
"""

import numpy as np
from minicv.utils.validation import validate_image


def kmeans_segment(image: np.ndarray,
                   k: int = 3,
                   max_iter: int = 100,
                   tol: float = 1e-4,
                   random_state: int | None = 42) -> tuple[np.ndarray, np.ndarray]:
    """Segment *image* into *k* clusters via K-means clustering.

    Each pixel (or pixel triplet for RGB) is treated as a feature vector.
    After convergence, every pixel is replaced by its cluster centroid
    value.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) array.  uint8 or float.
    k : int, optional
        Number of clusters.  Must be ≥ 1.  Default ``3``.
    max_iter : int, optional
        Maximum number of Lloyd's algorithm iterations.  Default ``100``.
    tol : float, optional
        Convergence tolerance: stop when centroid movement (L2 norm) is
        below this value.  Default ``1e-4``.
    random_state : int or None, optional
        Seed for reproducible centroid initialisation.  Default ``42``.

    Returns
    -------
    segmented : numpy.ndarray
        Reconstructed image where each pixel is replaced by its centroid.
        Same shape and dtype as *image*.
    labels : numpy.ndarray
        Integer label map of shape (H, W) with values in [0, k-1].

    Raises
    ------
    TypeError
        If *image* is not a numpy array.
    ValueError
        If *k* is out of range or *max_iter* < 1.

    Notes
    -----
    **Loop justification**: Lloyd's algorithm requires iterating until
    convergence; each iteration reassigns all pixels (vectorised) and
    recomputes centroids (vectorised).  The outer ``while`` loop over
    iterations is unavoidable but kept to at most *max_iter* steps.

    Initialisation uses K-means++ seeding for better convergence:
    subsequent centroids are chosen with probability proportional to the
    squared distance from the nearest existing centroid.
    """
    validate_image(image)
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be ≥ 1, got {max_iter}.")

    is_uint8 = image.dtype == np.uint8
    img_f    = image.astype(np.float64)

    H, W = image.shape[:2]
    if image.ndim == 2:
        pixels = img_f.reshape(-1, 1)   # (N, 1)
    else:
        pixels = img_f.reshape(-1, 3)   # (N, 3)

    N = pixels.shape[0]
    rng = np.random.default_rng(random_state)

    # K-means++ initialisation
    idx0 = rng.integers(0, N)
    centroids = [pixels[idx0]]
    for _ in range(1, k):
        # Distance from each pixel to its nearest centroid — shape (N,)
        cent_arr = np.array(centroids)                          # (c, C)
        dist_mat = np.sum(
            (pixels[:, np.newaxis, :] - cent_arr[np.newaxis, :, :]) ** 2,
            axis=2
        )                                                       # (N, c)
        dists = dist_mat.min(axis=1)                            # (N,)
        total_d = dists.sum()
        if total_d == 0:
            next_idx = int(rng.integers(0, N))
        else:
            probs = dists / total_d
            next_idx = int(rng.choice(N, p=probs))
        centroids.append(pixels[next_idx])
    centroids = np.array(centroids)  # (k, C)

    # Lloyd's iterations — outer loop is justified (convergence check)
    labels = np.zeros(N, dtype=np.int32)
    for _ in range(max_iter):
        # Assignment: (N, k) distance matrix, vectorised
        dists = np.sum((pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                       axis=2)   # (N, k)
        new_labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.array([
            pixels[new_labels == c].mean(axis=0) if (new_labels == c).any()
            else centroids[c]
            for c in range(k)
        ])

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels    = new_labels
        if shift < tol:
            break

    # Reconstruct image
    reconstructed = centroids[labels]  # (N, C)
    if image.ndim == 2:
        seg = reconstructed.reshape(H, W)
    else:
        seg = reconstructed.reshape(H, W, 3)

    if is_uint8:
        seg = np.clip(np.round(seg), 0, 255).astype(np.uint8)

    return seg, labels.reshape(H, W)
