"""
minicv.filtering
================
Spatial filters, thresholding, edge detection, and segmentation.

Public API
----------
mean_filter          : Box / mean blur.
gaussian_kernel      : Build a Gaussian kernel.
gaussian_filter      : Gaussian blur.
median_filter        : Median blur (sliding window, justified loop).
threshold_global     : Fixed-level thresholding.
threshold_otsu       : Otsu's automatic thresholding.
threshold_adaptive   : Adaptive (local) thresholding.
sobel_gradients      : Sobel edge gradients (magnitude + angle).
bit_plane_slice      : Extract a single bit plane.
histogram            : Compute intensity histogram.
histogram_equalization: Equalise grayscale histogram.
canny               : Canny edge detector.
kmeans_segment       : K-means colour/intensity segmentation.
"""

from minicv.filtering.blur        import mean_filter, gaussian_kernel, gaussian_filter, median_filter
from minicv.filtering.threshold   import threshold_global, threshold_otsu, threshold_adaptive
from minicv.filtering.edges       import sobel_gradients, canny
from minicv.filtering.bitplane    import bit_plane_slice
from minicv.filtering.histogram   import histogram, histogram_equalization
from minicv.filtering.segmentation import kmeans_segment

__all__ = [
    "mean_filter",
    "gaussian_kernel",
    "gaussian_filter",
    "median_filter",
    "threshold_global",
    "threshold_otsu",
    "threshold_adaptive",
    "sobel_gradients",
    "canny",
    "bit_plane_slice",
    "histogram",
    "histogram_equalization",
    "kmeans_segment",
]
