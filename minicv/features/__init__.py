"""
minicv.features
===============
Feature descriptor extraction.

Public API – Global descriptors
--------------------------------
color_histogram  : Multi-channel colour histogram descriptor.
pixel_statistics : Per-channel statistics (mean, std, skewness, kurtosis).

Public API – Gradient descriptors
-----------------------------------
hog_descriptor   : Histogram of Oriented Gradients (simplified).
lbp_descriptor   : Local Binary Patterns histogram.
"""

from minicv.features.global_descriptors   import color_histogram, pixel_statistics
from minicv.features.gradient_descriptors import hog_descriptor, lbp_descriptor

__all__ = [
    "color_histogram",
    "pixel_statistics",
    "hog_descriptor",
    "lbp_descriptor",
]
