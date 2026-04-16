"""
minicv — A lightweight image-processing library built on NumPy.

Emulates a well-defined subset of OpenCV functionality using only:
NumPy, Pandas, Matplotlib, and the Python standard library.

Modules
-------
io        : Image reading, writing, and color conversion
filtering : Spatial filters, thresholding, edge detection, segmentation
transforms: Geometric transformations (resize, rotate, translate)
features  : Global and gradient-based feature descriptors
drawing   : Drawing primitives and text placement on NumPy arrays
utils     : Core utilities (validation, padding, convolution, normalization)
"""

from minicv import io, filtering, transforms, features, drawing, utils

__version__ = "1.0.0"
__author__  = "minicv contributors"
__all__     = ["io", "filtering", "transforms", "features", "drawing", "utils"]
