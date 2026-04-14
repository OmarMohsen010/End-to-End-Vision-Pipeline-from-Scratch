"""
minicv.utils
============
Core utility functions shared across all minicv modules.

Public API
----------
validate_image          : Assert ndarray shape/dtype constraints
validate_kernel         : Assert kernel validity (odd size, numeric, non-empty)
to_float64              : Safe cast to float64 in [0, 1]
to_uint8                : Safe cast back to uint8 in [0, 255]
normalize               : Multi-mode image normalization
clip_pixels             : Clamp pixel values to a range
pad_image               : Multi-mode 2-D/3-D padding
convolve2d              : True 2-D convolution with boundary handling
spatial_filter          : Apply a kernel to grayscale or RGB images
"""

from minicv.utils.validation import validate_image, validate_kernel
from minicv.utils.dtype      import to_float64, to_uint8
from minicv.utils.normalize  import normalize
from minicv.utils.clip       import clip_pixels
from minicv.utils.padding    import pad_image
from minicv.utils.convolution import convolve2d, spatial_filter

__all__ = [
    "validate_image",
    "validate_kernel",
    "to_float64",
    "to_uint8",
    "normalize",
    "clip_pixels",
    "pad_image",
    "convolve2d",
    "spatial_filter",
]
