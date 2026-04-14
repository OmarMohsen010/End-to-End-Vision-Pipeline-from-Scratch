"""
minicv.io
=========
Image reading, writing, and color-space conversion.

Public API
----------
read_image   : Load an image from disk into a NumPy array.
write_image  : Save a NumPy array to disk as PNG or JPEG.
rgb_to_gray  : Convert an RGB image to grayscale.
gray_to_rgb  : Broadcast a grayscale image to a 3-channel RGB array.
"""

from minicv.io.readwrite import read_image, write_image
from minicv.io.color     import rgb_to_gray, gray_to_rgb

__all__ = ["read_image", "write_image", "rgb_to_gray", "gray_to_rgb"]
