"""
minicv.drawing
==============
Drawing primitives and text placement on NumPy image arrays.

Public API
----------
draw_point     : Draw a single point (filled circle).
draw_line      : Draw a line segment (Bresenham's algorithm).
draw_rectangle : Draw a filled or outlined rectangle.
draw_polygon   : Draw a polygon outline (or filled).
draw_text      : Render text onto an image using Matplotlib.
"""

from minicv.drawing.primitives import (
    draw_point,
    draw_line,
    draw_rectangle,
    draw_polygon,
)
from minicv.drawing.text import draw_text

__all__ = [
    "draw_point",
    "draw_line",
    "draw_rectangle",
    "draw_polygon",
    "draw_text",
]
