"""
minicv.drawing.text
====================
Text placement on NumPy image arrays via Matplotlib rasterisation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from minicv.utils.validation import validate_image


def draw_text(image: np.ndarray,
              text: str,
              x: int,
              y: int,
              font_scale: float = 1.0,
              color=(255, 255, 255),
              font_family: str = "monospace",
              thickness: int = 1) -> np.ndarray:
    """Render *text* onto *image* at position (*x*, *y*).

    Matplotlib is used to rasterise the glyphs onto a temporary canvas,
    which is then alpha-composited onto *image*.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale (H×W) or RGB (H×W×3) uint8 array.  Modified in place
        and returned.
    text : str
        The string to render.  Multi-line strings (with ``\\n``) are
        supported.
    x : int
        Left edge of the text bounding box in pixel columns.
    y : int
        Top edge of the text bounding box in pixel rows.
    font_scale : float, optional
        Multiplicative factor applied to the base font size (12 pt).
        Default ``1.0``.
    color : scalar or tuple, optional
        Text colour.  Scalar for grayscale images; RGB 3-tuple (values in
        [0, 255]) for colour images.  Default ``(255, 255, 255)`` (white).
    font_family : str, optional
        Matplotlib font family.  Default ``'monospace'``.
    thickness : int, optional
        Simulated stroke weight (1 = normal, 2 = bold effect via path
        effects).  Default ``1``.

    Returns
    -------
    numpy.ndarray
        The modified *image*.

    Raises
    ------
    TypeError
        If *image* is not a numpy array or *text* is not a string.
    ValueError
        If *font_scale* ≤ 0 or *image* shape is unsupported.

    Notes
    -----
    Strategy:

    1. Create a Matplotlib figure the same size as *image* (in pixels).
    2. Render *text* at the requested position with ``axes.text``.
    3. Rasterise to a uint8 RGBA array.
    4. Use the alpha channel as a mask to composite the text colour onto
       *image* in place.

    This approach supports the full Unicode range and all Matplotlib fonts
    without any bitmap font atlas.
    """
    validate_image(image)
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}.")
    if font_scale <= 0:
        raise ValueError(f"font_scale must be > 0, got {font_scale}.")
    if not isinstance(thickness, int) or thickness < 1:
        raise ValueError(f"thickness must be a positive integer, got {thickness}.")

    H, W  = image.shape[:2]
    dpi   = 100
    fig_w = W / dpi
    fig_h = H / dpi

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)   # invert y so (0,0) is top-left
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    # Normalise color to [0, 1] for Matplotlib
    if np.isscalar(color):
        mpl_color = (color / 255.0,) * 3
    else:
        mpl_color = tuple(c / 255.0 for c in color)

    font_size = 12 * font_scale
    path_effects = []
    if thickness > 1:
        path_effects = [pe.withStroke(linewidth=thickness - 1, foreground=mpl_color)]

    ax.text(
        x, y, text,
        color=mpl_color,
        fontsize=font_size,
        fontfamily=font_family,
        verticalalignment="top",
        path_effects=path_effects if path_effects else None,
    )

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(H, W, 4)  # ARGB
    plt.close(fig)

    alpha = buf[:, :, 0].astype(np.float64) / 255.0  # A channel
    rgb   = buf[:, :, 1:]                             # R, G, B channels

    # Composite: output = alpha * text_color + (1-alpha) * image
    if image.ndim == 2:
        text_val = float(color) if np.isscalar(color) else float(np.mean(color))
        gray_text = np.full((H, W), text_val, dtype=np.float64)
        composited = alpha * gray_text + (1 - alpha) * image.astype(np.float64)
        image[:, :] = np.clip(np.round(composited), 0, 255).astype(np.uint8)
    else:
        for c in range(3):
            tc = float(color[c]) if not np.isscalar(color) else float(color)
            channel = image[:, :, c].astype(np.float64)
            image[:, :, c] = np.clip(
                np.round(alpha * tc + (1 - alpha) * channel), 0, 255
            ).astype(np.uint8)

    return image
