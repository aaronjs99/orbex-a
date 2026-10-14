"""Lightweight, consistent identity and plot styles; no Matplotlib setup."""
import colorsys

_COLORS = ("#0072B2", "#D55E00", "#6D8B24", "#CC79A7", "#D99B00", "#56B4E9", "#8754A1", "#564B42")
_MARKERS = ("o", "s", "^", "v", "P", "X", ">", "<")
_LINES = ("-", "--", "-.", ":")
_INK, _MUTED = "#22303A", "#687580"
_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.labelcolor": _INK, "text.color": _INK, "axes.edgecolor": "#AAB4BC",
    "axes.titlesize": 12, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "grid.color": "#DEE4E8", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
}


def chaser_color(index: int) -> str:
    """Stable shared colors, with a deterministic extension for larger fleets."""
    if index < len(_COLORS):
        return _COLORS[index]
    hue = (0.11 + (index - len(_COLORS)) * 0.61803398875) % 1.0
    red, green, blue = colorsys.hls_to_rgb(hue, 0.43, 0.62)
    return "#{:02X}{:02X}{:02X}".format(round(red * 255), round(green * 255), round(blue * 255))


def chaser_label(identifier: str) -> str:
    return str(identifier).replace("chaser_", "Chaser ")
