"""canvas.py: Headless figures without selecting a process-wide Matplotlib backend.

Copyright: Aaron John Sabu
"""


def artifact_figure(*, figsize):
    """Return a Matplotlib Figure bound to an Agg canvas, with no pyplot backend.

    Inputs:
        figsize: (width, height) in inches.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


def artifact_subplots(*args, figsize, **kwargs):
    """Return a new Agg-backed figure and the axes from ``figure.subplots``.

    Inputs:
        *args, **kwargs: forwarded to ``Figure.subplots``.
        figsize: (width, height) in inches.
    """
    figure = artifact_figure(figsize=figsize)
    return figure, figure.subplots(*args, **kwargs)
