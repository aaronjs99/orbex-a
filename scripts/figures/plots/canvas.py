"""canvas.py: Headless figures without selecting a process-wide Matplotlib backend.

Copyright: Aaron John Sabu
"""


def artifact_figure(*, figsize):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


def artifact_subplots(*args, figsize, **kwargs):
    figure = artifact_figure(figsize=figsize)
    return figure, figure.subplots(*args, **kwargs)
