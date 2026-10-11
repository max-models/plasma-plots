"""The shared plot style and its interplay with notebook display."""

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from plasma_plots.plotting import PLOT_STYLE, _plot_style  # noqa: E402


def test_plot_style_applies_the_style_while_drawing_only():
    before = plt.rcParams["figure.dpi"]
    with _plot_style():
        assert plt.rcParams["figure.dpi"] == PLOT_STYLE["figure.dpi"]
    assert plt.rcParams["figure.dpi"] == before


def test_plot_style_keeps_the_interactive_state_set_while_drawing():
    """IPython enables interactive mode when a kernel's first figure is created.

    If that figure is drawn by plasma-plots, the switch happens inside the style context; it
    must survive the context, or later figures are no longer shown inline.
    """
    before = matplotlib.is_interactive()
    try:
        matplotlib.interactive(False)
        with _plot_style():
            matplotlib.interactive(True)
        assert matplotlib.is_interactive()

        with _plot_style():
            matplotlib.interactive(False)
        assert not matplotlib.is_interactive()
    finally:
        matplotlib.interactive(before)


def test_plot_style_keeps_the_interactive_state_on_errors():
    before = matplotlib.is_interactive()
    try:
        matplotlib.interactive(False)
        try:
            with _plot_style():
                matplotlib.interactive(True)
                raise RuntimeError("drawing failed")
        except RuntimeError:
            pass
        assert matplotlib.is_interactive()
    finally:
        matplotlib.interactive(before)
