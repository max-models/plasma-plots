"""The Plotly conversion of drawings: arrows, rotated text, axes turned off, pooled line z-order."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

pytest.importorskip("plotly")

from plasma_plots.plotly_backend import PX_PER_PT  # noqa: E402
from plasma_plots.plotting import PlotResult  # noqa: E402


def _converted(fig):
    return PlotResult(fig).to_plotly().fig


@pytest.mark.parametrize(
    ("style", "side"), [("<->", "end+start"), ("->", "end"), ("<-", "start")]
)
def test_arrows_become_annotation_arrows(style, side):
    fig, ax = plt.subplots()
    ax.add_patch(FancyArrowPatch((0.1, 0.2), (0.8, 0.2), arrowstyle=style, color="red"))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    (note,) = _converted(fig).layout.annotations
    assert (note.ax, note.ay, note.x, note.y) == (0.1, 0.2, 0.8, 0.2)
    assert note.axref == "x" and note.ayref == "y" and note.showarrow
    assert note.arrowside == side and note.arrowcolor.startswith("rgba(255, 0, 0")
    plt.close(fig)


def test_a_plain_arrow_is_a_line():
    fig, ax = plt.subplots()
    ax.add_patch(FancyArrowPatch((0.0, 0.0), (1.0, 1.0), arrowstyle="-"))
    converted = _converted(fig)
    assert not converted.layout.annotations
    (line,) = converted.layout.shapes
    assert line.type == "line" and (line.x1, line.y1) == (1.0, 1.0)
    plt.close(fig)


def test_rotated_text_keeps_its_angle_and_axes_off_are_hidden():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "4 mm", rotation=90)
    ax.text(0.2, 0.2, "flat")
    ax.set_axis_off()
    converted = _converted(fig)
    angles = {a.text: a.textangle for a in converted.layout.annotations}
    assert angles["4 mm"] == -90 and angles["flat"] is None
    assert (
        converted.layout.xaxis.visible is False
        and converted.layout.yaxis.visible is False
    )
    plt.close(fig)


def test_pooled_colored_lines_stay_below_later_artists():
    fig, ax = plt.subplots()
    colored = LineCollection([np.array([[0, 0], [1, 1]]), np.array([[0, 1], [1, 0]])])
    colored.set_array(np.array([0.0, 1.0]))
    ax.add_collection(colored)
    plain = LineCollection(
        [np.array([[0, 0.5], [1, 0.5]])], colors="C1", linewidths=3, zorder=5
    )
    ax.add_collection(plain)
    data = list(_converted(fig).data)
    orange = [
        k
        for k, t in enumerate(data)
        if (t.line.color or "").startswith("rgba(255, 127, 14")
    ]
    pooled = [
        k for k, t in enumerate(data) if t.showlegend is False and t.mode == "lines"
    ]
    pooled = [k for k in pooled if k not in orange]
    # the uniform collection keeps its width, and is drawn after (above) the pooled colors
    assert len(orange) == 1 and data[orange[0]].line.width == pytest.approx(
        3 * PX_PER_PT
    )
    assert pooled and orange[0] > max(pooled)
    plt.close(fig)
