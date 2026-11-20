"""Quadtree cells, the Barnes-Hut walk and plot_quadtree."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import plasma_plots
from plasma_plots.trees import (
    QuadtreeCells,
    barnes_hut_walk,
    build_quadtree,
    plot_quadtree,
    quadtree_cells,
)


@pytest.fixture
def points():
    rng = np.random.default_rng(1)
    return np.clip(rng.normal(0.0, 0.3, (2, 400)), -0.99, 0.99)


def test_build_quadtree_splits_until_the_leaves_are_small(points):
    cells = build_quadtree(points, leaf_capacity=6, bounds=(0.0, 0.0, 1.0))
    assert cells.counts[0] == 400
    assert np.all(cells.counts[cells.leaves] <= 6)
    # every inner cell's points are its children's
    for i in np.flatnonzero(~cells.leaves):
        children = cells.children[i][cells.children[i] >= 0]
        assert cells.counts[children].sum() == cells.counts[i]
        assert np.allclose(cells.half_widths[children], cells.half_widths[i] / 2)
        assert np.all(cells.depths[children] == cells.depths[i] + 1)
    # (N, 2) input and the default root
    assert len(build_quadtree(points.T[:3])) == 1
    with pytest.raises(ValueError, match="leaf_capacity"):
        build_quadtree(points, leaf_capacity=0)
    with pytest.raises(ValueError, match="no points"):
        build_quadtree(np.zeros((2, 0)))
    with pytest.raises(ValueError, match="shape"):
        build_quadtree(np.zeros((3, 3)))


def test_coincident_points_stop_at_max_depth():
    cells = build_quadtree(np.zeros((2, 20)) + 0.1, leaf_capacity=2, max_depth=5)
    assert cells.depths.max() == 5


class _FlatTree:
    """The attributes of tree-d-3d's FlatQuadtree: first-child indices, particle ranges."""

    node_center = np.array(
        [[0.0, 0.0], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
    )
    node_half = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    node_depth = np.array([0, 1, 1, 1, 1])
    node_child = np.array([1, -1, -1, -1, -1])
    node_start = np.array([0, 0, 3, 3, 5])
    node_end = np.array([7, 3, 3, 5, 7])
    node_count = 5


def test_cells_from_a_tree_and_from_arrays():
    cells = quadtree_cells(_FlatTree())
    assert cells.children.tolist()[0] == [1, 2, 3, 4]
    assert cells.counts.tolist() == [7, 3, 0, 2, 2]
    assert cells.leaves.tolist() == [False, True, True, True, True]
    assert quadtree_cells(cells) is cells
    plain = quadtree_cells(centers=[[0, 0]], half_widths=[1.0])
    assert isinstance(plain, QuadtreeCells) and plain.leaves.all() and len(plain) == 1
    with pytest.raises(ValueError, match="give a tree"):
        quadtree_cells()
    with pytest.raises(ValueError, match="centres"):
        quadtree_cells(centers=[[0, 0], [1, 1]], half_widths=[1.0])


def test_walk_accepts_far_cells_and_sums_near_leaves(points):
    cells = build_quadtree(points, leaf_capacity=4, bounds=(0.0, 0.0, 1.0))
    target = np.array([0.05, -0.02])
    walk = barnes_hut_walk(cells, target, theta=0.5)
    for i in np.flatnonzero(walk["accepted"]):
        distance = np.hypot(*(target - cells.centers[i]))
        assert 2 * cells.half_widths[i] < 0.5 * distance
    assert np.all(cells.leaves[walk["direct"]])
    # the accepted and direct cells cover every particle exactly once
    assert cells.counts[walk["accepted"] | walk["direct"]].sum() == 400
    # theta = 0: everything is summed directly
    exact = barnes_hut_walk(cells, target, theta=0.0)
    assert not exact["accepted"].any()
    assert exact["direct"].sum() == cells.leaves.sum()
    with pytest.raises(ValueError, match="theta"):
        barnes_hut_walk(cells, target, theta=-1)


def test_walk_skips_empty_quadrants():
    walk = barnes_hut_walk(_FlatTree(), (0.9, -0.9), theta=0.1)
    assert not walk["direct"][2]  # the empty quadrant


@pytest.mark.parametrize("color", ["depth", "count", None])
def test_plot_quadtree(points, color):
    cells = build_quadtree(points, leaf_capacity=8, bounds=(0.0, 0.0, 1.0))
    result = plot_quadtree(cells, points=points, color=color, max_depth=4)
    assert result.data["cells"] is cells
    assert result.ax.get_title().endswith("cells")
    assert result.ax.get_aspect() == 1.0


def test_plot_quadtree_walk_and_point_colors(points):
    charges = np.where(points[0] > 0, 1.0, -1.0)
    result = plot_quadtree(
        points=points, point_color=charges, target=(0.2, 0.1), theta=0.6
    )
    assert "multipole" in result.ax.get_title()
    assert set(result.data["walk"]) == {"accepted", "direct", "opened"}
    labels = [t.get_text() for t in result.ax.get_legend().get_texts()]
    assert labels[-1] == "target" and labels[0].startswith("multipole")


def test_plot_quadtree_errors(points):
    with pytest.raises(ValueError, match="color"):
        plot_quadtree(points=points, color="mass")
    with pytest.raises(ValueError, match="counts"):
        plot_quadtree(
            quadtree_cells(centers=[[0, 0]], half_widths=[1.0]), color="count"
        )
    with pytest.raises(ValueError, match="tree or points"):
        plot_quadtree()


def test_plot_quadtree_with_plotly_keeps_the_walk_on_top(points):
    pytest.importorskip("plotly")
    result = plot_quadtree(points=points, target=(0.2, 0.1), backend="plotly")
    names = [trace.name or "" for trace in result.fig.data]
    walk = [
        i for i, name in enumerate(names) if name.startswith(("multipole", "direct"))
    ]
    # the depth colors are pooled into many traces; the walk is drawn after all of them
    pooled = [
        i
        for i, t in enumerate(result.fig.data)
        if t.showlegend is False and t.mode == "lines"
    ]
    assert walk and pooled and min(walk) > max(pooled)
    assert plasma_plots.plot_quadtree is plot_quadtree
