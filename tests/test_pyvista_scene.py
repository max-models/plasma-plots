"""Tests for plasma_plots.scene: a PyVista plotter as a notebook output with a vtk.js page and a PNG."""

import base64
import importlib.util
import io

import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

import plasma_plots  # noqa: E402
from plasma_plots import pyvista_plots as p3  # noqa: E402

needs_trame = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("trame_pyvista", "nest_asyncio2")),
    reason='needs the scene extra: pip install "plasma-plots[scene]"',
)


def sphere_plotter():
    sphere = pv.Sphere()
    plotter = pv.Plotter(off_screen=True, window_size=[320, 240])
    plotter.add_mesh(sphere, scalars=sphere.points[:, 2], scalar_bar_args={"title": "z"})
    return plotter


def test_scene_is_exported_at_the_top_level():
    assert plasma_plots.scene is p3.scene
    assert "scene" in plasma_plots.__all__


@needs_trame
def test_the_mime_bundle_has_the_page_the_screenshot_and_the_metadata():
    plotter = sphere_plotter()
    data, metadata = plasma_plots.scene(plotter, alt="A sphere", title="Sphere")._repr_mimebundle_()
    assert set(data) == {"text/html", "image/png"}
    assert metadata == {"pyvista-scene": {"alt": "A sphere", "title": "Sphere"}}
    # a full standalone document, vtk.js inlined
    html = data["text/html"]
    assert html.lstrip().lower().startswith("<!doctype html")
    assert html.rstrip().lower().endswith("</html>")
    assert "<body" in html.lower()
    # the PNG decodes, at the plotter's size
    png = base64.b64decode(data["image/png"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    PIL = pytest.importorskip("PIL.Image")
    assert PIL.open(io.BytesIO(png)).size == (320, 240)
    # the export hides the scalar bar only while it runs
    assert all(actor.GetVisibility() for actor in plotter.scalar_bars.values())
    plotter.close()


@needs_trame
def test_the_metadata_leaves_out_what_is_not_given():
    _, metadata = plasma_plots.scene(sphere_plotter())._repr_mimebundle_()
    assert metadata == {"pyvista-scene": {}}


@needs_trame
def test_save_html_and_png_write_the_same_files_the_bundle_shows(tmp_path):
    view = plasma_plots.scene(sphere_plotter(), window_size=[200, 100])
    assert view.save_png(tmp_path / "s.png") == str(tmp_path / "s.png")
    view.save_html(tmp_path / "s.html")
    data, _ = view._repr_mimebundle_()
    assert (tmp_path / "s.png").read_bytes() == base64.b64decode(data["image/png"])
    assert (tmp_path / "s.html").read_text(encoding="utf-8") == data["text/html"]
    PIL = pytest.importorskip("PIL.Image")
    assert PIL.open(tmp_path / "s.png").size == (200, 100)


def test_without_trame_the_error_names_the_extra():
    class NoTrame:  # a plotter without trame-pyvista's .trame component
        scalar_bars = {}

    with pytest.raises(ImportError, match=r"plasma-plots\[scene\]"):
        plasma_plots.scene(NoTrame()).html()


def test_an_on_screen_plotter_asks_for_off_screen():
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(pv.Sphere())
    with pytest.raises(RuntimeError, match="off_screen=True"):
        plasma_plots.scene(plotter).png()
    plotter.close()
