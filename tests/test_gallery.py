"""The output helpers of the example gallery write the files the website reads."""

import json

import numpy as np
import pytest
import xarray as xr

go = pytest.importorskip("plotly.graph_objects")

from plasma_plots import gallery  # noqa: E402


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run in an empty directory, with the PNG export stubbed (kaleido needs a Chrome install)."""
    monkeypatch.chdir(tmp_path)
    written = []

    def write_image(self, path, **kwargs):
        written.append(
            (
                str(path),
                self.data[0].z if self.data and hasattr(self.data[0], "z") else None,
            )
        )
        open(path, "wb").close()

    monkeypatch.setattr(go.Figure, "write_image", write_image)
    return tmp_path, written


@pytest.fixture
def f():
    t = np.linspace(0.0, 2.0, 5)
    eta1 = np.linspace(0.0, 1.0, 8)
    v1 = np.linspace(-3.0, 3.0, 6)
    values = np.random.default_rng(0).random((t.size, eta1.size, v1.size))
    return xr.DataArray(
        values,
        dims=("t", "eta1", "v1"),
        coords={"t": t, "eta1": eta1, "v1": v1},
        name="f",
    )


def test_save_figure_writes_png_json_and_html_and_copies_the_thumbnail(workdir):
    tmp, _ = workdir
    (tmp / "../images/examples").mkdir(parents=True, exist_ok=True)
    gallery.save_figure(go.Figure(go.Scatter(x=[0, 1], y=[1e-6, 2e-6])), "demo")
    for name in ("demo.png", "demo.plotly.json", "demo.html"):
        assert (tmp / name).is_file()
    assert (tmp / "../images/examples/demo.png").is_file()
    layout = json.loads((tmp / "demo.plotly.json").read_text())["layout"]
    assert layout["yaxis"]["exponentformat"] == "power"


def test_save_extra_figure_returns_the_metadata_entry(workdir):
    tmp, _ = workdir
    entry = gallery.save_extra_figure(go.Figure(), "demo", "extra", alt="an image", caption="a caption")
    assert entry == {
        "key": "extra",
        "interactive": "/examples/demo-extra.plotly.json",
        "thumbnail": "/images/examples/demo-extra.png",
        "alt": "an image",
        "caption": "a caption",
    }
    assert (tmp / "demo-extra.plotly.json").is_file()


def test_heatmaps_take_eta_dimensions(f):
    heat = gallery.heatmap_figure(f.mean("eta1"), x="t", y="v1", title="f(v, t)", xaxis_title="t", yaxis_title="v")
    assert np.asarray(heat.data[0].z).shape == (f.sizes["v1"], f.sizes["t"])

    field = f.isel(v1=0) - 0.5
    space_time = gallery.space_time_figure(field, space="eta1", title="E(x, t)", colorbar_title="E")
    assert space_time.data[0].zmin == -space_time.data[0].zmax == -float(abs(field).max())


def test_heatmap_movie_keeps_at_most_max_frames_and_saves_its_still(workdir, f):
    _, written = workdir
    movie, static_z = gallery.heatmap_movie(
        f,
        x="eta1",
        y="v1",
        title="f(x, v)",
        xaxis_title="x",
        yaxis_title="v",
        max_frames=3,
    )
    assert len(movie.frames) == 3
    assert len(movie.layout.sliders[0].steps) == 3
    np.testing.assert_array_equal(static_z, f.transpose("t", "v1", "eta1").values[f.sizes["t"] // 2])

    initial = np.asarray(movie.data[0].z).copy()
    gallery.save_figure(movie, "demo", suffix="-movie", static_z=static_z)
    np.testing.assert_array_equal(np.asarray(written[-1][1]), static_z)  # the PNG shows the still
    np.testing.assert_array_equal(np.asarray(movie.data[0].z), initial)  # the animation is unchanged


def test_merge_metadata_keeps_existing_fields_and_refuses_nan(workdir):
    tmp, _ = workdir
    (tmp / "demo.metadata.json").write_text(json.dumps({"title": "Demo", "rate": 0.0}))
    path = gallery.merge_metadata("demo", rate=0.5, figures=[])
    assert json.loads(path.read_text()) == {"title": "Demo", "rate": 0.5, "figures": []}
    with pytest.raises(RuntimeError, match="rate"):
        gallery.merge_metadata("demo", rate=float("nan"))
    assert json.loads(path.read_text())["rate"] == 0.5


def test_a_serial_run_is_root():
    assert gallery.is_root()
    gallery.barrier()
