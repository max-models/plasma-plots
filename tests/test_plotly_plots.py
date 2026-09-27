"""Interactive Plotly figures: array.struphy.plotly and struphy_plots.plotly_plots."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("plotly")

import struphy_plots  # noqa: E402, F401
from struphy_plots.analysis import fit_dispersion_branches, power_spectrum  # noqa: E402
from struphy_plots.plotly_plots import dispersion, save_figure, space_time  # noqa: E402


@pytest.fixture
def waves():
    """Broadband light waves, omega = |k|, in both directions along z."""
    rng = np.random.default_rng(0)
    t = np.arange(400) * 0.05
    z = np.linspace(0.0, 20.0, 128, endpoint=False)
    T, Z = np.meshgrid(t, z, indexing="ij")
    values = np.zeros_like(T)
    for n in range(1, 40):
        k = 2 * np.pi * n / 20
        for sign in (1, -1):
            values += rng.normal() * np.cos(k * Z + sign * k * T + rng.uniform(0, 2 * np.pi))
    return xr.DataArray(values, dims=("t", "z"), coords={"t": t, "z": z}, name="E_x")


def test_space_time_is_symmetric_about_zero_and_labels_its_axes(waves):
    figure = space_time(waves, title="E_x(z, t)")
    heatmap = figure.data[0]
    assert heatmap.zmin == -heatmap.zmax == -float(np.abs(waves).max())
    assert np.asarray(heatmap.z).shape == (waves.sizes["t"], waves.sizes["z"])
    assert figure.layout.xaxis.title.text == "z" and figure.layout.yaxis.title.text == "t"
    assert figure.layout.title.text == "E_x(z, t)"


def test_space_time_needs_exactly_t_and_one_space_dimension(waves):
    with pytest.raises(ValueError, match="space is required"):
        space_time(waves.expand_dims(y=[0.0, 1.0]))
    with pytest.raises(ValueError, match="select every dimension"):
        space_time(waves.expand_dims(y=[0.0, 1.0]), space="z")


def test_dispersion_draws_the_positive_quadrant_with_branches_and_fits(waves):
    spectrum = power_spectrum(waves, dim="z")
    fits = fit_dispersion_branches(spectrum, n_branches=1)
    figure = dispersion(spectrum, branches={"light": lambda k: k}, fits=fits, dynamic_range=12)
    heatmap, light, fit = figure.data
    assert np.min(heatmap.x) >= 0 and np.min(heatmap.y) >= 0
    assert heatmap.zmax == 0 and heatmap.zmin == -12 and np.nanmax(heatmap.z) == 0
    np.testing.assert_allclose(light.y, light.x)
    assert fit.name.startswith("fit, v = ") and abs(fits[0].velocity - 1) < 0.02
    assert list(figure.layout.xaxis.range) == [0.0, float(spectrum.k.max())]


def test_the_accessor_draws_the_same_figures(waves):
    assert waves.struphy.plotly.space_time().to_dict() == space_time(waves).to_dict()
    spectrum = power_spectrum(waves, dim="z")
    assert spectrum.struphy.plotly.dispersion(omega_max=5).to_dict() == dispersion(spectrum, omega_max=5).to_dict()


def test_dispersion_needs_omega_and_k(waves):
    with pytest.raises(ValueError, match="omega"):
        dispersion(waves)


def test_save_figure_writes_html_png_and_json(waves, tmp_path, monkeypatch):
    import plotly.graph_objects as go
    import plotly.io as pio

    written = {}

    def write_image(self, path, **kwargs):  # kaleido needs a Chrome install
        written.update(kwargs)
        open(path, "wb").close()

    monkeypatch.setattr(go.Figure, "write_image", write_image)
    figure = space_time(waves, title="E_x(z, t)")
    paths = save_figure(figure, str(tmp_path / "figures" / "space-time"), width=800)
    assert [p.name for p in paths] == ["space-time.html", "space-time.png", "space-time.plotly.json"]
    assert all(p.is_file() for p in paths)
    assert written == {"width": 800, "height": 650, "scale": 2.0}
    assert pio.read_json(paths[2]).layout.title.text == "E_x(z, t)"


def test_save_figure_writes_only_on_mpi_rank_0(waves, tmp_path, monkeypatch):
    import sys
    import types

    rank_1 = types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(Get_rank=lambda: 1))
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", rank_1)
    assert save_figure(space_time(waves), str(tmp_path / "space-time")) == []
    assert not list(tmp_path.iterdir())


@pytest.fixture
def f():
    t = np.linspace(0.0, 2.0, 7)
    eta1 = np.linspace(0.0, 1.0, 8)
    v1 = np.linspace(-3.0, 3.0, 6)
    values = np.random.default_rng(1).random((t.size, eta1.size, v1.size))
    return xr.DataArray(values, dims=("t", "eta1", "v1"), coords={"t": t, "eta1": eta1, "v1": v1}, name="f")


def test_heatmap_draws_x_across_and_y_up(f):
    from struphy_plots.plotly_plots import heatmap

    figure = heatmap(f.mean("eta1"), x="t", y="v1", zmin=0.0)
    assert np.asarray(figure.data[0].z).shape == (f.sizes["v1"], f.sizes["t"])
    assert figure.layout.xaxis.title.text == "t" and figure.layout.yaxis.title.text == "v₁"
    assert figure.data[0].zmin == 0.0
    with pytest.raises(ValueError, match="select every dimension"):
        heatmap(f, x="t", y="v1")


def test_animation_keeps_at_most_max_frames(f):
    from struphy_plots.plotly_plots import animation

    movie = f.struphy.plotly.animation(x="eta1", y="v1", max_frames=3, xaxis_title="x")
    assert len(movie.frames) == len(movie.layout.sliders[0].steps) == 3
    assert movie.layout.xaxis.title.text == "x"
    np.testing.assert_array_equal(movie.frames[-1].data[0].z, f.isel(t=-1).transpose("v1", "eta1").values)
    assert movie.to_dict() == animation(f, x="eta1", y="v1", max_frames=3, xaxis_title="x").to_dict()


def test_save_figure_can_show_and_give_an_animation_a_still(f, tmp_path, monkeypatch):
    import plotly.graph_objects as go

    pngs, shown = [], []

    def write_image(self, path, **kwargs):
        pngs.append((np.asarray(self.data[0].z), self.layout.sliders[0].active))
        open(path, "wb").close()

    monkeypatch.setattr(go.Figure, "write_image", write_image)
    monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(self))
    movie = f.struphy.plotly.animation(x="eta1", y="v1", max_frames=3)
    initial = np.asarray(movie.data[0].z).copy()
    save_figure(movie, str(tmp_path / "movie"), show=True, still_frame=1)
    assert shown == [movie]
    np.testing.assert_array_equal(pngs[-1][0], movie.frames[1].data[0].z)
    assert pngs[-1][1] == 1
    save_figure(movie, str(tmp_path / "movie"), still_z=initial * 0 + 7, still_active=2)
    assert np.all(pngs[-1][0] == 7) and pngs[-1][1] == 2
    np.testing.assert_array_equal(np.asarray(movie.data[0].z), initial)  # the animation itself is unchanged
