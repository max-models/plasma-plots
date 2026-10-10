"""Field-line tracing on the mapped grid, Poincaré sections, islands, connection lengths and
profiles along the lines (plasma_plots.fieldlines, plasma_plots.fieldline_plots), on analytic
fields: a tokamak with a known q profile, the same with a seeded island, and a slab with open
field lines."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402, F401
from plasma_plots.fieldline_plots import plot_along_field_lines  # noqa: E402
from plasma_plots.fieldline_plots import (plot_connection_length,
                                          plot_field_lines, plot_footprint,
                                          plot_poincare, plot_surface_map)
from plasma_plots.fieldlines import classify_field_lines  # noqa: E402
from plasma_plots.fieldlines import (footprint, islands, parallel_wavenumber,
                                     poincare_section, rotational_transform,
                                     sample_along, seed_grid,
                                     trace_field_lines)
from plasma_plots.plotting import PlotResult  # noqa: E402

R0 = 3.0


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def q_of(r):
    return 1.0 + 2.0 * r**2


def tokamak(n1=24, n2=48, n3=32, epsilon=0.0, m=2, n=1):
    """B on a torus with q(r) = 1 + 2 r², r = 0.1 + 0.9 eta1, and a resonant radial perturbation
    epsilon sin(m theta - n phi) that opens an island chain at q = m/n."""
    e1 = np.linspace(0, 1, n1)
    e2, e3 = (np.arange(n2) + 0.5) / n2, (np.arange(n3) + 0.5) / n3
    E1, E2, E3 = np.meshgrid(e1, e2, e3, indexing="ij")
    r, th, ph = 0.1 + 0.9 * E1, 2 * np.pi * E2, 2 * np.pi * E3
    R = R0 + r * np.cos(th)
    e_r = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])
    e_th = np.stack([-np.sin(th) * np.cos(ph), -np.sin(th) * np.sin(ph), np.cos(th)])
    e_ph = np.stack([-np.sin(ph), np.cos(ph), 0 * ph])
    B = (
        R0 / R * e_ph
        + r / (q_of(r) * R0) * e_th
        + epsilon * np.sin(m * th - n * ph) * e_r
    )
    return xr.DataArray(
        B,
        dims=("component", "eta1", "eta2", "eta3"),
        coords={
            "component": [0, 1, 2],
            "eta1": e1,
            "eta2": e2,
            "eta3": e3,
            "X": (("eta1", "eta2", "eta3"), R * np.cos(ph)),
            "Y": (("eta1", "eta2", "eta3"), R * np.sin(ph)),
            "Z": (("eta1", "eta2", "eta3"), r * np.sin(th)),
        },
        name="B",
        attrs={"label": "$B$", "run": "analytic tokamak"},
    )


def slab(n=24, shear=0.0):
    """A periodic slab (eta2, eta3 periodic by attribute) whose field leans along x, so that the
    lines leave through the x faces: B = (0.2 + shear sin(2π eta2), 0, 1) on X = eta1.
    """
    e = (np.arange(n) + 0.5) / n
    E1, E2, E3 = np.meshgrid(e, e, e, indexing="ij")
    return xr.DataArray(
        np.stack([0.2 + shear * np.sin(2 * np.pi * E2), 0 * E1, 1 + 0 * E1]),
        dims=("component", "eta1", "eta2", "eta3"),
        coords={
            "component": [0, 1, 2],
            "eta1": e,
            "eta2": ("eta2", e, {"period": 1.0}),
            "eta3": ("eta3", e, {"period": 1.0}),
            "X": (("eta1", "eta2", "eta3"), E1),
            "Y": (("eta1", "eta2", "eta3"), 2 * np.pi * E2),
            "Z": (("eta1", "eta2", "eta3"), 2 * np.pi * E3),
        },
        name="B",
    )


@pytest.fixture(scope="module")
def lines():
    return trace_field_lines(tokamak(), seeds=6, turns=20)


@pytest.fixture(scope="module")
def island_lines():
    seeds = {"eta1": np.linspace(0.45, 0.80, 36), "eta2": 0.5 / 48, "eta3": 0.5 / 32}
    return trace_field_lines(tokamak(epsilon=0.001), seeds=seeds, turns=120)


# --- tracing ----------------------------------------------------------------------------------


def test_traced_lines_follow_the_rotational_transform(lines):
    assert lines.attrs["kind"] == "field_lines" and lines.attrs["logical_dims"] == [
        "eta1",
        "eta2",
        "eta3",
    ]
    assert set(lines.dims) == {"s", "line", "puncture"}
    r = 0.1 + 0.9 * np.asarray(lines.eta1_start)
    # dθ/dφ = (1 + ε cos θ)² / q exactly for this field: ι = (1 − ε²)^{3/2} / q
    expected = (1 - (r / R0) ** 2) ** 1.5 / q_of(r)
    np.testing.assert_allclose(lines.iota, expected, rtol=0.01)
    np.testing.assert_allclose(lines.transits, 20, atol=0.02)
    assert not lines.exited.any() and np.isnan(lines.connection_length).all()
    assert lines.attrs["toroidal_turn"] == pytest.approx(1.0)
    assert rotational_transform(lines).dims == ("line",)
    # the samples stay on their surface (eta1 is a flux label here) and in the torus
    assert np.nanstd(lines.eta1, axis=0).max() < 1e-3
    assert np.nanmax(np.hypot(lines.x, lines.y)) < R0 + 1.0
    # angles fold into [first grid value, first grid value + period)
    eta2 = lines.eta2.values[np.isfinite(lines.eta2.values)]
    assert eta2.min() >= 0.5 / 48 - 1e-12 and eta2.max() < 1 + 0.5 / 48


def test_punctures_land_on_the_section_and_count_the_transits(lines):
    section = poincare_section(lines)
    assert section.attrs["kind"] == "poincare_section"
    assert (
        section.sizes["puncture"] == 20
        and (np.isfinite(section.eta1).sum("puncture") == 20).all()
    )
    np.testing.assert_allclose(
        section.eta1, lines.eta1_start.broadcast_like(section.eta1), atol=1e-6
    )
    assert "iota" in section.coords and section.iota.dims == ("line",)
    # the same section cut from the saved samples, and another plane
    resampled = poincare_section(
        lines.drop_vars([v for v in lines if v.startswith("puncture")]).assign_attrs(
            section=np.nan
        ),
        angle=lines.attrs["section"],
    )
    assert resampled.sizes["puncture"] == 20
    other = poincare_section(lines, angle=0.5)
    assert other.sizes["puncture"] == 20 and other.attrs["section"] == 0.5
    assert poincare_section(section) is section
    with pytest.raises(ValueError, match="already a Poincaré section"):
        poincare_section(section, angle=0.3)


def test_seeds_in_every_form_and_the_other_options():
    B = tokamak(n1=12, n2=24, n3=16)
    by_dict = trace_field_lines(
        B, seeds={"eta1": [0.3, 0.6], "eta2": [0.0, 0.5]}, turns=2
    )
    assert by_dict.sizes["line"] == 4 and set(by_dict.eta3_start.values) == {0.5 / 16}
    by_array = trace_field_lines(B, seeds=[[0.3, 0.0, 0.0], [0.6, 0.0, 0.0]], turns=2)
    assert by_array.sizes["line"] == 2
    by_dataset = trace_field_lines(
        B,
        seeds=xr.Dataset(
            {"eta1": ("n", [0.3]), "eta2": ("n", [0.1]), "eta3": ("n", [0.2])}
        ),
        turns=2,
    )
    assert by_dataset.sizes["line"] == 1
    both = trace_field_lines(B, seeds=2, turns=1, direction="both")
    assert both.sizes["line"] == 4 and list(both.direction.values) == [1, 1, -1, -1]
    assert list(both.seed.values) == [0, 1, 0, 1]
    backward = trace_field_lines(B, seeds=2, turns=1, direction="backward")
    np.testing.assert_allclose(backward.iota, both.iota.isel(line=[2, 3]))
    by_length = trace_field_lines(B, seeds=1, length=3.0, stride=2)
    assert by_length.length.item() == pytest.approx(3.0, abs=by_length.attrs["step"])
    assert float(by_length.s[1]) == pytest.approx(2 * by_length.attrs["step"])
    contravariant = trace_field_lines(
        B, seeds=1, turns=1, components="contravariant"
    )  # some other field: it just needs to run
    assert contravariant.sizes["line"] == 1
    with pytest.raises(ValueError, match="select the dimensions"):
        trace_field_lines(B.expand_dims(t=[0.0, 1.0]), seeds=1)
    with pytest.raises(ValueError, match="component"):
        trace_field_lines(B.isel(component=0), seeds=1)
    with pytest.raises(ValueError, match="direction"):
        trace_field_lines(B, seeds=1, direction="sideways")
    with pytest.raises(ValueError, match="inside"):
        trace_field_lines(B, seeds=[[1.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="unknown seed coordinates"):
        trace_field_lines(B, seeds={"rho": 0.5})


# --- islands ----------------------------------------------------------------------------------


def test_surfaces_are_not_islands(lines):
    codes = classify_field_lines(lines)
    assert (
        codes.dims == ("line",)
        and codes.attrs["flag_meanings"] == "surface island chaotic"
    )
    assert (codes == 0).all()
    assert islands(lines).sizes["chain"] == 0


def test_a_seeded_island_chain_is_found_with_its_width(island_lines):
    section = poincare_section(island_lines)
    codes = classify_field_lines(section)
    inside = np.asarray(codes) == 1
    # q = m/n = 2 at r = 1/√2; ι = (1 − ε²)^{3/2}/q is 1/2 a little further in: eta1 ≈ 0.62
    starts = np.asarray(island_lines.eta1_start)
    assert inside.sum() >= 10
    assert 0.53 < starts[inside].min() and starts[inside].max() < 0.72
    assert (np.asarray(codes.m)[inside] == 2).all() and (
        np.asarray(codes.n)[inside] == 1
    ).all()
    assert (np.asarray(codes)[starts < 0.5] == 0).all()
    chains = islands(section)
    assert chains.sizes["chain"] >= 1
    main = chains.isel(chain=int(np.argmax(chains.lines.values)))
    assert int(main.n) == 1 and int(main.m) == 2
    assert 0.1 < float(main.width) < 0.2  # the locked range of ι is about 0.14 wide
    assert 0.58 < float(main.center) < 0.66
    assert (
        float(main.width_physical) > 0.5
    )  # 0.9 (r per eta1) × 0.15, a bit more off the midplane
    assert float(main.o_point_eta2) == pytest.approx(0.0, abs=0.03) or float(
        main.o_point_eta2
    ) == pytest.approx(1.0, abs=0.03)
    assert chains.to_dataframe().shape[0] == chains.sizes["chain"]


# --- open lines -------------------------------------------------------------------------------


def test_connection_lengths_and_footprints_in_a_slab():
    B = slab()
    edge = trace_field_lines(
        B,
        seeds={
            "eta1": 0.5,
            "eta2": np.linspace(0.1, 0.9, 5),
            "eta3": np.linspace(0.1, 0.9, 4),
        },
        direction="both",
        length=30.0,
    )
    assert edge.exited.all()
    # from X = 0.5 to the faces at X = 0.5/24 and 23.5/24, along B = (0.2, 0, 1)
    expected = 2 * (23.5 / 24 - 0.5) * np.sqrt(1 + 0.04) / 0.2
    np.testing.assert_allclose(edge.connection_length, expected, rtol=1e-6)
    np.testing.assert_allclose(edge.iota, 0.0, atol=1e-12)  # no poloidal advance
    hits = footprint(edge)
    assert hits.attrs["kind"] == "footprint" and set(hits.data_vars) >= {
        "eta1",
        "eta2",
        "eta3",
        "x",
        "R",
        "connection_length",
        "exited",
    }
    forward = hits.isel(line=np.flatnonzero(hits.direction.values == 1))
    np.testing.assert_allclose(forward.eta1, 23.5 / 24)
    np.testing.assert_allclose(forward.eta2, forward.eta2_start)  # B has no y component
    grid = seed_grid(edge)
    assert grid.dims == ("eta2", "eta3") and grid.shape == (5, 4)
    np.testing.assert_allclose(grid, expected, rtol=1e-6)
    with pytest.raises(ValueError, match="exactly two"):
        seed_grid(trace_field_lines(B, seeds=3, length=1.0))


def test_lines_stop_at_the_boundary_and_the_samples_end_there():
    edge = trace_field_lines(slab(), seeds=[[0.9, 0.5, 0.5]], length=30.0)
    assert bool(edge.exited.item()) and edge.eta1_end.item() == pytest.approx(23.5 / 24)
    finite = np.isfinite(edge.x.values[:, 0])
    assert finite[0] and not finite[-1]
    assert (
        float(edge.s[finite][-1])
        <= edge.length.item()
        <= edge.connection_length.item() + 1e-9
    )


# --- along the lines --------------------------------------------------------------------------


def test_a_mode_sampled_along_the_lines_gives_its_parallel_wavenumber(lines):
    B = tokamak()
    # cos(m θ − n φ) along θ = ι φ + θ₀: k∥ = |m ι − n| / R₀ (the line length is R₀ per radian)
    phi = (
        np.cos(3 * 2 * np.pi * B.eta2 - 2 * np.pi * B.eta3)
        + 0 * B.isel(component=0, drop=True)
    ).rename("phi")
    phi.attrs = {"label": r"$\phi$"}
    along = sample_along(phi, lines)
    assert along.dims == ("s", "line") and along.name == "phi"
    assert (
        "iota" in along.coords and along.attrs["label"] == r"$\phi$ along field lines"
    )
    k = parallel_wavenumber(along)
    expected = np.abs(3 * lines.iota.values - 1) / R0
    inner = (
        expected > 0.3
    )  # the inner lines, where the sidebands of the non-straight angle are weak
    np.testing.assert_allclose(k.values[inner], expected[inner], rtol=0.03)
    crossings = parallel_wavenumber(along, method="crossings")
    np.testing.assert_allclose(crossings.values[inner], expected[inner], rtol=0.03)
    # extra dimensions of the field are kept
    with_time = sample_along(
        xr.concat([phi, 2 * phi], dim="t").assign_coords(t=[0.0, 1.0]), lines
    )
    assert with_time.dims == ("t", "s", "line")
    np.testing.assert_allclose(with_time.isel(t=1), 2 * along)
    with pytest.raises(ValueError, match="scalar"):
        sample_along(B, lines)
    with pytest.raises(ValueError, match="method"):
        parallel_wavenumber(along, method="wavelet")


# --- plots ------------------------------------------------------------------------------------


def test_poincare_plots_in_both_coordinates_with_every_coloring(island_lines):
    physical = plot_poincare(
        island_lines,
        color_by="classification",
        islands_=True,
        boundary=tokamak(n1=4, n2=16, n3=4).isel(component=0),
    )
    assert isinstance(physical, PlotResult) and physical.ax.get_xlabel() == "$R$"
    assert "islands" in physical.data and physical.data["islands"].sizes["chain"] >= 1
    labels = [t.get_text() for t in physical.ax.get_legend().get_texts()]
    assert any(label.startswith("island") for label in labels)
    annotations = [a.get_text() for a in physical.artists if hasattr(a, "get_text")]
    assert any(text.startswith("1/2: w = ") for text in annotations)
    logical = plot_poincare(
        island_lines, coords="logical", color_by="iota", max_lines=5
    )
    assert (
        logical.ax.get_ylabel() == "$\\eta_1$" and len(logical.fig.axes) == 2
    )  # a color bar
    plain = plot_poincare(
        poincare_section(island_lines), color_by=None, title="punctures"
    )
    assert plain.ax.get_title() == "punctures"
    with pytest.raises(ValueError, match="coords"):
        plot_poincare(island_lines, coords="cylindrical")
    with pytest.raises(ValueError, match="color_by"):
        plot_poincare(island_lines, color_by="speed")


def test_field_line_projections(lines):
    for plane in ("RZ", "XY", "XZ", "YZ"):
        result = plot_field_lines(lines, plane=plane, color_by="iota")
        assert len(result.artists) == lines.sizes["line"]
    colored = plot_field_lines(lines, plane="RZ", color_by="absB", max_lines=2)
    assert len(colored.artists) == 2 and len(colored.fig.axes) == 2
    three_d = plot_field_lines(lines, plane="3d", max_lines=3)
    assert three_d.ax.name == "3d" and len(three_d.artists) == 3
    with pytest.raises(ValueError, match="plane"):
        plot_field_lines(lines, plane="RTheta")
    with pytest.raises(ValueError, match="2-D plane"):
        plot_field_lines(lines, plane="3d", color_by="absB")


def test_footprint_and_connection_length_plots():
    edge = trace_field_lines(
        slab(shear=0.1),
        seeds={
            "eta1": 0.5,
            "eta2": np.linspace(0.1, 0.9, 5),
            "eta3": np.linspace(0.1, 0.9, 4),
        },
        direction="both",
        length=30.0,
    )
    hits = plot_footprint(edge)
    assert hits.ax.get_title().startswith("footprint: 40 of 40 lines")
    assert "footprint" in hits.data
    mesh = plot_connection_length(edge)
    assert mesh.data["connection_length"].dims == ("eta2", "eta3")
    scattered = plot_connection_length(
        trace_field_lines(
            slab(),
            seeds=[[0.3, 0.2, 0.1], [0.5, 0.4, 0.2], [0.7, 0.8, 0.3]],
            length=30.0,
        ),
        log=False,
    )
    assert scattered.ax.get_xlabel() == "$\\eta_1$"
    with pytest.raises(ValueError, match="no line left"):
        plot_connection_length(
            trace_field_lines(tokamak(n1=8, n2=16, n3=8), seeds=2, turns=1)
        )


def test_along_and_surface_plots(lines):
    B = tokamak()
    phi = (
        np.cos(3 * 2 * np.pi * B.eta2 - 2 * np.pi * B.eta3)
        + 0 * B.isel(component=0, drop=True)
    ).rename("phi")
    along = sample_along(phi, lines)
    result = plot_along_field_lines(along, k_parallel=True, max_lines=3)
    assert len(result.artists) == 3 and "k_parallel" in result.data
    legend = [t.get_text() for t in result.ax.get_legend().get_texts()]
    assert all("$k_\\parallel$ = " in text and "$\\iota$ = " in text for text in legend)
    with pytest.raises(ValueError, match="select every dimension"):
        plot_along_field_lines(xr.concat([along, along], dim="t"))
    surface = plot_surface_map(
        phi.isel(eta1=12), iota=0.7, count=4, lines=lines.isel(line=[0]), turns=1.5
    )
    assert surface.data["iota"] == 0.7
    legend = [t.get_text() for t in surface.ax.get_legend().get_texts()]
    assert legend == ["$\\iota$ = 0.7", "traced field lines"]
    # a profile of ι is interpolated at the surface's radius
    iota = xr.DataArray([0.9, 0.5], dims="eta1", coords={"eta1": [0.0, 1.0]})
    profile = plot_surface_map(phi.isel(eta1=12), iota=iota)
    assert profile.data["iota"] == pytest.approx(0.9 - 0.4 * float(phi.eta1[12]))
    with pytest.raises(ValueError, match="exactly the angles"):
        plot_surface_map(phi)


def test_the_accessors_reach_the_field_line_tools():
    B = tokamak(n1=12, n2=24, n3=16).expand_dims(t=[0.0, 1.0])
    lines = B.plasma.analysis.field_lines(seeds=3, turns=3, t=-1)
    assert lines.sizes["line"] == 3
    assert isinstance(lines.plasma.plot.poincare(), PlotResult)
    assert isinstance(lines.plasma.plot.field_lines(), PlotResult)
    assert lines.plasma.analysis.poincare_section().sizes["puncture"] == 3
    assert lines.plasma.data.poincare().sizes["line"] == 3
    assert lines.plasma.analysis.rotational_transform().dims == ("line",)
    assert lines.plasma.analysis.classify_field_lines().dims == ("line",)
    assert lines.plasma.analysis.islands().sizes["chain"] == 0
    assert lines.plasma.analysis.footprint().sizes["line"] == 3
    assert lines.plasma.data.footprint().sizes["line"] == 3
    result = B.plasma.plot.poincare(seeds=2, turns=2, t=-1)
    assert "lines" in result.data and result.data["lines"].sizes["line"] == 2
    assert B.plasma.data.poincare(seeds=2, turns=2, t=-1).sizes["line"] == 2
    phi = (B.isel(component=0, drop=True) * 0 + np.cos(2 * np.pi * B.eta2)).rename(
        "phi"
    )
    along = phi.plasma.analysis.sample_along(lines)
    assert along.dims == ("t", "s", "line")
    assert phi.plasma.analysis.parallel_wavenumber(lines=lines).dims == ("t", "line")
    assert along.isel(t=0).plasma.analysis.parallel_wavenumber().dims == ("line",)
    assert isinstance(phi.plasma.plot.along_field_lines(lines, t=-1), PlotResult)
    assert isinstance(phi.plasma.plot.surface_map(eta1=0.5, t=-1, iota=0.6), PlotResult)
    edge = slab().plasma.analysis.field_lines(
        seeds={"eta1": 0.5, "eta2": [0.2, 0.4], "eta3": [0.1, 0.9]},
        direction="both",
        length=30.0,
    )
    assert edge.plasma.analysis.seed_grid().shape == (2, 2)
    assert isinstance(edge.plasma.plot.footprint(), PlotResult)
    assert isinstance(edge.plasma.plot.connection_length(), PlotResult)
