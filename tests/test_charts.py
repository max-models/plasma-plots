"""Charts of any labeled data: scans, breakdowns, twin and stacked profiles, cross sections,
matrices, the Child–Langmuir comparison and Dataset overviews."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402,F401  (registers the accessors)
from plasma_plots.charts import (  # noqa: E402
    matrix_entries,
    plot_breakdown,
    plot_child_langmuir,
    plot_cross_sections,
    plot_matrix,
    plot_scan,
    plot_stacked,
    plot_twin,
    plot_variables,
    prepare_breakdown,
    prepare_scan,
)
from plasma_plots.theory.parameters import proton_mass  # noqa: E402
from plasma_plots.theory.sheaths import child_langmuir  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def scan():
    voltage = np.linspace(1.0, 5.0, 5)
    seed = np.arange(4)
    values = 2.0 * voltage[:, None] + 0.1 * (seed[None, :] - 1.5)
    return xr.DataArray(
        values,
        dims=("voltage", "seed"),
        coords={"voltage": ("voltage", voltage, {"units": "kV"}), "seed": seed},
        name="divergence",
        attrs={"units": "mrad"},
    )


def profile(name, values, x=None, **attrs):
    x = np.linspace(0.0, 1.0, len(values)) if x is None else x
    return xr.DataArray(
        np.asarray(values, float), dims="x", coords={"x": x}, name=name, attrs=attrs
    )


# --- scans -----------------------------------------------------------------------------------


def test_a_scan_over_samples_draws_the_mean_and_the_spread(scan):
    result = plot_scan(scan, samples="seed")
    data = result.data["scan"]
    np.testing.assert_allclose(data.value, 2.0 * scan.voltage)
    np.testing.assert_allclose(data.lower, scan.std("seed"))
    assert result.ax.get_xlabel().startswith("voltage")
    assert result.ax.get_ylabel().endswith("[mrad]") or "mrad" in result.ax.get_ylabel()


def test_a_scan_draws_a_band_and_a_reference(scan):
    result = plot_scan(
        scan, samples="seed", band=True, label="PIC", reference=lambda v: 2 * v
    )
    assert any(type(a).__name__ == "Polygon" for a in result.artists)
    assert [t.get_text() for t in result.ax.get_legend().get_texts()] == [
        "PIC",
        "exact",
    ]


def test_a_scan_takes_explicit_asymmetric_errors(scan):
    mean = scan.mean("seed")
    data = prepare_scan(mean, error=(0.1 * np.ones(5), 0.3 * np.ones(5)))
    np.testing.assert_allclose(data.upper, 0.3)
    result = plot_scan(mean, error=0.2, logx=True, logy=True)
    assert result.ax.get_xscale() == "log" and result.ax.get_yscale() == "log"


def test_a_scan_without_errors_is_a_line(scan):
    result = plot_scan(scan.isel(seed=0))
    assert np.isnan(result.data["scan"].lower).all()


def test_a_scan_checks_its_dimensions(scan):
    with pytest.raises(ValueError, match="one dimension left"):
        prepare_scan(scan)
    with pytest.raises(ValueError, match="not a dimension"):
        prepare_scan(scan, samples="shot")
    with pytest.raises(ValueError, match="not the scanned"):
        prepare_scan(scan, samples="seed", x="gap")


def test_the_scan_accessor_and_its_data(scan):
    result = scan.plasma.plot.scan(samples="seed")
    assert result.data["scan"].sizes["voltage"] == 5
    data = scan.plasma.data.scan(seed=0)
    assert np.isnan(data.lower).all()


# --- breakdowns ------------------------------------------------------------------------------


def test_a_breakdown_from_a_dict_annotates_shares():
    result = plot_breakdown({"duct": 1.0, "dump": 3.0}, total=True)
    labels = [t.get_text() for t in result.ax.get_yticklabels()]
    assert labels == ["duct", "dump", "total"]  # from the top down
    notes = [t.get_text() for t in result.ax.texts]
    assert any("75%" in note for note in notes)
    assert len(result.artists) == 3


def test_a_breakdown_from_an_array_sorts_and_scales():
    power = xr.DataArray(
        [1e6, 4e6, 2e6],
        dims="component",
        coords={"component": ["duct", "dump", "neutralizer"]},
        name="P",
        attrs={"units": "W"},
    )
    result = power.plasma.plot.breakdown(sort=True, scale=1e-6, xlabel="P (MW)")
    assert [t.get_text() for t in result.ax.get_yticklabels()] == [
        "dump",
        "neutralizer",
        "duct",
    ]
    assert result.ax.patches[0].get_width() == pytest.approx(4.0)
    assert list(power.plasma.data.breakdown().component.values) == list(
        power.component.values
    )


def test_a_breakdown_of_a_dataset_of_scalars():
    deposited = xr.Dataset(
        {
            "duct": ((), 2.0, {"units": "MW"}),
            "dump": ((), 5.0, {"units": "MW", "label": "beam dump"}),
        }
    )
    parts = prepare_breakdown(deposited)
    assert list(parts.part.values) == ["duct", "beam dump"]
    assert parts.attrs["units"] == "MW"
    result = deposited.plasma.plot.breakdown(annotate=False)
    assert not result.ax.texts


def test_a_breakdown_rejects_profiles():
    with pytest.raises(ValueError, match="scalar"):
        prepare_breakdown(xr.Dataset({"a": ("x", [1.0, 2.0])}))
    with pytest.raises(ValueError, match="one-dimensional"):
        prepare_breakdown(xr.DataArray(np.ones((2, 2))))
    with pytest.raises(ValueError, match="not the array"):
        prepare_breakdown(xr.DataArray([1.0], dims="a"), dim="b")
    unnamed = prepare_breakdown(xr.DataArray([1.0, 2.0], dims="a"))
    assert list(unnamed.a.values) == ["0", "1"]


# --- twin and stacked profiles ---------------------------------------------------------------


def test_twin_axes_color_their_labels():
    n = profile("n", [1.0, 2.0, 3.0], units="m^-3")
    T = profile("T", [5.0, 4.0, 3.0], units="eV")
    result = plot_twin(n, T, logy=(True, False), step="mid")
    left, right = result.ax
    assert left.get_yscale() == "log" and right.get_yscale() == "linear"
    assert result.artists[0].get_drawstyle() == "steps-mid"
    assert matplotlib.colors.same_color(right.yaxis.label.get_color(), "C3")


def test_the_twin_accessor_selects_both():
    t = np.array([0.0, 1.0])
    n = xr.DataArray(
        np.ones((2, 3)), dims=("t", "x"), coords={"t": t, "x": [0, 1, 2]}, name="n"
    )
    T = xr.DataArray(np.ones(3), dims="x", coords={"x": [0, 1, 2]}, name="T")
    result = n.plasma.plot.twin(T, t=-1)
    assert len(result.ax) == 2
    with pytest.raises(ValueError, match="one-dimensional"):
        plot_twin(n, T)


def test_stacked_panels_share_x_and_take_several_lines_per_row():
    n = profile("n", [1.0, 2.0, 3.0])
    Te = profile("T_e", [5.0, 4.0, 3.0])
    Ti = profile("T_i", [2.0, 2.0, 2.0])
    result = plot_stacked(
        [n, {"T_e": Te, "T_i": Ti}, [Te, Ti]], logy=[False, True, False]
    )
    assert len(result.ax) == 3
    assert result.ax[1].get_yscale() == "log"
    assert [t.get_text() for t in result.ax[1].get_legend().get_texts()] == [
        "T_e",
        "T_i",
    ]
    assert result.ax[0].get_xlabel() == ""
    result = n.plasma.plot.stacked({"T_e": Te}, [Ti], title="profiles")
    assert result.fig._suptitle.get_text() == "profiles"
    with pytest.raises(ValueError, match="at least one"):
        plot_stacked([])


def test_a_dataset_stacks_one_row_per_variable():
    ds = xr.Dataset({"n": profile("n", [1, 2, 3]), "T": profile("T", [3, 2, 1])})
    assert len(plot_stacked(ds).ax) == 2


# --- cross sections --------------------------------------------------------------------------


def test_cross_sections_of_a_dataset_with_a_reference():
    energy = np.geomspace(1.0, 1e3, 12)
    catalogue = xr.Dataset(
        {
            "ionization": ("energy", 1e-20 * np.log(energy + 1)),
            "charge exchange": ("energy", 1e-19 / np.sqrt(energy)),
        },
        coords={"energy": ("energy", energy, {"units": "eV"})},
    )
    result = catalogue.plasma.plot.cross_sections(
        reference={"fit": lambda e: 1e-19 / np.sqrt(e)}
    )
    assert result.ax.get_xscale() == "log" and result.ax.get_yscale() == "log"
    assert [t.get_text() for t in result.ax.get_legend().get_texts()] == [
        "ionization",
        "charge exchange",
        "fit",
    ]
    assert "eV" in result.ax.get_xlabel()


def test_cross_sections_from_pairs_drop_non_positive_values_on_a_log_axis():
    energy = np.array([0.0, 1.0, 10.0, 100.0])
    result = plot_cross_sections(
        {"a": (energy, np.array([0.0, 1.0, 2.0, 3.0]))}, markers=False
    )
    assert len(result.artists[0].get_xdata()) == 3
    assert result.ax.get_xlabel() == "energy"
    linear = plot_cross_sections((energy, energy + 1), logx=False, logy=False)
    assert linear.ax.get_xscale() == "linear"


def test_cross_sections_over_a_process_dimension():
    sigma = xr.DataArray(
        np.ones((10, 2)),
        dims=("energy", "process"),
        coords={"energy": np.geomspace(1, 100, 10), "process": ["a", "b"]},
        name="sigma",
    )
    result = sigma.plasma.plot.cross_sections()
    assert len(result.artists) == 2
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        plot_cross_sections(xr.DataArray(np.ones((2, 2, 2))))


# --- matrices --------------------------------------------------------------------------------


def laplacian(n=12):
    return (
        np.diag(-2.0 * np.ones(n))
        + np.diag(np.ones(n - 1), 1)
        + np.diag(np.ones(n - 1), -1)
    )


def test_a_matrix_pattern_counts_its_entries_and_marks_ranks():
    result = plot_matrix(laplacian(), rank_boundaries=[6])
    assert result.data == {"shape": (12, 12), "nnz": 34}
    assert [t.get_text() for t in result.ax.get_legend().get_texts()] == ["> 0", "< 0"]
    assert len(result.ax.lines) == 2  # one dashed line each way


def test_matrix_entries_from_every_form():
    dense = laplacian(4)
    picnic = {"row_indices": [0, 1], "col_indices": [1, 0], "values": [1.0, -1.0]}
    rows, cols, values, shape = matrix_entries(picnic)
    assert shape == (2, 2) and list(values) == [1.0, -1.0]
    assert matrix_entries((np.array([0, 2]), np.array([1, 1]), np.array([3, 4])))[
        3
    ] == (3, 2)
    scipy = pytest.importorskip("scipy.sparse")
    assert matrix_entries(scipy.csr_matrix(dense))[3] == (4, 4)
    with pytest.raises(ValueError, match="two-dimensional"):
        matrix_entries(np.ones(3))


def test_matrix_values_and_the_accessor():
    array = xr.DataArray(laplacian(5), dims=("row", "col"))
    result = array.plasma.plot.matrix(kind="values")
    assert result.data["nnz"] == 13
    with pytest.raises(ValueError, match="kind"):
        plot_matrix(laplacian(), kind="spy")


# --- the Child–Langmuir law ------------------------------------------------------------------


def test_child_langmuir_over_voltage_and_over_gap():
    voltage = np.linspace(100.0, 1000.0, 6)
    gap = 0.01
    J = xr.DataArray(
        child_langmuir(voltage, gap, mass=proton_mass) * 1.02,
        dims="V",
        coords={"V": ("V", voltage, {"units": "V"})},
        name="J",
        attrs={"units": "A/m^2"},
    )
    result = J.plasma.plot.child_langmuir(gap=gap, mass=proton_mass)
    assert len(result.ax) == 2  # the error panel
    error = result.ax[1].lines[0].get_ydata()
    np.testing.assert_allclose(error, 0.02, rtol=1e-6)
    d = np.linspace(0.005, 0.02, 5)
    over_gap = plot_child_langmuir(
        {"pic": (d, child_langmuir(500.0, d))}, voltage=500.0, show_error=False
    )
    assert over_gap.ax.get_xlabel() == "d (m)"
    two = J.plasma.plot.child_langmuir(J.rename("J2"), gap=gap, mass=proton_mass)
    assert len(two.ax[1].lines) >= 2
    with pytest.raises(ValueError, match="exactly one"):
        plot_child_langmuir(J)


# --- Datasets --------------------------------------------------------------------------------


def test_variables_draws_profiles_and_slices():
    t = np.array([0.0, 1.0])
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 4)
    state = xr.Dataset(
        {
            "n": (("t", "x"), np.ones((2, 5))),
            "phi": (("t", "x", "y"), np.random.default_rng(0).random((2, 5, 4))),
            "E": (("x", "y"), np.ones((5, 4))),
        },
        coords={"t": t, "x": x, "y": y},
        attrs={"run": "test run"},
    )
    result = state.plasma.plot.variables(t=-1, cols=2, y="x")
    assert result.ax.shape == (2, 2)
    assert not result.ax[1, 1].get_visible()
    assert result.fig._suptitle.get_text() == "test run"
    assert result.ax[0, 1].get_xlabel() == "$y$"
    assert len(result.data["results"]) == 3
    one = plot_variables(state, ["n"], t=0.9, logy=True)
    assert one.ax[0, 0].get_yscale() == "log"
    with pytest.raises(ValueError, match="left"):
        plot_variables(state, ["phi"])
    with pytest.raises(ValueError, match="not data variables"):
        plot_variables(state, ["T"])
    with pytest.raises(ValueError, match="no data variables"):
        plot_variables(xr.Dataset())
