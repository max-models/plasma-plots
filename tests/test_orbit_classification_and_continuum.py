"""Tests for orbit classification and continuous-spectrum plots."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
import struphy_plots  # noqa: E402, F401  (registers the accessors)
from struphy.dispersion_relations.analytic import (  # noqa: E402
    MhdContinousSpectraCylinder, MhdContinousSpectraShearedSlab)
from struphy_plots.analysis import classify_orbits  # noqa: E402
from struphy_plots.plotting import (plot_continuous_spectrum,  # noqa: E402
                                    plot_orbit_classification,
                                    prepare_continuous_spectrum)


def guiding_center_orbits():
    """Four markers: passing, trapped (v_par reverses), lost (zeroed at t=2), and passing again."""
    t = np.arange(4.0)
    v_par = np.array(
        [
            [1.0, -0.5, 2.0, 0.3],
            [1.0, 0.5, 2.0, 0.3],
            [1.0, 0.2, 0.0, 0.3],
            [1.0, 0.1, 0.0, 0.3],
        ]
    )
    mu = np.array(
        [
            [0.1, 0.9, 0.5, 0.2],
            [0.1, 0.9, 0.5, 0.2],
            [0.1, 0.9, 0.0, 0.2],
            [0.1, 0.9, 0.0, 0.2],
        ]
    )
    x = np.where(mu == 0, 0.0, 1.0)
    return xr.Dataset(
        {
            "x": (("t", "marker"), x),
            "v_par": (("t", "marker"), v_par, {"label": r"$v_\parallel$"}),
            "mu": (("t", "marker"), mu, {"label": r"$\mu$"}),
        },
        coords={"t": t, "marker": np.arange(4)},
        attrs={"product": "orbits"},
    )


def test_classify_orbits_matches_struphy_criteria():
    codes = classify_orbits(guiding_center_orbits())
    assert codes.dims == ("marker",)
    assert codes.values.tolist() == [0, 1, -1, 0]
    assert codes.attrs["flag_meanings"] == "passing trapped lost"
    assert codes.equals(guiding_center_orbits().struphy.analysis.classify_orbits())


def test_classify_orbits_requires_parallel_velocity():
    with pytest.raises(ValueError, match="parallel velocity"):
        classify_orbits(guiding_center_orbits().drop_vars("v_par"))


def test_plot_orbit_classification_colors_each_class_and_counts_them():
    orbits = guiding_center_orbits()
    result = plot_orbit_classification(orbits)
    assert result.data["counts"] == {"passing": 2, "trapped": 1, "lost": 1}
    assert len(result.artists) == 3
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert labels[0].startswith("passing (2, 50%)")
    assert result.ax.get_ylabel().startswith(r"$\mu$")

    data = orbits.struphy.data.orbit_classification()
    assert set(data.data_vars) == {"v_par", "mu", "classification"}
    assert data["v_par"].values.tolist() == [1.0, -0.5, 2.0, 0.3]  # initial time by default
    assert len(orbits.struphy.plot.orbit_classification(t=-1).artists) == 3


def test_orbit_classification_defaults_to_v_perp_without_mu():
    orbits = guiding_center_orbits().rename({"mu": "v_perp"})
    data = orbits.struphy.data.orbit_classification()
    assert "v_perp" in data.data_vars
    with pytest.raises(ValueError, match="not data variables"):
        guiding_center_orbits().struphy.data.orbit_classification(x="p_phi")


def test_prepare_continuous_spectrum_evaluates_struphy_slab_continua():
    spectrum = MhdContinousSpectraShearedSlab()
    x = np.linspace(0.0, 1.0, 50)
    data = prepare_continuous_spectrum(spectrum, x, [(1, -1), (2, -1)])
    assert data.dims == ("mode", "branch", "x")
    assert data.mode.values.tolist() == ["1, -1", "2, -1"]
    assert data.branch.values.tolist() == ["shear_Alfvén", "slow_sound"]
    expected = spectrum(x, 2, -1)["shear_Alfvén"]
    np.testing.assert_allclose(data.sel(mode="2, -1", branch="shear_Alfvén"), expected)
    # the slow-sound continuum lies below the shear-Alfvén one everywhere
    assert (data.sel(branch="slow_sound") <= data.sel(branch="shear_Alfvén")).all()


def test_plot_continuous_spectrum_draws_every_branch_and_marks_frequencies():
    r = np.linspace(0.05, 1.0, 40)
    result = plot_continuous_spectrum(
        MhdContinousSpectraCylinder(),
        r,
        [(1, -1), (2, -1), (3, -1)],
        frequencies={"TAE": 0.2},
        xlabel="r",
    )
    assert len(result.artists) == 3 * 2 + 1
    assert result.data["spectrum"].sizes == {"mode": 3, "branch": 2, "x": 40}
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert "shear Alfvén, (m, n) = (1, -1)" in labels
    assert "TAE" in labels


def test_prepare_continuous_spectrum_accepts_any_callable_and_scalar_modes():
    def sound(x, k):
        return {"sound": k * np.ones_like(x)}

    data = prepare_continuous_spectrum(sound, [0.0, 1.0], [1, 2])
    assert data.sel(mode="2", branch="sound").values.tolist() == [2.0, 2.0]
    with pytest.raises(ValueError, match="at least one mode"):
        prepare_continuous_spectrum(sound, [0.0], [])
