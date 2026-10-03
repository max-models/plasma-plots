"""The plotting and analysis layer on real struphy output: small simulations, run on demand.

Skipped by default (they need compiled struphy kernels and take about a minute); run with
``pytest --run-simulations``. The docs workflow, which compiles the kernels anyway, runs them.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

pytestmark = pytest.mark.simulation

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

import plasma_plots  # noqa: E402, F401
from plasma_plots import pyvista_plots as p3  # noqa: E402


@pytest.fixture(scope="module")
def torus_run(tmp_path_factory):
    """LinearMHD in a sixth of a hollow torus, seeded with the m = 10, 11 harmonics of the
    TAE tutorial (coarse and short: seconds)."""
    from struphy import (
        BaseUnits,
        DerhamOptions,
        EnvironmentOptions,
        Time,
        domains,
        equils,
        grids,
        perturbations,
    )
    from struphy.models import LinearMHD
    from struphy.simulation.sim import Simulation

    model = LinearMHD(base_units=BaseUnits())
    model.propagators.shear_alf.options = model.propagators.shear_alf.Options()
    model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options()
    for field in (
        model.em_fields.b_field,
        model.mhd.density,
        model.mhd.velocity,
        model.mhd.pressure,
    ):
        field.save_data = True
    modes = (10, 11)
    model.mhd.velocity.add_perturbation(
        perturbations.TorusModesSin(
            ms=modes,
            ns=(-1, -1),
            amps=(1e-3, 1e-3),
            pfuns=("exp", "exp"),
            pfun_params=([0.5, 0.1], [0.5, 0.1]),
            comp=0,
            given_in_basis="2",
        )
    )
    model.mhd.velocity.add_perturbation(
        perturbations.TorusModesCos(
            ms=modes,
            ns=(-1, -1),
            amps=tuple(1e-3 / (2 * np.pi * m) for m in modes),
            pfuns=("d_exp", "d_exp"),
            pfun_params=([0.5, 0.1], [0.5, 0.1]),
            comp=1,
            given_in_basis="2",
        )
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(
            out_folders=str(tmp_path_factory.mktemp("runs")), sim_folder="torus"
        ),
        time_opts=Time(dt=0.5, Tend=4.0),
        domain=domains.HollowTorus(a1=0.1, a2=1.0, R0=10.0, tor_period=6),
        equil=equils.AdhocTorus(
            a=1.0,
            R0=10.0,
            B0=3.0,
            q_kind=0,
            q0=1.71,
            q1=1.87,
            n1=2.0,
            n2=1.0,
            na=0.2,
            p_kind=1,
            p1=0.95,
            p2=0.05,
            beta=0.0018,
        ),
        grid=grids.TensorProductGrid(num_elements=(6, 32, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    return sim.run().pproc(physical=True)


@pytest.fixture(scope="module")
def orbit_run(tmp_path_factory):
    """GuidingCenter markers in a tokamak-like torus, 60 of them with saved orbits."""
    from struphy import (
        BaseUnits,
        BoundaryParameters,
        DerhamOptions,
        EnvironmentOptions,
        LoadingParameters,
        SavingParameters,
        SortingParameters,
        Time,
        WeightsParameters,
        domains,
        equils,
        grids,
        maxwellians,
    )
    from struphy.models import GuidingCenter
    from struphy.simulation.sim import Simulation

    model = GuidingCenter(base_units=BaseUnits())
    model.kinetic_ions.var.save_data = True
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(
            out_folders=str(tmp_path_factory.mktemp("runs")),
            sim_folder="orbits",
            save_step=2,
        ),
        time_opts=Time(dt=0.1, Tend=60.0),
        domain=domains.HollowTorus(a1=0.1, a2=1.0, R0=3.0, tor_period=1),
        equil=equils.AdhocTorus(
            a=1.0, R0=3.0, B0=1.0, q_kind=0, q0=1.1, q1=1.9, p_kind=1, beta=0.001
        ),
        grid=grids.TensorProductGrid(num_elements=(8, 16, 8)),
        derham_opts=DerhamOptions(degree=(2, 2, 2)),
    )
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(Np=400, seed=1),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(),
        saving_params=SavingParameters(n_markers=60),
    )
    model.propagators.push_bxe.options = model.propagators.push_bxe.Options()
    model.propagators.push_parallel.options = model.propagators.push_parallel.Options()
    model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(1.0, None)))
    return sim.run().pproc()


def test_mode_spectrum_finds_the_seeded_torus_harmonics(torus_run):
    velocity = torus_run.evaluate("mhd/velocity").isel(t=0, component=0)
    top = (
        velocity.plasma.analysis.mode_spectrum()
        .plasma.analysis.mode_amplitudes(top=2)
        .max("eta1")
    )
    assert set(top.mode.values) == {"(10, -1)", "(11, -1)"}


def test_push_forward_matches_struphys_cartesian_product(torus_run):
    cartesian = torus_run.evaluate("mhd/velocity_xyz").isel(t=0)
    etas = [np.asarray(cartesian[d]) for d in ("eta1", "eta2", "eta3")]
    contravariant = torus_run.evaluate(
        "mhd/velocity",
        eta1=etas[0],
        eta2=etas[1],
        eta3=etas[2],
        representation="v",
        t=0,
    ).isel(t=0)
    pushed = p3.push_forward(contravariant)
    reference = cartesian.transpose(*pushed.dims)
    # spectral derivatives around the periodic angles make this exact up to round-off
    assert float(abs(pushed - reference).max() / abs(reference).max()) < 1e-10


def test_cell_centred_fields_close_the_periodic_seam(torus_run):
    velocity = torus_run.evaluate("mhd/velocity").isel(t=-1, component=0)
    grid = p3.structured_grid(velocity)
    assert grid.dimensions == (
        velocity.sizes["eta1"],
        velocity.sizes["eta2"] + 1,
        velocity.sizes["eta3"],
    )
    plane = p3.structured_grid(velocity.isel(eta3=0))
    assert plane.dimensions[1] == velocity.sizes["eta2"] + 1


def test_every_3d_view_renders_on_real_output(torus_run):
    velocity = torus_run.evaluate("mhd/velocity").isel(t=-1, component=0)
    cartesian = torus_run.evaluate("mhd/velocity_xyz").isel(t=-1)
    views = {
        "isosurface": velocity.plasma.plot.isosurface(values=4),
        "slices": velocity.plasma.plot.slices_3d(
            cuts={"eta3": [0.0, 0.5], "eta1": 0.5}
        ),
        "plane": velocity.isel(eta3=0).plasma.plot.isosurface(values=4),
        "glyphs": cartesian.plasma.plot.glyphs(stride=2),
        "streamlines": cartesian.plasma.plot.streamlines(n_points=40),
        "domain": torus_run.plot.domain_3d(n1=4, n2=16, n3=6),
    }
    lines = views["streamlines"].renderer.actors["streamlines"].mapper.dataset
    assert lines.n_lines > 10
    for plotter in views.values():
        plotter.close()


def test_orbit_classification_on_real_guiding_center_orbits(orbit_run):
    orbits = orbit_run.evaluate("kinetic_ions/orbits")
    assert {"x", "y", "z", "v_par", "mu"} <= set(orbits.data_vars)
    codes = orbits.plasma.analysis.classify_orbits()
    trapped = codes == 1
    assert 0 < int(trapped.sum()) < orbits.sizes["marker"]
    flipped = (orbits.v_par * orbits.v_par.isel(t=0) < 0).any("t")
    assert bool((trapped == (flipped & (codes != -1))).all())
    # trapped markers start with a smaller |v_par| relative to their mu: the trapped cone
    start = orbits.isel(t=0)
    pitch = abs(start.v_par) / np.sqrt(start.mu + 1e-12)
    assert float(pitch.where(trapped).median()) < float(pitch.where(~trapped).median())
    assert orbits.plasma.plot.orbit_classification().data["counts"]["trapped"] == int(
        trapped.sum()
    )
    orbits.plasma.plot.orbits_3d(color_by="classification").close()
    poloidal = orbits.plasma.plot.poloidal()
    assert {"passing", "trapped"} <= {line.get_label() for line in poloidal.ax.lines}
    quantities = orbits.plasma.plot.quantities(markers=4)
    # mu is an invariant of guiding-centre motion: its drift stays small
    assert max(abs(line.get_ydata()).max() for line in quantities.ax[1].lines) < 1e-6


def test_linear_mhd_energies_from_fields_match_the_saved_scalars(torus_run):
    energies = torus_run.analysis.linear_mhd_energies()
    for name in ("en_U", "en_B", "en_thermal", "en_tot"):
        saved = torus_run.scalars[name].interp(t=energies.t)
        assert float(abs(energies[name] - saved).max() / abs(saved).max()) < 1e-10, name
    # the energy of the filtered dominant mode is a part of the total
    etas, _ = torus_run.analysis.quadrature_grid()
    velocity = torus_run.evaluate(
        "mhd/velocity",
        eta1=etas["eta1"],
        eta2=etas["eta2"],
        eta3=etas["eta3"],
        representation="2",
    )
    filtered = velocity.plasma.analysis.filter_time(pad_bins=1).filtered
    mode = torus_run.analysis.linear_mhd_energies(
        velocity=filtered, b_field=None, pressure=None
    )
    assert 0 < float(mode.en_U.max()) <= 1.5 * float(energies.en_U.max())


def test_physical_slices_and_vtk_export_on_real_output(torus_run, tmp_path):
    velocity = torus_run.evaluate("mhd/velocity").isel(component=0)
    result = velocity.plasma.plot.slice(
        x="eta1", y="eta2", coords="physical", plane="RZ", t=-1, eta3=0, symmetric=True
    )
    mesh = result.artists[0]
    assert (
        mesh.get_coordinates().shape[1] == velocity.sizes["eta2"] + 2
    )  # seam closed: 33 points, 34 cell edges
    paths = velocity.plasma.data.to_vtk(tmp_path / "velocity")
    assert paths[0].endswith(".pvd") and len(paths) == velocity.sizes["t"] + 1


def test_linear_mhd_two_alfven_modes(tmp_path):
    """Ported from struphy's postprocessing-fft branch: the dominant-band filter separates two
    shear-Alfven modes of a real LinearMHD run, through out.analysis."""
    from struphy import (
        DerhamOptions,
        EnvironmentOptions,
        Time,
        domains,
        equils,
        grids,
        perturbations,
    )
    from struphy.models import LinearMHD
    from struphy.simulation.sim import Simulation

    model = LinearMHD()
    model.mhd.velocity.add_perturbation(
        perturbations.ModesSin(
            ns=(1, 3), amps=(1e-3, 3e-4), Lz=20, comp=0, given_in_basis="physical"
        )
    )
    simulation = Simulation(
        model=model,
        env=EnvironmentOptions(
            out_folders=str(tmp_path), sim_folder="two_modes", save_step=2
        ),
        time_opts=Time(dt=0.1, Tend=100),
        domain=domains.Cuboid(r3=20),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 32)),
        derham_opts=DerhamOptions(degree=(1, 1, 3)),
        equil=equils.HomogenSlab(B0x=0, B0y=0, B0z=1, beta=1, n0=1),
    )
    output = simulation.run().pproc(physical=True)
    for product in ("mhd/velocity", "mhd/velocity_xyz"):
        velocity = output.evaluate(product).isel(component=0, eta1=0, eta2=0)
        result = output.analysis.filter_time(velocity, pad_bins=2)
        spectrum = result.spectrum
        assert (
            abs(float(spectrum.dominant_frequency) - 2 * np.pi / 20)
            < spectrum.attrs["frequency_resolution"]
        )
        assert float(spectrum.omega_hi) < 3 * 2 * np.pi / 20

        def mode_amplitude(field, n):
            return (
                abs((field * np.sin(2 * np.pi * n * field.eta3)).sum("eta3"))
                .max()
                .item()
            )

        raw_ratio = mode_amplitude(velocity, 3) / mode_amplitude(velocity, 1)
        filtered_ratio = mode_amplitude(result.filtered, 3) / mode_amplitude(
            result.filtered, 1
        )
        assert raw_ratio > 0.2
        assert filtered_ratio < 0.05 * raw_ratio
        assert output.analysis.time_fft(velocity).attrs[
            "sample_spacing"
        ] == pytest.approx(0.2)


def test_the_command_line_on_real_runs(torus_run, orbit_run, tmp_path):
    command = shutil.which("plasma-plots")
    assert command, "plasma-plots is not on the PATH: pip install -e ."

    def run(*args):
        return subprocess.run(
            [command, *map(str, args)], check=True, capture_output=True, text=True
        )

    assert "mhd/velocity" in run("info", torus_run.path_out).stdout
    run("quicklook", torus_run.path_out, "-o", tmp_path / "torus")
    assert {
        "energies.png",
        "scalars.png",
        "equilibrium.png",
        "mhd-velocity-slice.png",
        "em_fields-b_field-slice.png",
    } <= set(os.listdir(tmp_path / "torus"))
    run("quicklook", orbit_run.path_out, "-o", tmp_path / "orbits")
    assert "kinetic_ions-trajectories.png" in os.listdir(tmp_path / "orbits")
    physical = tmp_path / "pressure.png"
    run(
        "plot",
        torus_run.path_out,
        "mhd/pressure",
        "slice",
        "coords=physical",
        "plane=XZ",
        "t=-1",
        "eta3=0",
        "-o",
        physical,
    )
    assert physical.stat().st_size > 0


def test_equilibrium_plots_of_a_reopened_run(torus_run):
    # out.equil restored from the run's metadata has no domain until the plot attaches it
    from struphy.post_processing.output import Output

    reopened = Output(torus_run.path_out)
    result = reopened.plot.equilibrium()
    assert len(result.ax.lines) >= 1
    assert np.isfinite(result.ax.lines[0].get_ydata()).all()
    reopened.plot.equilibrium_3d(scalars="p0").close()
