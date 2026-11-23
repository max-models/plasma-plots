"""openPMD files: particles as marker Datasets (from_openpmd), meshes on their grids (openpmd_meshes).

Most files are written by openPMD-api, the standard's reference implementation; one is written
by hand with h5py and checked by the openPMD validator, so the reader is tested without
openPMD-api too.
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import plasma_plots  # noqa: E402
from plasma_plots.openpmd import SPEED_OF_LIGHT, units_of  # noqa: E402

h5py = pytest.importorskip("h5py")

E = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
ITERATIONS = (0, 100, 200)


def _electrons(n: int, iteration: int):
    """Positions, momenta and ids of the electrons at an iteration: one lost at each step."""
    keep = n - ITERATIONS.index(iteration)
    k = np.arange(keep, dtype=float)
    x = 1e-3 * k + 1e-6 * iteration
    pz = M_E * SPEED_OF_LIGHT * (0.1 + 0.2 * k)  # up to relativistic
    return x, pz, np.arange(keep, dtype=np.uint64) + 7


def _write_series(path, *, encoding="groupBased"):
    """Electrons (with ids, a position offset, constant charge and mass) and ions, plus meshes."""
    io = pytest.importorskip("openpmd_api")
    series = io.Series(str(path), io.Access.create)
    series.iteration_encoding = getattr(io.Iteration_Encoding, {"groupBased": "group_based", "fileBased": "file_based"}[encoding])  # fmt: skip

    def store(component, values):
        values = np.ascontiguousarray(values)
        component.reset_dataset(io.Dataset(values.dtype, values.shape))
        component.store_chunk(values)

    def constant(component, value, count):
        component.reset_dataset(io.Dataset(np.dtype("float64"), [count]))
        component.make_constant(value)

    for iteration in ITERATIONS:
        it = series.iterations[iteration]
        it.time = float(iteration)
        it.dt = 1.0
        it.time_unit_SI = 1e-12
        x, pz, ids = _electrons(6, iteration)
        n = x.size
        electrons = it.particles["electrons"]
        store(electrons["position"]["x"], x)
        store(electrons["position"]["z"], np.full(n, 2e-3))
        electrons["position"].unit_dimension = {io.Unit_Dimension.L: 1}
        constant(electrons["positionOffset"]["x"], 0.5e-3, n)
        constant(electrons["positionOffset"]["z"], 0.0, n)
        electrons["positionOffset"].unit_dimension = {io.Unit_Dimension.L: 1}
        store(electrons["momentum"]["x"], np.zeros(n))
        store(electrons["momentum"]["z"], pz)
        electrons["momentum"].unit_dimension = {
            io.Unit_Dimension.L: 1, io.Unit_Dimension.M: 1, io.Unit_Dimension.T: -1,
        }  # fmt: skip
        store(electrons["weighting"][io.Record_Component.SCALAR], np.full(n, 1e6))
        constant(electrons["charge"][io.Record_Component.SCALAR], -E, n)
        electrons["charge"].unit_dimension = {
            io.Unit_Dimension.T: 1, io.Unit_Dimension.I: 1,
        }  # fmt: skip
        constant(electrons["mass"][io.Record_Component.SCALAR], M_E, n)
        electrons["mass"].unit_dimension = {io.Unit_Dimension.M: 1}
        store(electrons["id"][io.Record_Component.SCALAR], ids)

        ions = it.particles["ions"]
        store(ions["position"]["x"], np.array([1e-3, 2e-3]))
        store(ions["position"]["z"], np.array([0.0, 1e-3]))
        ions["position"].unit_dimension = {io.Unit_Dimension.L: 1}
        constant(ions["positionOffset"]["x"], 0.0, 2)
        constant(ions["positionOffset"]["z"], 0.0, 2)
        store(ions["momentum"]["x"], np.array([0.0, 0.0]))
        store(ions["momentum"]["z"], M_P * np.array([1e5, 2e5]))
        ions["momentum"].unit_dimension = electrons["momentum"].unit_dimension
        store(ions["weighting"][io.Record_Component.SCALAR], np.full(2, 1e6))

        # E on a Yee grid (E_x half a cell along x), B on the nodes, rho a scalar
        nx, nz, h = 5, 4, 1e-3
        field = it.meshes["E"]
        field.axis_labels = ["x", "z"]
        field.grid_spacing = [h, h]
        field.grid_global_offset = [0.0, -2e-3]
        field.grid_unit_SI = 1.0
        field.unit_dimension = {
            io.Unit_Dimension.L: 1, io.Unit_Dimension.M: 1, io.Unit_Dimension.T: -3,
            io.Unit_Dimension.I: -1,
        }  # fmt: skip
        grid = np.arange(nx * nz, dtype=float).reshape(nx, nz)
        store(field["x"], grid)
        field["x"].position = [0.5, 0.0]
        store(field["z"], 2.0 * grid)
        field["z"].position = [0.0, 0.5]
        magnetic = it.meshes["B"]
        magnetic.axis_labels = ["x", "z"]
        magnetic.grid_spacing = [h, h]
        magnetic.grid_global_offset = [0.0, -2e-3]
        magnetic.unit_dimension = {
            io.Unit_Dimension.M: 1, io.Unit_Dimension.T: -2, io.Unit_Dimension.I: -1,
        }  # fmt: skip
        for c in "xyz":
            store(magnetic[c], grid + iteration)
            magnetic[c].position = [0.0, 0.0]
        rho = it.meshes["rho"]
        rho.axis_labels = ["x", "z"]
        rho.grid_spacing = [h, h]
        rho.grid_global_offset = [0.0, -2e-3]
        rho.unit_dimension = {
            io.Unit_Dimension.L: -3, io.Unit_Dimension.T: 1, io.Unit_Dimension.I: 1,
        }  # fmt: skip
        store(rho[io.Mesh_Record_Component.SCALAR], -grid)
        rho[io.Mesh_Record_Component.SCALAR].position = [0.0, 0.0]
        it.close()
    series.close()


@pytest.fixture
def grouped(tmp_path):
    path = tmp_path / "data.h5"
    _write_series(path)
    return path


def test_units_of_an_openpmd_dimension() -> None:
    assert units_of((1, 1, -1, 0, 0, 0, 0)) == "kg m/s"
    assert units_of((1, 1, -3, -1, 0, 0, 0)) == "V/m"
    assert units_of((0, 0, 0, 0, 0, 0, 0)) == "1"
    assert units_of((1, 0, -2, 0, 0, 0, 0)) == "m s^-2"
    assert units_of((0, 0, 0)) == "1"


def test_the_particles_of_every_iteration(grouped) -> None:
    markers = plasma_plots.from_openpmd(grouped)
    assert markers.sizes == {"t": 3, "marker": 8}
    assert markers.t.values == pytest.approx(np.array(ITERATIONS) * 1e-12)
    assert list(markers.iteration.values) == list(ITERATIONS)
    electrons = markers.isel(
        marker=np.flatnonzero(markers.species.values == "electrons")
    )
    # matched by id: the electron lost after iteration 0 is NaN later, the others stay put
    x, pz, ids = _electrons(6, 0)
    assert list(electrons.id.values) == list(ids)
    assert electrons.x.isel(t=0).values == pytest.approx(x + 0.5e-3)
    assert np.isnan(electrons.x.isel(t=2).values[-2:]).all()
    assert electrons.pz.isel(t=0).values == pytest.approx(pz)
    gamma = np.sqrt(1.0 + (pz / (M_E * SPEED_OF_LIGHT)) ** 2)
    assert electrons.vz.isel(t=0).values == pytest.approx(pz / (gamma * M_E))
    assert electrons.charge.isel(t=0).values == pytest.approx(-E)
    assert markers.px.attrs["units"] == "kg m/s"
    assert markers.x.attrs["units"] == "m"
    assert markers.vz.attrs["label"] == "$v_{z}$"
    ions = markers.isel(marker=np.flatnonzero(markers.species.values == "ions"))
    assert "vz" in markers and np.isnan(ions.vz.values).all()  # no mass, no velocity


def test_one_species_at_one_iteration(grouped) -> None:
    ions = plasma_plots.from_openpmd(grouped, species="ions", iteration=-1)
    assert "t" not in ions.dims and int(ions.iteration) == 200
    assert ions.z.values == pytest.approx([0.0, 1e-3])
    assert set(ions.species.values) == {"ions"}
    with pytest.raises(ValueError, match="no species"):
        plasma_plots.from_openpmd(grouped, species="positrons", iteration=0)
    with pytest.raises(KeyError, match="no iteration"):
        plasma_plots.from_openpmd(grouped, iteration=50)


def test_the_markers_plot_in_phase_space(grouped) -> None:
    markers = plasma_plots.from_openpmd(grouped)
    result = markers.plasma.plot.phase_space(x="x", v="pz", species="species", t=0)
    assert result.fig is not None
    pytest.importorskip("plotly")
    result = markers.plasma.plot.phase_space(
        x="z", v="vz", kind="histogram", weight="weighting", t=-1, backend="plotly"
    )
    assert result.fig is not None


def test_a_file_based_series_from_one_file_or_the_pattern(tmp_path) -> None:
    _write_series(tmp_path / "simData_%06T.h5", encoding="fileBased")
    assert sorted(p.name for p in tmp_path.iterdir())[0] == "simData_000000.h5"
    from_pattern = plasma_plots.from_openpmd(tmp_path / "simData_%06T.h5")
    from_file = plasma_plots.from_openpmd(tmp_path / "simData_000100.h5")
    from_folder = plasma_plots.from_openpmd(tmp_path)
    assert list(from_file.iteration.values) == list(ITERATIONS)
    assert from_pattern.identical(from_file) and from_folder.identical(from_file)


def test_the_meshes_on_their_grids(grouped) -> None:
    fields = plasma_plots.openpmd_meshes(grouped, iteration=100)
    assert int(fields.iteration) == 100 and float(fields.t) == pytest.approx(1e-10)
    # E is staggered: its components apart, each on its own coordinates
    assert "E" not in fields
    assert fields.E_x.dims == ("x_0.5", "z")
    assert fields["x_0.5"].values == pytest.approx((np.arange(5) + 0.5) * 1e-3)
    assert fields.z.values == pytest.approx(-2e-3 + np.arange(4) * 1e-3)
    assert fields.E_z.dims == ("x", "z_0.5")
    assert fields.E_z.attrs["units"] == "V/m"
    # B shares its grid: one variable along component
    assert fields.B.dims == ("component", "x", "z")
    assert list(fields.component.values) == ["x", "y", "z"]
    assert fields.B.sel(component="y").values[0, 1] == pytest.approx(101.0)
    assert fields.B.attrs["units"] == "T"
    assert fields.rho.attrs["units"] == "C/m^3"
    assert set(plasma_plots.openpmd_meshes(grouped, names="rho").data_vars) == {"rho"}
    with pytest.raises(ValueError, match="no meshes"):
        plasma_plots.openpmd_meshes(grouped, names="J")
    result = fields.rho.plasma.plot.slice()
    assert result.fig is not None


def _hand_written(path) -> None:
    """A group-based file written with h5py: macro-weighted momenta, Fortran-ordered mesh."""
    with h5py.File(path, "w") as h5:
        attrs = {
            "openPMD": "1.1.0", "openPMDextension": np.uint32(0), "basePath": "/data/%T/",
            "meshesPath": "meshes/", "particlesPath": "particles/",
            "iterationEncoding": "groupBased", "iterationFormat": "/data/%T/",
            "author": "plasma-plots tests", "software": "h5py",
            "softwareVersion": h5py.__version__, "date": "2026-11-22 12:00:00 +0000",
        }  # fmt: skip
        for key, value in attrs.items():
            h5.attrs[key] = np.bytes_(value) if isinstance(value, str) else value
        it = h5.create_group("data/7")
        for key, value in {"time": 3.0, "dt": 1.0, "timeUnitSI": 1e-9}.items():
            it.attrs[key] = value

        def record(group, unit_dimension, weighting_power, macro):
            group.attrs["unitDimension"] = np.array(unit_dimension, dtype=float)
            group.attrs["timeOffset"] = np.float32(0.0)
            group.attrs["weightingPower"] = float(weighting_power)
            group.attrs["macroWeighted"] = np.uint32(macro)

        beam = it.create_group("particles/beam")
        weights = np.array([2.0, 4.0])
        position = beam.create_group("position")
        record(position, (1, 0, 0, 0, 0, 0, 0), 0.0, 0)
        offset = beam.create_group("positionOffset")
        record(offset, (1, 0, 0, 0, 0, 0, 0), 0.0, 0)
        momentum = beam.create_group("momentum")
        record(momentum, (1, 1, -1, 0, 0, 0, 0), 1.0, 1)
        for c, values in {"x": [1.0, 2.0], "y": [0.0, 0.5]}.items():
            data = position.create_dataset(c, data=np.array(values))  # in mm
            data.attrs["unitSI"] = 1e-3
            constant = offset.create_group(c)
            constant.attrs["value"] = 10.0
            constant.attrs["shape"] = np.array([2], dtype=np.uint64)
            constant.attrs["unitSI"] = 1e-3
            # macro-weighted: the macroparticle's momentum, w times the particle's
            data = momentum.create_dataset(c, data=np.array(values) * weights)
            data.attrs["unitSI"] = 1e-22
        weighting = beam.create_dataset("weighting", data=weights)
        record(weighting, (0, 0, 0, 0, 0, 0, 0), 1.0, 1)
        weighting.attrs["unitSI"] = 1.0

        phi = it.create_dataset(
            "meshes/phi", data=np.arange(6.0).reshape(3, 2).T
        )  # (z, x) in memory: x fastest
        record(phi, (2, 1, -3, -1, 0, 0, 0), 0.0, 0)
        for key, value in {
            "geometry": np.bytes_("cartesian"), "dataOrder": np.bytes_("F"),
            "axisLabels": np.array([b"x", b"z"]), "gridSpacing": np.array([1.0, 2.0]),
            "gridGlobalOffset": np.array([0.0, 0.0]), "gridUnitSI": 1e-3,
            "position": np.array([0.0, 0.0]), "unitSI": 1.0,
        }.items():  # fmt: skip
            phi.attrs[key] = value


def test_a_hand_written_file(tmp_path) -> None:
    path = tmp_path / "hand.h5"
    _hand_written(path)
    validator = pytest.importorskip("openpmd_validator.check_h5")
    errors, _ = validator.check_file(str(path), verbose=False)
    assert errors == 0
    beam = plasma_plots.from_openpmd(path, iteration=7)
    assert float(beam.t) == pytest.approx(3e-9)
    assert beam.x.values == pytest.approx([0.011, 0.012])
    assert beam.px.values == pytest.approx([1e-22, 2e-22])  # per particle
    assert beam.weighting.values == pytest.approx([2.0, 4.0])
    assert "vx" not in beam  # no mass
    phi = plasma_plots.openpmd_meshes(path).phi
    # Fortran order: the labels are listed fastest first, so the array reads as (z, x)
    assert phi.dims == ("z", "x")
    assert phi.x.values == pytest.approx([0.0, 1e-3, 2e-3])
    assert phi.z.values == pytest.approx([0.0, 2e-3])
    assert phi.attrs["units"] == "V"
    assert phi.transpose("x", "z").values == pytest.approx(np.arange(6.0).reshape(3, 2))


def test_errors_name_the_problem(tmp_path) -> None:
    with pytest.raises(ValueError, match="ADIOS2"):
        plasma_plots.from_openpmd(tmp_path / "run.bp")
    plain = tmp_path / "plain.h5"
    with h5py.File(plain, "w") as h5:
        h5["a"] = np.zeros(2)
    with pytest.raises(ValueError, match="not an openPMD file"):
        plasma_plots.from_openpmd(plain)
    with pytest.raises(FileNotFoundError):
        plasma_plots.from_openpmd(tmp_path / "none_%T.h5")
