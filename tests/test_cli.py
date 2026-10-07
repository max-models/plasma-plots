"""The ``plasma-plots`` command: parsing, plot dispatch and saving, on files and on Struphy runs."""

import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from plasma_plots import accessors, cli  # noqa: E402
from plasma_plots.cli import main  # noqa: E402
from plasma_plots.cli import (
    CLIError,
    _parse_options,
    parse_value,
    plot_methods,
    quicklook_plot,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def nc(tmp_path):
    t = np.linspace(0, 5, 12)
    e1 = np.linspace(0, 1, 10)
    e2 = np.linspace(0, 1, 16, endpoint=False)
    e3 = np.linspace(0, 1, 3, endpoint=False)
    T, E1, E2, _ = np.meshgrid(t, e1, e2, e3, indexing="ij")
    phi = np.exp(0.2 * T) * np.sin(np.pi * E1) * np.cos(2 * np.pi * (3 * E2 - T))
    path = tmp_path / "fields.nc"
    xr.Dataset(
        {
            "phi": (
                ("t", "eta1", "eta2", "eta3"),
                phi,
                {"units": "V", "long_name": "potential"},
            ),
            "energy": ("t", np.exp(0.4 * t), {"units": "J"}),
            "profile": (("t", "eta1"), phi[:, :, 0, 0]),
        },
        coords={"t": t, "eta1": e1, "eta2": e2, "eta3": e3},
    ).to_netcdf(path)
    return path


@pytest.fixture
def run(tmp_path):
    pytest.importorskip("struphy")
    from test_output_accessors import make_run

    return make_run(str(tmp_path))


def test_values_keep_integers_as_positions_and_decimals_as_values():
    assert parse_value("-1") == -1 and isinstance(parse_value("-1"), int)
    assert parse_value("0.35") == 0.35 and isinstance(parse_value("0.35"), float)
    assert parse_value("eta1") == "eta1"
    assert parse_value("true") is True and parse_value("none") is None
    assert parse_value("0,0.5") == [0, 0.5]
    assert type(parse_value("0,1")) is list  # lists, not numpy arrays
    assert parse_value("[1, 4]") == [1, 4]
    assert parse_value('{"eta3": [0, 0.5]}') == {"eta3": [0, 0.5]}


def test_options_are_key_value_pairs():
    assert _parse_options(["t=-1", "x=eta1"]) == {"t": -1, "x": "eta1"}
    with pytest.raises(CLIError, match="key=value"):
        _parse_options(["t"])
    with pytest.raises(CLIError, match="JSON"):
        parse_value("[1,")


def test_at_names_another_product_of_the_same_source(nc):
    source = cli.open_source(nc)
    other = parse_value("@energy", source)
    assert isinstance(other, xr.DataArray) and other.name == "energy"


def test_plot_methods_leave_out_3d_scenes_and_interactive_views():
    phi = xr.DataArray(np.zeros((2, 3)), dims=("t", "eta1"))
    methods = plot_methods(phi)
    assert {"slice", "lineout", "timeseries", "animation", "movie", "frames"} <= set(
        methods
    )
    left_out = {
        name
        for name in dir(accessors.ArrayPlots)
        if not name.startswith("_") and callable(getattr(accessors.ArrayPlots, name))
    } - set(methods)
    assert left_out == {
        "glyphs",
        "isosurface",
        "slices_3d",
        "streamlines",
        "view",
        "viewer",
        "volume",
    }


def test_quicklook_chooses_a_plot_by_dimensions(nc):
    ds = xr.open_dataset(nc)
    assert quicklook_plot(ds.phi) == (
        "slice",
        {"x": "eta1", "y": "eta2", "t": -1, "eta3": 0},
    )
    assert quicklook_plot(ds.profile) == ("lineout", {"x": "eta1", "t": -1})
    assert quicklook_plot(ds.energy) == ("timeseries", {})


def test_info_lists_variables_dimensions_and_units(nc, capsys):
    assert main(["info", str(nc)]) == 0
    text = capsys.readouterr().out
    assert "phi       t: 12, eta1: 10, eta2: 16, eta3: 3  V      potential" in text
    assert "t           0 … 5 (12)" in text


def test_plot_saves_the_figure(nc, tmp_path, capsys):
    out = tmp_path / "phi.png"
    assert (
        main(["plot", str(nc), "phi", "slice", "t=-1", "eta3=0", "-o", str(out)]) == 0
    )
    assert out.stat().st_size > 0
    assert f"wrote {out}" in capsys.readouterr().out


def test_plot_passes_other_products_and_json(nc, tmp_path):
    out = tmp_path / "energy.png"
    assert (
        main(["plot", str(nc), "energy", "compare", "other=@energy", "-o", str(out)])
        == 0
    )
    assert (
        main(["plot", str(nc), "energy", "timeseries", "fit=[1,4]", "-o", str(out)])
        == 0
    )


def test_an_html_output_draws_with_plotly(nc, tmp_path):
    pytest.importorskip("plotly")
    out = tmp_path / "phi.html"
    assert (
        main(["plot", str(nc), "phi", "slice", "t=2.5", "eta3=0", "-o", str(out)]) == 0
    )
    assert "plotly" in out.read_text()


def test_a_tex_or_tikz_output_draws_with_tikz(nc, tmp_path):
    pytest.importorskip("maxplotlib.backends.tikzfigure")
    tex, tikz = tmp_path / "phi.tex", tmp_path / "phi.tikz"
    assert (
        main(["plot", str(nc), "phi", "slice", "t=2.5", "eta3=0", "-o", str(tex)]) == 0
    )
    assert tex.read_text().startswith("\\documentclass")
    assert main(["plot", str(nc), "energy", "timeseries", "-o", str(tikz)]) == 0
    assert "\\begin{axis}" in tikz.read_text()
    assert list(tmp_path.glob("*.png")), "the slice's images are written next to it"


def test_an_animation_has_no_tikz_version(nc, tmp_path, capsys):
    pytest.importorskip("maxplotlib.backends.tikzfigure")
    out = str(tmp_path / "phi.tex")
    assert main(["plot", str(nc), "phi", "animation", "eta3=0", "-o", out]) == 1
    assert "still figures" in capsys.readouterr().err


def test_movie_animates_2d_and_1d_data(nc, tmp_path):
    two, one = tmp_path / "phi.gif", tmp_path / "profile.gif"
    assert main(["movie", str(nc), "phi", "eta3=0", "step=4", "-o", str(two)]) == 0
    assert main(["movie", str(nc), "profile", "step=4", "-o", str(one)]) == 0
    assert two.read_bytes()[:3] == b"GIF" and one.read_bytes()[:3] == b"GIF"


def test_errors_are_one_line_with_exit_code_1(nc, tmp_path, capsys):
    out = str(tmp_path / "x.png")
    assert main(["plot", str(nc), "phi", "isosurface", "-o", out]) == 1
    assert "interactive method; add --show" in capsys.readouterr().err
    assert main(["plot", str(nc), "nosuch", "slice", "-o", out]) == 1
    assert "the variables are phi, energy, profile" in capsys.readouterr().err
    assert main(["plot", str(nc), "phi", "slice", "bogus=1", "-o", out]) == 1
    err = capsys.readouterr().err
    assert "'bogus' is not a dimension" in err and "Traceback" not in err
    assert main(["info", str(tmp_path / "missing.nc")]) == 1


def test_list_prints_the_plot_methods(nc, capsys):
    assert main(["plot", str(nc), "phi", "--list"]) == 0
    assert "slice\n" in capsys.readouterr().out


def test_quicklook_of_a_file(nc, tmp_path):
    folder = tmp_path / "figures"
    assert main(["quicklook", str(nc), "-o", str(folder)]) == 0
    assert sorted(os.listdir(folder)) == [
        "energy-timeseries.png",
        "phi-slice.png",
        "profile-lineout.png",
    ]


def test_run_products_by_name(run, tmp_path, capsys):
    assert main(["info", str(run.path_out)]) == 0
    text = capsys.readouterr().out
    assert "em_fields/E" in text and "orbits" in text and "en_tot" in text
    out = tmp_path / "E.png"
    args = [
        "plot",
        str(run.path_out),
        "em_fields/E",
        "slice",
        "t=-1",
        "component=0",
        "eta3=0",
    ]
    assert main([*args, "-o", str(out)]) == 0 and out.exists()


def test_the_whole_run_plots_with_out_plot(run, tmp_path, capsys):
    assert main(["plot", str(run.path_out), ".", "--list"]) == 0
    assert {"energies", "scalars", "profile.gantt"} <= set(
        capsys.readouterr().out.split()
    )
    out = tmp_path / "energies.png"
    assert (
        main(["plot", str(run.path_out), ".", "energies", "-o", str(out)]) == 0
        and out.exists()
    )


def test_a_product_named_as_a_method_gets_a_hint(run, capsys):
    args = ["plot", str(run.path_out), ".", "en_phi", "-o", "x.png"]
    assert main(args) == 1
    assert "`plasma-plots plot PATH en_phi timeseries`" in capsys.readouterr().err


def test_quicklook_of_a_run(run, tmp_path):
    folder = tmp_path / "figures"
    assert main(["quicklook", str(run.path_out), "-o", str(folder)]) == 0
    assert {
        "energies.png",
        "scalars.png",
        "em_fields-E-slice.png",
        "kinetic_ions-trajectories.png",
        "kinetic_ions-e1_v1_density-f-slice.png",
    } <= set(os.listdir(folder))


@pytest.mark.parametrize("command", ["info", "plot", "movie", "quicklook"])
def test_commands_process_runs_automatically(run, tmp_path, monkeypatch, command):
    from struphy.post_processing.output import Output
    from struphy.post_processing.tests.test_output import write_manifest

    (run.path_pproc / "manifest.json").unlink()
    processed = []

    def pproc(output):
        # The fixture already contains products; stand in for the expensive reconstruction.
        processed.append(output.path_out)
        write_manifest(str(output.path_out))
        return output

    monkeypatch.setattr(Output, "pproc", pproc)
    args = [command, str(run.path_out)]
    if command == "plot":
        args += [
            "em_fields/E",
            "slice",
            "t=-1",
            "component=0",
            "eta3=0",
            "-o",
            str(tmp_path / "E.png"),
        ]
    elif command == "movie":
        args += ["em_fields/E", "component=0", "eta3=0", "-o", str(tmp_path / "E.gif")]
    elif command == "quicklook":
        args += ["-o", str(tmp_path / "figures")]
    assert main(args) == 0
    assert processed == [run.path_out]
    # A later command reuses the products instead of replacing custom processing options.
    assert main(["info", str(run.path_out)]) == 0
    assert processed == [run.path_out]


def test_open_source_processes_by_default(run, monkeypatch):
    from struphy.post_processing.output import Output
    from struphy.post_processing.tests.test_output import write_manifest

    (run.path_pproc / "manifest.json").unlink()
    monkeypatch.setattr(
        Output, "pproc", lambda output: write_manifest(str(output.path_out))
    )
    assert cli.open_source(run.path_out).is_processed


def test_python_m_runs_the_command():
    result = subprocess.run(
        [sys.executable, "-m", "plasma_plots", "api"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.startswith("# plasma-plots API index")
    bad = subprocess.run(
        [sys.executable, "-m", "plasma_plots", "bogus"], capture_output=True, text=True
    )
    assert bad.returncode == 2 and "invalid choice" in bad.stderr


def test_the_installed_command_runs(nc, tmp_path):
    # the console script of pyproject.toml, as `pip install` puts it on the PATH
    command = shutil.which("plasma-plots")
    assert command, "plasma-plots is not on the PATH: pip install -e ."
    out = tmp_path / "phi.png"
    subprocess.run(
        [command, "plot", str(nc), "phi", "slice", "t=-1", "eta3=0", "-o", str(out)],
        check=True,
        capture_output=True,
    )
    assert out.stat().st_size > 0
    bad = subprocess.run(
        [command, "info", str(tmp_path / "missing.nc")], capture_output=True, text=True
    )
    assert bad.returncode == 1 and bad.stderr.startswith("plasma-plots: error:")


@pytest.mark.parametrize("save", [False, True])
def test_show_displays_plot_and_optionally_saves(nc, tmp_path, monkeypatch, save):
    monkeypatch.setattr(cli, "_check_display", lambda: None)
    shown = []
    out = tmp_path / "shown.png"

    def show(*, block):
        assert block
        assert bool(out.exists()) == save  # save before the window can close
        shown.append(plt.gcf())

    monkeypatch.setattr(plt, "show", show)
    monkeypatch.setattr(
        matplotlib, "use", lambda *a, **kw: pytest.fail("forced backend with --show")
    )
    args = ["plot", str(nc), "phi", "slice", "t=-1", "eta3=0", "--show"]
    if save:
        args += ["-o", str(out)]
    assert main(args) == 0
    assert len(shown) == 1 and shown[0].axes


def test_show_movie_keeps_animation_alive(nc, monkeypatch):
    monkeypatch.setattr(cli, "_check_display", lambda: None)
    import gc

    from matplotlib.animation import FuncAnimation

    shown = []

    def show(*, block):
        assert block
        animations = [obj for obj in gc.get_objects() if isinstance(obj, FuncAnimation)]
        assert animations
        # Draw as a GUI would, starting the animation while it is still referenced.
        plt.gcf().canvas.draw()
        shown.append(True)

    monkeypatch.setattr(plt, "show", show)
    assert main(["movie", str(nc), "phi", "eta3=0", "--show"]) == 0
    assert shown == [True]


def test_show_plotly_opens_browser(nc, monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, **kw: shown.append(kw))
    assert (
        main(["plot", str(nc), "energy", "timeseries", "backend=plotly", "--show"]) == 0
    )
    assert shown == [{"renderer": "browser"}]


def test_quicklook_can_show_without_output(nc, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_check_display", lambda: None)
    shown = []
    monkeypatch.setattr(plt, "show", lambda **kw: shown.append(plt.gcf()))
    assert main(["quicklook", str(nc), "--show"]) == 0
    assert len(shown) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fields.nc"]


@pytest.mark.parametrize(
    "command,options",
    [
        ("plot", ["energy", "timeseries"]),
        ("movie", ["phi", "eta3=0"]),
        ("quicklook", []),
    ],
)
def test_requires_save_or_show(nc, capsys, command, options):
    assert main([command, str(nc), *options]) == 1
    assert "--show" in capsys.readouterr().err


def test_file_writing_method_requires_output_with_show(nc, capsys):
    assert main(["plot", str(nc), "phi", "frames", "eta3=0", "--show"]) == 1
    assert "requires -o" in capsys.readouterr().err


def test_show_result_accepts_figure_and_tuple(monkeypatch):
    monkeypatch.setattr(cli, "_check_display", lambda: None)
    calls = []
    monkeypatch.setattr(plt, "show", lambda **kw: calls.append(kw))
    fig, ax = plt.subplots()
    cli._show_result(fig)
    cli._show_result((fig, ax))
    assert calls == [{"block": True}, {"block": True}]


def test_show_result_skips_other_mpi_ranks(monkeypatch):
    from plasma_plots import mpi

    monkeypatch.setattr(mpi, "is_plotting_rank", lambda: False)
    monkeypatch.setattr(
        plt, "show", lambda **kw: pytest.fail("displayed on another rank")
    )
    cli._show_result(plt.figure())


@pytest.mark.parametrize(
    "args",
    [
        ["plot", "missing", "phi", "slice"],
        ["plot", "missing", "phi", "slic", "--show"],
        ["plot", "missing", "phi", "slice", "t=0", "t=1", "--show"],
        ["plot", "missing", "phi", "slice", "levels=[oops", "--show"],
        ["plot", "missing", "phi", "slice", "bad-option", "--show"],
        ["plot", "missing", "phi", "frames", "--show", "-o", "frames"],
        ["movie", "missing", "phi", "--dpi", "0", "--show"],
        ["quicklook", "missing", "--format", ",", "--show"],
        ["quicklook", "missing", "--select", "t=last", "--show"],
    ],
)
def test_usage_errors_do_not_open_data(args, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "open_source", lambda *a: pytest.fail("opened data for invalid command")
    )
    assert main(args) == 1
    assert "error:" in capsys.readouterr().err


def test_method_help_needs_no_data(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "open_source", lambda *a: pytest.fail("opened data for help")
    )
    assert main(["help", "slice"]) == 0
    text = capsys.readouterr().out
    assert "Parameters" in text and "x:" in text and "t=-1" in text
    assert main(["help", "energies"]) == 0
    assert "run plot: energies" in capsys.readouterr().out
    assert main(["help", "profile.gantt"]) == 0
    assert "min_duration" in capsys.readouterr().out
    with pytest.raises(SystemExit) as done:
        main(["plot", "missing.nc", "phi", "slice", "--help"])
    assert done.value.code == 0
    assert "array plot: slice" in capsys.readouterr().out


def test_method_and_product_typos_offer_suggestions(nc, capsys):
    assert main(["help", "slic"]) == 1
    assert "did you mean 'slice'" in capsys.readouterr().err
    assert main(["plot", str(nc), "energi", "timeseries", "--show"]) == 1
    assert "did you mean 'energy'" in capsys.readouterr().err


@pytest.mark.parametrize("method", ["viewer", "view"])
def test_interactive_view_can_show_and_save(nc, tmp_path, monkeypatch, method):
    monkeypatch.setattr(cli, "_check_display", lambda: None)
    shown = []
    monkeypatch.setattr(plt, "show", lambda **kw: shown.append(plt.gcf()))
    out = tmp_path / "viewer.png"
    assert (
        main(["plot", str(nc), "phi", method, "eta3=0", "--show", "-o", str(out)]) == 0
    )
    assert out.stat().st_size > 0
    assert shown and len(shown[0].axes) >= 2


def test_interactive_methods_are_listed_with_show(nc, capsys):
    assert main(["plot", str(nc), "phi", "--list", "--show"]) == 0
    assert {"viewer", "view", "volume", "isosurface"} <= set(
        capsys.readouterr().out.split()
    )


def test_headless_show_fails_with_recovery_instructions(nc, capsys):
    assert main(["plot", str(nc), "energy", "timeseries", "--show"]) == 1
    error = capsys.readouterr().err
    assert "cannot open a window" in error
    assert "-o figure.png" in error and "backend=plotly" in error


def test_pyvista_show_dispatch(nc, monkeypatch, tmp_path):
    calls = []

    class Scene:
        __module__ = "pyvista.plotting.plotter"

        def show(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(accessors.ArrayPlots, "volume", lambda self, **kwargs: Scene())
    output = tmp_path / "scene.png"
    assert (
        main(["plot", str(nc), "phi", "volume", "t=-1", "--show", "-o", str(output)])
        == 0
    )
    assert calls == [{"screenshot": str(output)}]


def test_quicklook_product_and_dimension_selection(nc, tmp_path, monkeypatch):
    calls = []
    original = cli._call_plot

    def record(obj, name, options, output, **kwargs):
        calls.append((obj, name, options))
        return original(obj, name, options, output, **kwargs)

    monkeypatch.setattr(cli, "_call_plot", record)
    folder = tmp_path / "selected"
    assert (
        main(
            [
                "quicklook",
                str(nc),
                "--products",
                "phi",
                "--select",
                "t=2.4",
                "eta3=0",
                "-o",
                str(folder),
            ]
        )
        == 0
    )
    assert [p.name for p in folder.iterdir()] == ["phi-slice.png"]
    assert len(calls) == 1
    data, method, options = calls[0]
    assert data.dims == ("eta1", "eta2")
    assert float(data.t) == pytest.approx(25 / 11)
    assert method == "slice" and "t" not in options


def test_quicklook_unknown_selection_is_error(nc, tmp_path, capsys):
    assert (
        main(
            [
                "quicklook",
                str(nc),
                "--select",
                "oops=0",
                "-o",
                str(tmp_path / "figures"),
            ]
        )
        == 1
    )
    assert "unknown selection dimensions: oops" in capsys.readouterr().err


def test_empty_option_value_and_duplicate_options():
    assert parse_value("") == ""
    with pytest.raises(CLIError, match="duplicate option 't'"):
        _parse_options(["t=0", "t=1"])


def test_quicklook_headless_error_reaches_stderr(nc, capsys):
    assert main(["quicklook", str(nc), "--show"]) == 1
    assert "cannot open a window" in capsys.readouterr().err


def test_quicklook_selection_error_names_product(nc, tmp_path, capsys):
    assert (
        main(
            [
                "quicklook",
                str(nc),
                "--products",
                "phi",
                "--select",
                "t=999",
                "-o",
                str(tmp_path / "plots"),
            ]
        )
        == 1
    )
    assert "cannot select 'phi'" in capsys.readouterr().err


def test_traceback_preserves_plot_failure(nc):
    with pytest.raises(CLIError, match="Available dimensions"):
        main(
            [
                "plot",
                str(nc),
                "phi",
                "slice",
                "bogus=1",
                "-o",
                "unused.png",
                "--traceback",
            ]
        )


def test_missing_netcdf_backend_suggests_extra(tmp_path, monkeypatch):
    path = tmp_path / "fields.nc"
    path.touch()

    def missing_backend(*args, **kwargs):
        raise ValueError(
            "found matches with xarray's IO backends, but their dependencies may not be installed"
        )

    monkeypatch.setattr(xr, "open_dataset", missing_backend)
    with pytest.raises(CLIError, match=r'pip install "plasma-plots\[netcdf\]"'):
        cli.open_source(path)


def test_invalid_file_does_not_suggest_installing_netcdf(tmp_path, monkeypatch):
    path = tmp_path / "broken.nc"
    path.touch()

    def invalid_file(*args, **kwargs):
        raise OSError("corrupt file")

    monkeypatch.setattr(xr, "open_dataset", invalid_file)
    with pytest.raises(CLIError, match="corrupt file") as error:
        cli.open_source(path)
    assert "pip install" not in str(error.value)
