"""The ``plasma-plots`` command: quick looks at simulation output without writing Python.

Installed as ``plasma-plots`` (also ``python -m plasma_plots``). It opens a Struphy run folder (as
a Struphy ``Output``) or any file xarray can read (netCDF, zarr, ...) and saves or displays figures of it. It
has no plot options of its own: ``plot`` calls an accessor method by name with ``key=value``
arguments, so every plot method works from the command line::

    plasma-plots info sim_1
    plasma-plots plot sim_1 em_fields/phi slice t=-1 eta3=0 -o phi.png
    plasma-plots plot sim_1 . energies -o energies.html
    plasma-plots plot sim_1 . energies -o energies.tex
    plasma-plots plot sim_1 em_fields/phi slice t=-1 eta3=0 --show
    plasma-plots movie sim_1 em_fields/phi eta3=0 -o phi.gif
    plasma-plots quicklook sim_1 -o figures/

``plasma-plots guide`` prints the package guide, ``plasma-plots api`` the API index.
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
import os
import sys
from pathlib import Path

WHOLE = "."
"""The PRODUCT that names the whole source: the run (``out.plot``) or the file's Dataset."""

_PLOTLY_SUFFIXES = (".html", ".htm", ".json")
_TIKZ_SUFFIXES = (".tex", ".tikz")
_ANIMATION_WRITERS = {".gif": "pillow", ".apng": "pillow", ".webp": "pillow"}
_INTERACTIVE_RETURNS = ("Plotter", "SliceView", "InteractiveSliceViewer")
_NOT_SPATIAL = ("t", "component", "marker")


class _MethodHelp(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        method = getattr(namespace, "method", None)
        if method:
            product = getattr(namespace, "product", None)
            kind = "array"
            if product == ".":
                kind = "run" if _is_struphy_run(namespace.path) else "dataset"
            try:
                _cmd_help(argparse.Namespace(method=method, kind=kind))
            except CLIError as error:
                parser.error(str(error))
        else:
            parser.print_help()
        parser.exit()


class CLIError(Exception):
    """A problem with the command line or its data, reported without a traceback."""


class _DisplayError(CLIError):
    """A display configuration error that should stop even a batch quicklook."""


# ---------------------------------------------------------------------------------------------
# Opening output


def _is_struphy_run(path) -> bool:
    """Whether ``path`` is a Struphy run folder (it has ``run_metadata.json`` or ``data/``)."""
    path = Path(path)
    return path.is_dir() and ((path / "run_metadata.json").exists() or (path / "data").is_dir())


def open_source(path):
    """Open ``path``: a Struphy ``Output`` for a run folder, else an ``xarray.Dataset``.

    Struphy runs are automatically post-processed with default options when needed.
    Existing processed output is reused.

    Parameters
    ----------
    path : str or pathlib.Path
        A Struphy run folder, or a file (or zarr store) that ``xarray.open_dataset`` reads.

    Returns
    -------
    struphy.Output or xarray.Dataset
        The opened output. A GVEC Dataset is converted with :func:`plasma_plots.from_gvec`.
    """
    path = Path(path)
    if not path.exists():
        raise CLIError(f"{path}: no such file or folder")
    if _is_struphy_run(path):
        try:
            from struphy.post_processing.output import Output
        except ImportError as error:
            raise CLIError(f"{path} is a Struphy run folder; opening it needs struphy ({error})") from None
        out = Output(path)
        if not out.is_processed:
            out.pproc()
        return out
    import xarray as xr

    from .gvec import from_gvec, is_gvec

    try:
        dataset = xr.open_dataset(path)
    except Exception as error:
        hint = ""
        if isinstance(error, (ImportError, ValueError)) and (
            "dependencies may not be installed" in str(error)
            or "currently installed IO backends" in str(error)
            or "netCDF4" in str(error)
        ):
            hint = '\nFor netCDF files, install the backend with: pip install "plasma-plots[netcdf]"'
        raise CLIError(f"{path}: xarray cannot open it ({error}){hint}") from None
    return from_gvec(dataset) if is_gvec(dataset) else dataset


def _is_output(source) -> bool:
    import xarray as xr

    return not isinstance(source, (xr.Dataset, xr.DataArray))


def _get_product(source, name: str):
    """The product ``name`` of ``source``: a ``DataArray``, a ``Dataset`` or the run itself."""
    if name == WHOLE:
        return source
    if not _is_output(source):
        if name not in source.data_vars:
            raise CLIError(
                f"no variable {name!r}{_suggest(name, source.data_vars)}; the variables are {', '.join(source.data_vars)}"
            )
        return source[name]
    if name == "scalars":
        return source.scalars
    if name in source.scalars.data_vars:
        return source.scalars[name]
    for catalog in (
        source.field_catalog,
        source.distribution_catalog,
        source.density_catalog,
        source.orbit_catalog,
    ):
        if name in catalog:
            return catalog[name]
    try:
        return source.evaluate(name)
    except Exception as error:
        raise CLIError(f"no product {name!r} ({error}); `plasma-plots info` lists them") from None


# ---------------------------------------------------------------------------------------------
# Arguments


def parse_value(text: str, source=None):
    """A ``key=value`` value as Python: an int, a float, a bool, ``None``, a list or a string.

    An integer stays an ``int`` (a position, ``t=-1``) and a decimal number becomes a ``float``
    (the nearest coordinate value, ``t=0.35``). ``true``/``false``/``none`` are ``True``,
    ``False`` and ``None``; text starting with ``[`` or ``{`` is JSON; commas make a list
    (``eta3=0,0.25,0.5``); ``@name`` is another product of the same source (e.g. ``other=@phi``).
    Anything else is a string.

    Parameters
    ----------
    text : str
        The value as typed.
    source : struphy.Output or xarray.Dataset, optional
        Where ``@name`` looks up products.

    Returns
    -------
    object
        The value.

    Examples
    --------
    >>> parse_value("-1"), parse_value("0.35"), parse_value("eta1")
    (-1, 0.35, 'eta1')
    >>> parse_value("0,0.5"), parse_value("none")
    ([0, 0.5], None)
    """
    if text and text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise CLIError(f"{text!r} is not valid JSON ({error})") from None
    if "," in text:
        return [_parse_scalar(part, source) for part in text.split(",") if part != ""]
    return _parse_scalar(text, source)


def _parse_scalar(text: str, source):
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if text.startswith("@"):
        if source is None:
            raise CLIError(f"{text}: products can only be named with an open source")
        return _get_product(source, text[1:])
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _parse_options(pairs, source=None) -> dict:
    """``["t=-1", "x=eta1"]`` as keyword arguments, each value through :func:`parse_value`."""
    options = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.isidentifier():
            raise CLIError(f"{pair!r}: options are key=value, e.g. t=-1 or x=eta1")
        if key in options:
            raise CLIError(f"duplicate option {key!r}; give each option once")
        options[key] = parse_value(value, source)
    return options


# ---------------------------------------------------------------------------------------------
# Plot methods


def _plots_of(obj):
    """The plot namespace of a product: ``.plasma.plot``, or ``out.plot`` for a run."""
    return obj.plot if _is_output(obj) else obj.plasma.plot


def _is_command_line_method(function) -> bool:
    """Whether a plot method gives something to save: not a PyVista scene or an interactive view."""
    doc = inspect.getdoc(function) or ""
    returns = doc.split("Returns\n-------\n", 1)
    first = returns[1].split("\n", 1)[0] if len(returns) == 2 else ""
    return not any(kind in first for kind in _INTERACTIVE_RETURNS)


def plot_methods(obj, *, show=False) -> list[str]:
    """The plot methods of ``obj`` available to ``plasma-plots plot``.

    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset or struphy.Output
        A product, or a run.
    show : bool, optional
        Include interactive viewers and PyVista scenes. Default: ``False``.

    Returns
    -------
    list of str
        Method names; ``profile.gantt`` style for the profiling plots of a run.
    """
    plots = _plots_of(obj)
    names = []
    for name, member in inspect.getmembers(type(plots)):
        if name.startswith("_"):
            continue
        if isinstance(member, property) and name == "profile":
            names += [
                f"profile.{n}"
                for n, f in inspect.getmembers(member.fget(plots), inspect.ismethod)
                if not n.startswith("_")
            ]
        elif inspect.isfunction(member) and (show or _is_command_line_method(member)):
            names.append(name)
    return sorted(names)


def _method(obj, name, *, show=False):
    methods = plot_methods(obj, show=show)
    if name not in methods:
        hint = _suggest(name, methods)
        if not show and name in plot_methods(obj, show=True):
            hint += "; this interactive method requires --show"
        if _is_output(obj) and name in ["scalars", *obj.keys()]:
            hint = f" ({name!r} is a product: `plasma-plots plot PATH {name} timeseries`)"
        raise CLIError(f"{name!r} is not a plot this command can save{hint}; it can use: {', '.join(methods)}")
    target = _plots_of(obj)
    for part in name.split("."):
        target = getattr(target, part)
    return target


def _call_plot(obj, name: str, options: dict, output, *, show=False):
    """Call the plot method ``name`` of ``obj`` with ``options``, for saving to ``output``.

    A method whose first parameter is a file or folder (``movie``, ``frames``) gets ``output``
    there and writes it itself. A Plotly output (``.html``, ``.json``) draws with
    ``backend="plotly"``, a LaTeX output (``.tex``, ``.tikz``) with ``backend="tikz"``, unless
    ``options`` choose a backend.
    """
    method = _method(obj, name, show=show)
    parameters = inspect.signature(method).parameters
    first = next(iter(parameters), None)
    options = dict(options)
    suffix = Path(output).suffix.lower() if output is not None else ""
    if "backend" in parameters:
        if suffix in _PLOTLY_SUFFIXES:
            options.setdefault("backend", "plotly")
        elif suffix in _TIKZ_SUFFIXES:
            options.setdefault("backend", "tikz")
    if first in ("path", "directory"):
        if show:
            raise CLIError(f"{name} writes files and cannot use --show; use the movie command or an animation method")
        if output is None:
            raise CLIError(f"{name} writes files and requires -o; use an animation method to display a movie")
        return method(output, **options)
    try:
        inspect.signature(method).bind(**options)
    except TypeError as error:
        raise CLIError(f"{name}: {error}. Run `plasma-plots help {name}` for parameters.") from None
    try:
        result = method(**options)
    except (ValueError, TypeError, IndexError, KeyError) as error:
        dims = getattr(obj, "sizes", {})
        dimensions = ", ".join(f"{key}={value}" for key, value in dims.items())
        raise CLIError(
            f"{name}: {error}\n"
            + (f"Available dimensions (sizes): {dimensions}.\n" if dimensions else "")
            + f"Use integer indices (t=-1) or coordinate values (t=0.5). "
            f"Run `plasma-plots help {name}` for parameters."
        ) from error
    from .accessors import SliceView
    from .plotting import InteractiveSliceViewer

    if isinstance(result, SliceView):
        result = result.viewer()
    if isinstance(result, InteractiveSliceViewer):
        result = result.draw()
    return result


def _save_result(result, path, *, dpi=None) -> list[str]:
    """Save what a plot method returned to ``path``; return the files written.

    Handles a ``PlotResult`` (and anything with ``.save``), a Matplotlib ``FuncAnimation``
    (``.gif`` through Pillow, videos through ffmpeg), a Matplotlib or Plotly figure, a
    ``(fig, axes)`` pair, and paths a method wrote itself. Nothing is written on MPI ranks other
    than 0.
    """
    from matplotlib.animation import FuncAnimation

    from .mpi import SkippedPlot, is_plotting_rank
    from .plotting import PlotResult

    if result is None or isinstance(result, SkippedPlot) or not is_plotting_rank():
        return []
    if isinstance(result, (str, os.PathLike)):
        return [str(result)]
    if isinstance(result, list) and all(isinstance(p, (str, os.PathLike)) for p in result):
        return [str(p) for p in result]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(path).suffix.lower()
    if isinstance(result, FuncAnimation):
        if suffix in _PLOTLY_SUFFIXES:
            raise CLIError("this animation has no Plotly version; save it as .gif or .mp4")
        if suffix in _TIKZ_SUFFIXES:
            raise CLIError("an animation has no TikZ version; save it as .gif or .mp4")
        writer = _ANIMATION_WRITERS.get(suffix)
        result.save(str(path), writer=writer, **({"dpi": dpi} if dpi else {}))
        return [str(path)]
    if isinstance(result, tuple):
        result = result[0]
    if not hasattr(result, "save"):
        result = PlotResult(result)
    if suffix in _PLOTLY_SUFFIXES and not type(result.fig).__module__.startswith("plotly"):
        raise CLIError(f"this plot has no Plotly version; save it as .png, .pdf or .svg instead of {suffix}")
    if suffix in _TIKZ_SUFFIXES and not type(result.fig).__module__.startswith("tikzfigure"):
        if type(result.fig).__module__.startswith("plotly"):
            raise CLIError(f"a Plotly figure has no TikZ version; draw it with backend=tikz for {suffix}")
        result = result.to_tikz()
    return [result.save(path, **({"dpi": dpi} if dpi else {}))]


def _show_result(result):
    """Display a result, keeping Matplotlib figures and animations alive until closed."""
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    from .mpi import SkippedPlot, is_plotting_rank

    if result is None or isinstance(result, SkippedPlot) or not is_plotting_rank():
        return
    if isinstance(result, tuple):
        result = result[0]
    figure = getattr(result, "fig", result)
    if type(figure).__module__.startswith("plotly"):
        figure.show(renderer="browser")
    elif isinstance(result, FuncAnimation) or isinstance(figure, plt.Figure):
        _check_display()
        plt.show(block=True)
    elif type(result).__module__.startswith("pyvista") and hasattr(result, "show"):
        result.show()
    else:
        raise CLIError("this method writes files and has no figure to show; use a plotting or animation method")


def _check_display():
    import matplotlib
    from matplotlib.backends.registry import backend_registry

    backend = matplotlib.get_backend()
    _, gui = backend_registry.resolve_backend(backend)
    if gui is None or "inline" in backend.lower():
        raise _DisplayError(
            f"--show cannot open a window with Matplotlib backend {backend!r}. "
            "Use -o figure.png to save, backend=plotly --show to open a browser, "
            "or configure an interactive backend (e.g. MPLBACKEND=TkAgg) with a working display."
        )


def _finish_result(result, args):
    if type(result).__module__.startswith("pyvista") and hasattr(result, "show"):
        if args.output and Path(args.output).suffix.lower() not in (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        ):
            raise CLIError("3-D screenshots require an image output such as -o scene.png")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.show(screenshot=args.output)
        if args.output:
            print(f"wrote {args.output}")
        return
    if args.output is not None:
        for file in _save_result(result, args.output, dpi=args.dpi):
            print(f"wrote {file}")
    if args.show:
        _show_result(result)


# ---------------------------------------------------------------------------------------------
# Quick looks


def _spatial_dims(array) -> list[str]:
    """The dimensions of ``array`` a plot can draw: logical ones first, size > 1."""
    from .arrays import logical_dims

    logical = [d for d in logical_dims(array) if d in array.dims]
    rest = [d for d in array.dims if d not in logical and d not in _NOT_SPATIAL]
    return [d for d in logical + rest if array.sizes[d] > 1]


def quicklook_plot(array) -> tuple[str, dict] | None:
    """The plot method and options of a quick look at ``array``, or ``None`` if there is none.

    A time series draws ``timeseries``; a field with one drawn dimension a ``lineout`` and one
    with two or more a ``slice`` of its first two (logical ones first), each at ``t=-1`` and at
    position 0 of every other dimension (e.g. ``component=0``, ``eta3=0``). Marker Datasets draw
    their ``trajectories``.

    Parameters
    ----------
    array : xarray.DataArray or xarray.Dataset
        A product.

    Returns
    -------
    tuple of (str, dict) or None
        The plot method's name and its keyword arguments.

    Examples
    --------
    >>> quicklook_plot(phi)  # dims t, eta1, eta2, eta3
    ('slice', {'x': 'eta1', 'y': 'eta2', 't': -1, 'eta3': 0})
    """
    import xarray as xr

    if isinstance(array, xr.Dataset):
        return ("trajectories", {}) if "marker" in array.dims else None
    drawn = _spatial_dims(array)
    rest = {d: 0 for d in array.dims if d != "t" and d not in drawn[:2]}
    at_end = {"t": -1} if "t" in array.dims else {}
    if not drawn:
        return ("timeseries", rest) if "t" in array.dims and array.sizes["t"] > 1 else None
    if len(drawn) == 1:
        return "lineout", {"x": drawn[0], **at_end, **rest}
    return "slice", {"x": drawn[0], "y": drawn[1], **at_end, **rest}


def _file_stem(name: str) -> str:
    return name.strip("/").replace("/", "-") or "product"


def quicklook(
    source,
    directory=None,
    *,
    formats=("png",),
    dpi=None,
    log=None,
    show=False,
    products=None,
    selection=None,
) -> list[str]:
    """Save or display the standard figures of ``source``; return the files written.

    For a Struphy run: the energy budget and the scalars, the equilibrium,
    and a quick look (:func:`quicklook_plot`) at every other product; for a
    Dataset, a quick look at every variable. A figure that fails is reported through ``log`` and
    skipped.

    Parameters
    ----------
    source : struphy.Output or xarray.Dataset
        What to plot, as :func:`open_source` returns it.
    directory : str or pathlib.Path, optional
        Where to save the figures; created if needed. Omit to only display them.
    formats : sequence of str, optional
        File formats, each of every figure, e.g. ``("png", "html")``. Default: ``("png",)``.
    dpi : int, optional
        Resolution of images.
    log : callable, optional
        Receives one line per figure written or skipped. Default: ``print``.
    show : bool, optional
        Display each figure; close its window to continue. Default: ``False``.
    products : sequence of str, optional
        Only draw these products, omitting the standard run overview plots.
    selection : dict, optional
        Dimension selections shared by products: integers index, other numbers select nearest coordinates.

    Returns
    -------
    list of str
        The files written.

    Examples
    --------
    >>> quicklook(out, "figures", formats=("png", "html"))
    """
    from .plotting import energy_names

    log = print if log is None else log
    if directory is None and not show:
        raise CLIError("give -o DIR to save figures or --show to display them")
    directory = Path(directory) if directory is not None else None
    requested = products
    selection = selection or {}
    jobs = []  # (file stem, product, method, options)
    if requested is not None:
        products = list(dict.fromkeys(requested))
        for name in products:
            _get_product(source, name)  # fail clearly on an explicitly requested missing product
    elif _is_output(source):
        names = list(source.scalars.data_vars)
        if energy_names(names):
            jobs.append(("energies", source, "energies", {}))
        if names:
            jobs.append(("scalars", source, "scalars", {}))
        if source.equil is not None:
            jobs.append(("equilibrium", source, "equilibrium", {}))
        products = [n for n in source.keys() if n not in names]
    else:
        products = list(source.data_vars)
        if "marker" in source.dims:  # a marker Dataset: its orbits, not each quantity
            jobs.append(("trajectories", source, "trajectories", {}))
            products = [n for n in products if "marker" not in source[n].dims]
    available_dims = set()
    for name in products:
        try:
            product = _get_product(source, name)
            available_dims.update(product.dims)
            for dim, value in selection.items():
                if dim in product.dims:
                    if isinstance(value, int):
                        product = product.isel({dim: value})
                    else:
                        product = product.sel({dim: value}, method="nearest")
        except (IndexError, ValueError, KeyError) as error:
            raise CLIError(
                f"cannot select {name!r}: {error}; use --select with valid dimension indices or coordinates"
            ) from error
        except CLIError as error:
            log(f"skipped {name}: {error}")
            continue
        chosen = quicklook_plot(product)
        if chosen is None:
            log(f"skipped {name}: nothing to draw")
            continue
        jobs.append((f"{_file_stem(name)}-{chosen[0]}", product, *chosen))

    unknown = set(selection) - available_dims
    if unknown:
        raise CLIError(
            f"unknown selection dimensions: {', '.join(sorted(unknown))}; available: {', '.join(sorted(available_dims))}"
        )
    import matplotlib.pyplot as plt

    written = []
    displayed = 0
    for stem, product, method, options in jobs:
        for fmt in formats if directory is not None else (None,):
            path = directory / f"{stem}.{fmt}" if directory is not None else None
            try:
                result = _call_plot(product, method, options, path)
                files = _save_result(result, path, dpi=dpi) if path is not None else []
                if show:
                    _show_result(result)
                    displayed += 1
            except _DisplayError:
                raise
            except Exception as error:  # one broken figure must not stop the rest
                log(f"skipped {path.name if path is not None else stem}: {type(error).__name__}: {error}")
                continue
            finally:
                plt.close("all")
            written += files
            for file in files:
                log(f"wrote {file}")
    if show and not displayed and not written:
        raise CLIError("no figure could be drawn")
    return written


# ---------------------------------------------------------------------------------------------
# info


def _dims_text(obj) -> str:
    return ", ".join(f"{d}: {n}" for d, n in obj.sizes.items()) or "-"


def _coordinate_range(coord) -> str:
    import numpy as np

    values = np.asarray(coord.values)
    if values.size == 0:
        return "(empty)"
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        return f"({values.size} values)"
    units = coord.attrs.get("units", "")
    return f"{values[0]:.4g} … {values[-1]:.4g} ({values.size}){' ' + units if units else ''}"


def _table(rows, header) -> str:
    rows = [header, *rows]
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _describe(source, path) -> str:
    """The text of ``plasma-plots info``: every product with its dimensions and units."""
    import xarray as xr

    lines = []
    if _is_output(source):
        lines += [f"Struphy run: {Path(path)}", f"Label: {source.label}", ""]
        kinds = (
            ("field", source.field_catalog),
            ("distribution", source.distribution_catalog),
            ("density", source.density_catalog),
            ("orbits", source.orbit_catalog),
        )
        rows = [
            (
                name,
                "scalar",
                _dims_text(source.scalars[name]),
                source.scalars[name].attrs.get("units", ""),
            )
            for name in source.scalars.data_vars
        ]
        for kind, catalog in kinds:
            for name in catalog:
                product = catalog[name]
                units = "" if isinstance(product, xr.Dataset) else product.attrs.get("units", "")
                rows.append((name, kind, _dims_text(product), units))
        lines.append(_table(rows, ("Product", "Kind", "Dimensions", "Units")))
        coords = next((catalog[n] for _, catalog in kinds for n in catalog), source.scalars).coords
    else:
        lines += [f"File: {Path(path)}", ""]
        rows = [
            (
                name,
                _dims_text(var),
                var.attrs.get("units", ""),
                var.attrs.get("long_name", var.attrs.get("label", "")),
            )
            for name, var in source.data_vars.items()
        ]
        lines.append(_table(rows, ("Variable", "Dimensions", "Units", "Description")))
        coords = source.coords
    dims = [c for c in coords if coords[c].ndim == 1 and c in coords.dims]
    if dims:
        lines += [
            "",
            _table(
                [(c, _coordinate_range(coords[c])) for c in dims],
                ("Coordinate", "Range"),
            ),
        ]
    lines += [
        "",
        f"Plot one with `plasma-plots plot {Path(path)} PRODUCT METHOD key=value ... -o FILE`;",
        f"`plasma-plots plot {Path(path)} PRODUCT --list` lists its plot methods.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Commands


def _cmd_guide(args):
    import plasma_plots

    print(plasma_plots.__doc__)


def _suggest(name, choices):
    matches = difflib.get_close_matches(name, choices, n=1)
    return f"; did you mean {matches[0]!r}?" if matches else ""


def _cmd_help(args):
    from .accessors import ArrayPlots, DatasetPlots
    from .output_accessors import OutputPlots, ProfilePlots

    classes = {"array": ArrayPlots, "dataset": DatasetPlots, "run": OutputPlots}
    kind = args.kind or ("run" if args.method.startswith("profile.") else "array")
    if args.kind is None and not args.method.startswith("profile."):
        matches = [key for key, cls in classes.items() if inspect.isfunction(getattr(cls, args.method, None))]
        if len(matches) == 1:
            kind = matches[0]
    cls = classes[kind]
    name = args.method
    if kind == "run" and name.startswith("profile."):
        cls, name = ProfilePlots, name.removeprefix("profile.")
    method = getattr(cls, name, None) if not name.startswith("_") else None
    if not inspect.isfunction(method):
        choices = [n for n, m in inspect.getmembers(cls, inspect.isfunction) if not n.startswith("_")]
        raise CLIError(
            f"no {kind} plot method {args.method!r}{_suggest(name, choices)}; choose --kind array, dataset or run. Methods: {', '.join(choices)}"
        )
    signature = inspect.signature(method)
    signature = signature.replace(parameters=[p for n, p in signature.parameters.items() if n != "self"])
    print(f"{kind} plot: {args.method}{signature}\n")
    print(inspect.getdoc(method) or "No additional documentation.")
    destination = "-o FILE" if next(iter(signature.parameters), None) in ("path", "directory") else "--show"
    product = "." if kind in ("run", "dataset") else "PRODUCT"
    print(f"\nCLI: plasma-plots plot PATH {product} {args.method} key=value ... {destination}")
    print("Use -o FILE to save. PRODUCT '.' means the whole run or Dataset.")
    print("Integer selections index (t=-1); decimals select nearest coordinates (t=0.5).")
    print("Quote lists/dicts in your shell: 'levels=[0.1,0.5]' or 'cuts={\"eta3\":[0,0.5]}'.")


def _cmd_api(args):
    from ._api import api_index

    print(api_index(), end="")


def _cmd_info(args):
    source = open_source(args.path)
    print(_describe(source, args.path))


def _cmd_plot(args):
    source = open_source(args.path)
    product = _get_product(source, args.product)
    if args.list or args.method is None:
        print("\n".join(plot_methods(product, show=args.show)))
        if args.method is None and not args.list:
            print(f"\nChoose one: plasma-plots plot {args.path} {args.product} METHOD ... (-o FILE or --show)")
        return
    options = _parse_options(args.options, source)
    result = _call_plot(product, args.method, options, args.output, show=args.show)
    _finish_result(result, args)


def _cmd_movie(args):
    source = open_source(args.path)
    product = _get_product(source, args.product)
    options = _parse_options(args.options, source)
    sweep = options.get("sweep", "t")
    # the dimensions left to draw once the options select the others
    free = [d for d in _spatial_dims(product) if d not in options and d != sweep]
    method = "animation" if len(free) >= 2 else "line_animation"
    result = _call_plot(product, method, options, args.output, show=args.show)
    _finish_result(result, args)


def _cmd_quicklook(args):
    source = open_source(args.path)
    formats = [f.strip().lstrip(".") for f in args.format.split(",") if f.strip()]
    written = quicklook(
        source,
        args.output,
        formats=formats,
        dpi=args.dpi,
        show=args.show,
        products=args.products,
        selection=_parse_options(args.select, source),
    )
    if not written and not args.show:
        raise CLIError("no figure could be drawn")


def _build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``plasma-plots`` command."""
    parser = argparse.ArgumentParser(
        prog="plasma-plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Quick looks at plasma simulation output: a Struphy run folder, or a file xarray "
            "can read (netCDF, zarr). PRODUCT is a product or variable name, or '.' for the "
            "whole run (out.plot) or file (dataset.plasma.plot)."
        ),
        epilog=(
            "Examples:\n"
            "  plasma-plots info sim_1\n"
            "  plasma-plots plot sim_1 em_fields/b_field slice t=-1 component=2 eta3=0 --show\n"
            "  plasma-plots quicklook sim_1 -o figures/\n"
            "  plasma-plots help slice\n\n"
            "Values of key=value options: an integer is a position (t=-1), a decimal number "
            "the nearest coordinate value (t=0.35); true, false, none; commas make a list "
            "(eta3=0,0.5); [..] or {..} is JSON; @NAME another product (other=@em_fields/B)."
        ),
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    def command(name, func, help):
        sub = commands.add_parser(
            name,
            help=help,
            description=help,
            add_help=name != "plot",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Examples:\n  plasma-plots plot run.nc phi slice t=-1 eta3=0 --show\n  plasma-plots help slice\n  plasma-plots info run.nc\n\nSelections: t=-1 indexes the last step; t=0.5 selects the nearest time.\nQuote lists and JSON to protect them from shell expansion.",
        )
        if name == "plot":
            sub.add_argument(
                "-h",
                "--help",
                action=_MethodHelp,
                nargs=0,
                help="show command help, or help for the selected METHOD",
            )
        sub.set_defaults(func=func)
        return sub

    def source_arguments(sub, product=True):
        sub.add_argument("path", metavar="PATH", help="a Struphy run folder, or a netCDF/zarr file")
        if product:
            sub.add_argument(
                "product",
                metavar="PRODUCT",
                help="a product name, or '.' for all of PATH",
            )
        sub.add_argument(
            "--traceback",
            action="store_true",
            help="show the full traceback of an error",
        )

    def output_arguments(sub):
        sub.add_argument(
            "-o",
            "--output",
            metavar="FILE",
            help="the file to write; its extension picks the format",
        )
        sub.add_argument("--dpi", type=int, help="resolution of images")
        sub.add_argument("--show", action="store_true", help="display the figure; -o is optional")

    sub = command(
        "help",
        _cmd_help,
        "show a plot method's parameters and examples without opening data",
    )
    sub.add_argument("method", metavar="METHOD", help="e.g. slice, viewer, energies, profile.gantt")
    sub.add_argument(
        "--kind",
        choices=("array", "dataset", "run"),
        help="plot namespace (default: array; run for profile.*)",
    )

    command("guide", _cmd_guide, "print the Python package guide")
    command(
        "api",
        _cmd_api,
        "print the API index: every method and function with its signature",
    )

    sub = command("info", _cmd_info, "list the products of PATH with their dimensions and units")
    source_arguments(sub, product=False)

    sub = command(
        "plot",
        _cmd_plot,
        "call one plot method of PRODUCT with key=value options and save or show the figure",
    )
    source_arguments(sub)
    sub.add_argument(
        "method",
        metavar="METHOD",
        nargs="?",
        help="a plot method, e.g. slice, lineout, timeseries, energies",
    )
    sub.add_argument(
        "options",
        metavar="key=value",
        nargs="*",
        help="the method's options, e.g. t=-1 x=eta1",
    )
    sub.add_argument("--list", action="store_true", help="list the plot methods of PRODUCT")
    output_arguments(sub)

    sub = command(
        "movie",
        _cmd_movie,
        "animate PRODUCT over t; save to .gif, .mp4 or .html, or display with --show",
    )
    source_arguments(sub)
    sub.add_argument(
        "options",
        metavar="key=value",
        nargs="*",
        help="the animation's options, e.g. eta3=0 step=2",
    )
    output_arguments(sub)

    sub = command(
        "quicklook",
        _cmd_quicklook,
        "save or show the standard figures of PATH (energies, scalars, a slice of every field, ...)",
    )
    source_arguments(sub, product=False)
    sub.add_argument("-o", "--output", metavar="DIR", help="the folder to write into")
    sub.add_argument(
        "--format",
        default="png",
        help="file formats, comma separated, e.g. png,html (default: png)",
    )
    sub.add_argument("--dpi", type=int, help="resolution of images")
    sub.add_argument("--show", action="store_true", help="display each figure; -o is optional")
    sub.add_argument(
        "--products",
        nargs="+",
        metavar="PRODUCT",
        help="only these products; omit run overview plots",
    )
    sub.add_argument(
        "--select",
        nargs="+",
        default=[],
        metavar="DIM=VALUE",
        help="select dimensions before plotting, e.g. t=-1 component=2 eta3=0",
    )
    return parser


def _validate_args(args):
    """Reject usage errors before opening or post-processing a source."""
    if args.command not in ("plot", "movie", "quicklook"):
        return
    listing = args.command == "plot" and (args.list or args.method is None)
    if not listing and args.output is None and not args.show:
        raise CLIError("choose --show to display or -o FILE to save (use -o DIR for quicklook)")
    if args.command == "plot" and args.method and not args.list:
        from .accessors import ArrayPlots, DatasetPlots
        from .output_accessors import OutputPlots, ProfilePlots

        methods = {}
        for cls, prefix in (
            (ArrayPlots, ""),
            (DatasetPlots, ""),
            (OutputPlots, ""),
            (ProfilePlots, "profile."),
        ):
            methods.update(
                {prefix + n: m for n, m in inspect.getmembers(cls, inspect.isfunction) if not n.startswith("_")}
            )
        if args.method not in methods:
            raise CLIError(
                f"unknown plot method {args.method!r}{_suggest(args.method, methods)}. Use `plasma-plots plot PATH PRODUCT --list` or `plasma-plots help METHOD`. "
                f"If {args.method!r} is a product, try `plasma-plots plot PATH {args.method} timeseries`."
            )
        method = methods[args.method]
        if not _is_command_line_method(method) and not args.show:
            raise CLIError(f"{args.method} is an interactive method; add --show to display it")
        parameters = list(inspect.signature(method).parameters)
        if args.show and len(parameters) > 1 and parameters[1] in ("path", "directory"):
            raise CLIError(
                f"{args.method} requires -o and cannot use --show; use the movie command or an animation method"
            )
    if args.dpi is not None and args.dpi <= 0:
        raise CLIError("--dpi must be a positive integer, e.g. --dpi 150")
    pairs = getattr(args, "options", []) + getattr(args, "select", [])
    # Parse references only after opening the source, but validate their surrounding syntax now.
    parsed = _parse_options([pair.split("=", 1)[0] + "=reference" if "=@" in pair else pair for pair in pairs])
    if args.command == "quicklook":
        for key, value in parsed.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CLIError(f"--select {key} needs a numeric index or coordinate, e.g. {key}=0")
        formats = [f.strip().lstrip(".") for f in args.format.split(",") if f.strip()]
        if not formats:
            raise CLIError("--format needs at least one format, e.g. png or png,html")
        if args.products and "." in args.products:
            raise CLIError("--products expects individual products; omit it for the standard figures")


def main(argv=None) -> int:
    """Run the ``plasma-plots`` command.

    Parameters
    ----------
    argv : list of str, optional
        The arguments after the command name. Default: ``sys.argv[1:]``.

    Returns
    -------
    int
        The exit code: 0 on success, 1 on an error (with a diagnostic), 2 on a usage error
        (from argparse, which exits itself).

    Examples
    --------
    >>> main(
    ...     [
    ...         "plot",
    ...         "sim_1",
    ...         "em_fields/phi",
    ...         "slice",
    ...         "t=-1",
    ...         "eta3=0",
    ...         "-o",
    ...         "phi.png",
    ...     ]
    ... )
    0
    """
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        parser.print_help()
        return 0
    import matplotlib

    if args.func not in (_cmd_guide, _cmd_api, _cmd_help) and not getattr(args, "show", False):
        matplotlib.use("Agg")  # files only, also without a display
    try:
        _validate_args(args)
        args.func(args)
    except CLIError as error:
        if getattr(args, "traceback", False):
            raise
        print(f"plasma-plots: error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # e.g. an option the plot method does not take
        if getattr(args, "traceback", False):
            raise
        print(
            f"plasma-plots: error: {type(error).__name__}: {error}\n"
            "(Use --traceback for details, `plasma-plots help METHOD` for plot options, or COMMAND --help for usage.)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
