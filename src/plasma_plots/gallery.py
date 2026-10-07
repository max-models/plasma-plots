"""Output helpers of the struphy-hub example gallery: figure files, profiling exports and metadata.

The example scripts on https://struphy-hub.github.io/examples/ use these helpers to write
their results, so that a copied example runs on its own with ``pip install
"plasma-plots[gallery]"``. Every helper writes to the current directory and names its files
after the example's stem: ``<stem>.png``, ``<stem>.plotly.json`` and ``<stem>.html`` for each
figure, ``<stem>.metadata.json`` for the measured values, and ``<stem>-profile.h5`` with its
plot data for the profiling. The website's build reads exactly these files.

The scripts also run under MPI (``mpirun -n 4 python <script>.py``): the simulation, the
post-processing and the analysis run on every rank, and only rank 0 writes files. The
helpers take care of that, so a script needs no rank checks of its own.

Importing this module sets Struphy's logging level to INFO, which prints one block per time
step (step number, times, wall clock and scalar quantities), as one wants to see in a CI log.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

try:
    from struphy import set_logging_level as _set_logging_level
    from struphy.utils.mpi_launch import launched_under_mpi as _launched_under_mpi
except (
    ImportError
):  # the figure helpers work without struphy; the example scripts always have it
    _set_logging_level = None

    def _launched_under_mpi() -> bool:
        return False


__all__ = [
    "GANTT_MAX_INTERVALS",
    "barrier",
    "export_profiling",
    "heatmap_figure",
    "heatmap_movie",
    "is_root",
    "merge_metadata",
    "save_extra_figure",
    "save_figure",
    "space_time_figure",
]

# Importing mpi4py initializes MPI. In particular, a serial command run inside
# a Slurm allocation must not initialize MPI merely because mpi4py is present.
# Struphy's launcher detector also supports STRUPHY_MPI=0 for that case.
if _launched_under_mpi():
    try:
        from mpi4py import MPI

        _COMM = MPI.COMM_WORLD
    except ImportError:  # a serial install without MPI
        _COMM = None
else:
    _COMM = None

#: Gantt bars kept by :func:`export_profiling`. A gantt bar is one call, not an aggregate, and a run
#: with thousands of steps draws tens of thousands of near-identical bars -- heavy enough to hang
#: the tab. Keeping the first ones in time order leaves the setup phase plus several complete
#: step-loop iterations.
GANTT_MAX_INTERVALS = 5000

if _set_logging_level is not None:
    _set_logging_level(logging.INFO)


def is_root() -> bool:
    """Tell whether this process writes files: MPI rank 0, or a serial run.

    Returns
    -------
    bool
        True on rank 0 and in a serial run.

    Examples
    --------
    >>> if is_root():
    ...     print(f"measured rate: {rate:.4f}")
    """
    return _COMM is None or _COMM.Get_rank() == 0


def barrier() -> None:
    """Wait until every MPI rank got here; does nothing in a serial run.

    Use it, for example, before rank 0 reads a file that the ranks wrote together.
    """
    if _COMM is not None:
        _COMM.Barrier()


def _plot_durations(*args, metrics=("total",), **kwargs):
    """Export several metrics with scope-profiler's one-metric API."""
    from scope_profiler import plot_durations

    if isinstance(metrics, str):
        metrics = (metrics,)
    metrics = tuple(metrics)
    data_filepath = kwargs.pop("data_filepath", None)
    if not data_filepath or len(metrics) == 1:
        return plot_durations(
            *args, metric=metrics[0], data_filepath=data_filepath, **kwargs
        )

    path = Path(data_filepath)
    payloads = []
    results = []
    for metric in metrics:
        temporary_path = path.with_name(f"{path.stem}.{metric}{path.suffix}")
        results.extend(
            plot_durations(*args, metric=metric, data_filepath=temporary_path, **kwargs)
        )
        payloads.append(json.loads(temporary_path.read_text()))
        temporary_path.unlink()

    merged = payloads[0]
    merged["bars"] = [bar for payload in payloads for bar in payload["bars"]]
    merged["metrics"] = list(metrics)
    path.write_text(json.dumps(merged))
    return results


def _scientific_ticks(figure) -> None:
    """Label tick values as 1×10⁻⁶ instead of Plotly's default SI prefixes (µ, n, k, M, ...).

    Only fills in `exponentformat` where the example has not chosen one itself.
    """

    def default(obj) -> None:
        if obj.exponentformat is None:
            obj.exponentformat = "power"

    figure.for_each_xaxis(default)
    figure.for_each_yaxis(default)
    figure.for_each_scene(
        lambda scene: [
            default(axis) for axis in (scene.xaxis, scene.yaxis, scene.zaxis)
        ]
    )
    figure.for_each_coloraxis(lambda coloraxis: default(coloraxis.colorbar))
    for trace in figure.data:
        # Touching `colorbar` on a trace that has none would make Plotly draw an empty one.
        marker = getattr(trace, "marker", None)
        if marker is not None and marker.showscale:
            default(marker.colorbar)
        if (
            "colorbar" in trace._valid_props
            and trace.showscale is not False
            and trace.type != "scatter"
        ):
            default(trace.colorbar)


def save_figure(
    figure,
    stem: str,
    *,
    width: int = 1100,
    height: int = 650,
    suffix: str = "",
    static_z=None,
    static_data=None,
    static_active=None,
) -> None:
    """Write static PNG, Plotly JSON and standalone HTML versions of a figure.

    The files are ``<stem><suffix>.png``, ``.plotly.json`` and ``.html`` in the current
    directory. The PNG is also copied to ``../images/examples/`` when that directory exists,
    as the gallery thumbnail. Only rank 0 writes. Tick labels get scientific notation where
    the figure has not chosen a format.

    An animation that starts at t = 0 can still have an informative static image: ``static_z``
    or ``static_data`` replace what the PNG shows, and the JSON and HTML keep the animation.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure to save.
    stem : str
        The example's stem, which names the files.
    width : int, optional
        Width of the PNG in pixels, before the factor 2 of its scale. Default: 1100.
    height : int, optional
        Height of the PNG in pixels, before the factor 2 of its scale. Default: 650.
    suffix : str, optional
        Appended to the stem, e.g. ``"-space-time"`` for an additional figure.
    static_z : array, optional
        Replaces the heatmap of the first trace in the PNG only, e.g. the ``static_z`` that
        :func:`heatmap_movie` returns.
    static_data : list of plotly traces, optional
        The traces of the PNG, e.g. one frame's ``data``, drawn with the figure's layout.
    static_active : int, optional
        The slider position shown in the PNG, with ``static_z`` or ``static_data``.

    See Also
    --------
    save_extra_figure : Save a further figure and return its metadata entry.

    Examples
    --------
    >>> save_figure(figure, "weak-landau-damping")
    """
    if not is_root():
        return
    _scientific_ticks(figure)
    png_path = Path(f"{stem}{suffix}.png")
    json_path = Path(f"{stem}{suffix}.plotly.json")
    html_path = Path(f"{stem}{suffix}.html")
    if static_data is not None:
        still = go.Figure(data=static_data, layout=figure.layout)
        if static_active is not None and still.layout.sliders:
            still.layout.sliders[0].active = (
                static_active  # the slider shows the still's frame
            )
        still.write_image(png_path, width=width, height=height, scale=2)
    elif static_z is not None:
        initial_z = figure.data[0].z
        initial_active = (
            figure.layout.sliders[0].active if figure.layout.sliders else None
        )
        figure.data[0].z = static_z
        if static_active is not None and figure.layout.sliders:
            figure.layout.sliders[0].active = static_active
        figure.write_image(png_path, width=width, height=height, scale=2)
        figure.data[0].z = initial_z
        if figure.layout.sliders:
            figure.layout.sliders[0].active = initial_active
    else:
        figure.write_image(png_path, width=width, height=height, scale=2)
    figure.write_json(json_path, pretty=False)
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
        auto_play=False,
        config={"responsive": True},
    )
    print(f"Saved {png_path.resolve()}")
    print(f"Saved {json_path.resolve()}")
    print(f"Saved {html_path.resolve()}")
    # The gallery, the model pages and the static fallbacks of the example page (shown on phones and
    # without JavaScript, in place of the interactive plot) read the PNG from the images directory,
    # under the same name as here. Both directories are generated and untracked.
    images = Path("../images/examples")
    if images.is_dir():
        shutil.copyfile(png_path, images / png_path.name)


def save_extra_figure(
    figure,
    stem: str,
    key: str,
    *,
    alt: str,
    caption: str,
    static_z=None,
    static_active=None,
) -> dict:
    """Save an additional figure of an example and return its entry for the ``figures`` metadata.

    Writes ``<stem>-<key>.png``, ``.plotly.json`` and ``.html`` with :func:`save_figure`. The
    example page shows every entry of ``figures`` below its main figure: pass the list to
    ``merge_metadata(stem, figures=[...])``.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure to save.
    stem : str
        The example's stem.
    key : str
        Names the figure within the example, and its files.
    alt : str
        Alternative text of the image.
    caption : str
        Caption below the figure on the example page.
    static_z : array, optional
        Replaces the heatmap of the first trace in the PNG only; see :func:`save_figure`.
    static_active : int, optional
        The slider position shown in the PNG; see :func:`save_figure`.

    Returns
    -------
    dict
        The entry for the ``figures`` list of :func:`merge_metadata`: ``key``,
        ``interactive`` and ``thumbnail`` paths, ``alt`` and ``caption``.

    Examples
    --------
    >>> figures = [
    ...     save_extra_figure(
    ...         space_time,
    ...         "weak-landau-damping",
    ...         "space-time",
    ...         alt="Space-time map of E",
    ...         caption="E(x, t) of the run.",
    ...     )
    ... ]
    >>> merge_metadata("weak-landau-damping", figures=figures)
    """
    save_figure(
        figure,
        stem,
        suffix=f"-{key}",
        static_z=static_z,
        static_active=static_active,
    )
    return {
        "key": key,
        "interactive": f"/examples/{stem}-{key}.plotly.json",
        "thumbnail": f"/images/examples/{stem}-{key}.png",
        "alt": alt,
        "caption": caption,
    }


def _finish_layout(figure, title, xaxis_title, yaxis_title, **layout):
    layout.setdefault("margin", {"l": 70, "r": 30, "t": 80, "b": 60})
    figure.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        autosize=True,
        **layout,
    )
    return figure


def heatmap_figure(
    data,
    *,
    x: str,
    y: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colorbar_title: str = "",
    colorscale: str = "Viridis",
    zmin=None,
    zmax=None,
    x_values=None,
    y_values=None,
):
    """Draw a two-dimensional array as a Plotly heatmap.

    Parameters
    ----------
    data : xarray.DataArray
        The values, with the two dimensions ``x`` and ``y``.
    x : str
        The dimension along the horizontal axis.
    y : str
        The dimension along the vertical axis.
    title : str
        Title of the figure.
    xaxis_title : str
        Title of the horizontal axis.
    yaxis_title : str
        Title of the vertical axis.
    colorbar_title : str, optional
        Title of the color bar.
    colorscale : str, optional
        Plotly color scale. Default: ``"Viridis"``.
    zmin : float, optional
        Lower end of the color scale. Default: the data's minimum.
    zmax : float, optional
        Upper end of the color scale. Default: the data's maximum.
    x_values : array, optional
        Replaces the coordinate of ``x``, e.g. to plot a logical coordinate in physical length.
    y_values : array, optional
        Replaces the coordinate of ``y``.

    Returns
    -------
    plotly.graph_objects.Figure
        The heatmap.

    Examples
    --------
    >>> heatmap_figure(
    ...     f.plasma.analysis.spatial_average(),
    ...     x="t",
    ...     y="v1",
    ...     title="f(v, t)",
    ...     xaxis_title="t [a.u.]",
    ...     yaxis_title="v [a.u.]",
    ... )
    """
    x_values = data[x].values if x_values is None else x_values
    y_values = data[y].values if y_values is None else y_values
    figure = go.Figure(
        go.Heatmap(
            z=data.transpose(y, x).values,
            x=x_values,
            y=y_values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": colorbar_title},
        )
    )
    return _finish_layout(figure, title, xaxis_title, yaxis_title)


def space_time_figure(
    data,
    *,
    space: str,
    title: str,
    colorbar_title: str,
    xaxis_title: str = "x [a.u.]",
    x_values=None,
    colorscale: str = "RdBu",
):
    """Draw a space-time map of a field: space along x, time up, colors symmetric about zero.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with the dimensions ``t`` and ``space``.
    space : str
        The spatial dimension, e.g. ``"eta1"``.
    title : str
        Title of the figure.
    colorbar_title : str
        Title of the color bar.
    xaxis_title : str, optional
        Title of the horizontal axis. Default: ``"x [a.u.]"``.
    x_values : array, optional
        Replaces the coordinate of ``space``, e.g. in physical length.
    colorscale : str, optional
        Plotly color scale. Default: ``"RdBu"``.

    Returns
    -------
    plotly.graph_objects.Figure
        The heatmap.

    Examples
    --------
    >>> e_x = out.evaluate("em_fields/e_field").isel(component=0, eta2=0, eta3=0)
    >>> space_time_figure(e_x, space="eta1", title="E(x, t)", colorbar_title="E_x")
    """
    limit = float(abs(data).max())
    return heatmap_figure(
        data,
        x=space,
        y="t",
        x_values=x_values,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="t [a.u.]",
        colorbar_title=colorbar_title,
        colorscale=colorscale,
        zmin=-limit,
        zmax=limit,
    )


def heatmap_movie(
    data,
    *,
    x: str,
    y: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colorbar_title: str = "",
    sweep: str = "t",
    colorscale: str = "Viridis",
    zmin=0.0,
    zmax=None,
    x_values=None,
    y_values=None,
    max_frames: int = 150,
):
    """Animate a three-dimensional array as a Plotly heatmap, one frame per ``sweep`` value.

    A frame per saved step would embed tens of megabytes in the page, so at most
    ``max_frames`` evenly spaced frames are kept.

    Parameters
    ----------
    data : xarray.DataArray
        The values, with the dimensions ``x``, ``y`` and ``sweep``.
    x : str
        The dimension along the horizontal axis.
    y : str
        The dimension along the vertical axis.
    title : str
        Title of the figure.
    xaxis_title : str
        Title of the horizontal axis.
    yaxis_title : str
        Title of the vertical axis.
    colorbar_title : str, optional
        Title of the color bar.
    sweep : str, optional
        The dimension the frames run over. Default: ``"t"``.
    colorscale : str, optional
        Plotly color scale. Default: ``"Viridis"``.
    zmin : float, optional
        Lower end of the color scale. Default: 0.
    zmax : float, optional
        Upper end of the color scale. Default: each frame's maximum.
    x_values : array, optional
        Replaces the coordinate of ``x``, e.g. in physical length.
    y_values : array, optional
        Replaces the coordinate of ``y``.
    max_frames : int, optional
        The most frames kept. Default: 150.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        The animated heatmap, with a play button and a slider.
    static_z : numpy.ndarray
        A well-developed frame from the middle of the sweep, for
        ``save_figure(..., static_z=static_z)``.

    Examples
    --------
    >>> f = out.evaluate("kinetic_ions/e1_v1_density/f")
    >>> movie, static_z = heatmap_movie(
    ...     f,
    ...     x="eta1",
    ...     y="v1",
    ...     title="f(x, v)",
    ...     xaxis_title="x [a.u.]",
    ...     yaxis_title="v [a.u.]",
    ... )
    >>> save_figure(
    ...     movie,
    ...     "two-stream-instability",
    ...     suffix="-phase-space",
    ...     static_z=static_z,
    ... )
    """
    x_values = data[x].values if x_values is None else x_values
    y_values = data[y].values if y_values is None else y_values
    frames_data = data.transpose(sweep, y, x).values
    labels = data[sweep].values
    picks = np.linspace(0, len(labels) - 1, min(max_frames, len(labels)), dtype=int)

    def heatmap(z, **extra):
        return go.Heatmap(
            z=z,
            x=x_values,
            y=y_values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            **extra,
        )

    frames = [
        go.Frame(name=f"{labels[i]:.1f}", data=[heatmap(frames_data[i])]) for i in picks
    ]
    figure = go.Figure(
        data=[heatmap(frames_data[0], colorbar={"title": colorbar_title})],
        frames=frames,
    )
    _finish_layout(
        figure,
        title,
        xaxis_title,
        yaxis_title,
        margin={"l": 70, "r": 30, "t": 80, "b": 130},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.28,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 30, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                ],
            },
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in frames
                ],
                "x": 0.12,
                "len": 0.88,
                "y": -0.18,
                "currentvalue": {"prefix": f"{sweep} = "},
            },
        ],
    )
    return figure, frames_data[len(frames_data) // 2]


def merge_metadata(stem: str, **fields) -> Path:
    """Add result fields to ``<stem>.metadata.json``, keeping the fields already there.

    Only rank 0 writes.

    Parameters
    ----------
    stem : str
        The example's stem, which names the file.
    **fields
        The fields to add or replace, e.g. measured values, ``figures`` from
        :func:`save_extra_figure` and the fields :func:`export_profiling` returns.

    Returns
    -------
    pathlib.Path
        The metadata file.

    Raises
    ------
    RuntimeError
        If a value is a non-finite float: NaN and Infinity are not valid JSON, and a
        non-finite result is a broken run.

    Examples
    --------
    >>> merge_metadata(
    ...     "weak-landau-damping", measuredDampingRate=rate, **profiling
    ... )
    """
    path = Path(f"{stem}.metadata.json")
    if not is_root():
        return path
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update(fields)
    # NaN and Infinity are not valid JSON: Python writes them anyway, and the site build then fails to
    # parse the file. A non-finite result is a broken run, so stop here and name it.
    broken = [
        key
        for key, value in metadata.items()
        if isinstance(value, float) and not np.isfinite(value)
    ]
    if broken:
        raise RuntimeError(
            f"Non-finite values in the metadata of {stem}: {', '.join(broken)}; refusing to publish the run"
        )
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved {path.resolve()}")
    return path


def export_profiling(sim, stem: str) -> dict:
    """Export the scope-profiler data of a run made with ``profiling_activated=True``.

    Writes the raw HDF5 (``<stem>-profile.h5``) and the plot data that the example page draws
    as Plotly figures: durations, gantt and region statistics as JSON. Needs
    ``scope-profiler[pproc]``, which the ``gallery`` extra installs. Call it on every rank;
    only rank 0 writes.

    Parameters
    ----------
    sim : struphy.Simulation
        The simulation, after ``sim.run(profiling_activated=True)``.
    stem : str
        The example's stem, which names the files.

    Returns
    -------
    dict
        The metadata fields that point to the files, for :func:`merge_metadata`.

    Examples
    --------
    >>> profiling = export_profiling(sim, "weak-landau-damping")
    >>> merge_metadata("weak-landau-damping", **profiling)
    """
    from scope_profiler import plot_gantt, read_h5, write_region_statistics_json

    barrier()  # the ranks write the profiling file together
    profile_h5_path = Path(f"{stem}-profile.h5")
    durations_path = Path(f"{stem}-durations.json")
    gantt_path = Path(f"{stem}-gantt.json")
    region_stats_path = Path(f"{stem}-region-stats.json")
    fields = {
        "profilingData": f"/examples/{profile_h5_path.name}",
        "profilingDurations": f"/examples/{durations_path.name}",
        "profilingGantt": f"/examples/{gantt_path.name}",
        "profilingRegionStats": f"/examples/{region_stats_path.name}",
    }
    if not is_root():
        return fields

    profile_reader = read_h5(sim.profiling_filepath)
    shutil.copyfile(sim.profiling_filepath, profile_h5_path)

    # `stack_children` splits each bar into the region's own time plus one segment per region it
    # calls, which is what the page's durations chart stacks; it only decomposes total/avg.
    # `sort_by` fixes the region order the chart draws, biggest total first.
    _plot_durations(
        [profile_reader],
        ranks=[0],
        metrics=("total", "avg"),
        sort_by="total",
        stack_children=True,
        data_filepath=durations_path,
        data_format="json",
        verbose=False,
    )
    # The stacked export is a dense region x segment grid, but a call graph is sparse (~90% zeros
    # for regions that never call each other); the chart reads a missing pair as null, so dropping
    # them cuts the file ~10x with no change on the page.
    durations_payload = json.loads(durations_path.read_text())
    durations_payload["bars"] = [
        bar for bar in durations_payload["bars"] if bar["value_seconds"]
    ]
    durations_path.write_text(json.dumps(durations_payload))

    plot_gantt(
        [profile_reader],
        ranks=[0],
        data_filepath=gantt_path,
        data_format="json",
        verbose=False,
    )
    write_region_statistics_json([profile_reader], region_stats_path, ranks=[0])
    # Keep the run's recorded host alongside its time and rank count for the gallery summary.
    # Read it from the profile, since exports can be regenerated on a different machine.
    region_stats_payload = json.loads(region_stats_path.read_text())
    region_stats_payload["files"][0]["hostname"] = profile_reader.metadata.get(
        "hostname"
    )
    region_stats_path.write_text(json.dumps(region_stats_payload))

    gantt_payload = json.loads(gantt_path.read_text())
    if len(gantt_payload["intervals"]) > GANTT_MAX_INTERVALS:
        gantt_payload["intervals"] = sorted(
            gantt_payload["intervals"], key=lambda c: c["start_seconds"]
        )[:GANTT_MAX_INTERVALS]
        gantt_path.write_text(json.dumps(gantt_payload))

    print(f"Saved {profile_h5_path.resolve()}")
    print(
        f"Saved {durations_path.resolve()}, {gantt_path.resolve()}, {region_stats_path.resolve()}"
    )
    return fields
