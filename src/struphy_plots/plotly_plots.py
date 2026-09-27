"""Interactive Plotly figures of labeled arrays, for web pages and notebooks.

The functions here return a :class:`plotly.graph_objects.Figure`; call ``.show()`` on it, or
save it with :func:`save_figure`. They need Plotly (``pip install "struphy-plots[plotly]"``);
the PNG of :func:`save_figure` also needs Kaleido. The same plots are methods of
``array.struphy.plotly``, e.g. ``field.struphy.plotly.space_time()``. Axis titles are plain
text: Plotly draws LaTeX only where MathJax is loaded.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from .arrays import validate_array

#: Plain-text names of the dimensions Struphy output uses, for axis titles.
PLAIN_LABELS = {
    "t": "t",
    "eta1": "η₁",
    "eta2": "η₂",
    "eta3": "η₃",
    "v1": "v₁",
    "v2": "v₂",
    "v3": "v₃",
    "omega": "ω",
    "k": "k",
}

_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError('Plotly figures need plotly: pip install "struphy-plots[plotly]"') from error
    return go


def _axis_title(data: xr.DataArray, dim: str) -> str:
    """The dimension's plain-text name, with the coordinate's units when it has any."""
    coord = data.coords.get(dim)
    long_name = "" if coord is None else coord.attrs.get("long_name", "")
    label = PLAIN_LABELS.get(dim) or (long_name if long_name and "$" not in long_name else dim)
    unit = "" if coord is None else coord.attrs.get("units", "")
    return f"{label} [{unit}]" if unit and unit != "rad / coordinate unit" else label


def _layout(figure, title, xaxis_title, yaxis_title):
    figure.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        autosize=True,
        margin={"l": 75, "r": 45, "t": 80, "b": 70},
    )
    # 1×10⁻⁶ instead of Plotly's SI prefixes (µ, n, k, M, ...)
    figure.update_xaxes(exponentformat="power")
    figure.update_yaxes(exponentformat="power")
    return figure


def _writes_files() -> bool:
    """Whether this process writes files: MPI rank 0, or a serial run (MPI is not initialized here)."""
    mpi = sys.modules.get("mpi4py.MPI")
    return mpi is None or mpi.COMM_WORLD.Get_rank() == 0


def save_figure(
    figure,
    name: str,
    *,
    width: int = 1100,
    height: int = 650,
    scale: float = 2.0,
    show: bool = False,
    still_frame: int | None = None,
    still_data=None,
    still_z=None,
    still_active: int | None = None,
) -> list[Path]:
    """Save a figure as ``<name>.html``, ``<name>.png`` and ``<name>.plotly.json``.

    The HTML is the interactive figure as a standalone page (Plotly's JavaScript from a CDN), the
    PNG a static image (it needs Kaleido and a Chrome install, see ``plotly_get_chrome``), and the
    JSON the figure itself, to load again with ``plotly.io.read_json``. Under MPI only rank 0
    writes, so a script run on several ranks can call this on every rank.

    An animation that starts from a featureless state can still have an informative PNG: the
    ``still_*`` options change what the PNG shows, while the HTML and JSON keep the animation.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure to save.
    name : str
        The path of the files without their extensions, e.g. ``"maxwell-wave"`` or
        ``"figures/maxwell-wave"``.
    width : int, optional
        Width of the PNG in layout pixels. Default: 1100.
    height : int, optional
        Height of the PNG in layout pixels. Default: 650.
    scale : float, optional
        Resolution factor of the PNG. Default: 2.
    show : bool, optional
        Show the figure (``figure.show()``) before saving it. Default: False.
    still_frame : int, optional
        The PNG shows this frame of the animation, with its slider there.
    still_data : list of plotly traces, optional
        The PNG shows these traces, with the figure's layout.
    still_z : array, optional
        The PNG shows this array as the heatmap of the first trace.
    still_active : int, optional
        The slider position the PNG shows, with ``still_data`` or ``still_z``.

    Returns
    -------
    list of pathlib.Path
        The files written (none on the other MPI ranks).

    Examples
    --------
    >>> save_figure(e_x.struphy.plotly.space_time(), "space-time", show=True)
    >>> movie = f.struphy.plotly.animation(x="eta1", y="v1")
    >>> save_figure(movie, "phase-space", still_frame=len(movie.frames) // 2)
    """
    if show:
        figure.show()
    if not _writes_files():
        return []
    go = _go()
    base = Path(name)
    base.parent.mkdir(parents=True, exist_ok=True)
    html, png, json = (base.with_name(base.name + suffix) for suffix in (".html", ".png", ".plotly.json"))
    image = {"width": width, "height": height, "scale": scale}
    if still_frame is not None:
        still_data, still_active = figure.frames[still_frame].data, still_frame
    if still_data is not None:
        still = go.Figure(data=still_data, layout=figure.layout)
        if still_active is not None and still.layout.sliders:
            still.layout.sliders[0].active = still_active
        still.write_image(png, **image)
    elif still_z is not None:
        initial_z = figure.data[0].z
        initial_active = figure.layout.sliders[0].active if figure.layout.sliders else None
        figure.data[0].z = still_z
        if still_active is not None and figure.layout.sliders:
            figure.layout.sliders[0].active = still_active
        figure.write_image(png, **image)
        figure.data[0].z = initial_z
        if figure.layout.sliders:
            figure.layout.sliders[0].active = initial_active
    else:
        figure.write_image(png, **image)
    figure.write_html(html, include_plotlyjs="cdn", full_html=True, auto_play=False, config={"responsive": True})
    figure.write_json(json, pretty=False)
    for path in (html, png, json):
        print(f"Saved {path.resolve()}")
    return [html, png, json]


def _two_dims(data: xr.DataArray, x: str, y: str, sweep: str | None = None) -> None:
    wanted = {x, y} | ({sweep} if sweep else set())
    if set(data.dims) != wanted:
        raise ValueError(f"select every dimension except {sorted(wanted)} first; got {data.dims}")


def space_time(
    data: xr.DataArray,
    *,
    space: str | None = None,
    title: str | None = None,
    colorbar_title: str | None = None,
    colorscale: str = "RdBu",
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
):
    """Draw a field over one spatial coordinate and time: space along x, time up.

    The colors are symmetric about zero, so waves show as alternating stripes whose slope is
    their speed. To draw a logical coordinate in physical length, replace its values first, e.g.
    ``data.assign_coords(eta1=data.eta1 * length)``.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with exactly the two dimensions ``t`` and ``space``; select the others first.
    space : str, optional
        The spatial dimension. Default: the sole dimension other than ``t``.
    title : str, optional
        Title of the figure. Default: the array's label or name.
    colorbar_title : str, optional
        Title of the color bar. Default: the array's name.
    colorscale : str, optional
        Plotly color scale. Default: ``"RdBu"``.
    xaxis_title : str, optional
        Title of the horizontal axis. Default: the name of ``space`` and its units.
    yaxis_title : str, optional
        Title of the vertical axis. Default: ``"t"`` and its units.

    Returns
    -------
    plotly.graph_objects.Figure
        The heatmap.

    Raises
    ------
    ValueError
        If ``data`` has dimensions other than ``t`` and ``space``.

    Examples
    --------
    >>> e_x = out.evaluate("em_fields/e_field", eta1=0.0, eta2=0.0, eta3=np.linspace(0, 1, 129),
    ...                    representation="1").isel(component=0)
    >>> space_time(e_x, title="E_x(z, t)").show()
    """
    validate_array(data, required_dims=("t",))
    others = [dim for dim in data.dims if dim != "t"]
    if space is None:
        if len(others) != 1:
            raise ValueError(f"space is required unless data has exactly one dimension besides 't'; got {data.dims}")
        space = others[0]
    if set(data.dims) != {"t", space}:
        raise ValueError(f"select every dimension except 't' and {space!r} first; got {data.dims}")
    go = _go()
    values = data.transpose("t", space).values
    limit = float(np.nanmax(np.abs(values))) or 1.0
    name = data.attrs.get("label") or data.name or ""
    figure = go.Figure(
        go.Heatmap(
            x=np.asarray(data[space]),
            y=np.asarray(data.t),
            z=values,
            colorscale=colorscale,
            zmin=-limit,
            zmax=limit,
            colorbar={
                "title": {"text": colorbar_title if colorbar_title is not None else str(data.name or "")},
                "exponentformat": "power",
            },
        )
    )
    return _layout(
        figure,
        title if title is not None else name,
        xaxis_title if xaxis_title is not None else _axis_title(data, space),
        yaxis_title if yaxis_title is not None else _axis_title(data, "t"),
    )


def heatmap(
    data: xr.DataArray,
    *,
    x: str,
    y: str,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorbar_title: str | None = None,
    colorscale: str = "Viridis",
    zmin: float | None = None,
    zmax: float | None = None,
):
    """Draw a two-dimensional array as a heatmap.

    To draw a logical coordinate in physical length, replace its values first, e.g.
    ``data.assign_coords(eta1=data.eta1 * length)``.

    Parameters
    ----------
    data : xarray.DataArray
        The values, with exactly the two dimensions ``x`` and ``y``; select the others first.
    x : str
        The dimension along the horizontal axis.
    y : str
        The dimension along the vertical axis.
    title : str, optional
        Title of the figure. Default: the array's label or name.
    xaxis_title : str, optional
        Title of the horizontal axis. Default: the name of ``x`` and its units.
    yaxis_title : str, optional
        Title of the vertical axis. Default: the name of ``y`` and its units.
    colorbar_title : str, optional
        Title of the color bar. Default: the array's name.
    colorscale : str, optional
        Plotly color scale. Default: ``"Viridis"``.
    zmin : float, optional
        Lower end of the color scale. Default: the data's minimum.
    zmax : float, optional
        Upper end of the color scale. Default: the data's maximum.

    Returns
    -------
    plotly.graph_objects.Figure
        The heatmap.

    Raises
    ------
    ValueError
        If ``data`` has dimensions other than ``x`` and ``y``.

    Examples
    --------
    >>> heatmap(f.struphy.analysis.spatial_average(), x="t", y="v1", title="f(v, t)").show()
    """
    _two_dims(data, x, y)
    go = _go()
    figure = go.Figure(
        go.Heatmap(
            x=np.asarray(data[x]),
            y=np.asarray(data[y]),
            z=data.transpose(y, x).values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar={
                "title": {"text": colorbar_title if colorbar_title is not None else str(data.name or "")},
                "exponentformat": "power",
            },
        )
    )
    return _layout(
        figure,
        title if title is not None else (data.attrs.get("label") or data.name or ""),
        xaxis_title if xaxis_title is not None else _axis_title(data, x),
        yaxis_title if yaxis_title is not None else _axis_title(data, y),
    )


def animation(
    data: xr.DataArray,
    *,
    x: str,
    y: str,
    sweep: str = "t",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorbar_title: str | None = None,
    colorscale: str = "Viridis",
    zmin: float | None = 0.0,
    zmax: float | None = None,
    max_frames: int = 150,
):
    """Animate a three-dimensional array as a heatmap, one frame per value of ``sweep``.

    A frame per saved step would make the figure tens of megabytes, so at most ``max_frames``
    evenly spaced frames are kept. The figure has a play button and a slider. Its first frame is
    often featureless; ``save_figure(movie, name, still_frame=len(movie.frames) // 2)`` gives it a
    PNG of a developed frame.

    Parameters
    ----------
    data : xarray.DataArray
        The values, with exactly the dimensions ``x``, ``y`` and ``sweep``.
    x : str
        The dimension along the horizontal axis.
    y : str
        The dimension along the vertical axis.
    sweep : str, optional
        The dimension the frames run over. Default: ``"t"``.
    title : str, optional
        Title of the figure. Default: the array's label or name.
    xaxis_title : str, optional
        Title of the horizontal axis. Default: the name of ``x`` and its units.
    yaxis_title : str, optional
        Title of the vertical axis. Default: the name of ``y`` and its units.
    colorbar_title : str, optional
        Title of the color bar. Default: the array's name.
    colorscale : str, optional
        Plotly color scale. Default: ``"Viridis"``.
    zmin : float or None, optional
        Lower end of the color scale. Default: 0.
    zmax : float, optional
        Upper end of the color scale. Default: each frame's maximum.
    max_frames : int, optional
        The most frames kept. Default: 150.

    Returns
    -------
    plotly.graph_objects.Figure
        The animated heatmap.

    Raises
    ------
    ValueError
        If ``data`` has dimensions other than ``x``, ``y`` and ``sweep``.

    Examples
    --------
    >>> f = out.evaluate("kinetic_ions/e1_v1_density/f")
    >>> movie = animation(f, x="eta1", y="v1", title="f(x, v)")
    >>> save_figure(movie, "phase-space", still_frame=len(movie.frames) // 2)
    """
    _two_dims(data, x, y, sweep)
    go = _go()
    x_values, y_values = np.asarray(data[x]), np.asarray(data[y])
    frames_data = data.transpose(sweep, y, x).values
    labels = np.asarray(data[sweep])
    picks = np.linspace(0, len(labels) - 1, min(max_frames, len(labels)), dtype=int)

    def trace(z, **extra):
        return go.Heatmap(z=z, x=x_values, y=y_values, colorscale=colorscale, zmin=zmin, zmax=zmax, **extra)

    frames = [go.Frame(name=f"{labels[i]:.1f}", data=[trace(frames_data[i])]) for i in picks]
    colorbar = {
        "title": {"text": colorbar_title if colorbar_title is not None else str(data.name or "")},
        "exponentformat": "power",
    }
    figure = go.Figure(data=[trace(frames_data[0], colorbar=colorbar)], frames=frames)
    _layout(
        figure,
        title if title is not None else (data.attrs.get("label") or data.name or ""),
        xaxis_title if xaxis_title is not None else _axis_title(data, x),
        yaxis_title if yaxis_title is not None else _axis_title(data, y),
    )
    play = {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}
    figure.update_layout(
        margin={"l": 70, "r": 30, "t": 80, "b": 130},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.28,
                "yanchor": "top",
                "buttons": [{"label": "Play", "method": "animate", "args": [None, play]}],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in frames
                ],
                "x": 0.12,
                "len": 0.88,
                "y": -0.18,
                "currentvalue": {"prefix": f"{sweep} = "},
            }
        ],
    )
    return figure


def dispersion(
    spectrum: xr.DataArray,
    *,
    branches: Mapping[str, Callable | tuple] | None = None,
    fits: Sequence = (),
    dynamic_range: float = 15.0,
    kmax: float | None = None,
    omega_max: float | None = None,
    title: str | None = None,
    colorscale: str = "Plasma",
):
    """Draw an ``(omega, k)`` power spectrum as a dispersion relation, for ``omega, k >= 0``.

    The power is normalized to its maximum and drawn on a logarithmic color scale. Theoretical
    branches are overlaid dashed, fitted branches dotted.

    Parameters
    ----------
    spectrum : xarray.DataArray
        The power over ``omega`` and ``k``, e.g. from :func:`struphy_plots.analysis.power_spectrum`.
    branches : dict, optional
        Theoretical curves to compare with: a mapping of label to a callable ``omega(k)`` (of a
        complex frequency, the real part is drawn), or to an explicit ``(k, omega)`` pair.
    fits : sequence of BranchFit, optional
        Fitted straight branches, from :func:`struphy_plots.analysis.fit_dispersion_branches`,
        drawn as ``omega = velocity * k``.
    dynamic_range : float, optional
        Decades of power below the maximum that the color scale shows. Default: 15.
    kmax : float, optional
        Upper end of the ``k`` axis. Default: the largest ``k``.
    omega_max : float, optional
        Upper end of the ``omega`` axis. Default: the largest ``omega``.
    title : str, optional
        Title of the figure. Default: the spectrum's label.
    colorscale : str, optional
        Plotly color scale. Default: ``"Plasma"``.

    Returns
    -------
    plotly.graph_objects.Figure
        The heatmap with its overlays.

    Raises
    ------
    ValueError
        If ``spectrum`` lacks an ``omega`` or ``k`` dimension.

    Examples
    --------
    >>> spectrum = power_spectrum(e_x, dim="z")
    >>> fits = fit_dispersion_branches(spectrum, n_branches=1)
    >>> dispersion(spectrum, branches={"light wave, c = 1": lambda k: k}, fits=fits).show()
    """
    if not {"omega", "k"} <= set(spectrum.dims):
        raise ValueError(f"spectrum must have dims 'omega' and 'k'; got {spectrum.dims}")
    go = _go()
    quadrant = spectrum.sel(omega=spectrum.omega >= 0, k=spectrum.k >= 0).transpose("omega", "k")
    k, omega = np.asarray(quadrant.k, dtype=float), np.asarray(quadrant.omega, dtype=float)
    peak = float(quadrant.max()) or 1.0
    log_power = np.log10(np.clip(np.asarray(quadrant, dtype=float) / peak, 10.0 ** -dynamic_range, None))
    ticks = np.arange(0, -dynamic_range - 1e-9, -3.0)[::-1]
    figure = go.Figure(
        go.Heatmap(
            x=k,
            y=omega,
            z=log_power,
            zmin=-dynamic_range,
            zmax=0.0,
            colorscale=colorscale,
            colorbar={
                "title": {"text": "log₁₀ P"},
                "tickvals": list(ticks),
                "ticktext": ["1" if tick == 0 else f"10{str(int(tick)).translate(_SUPERSCRIPTS)}" for tick in ticks],
            },
            hovertemplate="k=%{x:.3f}<br>ω=%{y:.3f}<br>log₁₀ P=%{z:.2f}<extra></extra>",
        )
    )
    kmax = float(k[-1]) if kmax is None else float(kmax)
    omega_max = float(omega[-1]) if omega_max is None else float(omega_max)
    k_line = np.linspace(0.0, kmax, 200)
    for label, branch in (branches or {}).items():
        ks, omegas = branch if isinstance(branch, tuple) else (k_line, np.real(np.asarray(branch(k_line))) * np.ones_like(k_line))
        figure.add_scatter(x=ks, y=omegas, mode="lines", name=label, line={"width": 3, "dash": "dash"})
    for fit in fits:
        figure.add_scatter(
            x=k_line, y=fit.velocity * k_line, mode="lines", name=f"fit, v = {fit.velocity:.5f}", line={"width": 3, "dash": "dot"}
        )
    _layout(figure, title if title is not None else spectrum.attrs.get("label", ""), "k", "ω")
    figure.update_layout(legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.82)"})
    figure.update_xaxes(range=[0.0, kmax])
    figure.update_yaxes(range=[0.0, omega_max])
    return figure
