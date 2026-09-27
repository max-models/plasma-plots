"""Interactive Plotly figures of labeled arrays, for web pages and notebooks.

The functions here return a :class:`plotly.graph_objects.Figure`; call ``.show()`` on it, or
``.write_html(...)``. They need Plotly (``pip install "struphy-plots[plotly]"``). The same
plots are methods of ``array.struphy.plotly``, e.g. ``field.struphy.plotly.space_time()``.
Axis titles are plain text: Plotly draws LaTeX only where MathJax is loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

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
    return figure


def space_time(
    data: xr.DataArray,
    *,
    space: str | None = None,
    title: str | None = None,
    colorbar_title: str | None = None,
    colorscale: str = "RdBu",
):
    """Draw a field over one spatial coordinate and time: space along x, time up.

    The colors are symmetric about zero, so waves show as alternating stripes whose slope is
    their speed.

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
            colorbar={"title": {"text": colorbar_title if colorbar_title is not None else str(data.name or "")}},
        )
    )
    return _layout(figure, title if title is not None else name, _axis_title(data, space), _axis_title(data, "t"))


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
