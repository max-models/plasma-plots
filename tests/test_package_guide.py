"""The package docstring is a guide for people and agents: its example runs, and every method it
names exists."""

import doctest
import re
import subprocess
import sys

import plasma_plots
from plasma_plots import accessors, output_accessors


def section(title):
    doc = plasma_plots.__doc__
    start = doc.index(f"\n{title}\n{'-' * len(title)}\n")
    rest = doc[start + len(title) * 2 + 3 :]
    end = re.search(r"\n\S[^\n]*\n-{3,}\n", rest)
    return rest[: end.start()] if end else rest


def test_the_example_runs():
    parser = doctest.DocTestParser()
    test = parser.get_doctest(section("Example"), {}, "plasma_plots guide", None, 0)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    runner.run(test)
    assert runner.failures == 0 and runner.tries > 5


def test_every_method_the_guide_names_exists():
    doc = plasma_plots.__doc__
    owners = {
        "out.plot": output_accessors.OutputPlots,
        "out.analysis": output_accessors.OutputAnalysis,
        "dataset.plasma.plot": accessors.DatasetPlots,
        "dataset.plasma.analysis": accessors.DatasetAnalysis,
        "plot": accessors.ArrayPlots,
        "analysis": accessors.ArrayAnalysis,
        "data": accessors.ArrayData,
    }
    named = re.findall(r"``((?:out|dataset\.plasma)\.(?:plot|analysis)|plot|analysis|data)\.(\w+)", doc)
    # the "Particles" item continues a dataset accessor across ``.name`` references
    named += [
        ("dataset.plasma.plot", n) for n in re.findall(r"``\.(\w+)``", doc.split("* Particles:")[1].split(";")[0])
    ]
    named += [
        ("dataset.plasma.analysis", n)
        for n in re.findall(r"``\.?(\w+)``", doc.split("dataset.plasma.analysis.")[1].split(";")[0])
    ]
    missing = [f"{owner}.{name}" for owner, name in named if not hasattr(owners[owner], name)]
    assert len(named) > 40 and not missing, missing


def test_python_m_prints_the_guide():
    result = subprocess.run(
        [sys.executable, "-m", "plasma_plots", "guide"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.startswith("Plots and diagnostics of labeled xarray data from plasma simulations")


def test_accessors_print_a_menu_of_their_methods():
    import numpy as np
    import xarray as xr

    b = xr.DataArray(np.zeros(3), dims="t", coords={"t": [0.0, 1.0, 2.0]}, name="b_field")
    orbits = xr.Dataset(
        {"x": (("t", "marker"), np.zeros((3, 2)))},
        coords={"t": [0.0, 1.0, 2.0], "marker": [0, 1]},
    )
    for accessor, expected in (
        (b.plasma, "plot"),
        (b.plasma.plot, "slice"),
        (b.plasma.analysis, "growth_rate"),
        (b.plasma.data, "lineout"),
        (orbits.plasma.plot, "poloidal"),
        (orbits.plasma.analysis, "classify_orbits"),
    ):
        text = repr(accessor)
        assert "object at 0x" not in text and f"  {expected} " in text
        assert "``" not in text and ":meth:" not in text  # plain text, no reST markup
    assert repr(b.plasma.plot).startswith("array.plasma.plot of DataArray 'b_field' (t: 3)")
    assert "help(b_field.plasma.plot." in repr(b.plasma.plot)
