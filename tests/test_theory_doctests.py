"""Every example in the docstrings of struphy_plots.theory runs and prints what it shows."""

import doctest
import importlib

import pytest

MODULES = ["special", "parameters", "kinetic", "waves", "orbits", "exact", "numerics"]


@pytest.mark.parametrize("name", ["__init__", *MODULES])
def test_docstring_examples_run(name):
    module = importlib.import_module("struphy_plots.theory" + ("" if name == "__init__" else f".{name}"))
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0 and result.failed == 0
