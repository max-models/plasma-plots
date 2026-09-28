"""Opt-in real-simulation tests: ``pytest --run-simulations`` (or ``-m simulation``)."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-simulations",
        action="store_true",
        default=False,
        help="also run tests marked 'simulation' (real struphy runs; need compiled kernels)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-simulations") or "simulation" in (
        config.getoption("-m") or ""
    ):
        return
    skip = pytest.mark.skip(
        reason="real struphy simulation; run with --run-simulations"
    )
    for item in items:
        if "simulation" in item.keywords:
            item.add_marker(skip)
