"""Plot on MPI rank 0 only.

A post-processing script started with ``mpirun -n 4 python script.py`` runs every plot on every
rank; without care, four processes draw the same figure and write the same files at once. Every
plotting function here draws on rank 0 and returns a :class:`SkippedPlot` on the other ranks, so
the same script runs unchanged in serial and under MPI::

    out.plot.scalars().save("scalars.png")  # written once, by rank 0

The rank is read without importing ``mpi4py``, which would initialize MPI in a serial run: from
``mpi4py.MPI.COMM_WORLD`` when the application has already initialized MPI, otherwise from the
per-rank variables that MPI launchers export. As in Struphy, ``STRUPHY_MPI=0`` disables the
detection, and every process plots.

Nothing waits for rank 0: a rank that reads a file rank 0 writes must synchronize first, e.g.
with ``MPI.COMM_WORLD.Barrier()``.
"""

from __future__ import annotations

import functools
import os
import sys

# Global rank exported by the process managers behind the common launchers, see
# struphy.utils.mpi_launch. Hydra's MPI_LOCALRANKID is absent: it is the rank within a node.
_RANK_ENV_VARS = (
    "OMPI_COMM_WORLD_RANK",  # Open MPI
    "PMI_RANK",  # MPICH, Intel MPI, Cray, srun (pmi2)
    "PMIX_RANK",  # PMIx: srun --mpi=pmix, Open MPI 5
    "MV2_COMM_WORLD_RANK",  # MVAPICH2
    "PALS_RANKID",  # Cray PALS
    "ALPS_APP_PE",  # Cray ALPS
)
_OVERRIDE_ENV_VAR = "STRUPHY_MPI"


def mpi_rank() -> int:
    """This process' rank in ``MPI_COMM_WORLD``, or 0 outside an MPI job; never initializes MPI."""
    if os.environ.get(_OVERRIDE_ENV_VAR, "").strip().lower() in ("0", "false", "no", "off"):
        return 0
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is not None:
        try:
            if mpi.Is_initialized() and not mpi.Is_finalized():
                return mpi.COMM_WORLD.Get_rank()
        except AttributeError:
            pass
    for var in _RANK_ENV_VARS:
        try:
            return int(os.environ[var])
        except (KeyError, ValueError):
            continue
    return 0


def is_plotting_rank() -> bool:
    """Whether this process draws: rank 0 of an MPI job, or any process outside one."""
    return mpi_rank() == 0


class SkippedPlot:
    """What a plot returns on MPI ranks other than 0, where nothing is drawn.

    Any public attribute or call returns the same object, so ``plot(...).save(path)``,
    ``plotter.show()`` or ``animation.save(path)`` run on every rank but act only on rank 0. It is
    false and iterates as empty, like the (empty) list of files it wrote.
    """

    def __init__(self, name: str, rank: int):
        self._name = name
        self._rank = rank

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __bool__(self):
        return False

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __repr__(self):
        return f"SkippedPlot({self._name}, rank={self._rank})"


def rank_zero(func):
    """Run ``func`` on the plotting rank only; other ranks get a :class:`SkippedPlot`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = mpi_rank()
        if rank == 0:
            return func(*args, **kwargs)
        return SkippedPlot(func.__qualname__, rank)

    return wrapper
