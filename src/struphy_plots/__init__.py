"""Optional plotting and diagnostic helpers for Struphy xarray output.

Under MPI, plots are drawn on rank 0 only; see :mod:`struphy_plots.mpi`.
"""

from .accessors import StruphyAccessor
from .mpi import SkippedPlot, is_plotting_rank, mpi_rank

__all__ = ["SkippedPlot", "StruphyAccessor", "is_plotting_rank", "mpi_rank"]
