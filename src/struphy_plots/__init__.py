"""Optional plotting and diagnostic helpers for Struphy xarray output."""

from . import output_accessors  # noqa: F401  (registers Output.plot, if struphy is installed)
from .accessors import StruphyAccessor

__all__ = ["StruphyAccessor"]
