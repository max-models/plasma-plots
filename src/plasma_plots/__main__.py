"""``python -m plasma_plots``: the ``plasma-plots`` command (see :mod:`plasma_plots.cli`).

Without a command it prints the package guide (the same as ``help(plasma_plots)``);
``python -m plasma_plots api`` prints the API index (see ``API.md``).
"""

import sys

from plasma_plots.cli import main

sys.exit(main())
