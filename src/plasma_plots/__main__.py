"""``python -m plasma_plots``: print the package guide (the same as ``help(plasma_plots)``).

``python -m plasma_plots api`` prints the API index instead: every accessor method and public
function with its signature and one-line summary (see ``API.md``).
"""

import sys

import plasma_plots

if sys.argv[1:] == ["api"]:
    from plasma_plots._api import api_index

    print(api_index(), end="")
elif sys.argv[1:] in ([], ["guide"]):
    print(plasma_plots.__doc__)
else:
    print("usage: python -m plasma_plots [guide | api]\n  guide  the package guide (default)\n  api    the API index")
    sys.exit(2)
