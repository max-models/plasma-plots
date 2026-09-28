"""``python -m struphy_plots``: print the package guide (the same as ``help(struphy_plots)``).

``python -m struphy_plots api`` prints the API index instead: every accessor method and public
function with its signature and one-line summary (see ``API.md``).
"""

import sys

import struphy_plots

if sys.argv[1:] == ["api"]:
    from struphy_plots._api import api_index

    print(api_index(), end="")
elif sys.argv[1:] in ([], ["guide"]):
    print(struphy_plots.__doc__)
else:
    print("usage: python -m struphy_plots [guide | api]\n  guide  the package guide (default)\n  api    the API index")
    sys.exit(2)
