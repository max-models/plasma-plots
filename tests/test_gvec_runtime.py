"""Exercise GVEC's compiled solver, which synthetic Dataset tests cannot cover."""

import os
import subprocess
import sys

import pytest


def test_native_equilibrium_solver(tmp_path):
    pytest.importorskip("gvec")
    # A subprocess makes a native crash (such as SIGILL) an ordinary test failure.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gvec
import numpy as np
import sys

parameters = {
    "ProjectName": "native_smoke",
    "which_hmap": 1,
    "PhiEdge": 1.0,
    "iota": {"type": "polynomial", "coefs": [0.625, 0.35]},
    "pres": {"type": "polynomial", "coefs": [1.0, -1.0], "scale": 1000.0},
    "nfp": 3,
    "X1_b_cos": {(0, 0): 3.0, (1, 0): 1.0, (1, 1): 0.4},
    "X2_b_sin": {(1, 0): 1.0, (1, 1): -0.4, (0, 1): -0.25},
    "init_average_axis": True,
    "sgrid_nElems": 2,
    "X1_mn_max": [3, 3],
    "X2_mn_max": [3, 3],
    "LA_mn_max": [3, 3],
    "X1X2_deg": 5,
    "LA_deg": 5,
    "totalIter": 5,
    "minimize_tol": 0.1,
}
run = gvec.run(parameters, runpath=sys.argv[1], quiet=True)
data = run.state.evaluate("mod_B", "pos", rho=[0.5], theta=4, zeta=3)
assert data.mod_B.size == 12
assert np.isfinite(data.mod_B).all()
assert np.isfinite(data.pos).all()
""",
            str(tmp_path / "equilibrium"),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"GVEC exited with status {result.returncode}:\n{result.stdout}\n{result.stderr}"
