FROM ubuntu:latest

# Build locally: docker build --pull -t plasma-plots:local .
# Run a shell: docker run --rm -it -v "$PWD:/work" plasma-plots:local
# Run a command: docker run --rm -v "$PWD:/work" plasma-plots:local \
#   plasma-plots quicklook sim_1 -o figures/
# Run MPI: docker run --rm -v "$PWD:/work" plasma-plots:local \
#   mpirun --allow-run-as-root --oversubscribe -n 2 python simulation.py
# Containers run as root by default. Add --user "$(id -u):$(id -g)" -e HOME=/tmp
# to write mounted files as your Linux user; omit --allow-run-as-root for MPI.
# For renderers needing a display, run xvfb-run -a python script.py; add --init
# to docker run when launching xvfb-run directly.
#
# .github/workflows/docker.yml builds and tests on pushes to main, then publishes
# ghcr.io/max-models/plasma-plots:latest and sha-<commit> tags (linux/amd64).
# Pull requests check the build without publishing. GITHUB_TOKEN needs package
# write permission. For publishing from a repository outside max-models, set
# GHCR_TOKEN to a classic personal access token with write:packages permission
# for max-models (or grant the repository Actions access to an existing package).
# After the first publish, set the package visibility to public for anonymous pulls.

# Python's virtual environment is already on PATH. Matplotlib uses Agg and
# PyVista renders off-screen.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    MPLBACKEND=Agg \
    PYVISTA_OFF_SCREEN=true

# C/Fortran compilers, BLAS/LAPACK, MPI and netCDF support Struphy; TeX Live
# provides pdflatex and pgfplots. FFmpeg and Poppler support figure exports.
# Mesa/Xvfb support headless rendering; the remaining libraries support Chrome.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates git build-essential gfortran cmake \
        python3 python3-dev python3-pip python3-venv \
        libblas-dev liblapack-dev libopenmpi-dev openmpi-bin \
        libnetcdf-dev libnetcdff-dev \
        texlive-latex-base texlive-latex-extra texlive-pictures \
        texlive-fonts-recommended lmodern poppler-utils ffmpeg \
        xvfb xauth libgl1 libglx-mesa0 libegl1 libosmesa6 \
        libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 \
        libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Keep the expensive runtime build cached independently of plasma-plots sources.
RUN python -m pip install --no-cache-dir "struphy[mpi]>=3.4.0" \
    && struphy compile -y --language fortran \
    && struphy compile --status | grep -E '^0 of [1-9][0-9]* Struphy kernels are not compiled'

COPY pyproject.toml README.md LICENSE API.md /tmp/plasma-plots/
COPY src /tmp/plasma-plots/src
# Install the checked-out package with plotting and development extras, plus
# Chrome for Plotly exports. Omit gallery/profiling: their current profiling
# dependency pins a maxplotlib version incompatible with the tikz extra.
RUN python -m pip install --no-cache-dir \
        "/tmp/plasma-plots[dev,netcdf,plotly,tikz,pyvista]" \
    && python -m pip check \
    && plotly_get_chrome -y \
    && rm -rf /tmp/plasma-plots

WORKDIR /work
CMD ["bash"]
