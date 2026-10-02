FROM ubuntu:latest

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    MPLBACKEND=Agg \
    PYVISTA_OFF_SCREEN=true

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
RUN python -m pip install --no-cache-dir \
        "/tmp/plasma-plots[dev,netcdf,plotly,tikz,pyvista]" \
    && python -m pip check \
    && plotly_get_chrome -y \
    && rm -rf /tmp/plasma-plots

WORKDIR /work
CMD ["bash"]
