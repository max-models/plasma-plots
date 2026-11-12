"""Fetch the CoilStellaration sample that the "Stellarator coils" guide draws (scripts/data).

One benchmark instance of CoilStellaration's ``eval`` split: the first, by boundary ID, of the
``fixed_shape`` track (3 field periods, 5 coils per half period) in the ``tight`` stratum. It
writes its DESC-optimized coilset (``coilstellaration_coilset.json``), its ConStellaration
plasma boundary (``coilstellaration_boundary.json``) and the dataset's own metrics of the
coilset (``coilstellaration_sample.json``) to scripts/data, so that the docs build needs no
download. Both datasets are MIT-licensed (see scripts/data/coilstellaration_sample.md).

It needs ``pip install pandas pyarrow huggingface_hub`` and reads about 0.7 GB: the boundary
column of ConStellaration's data files has no index by ID.

Run from the repo root: python scripts/fetch_coilstellaration_sample.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COILSTELLARATION = "hf://datasets/proxima-fusion/coilstellaration"
CONSTELLARATION = "hf://datasets/proxima-fusion/constellaration"
DATA = Path(__file__).resolve().parent / "data"
METRICS = [
    "desc_metrics/normalized_coil_to_coil_min_distances/min",
    "desc_metrics/normalized_coil_to_plasma_min_distances/min",
    "desc_metrics/normalized_coil_curvatures/max",
    "desc_metrics/normalized_field_error/mean",
]


def pick_sample() -> pd.Series:
    """The first fixed-shape, tight eval instance with a DESC coilset, by boundary ID."""
    results = pd.read_parquet(
        f"{COILSTELLARATION}/results/eval-part-0.parquet",
        columns=[
            "constellaration_boundary_id",
            "desc_coilset_id",
            "reqs/n_coils_per_half_period",
            "boundary.n_field_periods",
            "benchmark/fixed_shape_track",
            "benchmark/stratification",
            *METRICS,
        ],
    )
    chosen = results[
        results["desc_coilset_id"].notna()
        & results["benchmark/fixed_shape_track"]
        & (results["benchmark/stratification"] == "tight")
    ]
    return chosen.sort_values("constellaration_boundary_id").iloc[0]


def load_coilset(coilset_id: str) -> dict:
    """A coilset by ID, through the subset's ``id_to_file_map.parquet``."""
    where = [("id", "==", coilset_id)]
    index = pd.read_parquet(
        f"{COILSTELLARATION}/coilsets/id_to_file_map.parquet", filters=where
    )
    row = pd.read_parquet(
        f"{COILSTELLARATION}/{index['path'].iloc[0]}", filters=where
    )
    return json.loads(row["json"].iloc[0])


def load_boundary(boundary_id: str) -> dict:
    """ConStellaration's ``boundary.json`` of a plasma configuration, by ID."""
    for part in range(3):
        rows = pd.read_parquet(
            f"{CONSTELLARATION}/data/train-{part:05d}-of-00003.parquet",
            columns=["plasma_config_id", "boundary.json"],
            filters=[("plasma_config_id", "==", boundary_id)],
        )
        if len(rows):
            return json.loads(rows["boundary.json"].iloc[0])
    raise KeyError(f"boundary {boundary_id} is not in ConStellaration")


def main():
    sample = pick_sample()
    print(sample)
    coilset = load_coilset(sample["desc_coilset_id"])
    boundary = load_boundary(sample["constellaration_boundary_id"])
    meta = {
        "constellaration_boundary_id": sample["constellaration_boundary_id"],
        "desc_coilset_id": sample["desc_coilset_id"],
        "n_coils_per_half_period": int(sample["reqs/n_coils_per_half_period"]),
        "n_field_periods": int(sample["boundary.n_field_periods"]),
        **{name: float(sample[name]) for name in METRICS},
    }
    for name, content in (
        ("coilstellaration_coilset.json", coilset),
        ("coilstellaration_boundary.json", boundary),
        ("coilstellaration_sample.json", meta),
    ):
        (DATA / name).write_text(json.dumps(content, indent=1) + "\n")
        print(f"wrote {DATA / name}")


if __name__ == "__main__":
    main()
