# coilstellaration_*.json

One benchmark instance of [CoilStellaration](https://huggingface.co/datasets/proxima-fusion/coilstellaration),
written by `scripts/fetch_coilstellaration_sample.py`: the first instance, by boundary ID, of the
`eval` split in the `fixed_shape` track (3 field periods, 5 coils per half period) and the
`tight` stratum (mean field error below 0.31 %).

- `coilstellaration_coilset.json`: its DESC-optimized coilset, the `json` of its row in the
  `coilsets` subset (CoilStellaration's layout, read by `plasma_plots.from_coilset`);
- `coilstellaration_boundary.json`: its plasma boundary, the `boundary.json` of its row in
  [ConStellaration](https://huggingface.co/datasets/proxima-fusion/constellaration) (read by
  `plasma_plots.from_boundary`);
- `coilstellaration_sample.json`: the IDs and the dataset's own metrics of the coilset, from the
  `results` subset.

`scripts/generate_coil_figures.py` draws them in the "Stellarator coils" guide.

Both datasets are MIT-licensed, Copyright (c) Proxima Fusion GmbH:
https://huggingface.co/datasets/proxima-fusion/coilstellaration and
https://huggingface.co/datasets/proxima-fusion/constellaration.
