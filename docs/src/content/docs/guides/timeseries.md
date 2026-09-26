---
title: Time series & comparisons
description: Growth-rate fits and comparing scalars between runs.
---

## Time series with a growth-rate fit

```python
energy.struphy.plot.timeseries(logy=True, fit=(0.0, 2.0))
```

Overlays an exponential fit on a window of the series and reports the rate in
`result.fit_results`. See [Diagnostics](/struphy-plots/guides/analysis/) for
the damping-rate variant (fit the envelope of an oscillating signal instead).

![A time series with an exponential growth-rate fit overlaid](../../../assets/figures/timeseries_growth.png)

## Several series in one axes

```python
energy.struphy.plot.timeseries(other_run_energy, logy=True)
```

Passing further arrays plots them together; if they come from different runs
(different `attrs["run_name"]`), each line is labeled by its run.

## Comparing two runs directly

```python
field.struphy.plot.compare(reference_field, mode="difference")
field.struphy.plot.compare(reference_field, mode="ratio")
```

Aligns the two arrays and plots their difference or ratio as a 1-D lineout —
useful for a convergence study or comparing a run against a reference.

![Difference of a diagnostic between two runs](../../../assets/figures/compare_difference.png)

![Ratio of a diagnostic between two runs](../../../assets/figures/compare_ratio.png)
