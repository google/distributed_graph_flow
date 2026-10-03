# Changelog

This file lists the user-visible features in each version of the package.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Usage is described in [`README.md`](./README.md), and the mathematical arguments
are in [`THEORY.md`](./THEORY.md), [`THEORY_INDEPENDENT.md`](./THEORY_INDEPENDENT.md),
[`THEORY_SPACETIME.md`](./THEORY_SPACETIME.md), and
[`THEORY_RANGE_ENVELOPE.md`](./THEORY_RANGE_ENVELOPE.md).

## [0.1.0]

### Added

*   **Confidence intervals for dependent evaluation data.** Finite-sample confidence
    intervals for evaluation metrics: mean absolute error (MAE), mean squared error
    (MSE), root mean squared error (RMSE), accuracy, and the coefficient of
    determination (R²). Supports time-ordered items (`stats.temporal`), graph-linked
    nodes (`stats.graph`), and spatio-temporal observations (`stats.spacetime`).
*   **Independent evaluation (`stats.independent`).** Finite-sample confidence intervals
    for independent evaluation streams with streaming sketches (`IndependentSketch`,
    `R2Sketch`), JAX accumulators, distributed shard aggregation, and Clopper–Pearson
    exact binomial intervals for accuracy.
*   **Space-time evaluation (`stats.spacetime`).** Confidence intervals for graph nodes
    observed over time. Features candidate partition generation, automatic partition
    selection maximizing effective sample size without loss feedback, separable
    covariance certification, topological spectral bounds for incomplete grids, and
    drift diagnostics.
*   **Range-envelope variance inflation (`stats.range_envelope`).** Derives certified
    variance inflation bounds $\kappa$ from model receptive fields and noise correlation
    ranges for fixed spatio-temporal test sets evaluated in full.
*   **Graph variance inflation (`stats.graph`).** Computes certified variance inflation
    bounds $\kappa$ from graph topology and per-edge correlation bounds via Perron–Frobenius
    spectral bounds, including hub-ratio guards and distributed partitioners.
*   **Time-bucket evaluation (`temporal.temporal_interval_from_buckets`).** Certified
    confidence intervals for pre-aggregated time series and hourly counts without
    requiring per-item loss records.
*   **Assumption declaration and calibration (`stats.declare`).** Tools to bootstrap
    conservative relative moment bounds $M$ from historical archive data, serialize
    them with archive fingerprints, and supply them cleanly to evaluation routines.
*   **Diagnostic and refutation checks (`stats.refutation`).** Automated checks for
    correlation and tail-moment assumptions that report diagnostic statuses and
    flag potential assumption violations without invalidating conservative bounds.
