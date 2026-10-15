# Benchmark Executables

Each binary in this directory regenerates one section of
[../EVIDENCE.md](../EVIDENCE.md). Every binary accepts `--mode=quick` (a few
repetitions per cell, for a smoke check) and `--mode=full` (the repetition
counts reported in `EVIDENCE.md`). Seeds are fixed module constants, so a
full run reproduces the reported numbers up to Monte Carlo and platform
differences.

```bash
bazel run -c opt //dgf/src/stats/benchmark:<binary> -- --mode=full
```

Most binaries write one JSON record per cell with `--output=<path>`;
`textbook_rmse_coverage` takes `--output_dependent=<path>` and
`--output_iid=<path>` for its two parts, and `cluster_sweep` selects one of
its three suites with `--suite=granularity|calibration|refutation`.

| Binary | Regenerates (`EVIDENCE.md`) | Seed constant |
| :--- | :--- | :--- |
| `coverage_independent` | [§2.1 Independent-Sample Coverage Study](../EVIDENCE.md#21-independent-sample-coverage-study) | `DEFAULT_SEED = 80017972348887` |
| `hypergeom_cp` | [§2.2 Exact Hypergeometric Clopper-Pearson Enumeration](../EVIDENCE.md#22-exact-hypergeometric-clopper-pearson-enumeration) | Deterministic (no RNG) |
| `coverage_dependent` | [§3.1 Temporal and Graph Coverage and Refutation Study](../EVIDENCE.md#31-temporal-and-graph-coverage-and-refutation-study) | `DEFAULT_SEED = 54077361271078` |
| `partitioner_robustness` | [§3.2 Graph Partitioner Cluster-Size and Count Bounds](../EVIDENCE.md#32-graph-partitioner-cluster-size-and-count-bounds) | `_DEFAULT_SEEDS = tuple(list(range(20)) + [20260923])` |
| `validation_r2_buckets` | [§4 Correlated $R^2$, Midpoint-Bucketed Windows, and Topological $\kappa$](../EVIDENCE.md#4-correlated-r2-midpoint-bucketed-windows-and-topological-kappa) | `DEFAULT_SEED = 0x21DA7A` |
| `validation_spacetime` | [§5 Space-Time Panels, Incomplete Grids, and Per-Window Centring](../EVIDENCE.md#5-space-time-panels-incomplete-grids-and-per-window-centring) | `DEFAULT_SEED = 0x59AC371E` |
| `range_coverage` | [§6 Range-Envelope Coverage](../EVIDENCE.md#6-range-envelope-coverage) | `DEFAULT_SEED = 0x820000` |
| `textbook_baselines` | [§7.1 Dependent Temporal and Real-Topology Graph Comparisons](../EVIDENCE.md#71-dependent-temporal-and-real-topology-graph-comparisons) | `DEFAULT_SEED = 54077361271078` |
| `textbook_rmse_coverage` | [§7.2 RMSE Coverage Under Dependence and Heavy-Tailed i.i.d. Designs](../EVIDENCE.md#72-rmse-coverage-under-dependence-and-heavy-tailed-iid-designs) | `DEFAULT_SEED_PART_A = 0x820000`, `DEFAULT_SEED_PART_B = 0x880000` |
| `edge_guard_sweep` | [§8.1 Cluster-Graph Edge-Dominance Guard Sweep](../EVIDENCE.md#81-cluster-graph-edge-dominance-guard-sweep) | `DEFAULT_SEED = 20260926` |
| `theory_sims` | [§8.2 Diagnostic and Refutation Calibration Simulations](../EVIDENCE.md#82-diagnostic-and-refutation-calibration-simulations) | `DEFAULT_SEED = 42` |
| `cluster_sweep` | [§9 Cluster-Granularity Sweep](../EVIDENCE.md#9-cluster-granularity-sweep) | `DEFAULT_SEED = 20260923` |

## Library test modes

By default, `bazel test //dgf/src/stats/...` runs fast
deterministic unit, contract and small-sample Monte Carlo tests. Setting
`DGF_STATS_LONG_TESTS=1` enables the long statistical property and
multi-thousand-repetition coverage checks:

```bash
bazel test //dgf/src/stats/... --test_env=DGF_STATS_LONG_TESTS=1 --test_timeout=3600
```
