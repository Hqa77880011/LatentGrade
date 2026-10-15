# LatentGrade

Implementation of **Identifiability of a shared continuous condition from ordinal standards**.

Fits positive-slope ordered-logistic standards with a gauge-fixed shared latent score, checks anchor-graph connectivity and structural Fisher conditioning and refuses unsupported shared scales.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/LatentGrade.git
cd LatentGrade
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

ratings is an item/standard integer matrix with -1 for unobserved grades. heldout grades never enter fitting. categories fixes the ordinal scale; truth_score is an evaluator-only controlled latent variable.

```bash
python -m latentgrade prepare --count 1 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m latentgrade run --data data/workload.jsonl --output runs/main
python -m latentgrade suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. latentgrade, shared_thresholds and no_anchor. Shared thresholds test naive equivalence of standards; no_anchor removes cross-standard observations.

## Evaluation and visualization

Held-out ordinal accuracy/NLL, score RMSE/rank correlation on controlled data, shared-scale refusal, graph components and gauge-projected Fisher diagnostics.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m latentgrade plot --inputs runs/comparison/latentgrade --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

Scores are normalized to population mean zero and variance one on every likelihood evaluation. Fisher information removes the two latent affine gauge directions. The numerical example estimates per-item latent scores; an amortized image encoder is not assumed. A connected graph alone does not override weak information or bad conditioning.


## Uncertainty and paired comparisons

```bash
python -m latentgrade.statistics --records runs/main/latentgrade/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.
