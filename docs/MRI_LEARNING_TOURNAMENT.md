# MRI Learning Tournament — Experimental Contract

## Objective

Determine whether MRI-0 structured dense-weight initialization deserves a larger learning trial. This experiment does **not** claim transformer-scale superiority.

The controlled factorization is:

```text
Architecture  ∈ {standard, gst, fractal}
Initializer  ∈ {xavier, mri}
Seed         ∈ {440, 1337, 20250822}
```

That produces 18 training conditions per canonical run.

## Architecture definitions

- **standard** — one tanh hidden layer followed by a classifier head.
- **gst** — **Gated State Transition**, defined here as a tanh candidate state multiplied elementwise by a sigmoid gate before the classifier head.
- **fractal** — two parallel first-scale tanh branches whose average feeds a recursive second-scale branch; all three paths are merged before the classifier head.

These are compact experimental architecture families, not claims that the repository already contained canonical GST or Fractal Transformer implementations.

## Controlled variables

Every condition receives the same:

- synthetic teacher-generated classification dataset;
- train/validation split;
- number of optimizer steps;
- batch size;
- learning rate and Adam hyperparameters;
- convergence threshold;
- gradient clipping threshold;
- batch-order seed for a given trial seed.

Initializer comparisons are therefore interpreted **within each architecture**. Cross-architecture rankings remain exploratory because parameter counts differ.

## Initializer conditions

### Xavier

Classic Glorot uniform dense weights:

```text
limit = sqrt(6 / (fan_in + fan_out))
W ~ U(-limit, +limit)
```

### MRI

MRI-0 uses `MathRevolutionaryInitializer` with all current MRI components enabled for dense matrices. Structured/Laplacian behavior is enabled on hidden transforms.

This microtraining tournament tests the dense-weight portion of MRI. Token embedding, positional, and LayerNorm-specific MRI mechanisms are outside this experiment and require a later transformer-scale trial.

## Metrics

Each trial records:

- initial and final training loss;
- initial and final training accuracy;
- validation loss and accuracy;
- convergence step to 80% of initial loss;
- gradient norm mean, standard deviation, maximum, coefficient of variation, and finite fraction;
- hidden activation variance over training and validation;
- parameter count and parameter bytes;
- deterministic NumPy working-set bytes for parameters, Adam state, final gradients, and cached activations;
- elapsed wall time;
- training samples/second and steps/second.

The tournament aggregates seed variance for validation loss and accuracy, then computes MRI-minus-Xavier deltas for each architecture.

## Decision rule

Negative results are valid research outcomes and do not fail CI.

The scientific decision is:

- `REJECT_NUMERICAL_INSTABILITY` if any trial produces non-finite gradients;
- `PROMOTE_FOR_LARGER_TRIAL` if MRI wins mean validation loss in at least 2 of 3 architectures and mean validation-accuracy delta is no worse than -0.01;
- `REJECT_CURRENT_FORM` if MRI wins validation loss in 0 of 3 architectures;
- `INCONCLUSIVE` otherwise.

CI fails only on experiment-integrity failure, not because MRI loses.

## Reproducibility receipt

The report contains `scientific_receipt_sha256`. The hash excludes wall-clock throughput fields so repeated executions with identical scientific inputs can reproduce the same scientific receipt while still preserving runtime-performance measurements in the full JSON artifact.

## Local execution

```bash
python -m pip install -e ".[dev]"
python benchmarks/mri_learning_tournament.py \
  --steps 120 \
  --seeds 440,1337,20250822 \
  --output artifacts/mri-learning-tournament/report.json
```

Fast smoke run:

```bash
python benchmarks/mri_learning_tournament.py --small --steps 36
```

## Promotion boundary

A `PROMOTE_FOR_LARGER_TRIAL` result means only that MRI survived this compact causal screen. The next gate should use a real sequence model with matched parameter budgets, real training/validation corpora, multiple dataset orders, and explicit ablations of each MRI component.
