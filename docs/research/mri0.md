# MRI-0 Experimental Initializer

MRI-0 is a quarantined research implementation derived from the original
`MathRevolutionaryInitializer v1.0.0-GODCORE` prototype. It is **not** part of
the canonical Starpower/Victor runtime and it makes no claim of improved model
quality until controlled benchmarks demonstrate one.

## What changed

- Julia escape-time updates stop after a cell escapes, eliminating avoidable
  overflow propagation.
- Every matrix is fan-aware after structured transformations, so the experiment
  does not accidentally compare architecture against wildly different scales.
- `zipf_s`, `phi_scale`, and Laplacian strength are real configuration values.
- The duplicate positional-encoding implementation is removed.
- Artifact serialization is implemented atomically with SHA-256 output.
- Structured mechanisms have independent ablation switches.
- Xavier-style random initialization is available as the controlled baseline.
- Determinism and finite-value checks fail closed.

## Current claim boundary

MRI-0 currently tests the hypothesis that deterministic structured initial
conditions can be compared fairly against conventional initialization. The
included benchmark measures reproducibility, numerical finiteness, tensor
statistics, and generation cost. It **does not** establish downstream learning
quality.

## Run

```bash
python benchmarks/mri0_benchmark.py --small
pytest -q tests/test_mri.py
```

## Next experimental gate

Run the same architecture, dataset, batch order, optimizer, seed family, step
budget, and hardware across:

1. Xavier baseline
2. fractal only
3. fractal + entropy rescale
4. prime + chaos embedding
5. full MRI

Then report convergence, validation loss, gradient norms, activation variance,
throughput, peak memory, and seed-to-seed variance. Do not promote MRI into a
production cognitive organ until that gate is passed reproducibly.
