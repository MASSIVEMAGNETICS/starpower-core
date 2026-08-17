# MRI-1 Multi-Task Screening Study

MRI-1 is the second evidence gate for `MathRevolutionaryInitializer` (MRI). It
does not promote MRI into production cognition. It asks a narrower question:

> Does MRI produce a repeatable optimization advantage over strong conventional
> initializers under paired data, batch-order, architecture, optimizer, and
> compute budgets?

## Scope

MRI-1 retains the compact CPU-only architecture proxies from MRI-0:

- `standard`
- `gst`
- `fractal`

These are screening models, not transformer-scale implementations. Results may
justify a larger PyTorch experiment, but cannot establish transformer-scale
superiority.

The full study crosses:

- 3 architecture families
- 4 initializers: Xavier, Kaiming, orthogonal, MRI
- 4 task families: nonlinear classification, nonlinear regression, sequence
  prediction, noisy XOR/parity
- 3 widths: 24, 64, 128
- 2 optimizers: AdamW and momentum SGD
- 10 paired seeds

That produces 2,880 main training trials. MRI component ablations are run
separately on representative classification and sequence conditions.

## Pairing and fairness

Within a task/width/optimizer/seed cell:

- data are identical across initializers;
- batch order is identical across initializers;
- training steps and batch size are identical;
- optimizer hyperparameters are fixed;
- comparisons are made within an architecture, never by pretending parameter
  counts are equal across architecture families.

The best conventional baseline is selected from Xavier, Kaiming, and
orthogonal initialization using mean validation loss within the paired cell.

## Metrics

Each trial records:

- initial and final train loss;
- validation loss;
- accuracy for classification tasks;
- RMSE for regression and sequence tasks;
- convergence step;
- gradient norm mean/std/max/CV;
- finite-gradient fraction;
- activation variance;
- parameter and optimizer-state bytes;
- elapsed time and throughput.

Wall-clock timing is retained for operational analysis but excluded from the
scientific receipt hash.

## Promotion gate

A cell is a scale signal only when MRI beats the best conventional baseline by
at least 5% mean relative validation-loss improvement across at least five
paired seeds, with a one-sided normal-approximation probability of improvement
of at least 0.95 and no material task-metric regression.

MRI-1 emits `PROMOTE_FOR_PYTORCH_TRIAL` only when:

1. scale signals occur in at least two distinct task families; and
2. the ablation suite detects at least one active MRI component whose removal
   worsens validation loss by at least 2% with probability at least 0.90.

Other decisions are:

- `REJECT_NUMERICAL_INSTABILITY`
- `REJECT_CURRENT_FORM`
- `INCONCLUSIVE`

A negative scientific result does not fail CI. CI fails on experiment-integrity
problems such as non-finite gradients, test failures, or missing artifacts.

## Ablations

The linear-weight path currently exercises three MRI mechanisms directly:

- fractal structure;
- Laplacian coupling;
- entropy rescaling.

MRI-1 therefore tests:

- full MRI;
- MRI without fractal initialization;
- MRI without Laplacian coupling;
- MRI without entropy rescaling.

Prime/chaos embedding and golden LayerNorm mechanisms are not claimed by this
linear microtraining study because they are not active in this path. They
belong in the later transformer-scale experiment.

## Automation

`.github/workflows/mri1-study.yml` runs:

- smoke mode on pull requests and `agent/**` pushes;
- full mode on `main`;
- manual smoke/full dispatch;
- a full weekly replication on Sunday at 06:17 UTC.

The JSON report is retained as a GitHub Actions artifact for 90 days and carries
a deterministic `scientific_receipt_sha256`.

## Interpretation boundary

`PROMOTE_FOR_PYTORCH_TRIAL` means only that MRI earned a more expensive
experiment. It does not mean MRI is generally superior, that GST or the fractal
proxy is a superior architecture, or that any result transfers to language
model scale without further evidence.
