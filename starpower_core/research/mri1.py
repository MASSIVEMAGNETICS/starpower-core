"""MRI-1: multi-task screening study for structured initialization.

This module expands MRI-0 from a single-task Xavier comparison into a paired,
multi-baseline screening experiment. It remains CPU-only and deliberately small:
the architecture families are compact research proxies, not production
transformers. Promotion means "worth a PyTorch-scale trial", never "proven
superior".
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np

from .mri import MathRevolutionaryInitializer, MRICfg, MRIComponents
from .tournament import (
    _backward,
    _clip_gradients,
    _forward,
    _gradient_norm,
)

Array = np.ndarray
Architecture = Literal["standard", "gst", "fractal"]
InitializerKind = Literal["xavier", "kaiming", "orthogonal", "mri"]
OptimizerKind = Literal["adamw", "sgd"]
TaskKind = Literal["classification", "regression", "sequence", "xor"]
AblationKind = Literal["full", "no_fractal", "no_laplacian", "no_entropy"]

ARCHITECTURES: tuple[Architecture, ...] = ("standard", "gst", "fractal")
INITIALIZERS: tuple[InitializerKind, ...] = ("xavier", "kaiming", "orthogonal", "mri")
OPTIMIZERS: tuple[OptimizerKind, ...] = ("adamw", "sgd")
TASKS: tuple[TaskKind, ...] = ("classification", "regression", "sequence", "xor")
WIDTHS = (24, 64, 128)
DEFAULT_SEEDS = (
    440,
    1337,
    20250822,
    31415,
    271828,
    161803,
    424242,
    8675309,
    314159,
    1234567,
)
ABLATIONS: tuple[AblationKind, ...] = ("full", "no_fractal", "no_laplacian", "no_entropy")


@dataclass(frozen=True)
class StudyCfg:
    """Fixed scientific budget for one MRI-1 trial."""

    input_dim: int = 12
    hidden_dim: int = 24
    train_size: int = 512
    val_size: int = 256
    batch_size: int = 64
    steps: int = 60
    adamw_learning_rate: float = 0.008
    sgd_learning_rate: float = 0.04
    weight_decay: float = 1e-4
    momentum: float = 0.9
    data_seed: int = 73019
    convergence_fraction: float = 0.80
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        positive_ints = (
            self.input_dim,
            self.hidden_dim,
            self.train_size,
            self.val_size,
            self.batch_size,
            self.steps,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("dimensions and budgets must be positive")
        if self.batch_size > self.train_size:
            raise ValueError("batch_size cannot exceed train_size")
        if self.adamw_learning_rate <= 0.0 or self.sgd_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be within [0, 1)")
        if not 0.0 < self.convergence_fraction < 1.0:
            raise ValueError("convergence_fraction must be within (0, 1)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


def _components(ablation: AblationKind) -> MRIComponents:
    base = MRIComponents()
    if ablation == "full":
        return base
    if ablation == "no_fractal":
        return replace(base, fractal=False)
    if ablation == "no_laplacian":
        return replace(base, laplacian=False)
    if ablation == "no_entropy":
        return replace(base, entropy_rescale=False)
    raise ValueError(f"unknown ablation: {ablation}")


class _WeightFactory:
    def __init__(
        self,
        initializer: InitializerKind,
        cfg: StudyCfg,
        seed: int,
        *,
        ablation: AblationKind = "full",
    ) -> None:
        self.initializer = initializer
        self.rng = np.random.default_rng(seed)
        n_heads = 4 if cfg.hidden_dim % 4 == 0 else 1
        mri_cfg = MRICfg(
            vocab_size=128,
            d_model=cfg.hidden_dim,
            n_heads=n_heads,
            d_ff=max(cfg.hidden_dim * 2, 8),
            max_len=32,
            seed=seed,
            julia_iters=16,
        )
        self.mri = MathRevolutionaryInitializer(mri_cfg, _components(ablation))

    def linear(
        self,
        in_dim: int,
        out_dim: int,
        *,
        structured: bool,
    ) -> tuple[Array, Array]:
        if self.initializer == "xavier":
            limit = math.sqrt(6.0 / float(in_dim + out_dim))
            weight = self.rng.uniform(-limit, limit, size=(in_dim, out_dim))
        elif self.initializer == "kaiming":
            std = math.sqrt(2.0 / float(in_dim))
            weight = self.rng.normal(0.0, std, size=(in_dim, out_dim))
        elif self.initializer == "orthogonal":
            raw = self.rng.normal(0.0, 1.0, size=(in_dim, out_dim))
            if in_dim >= out_dim:
                q, _ = np.linalg.qr(raw, mode="reduced")
            else:
                q_t, _ = np.linalg.qr(raw.T, mode="reduced")
                q = q_t.T
            weight = q
        elif self.initializer == "mri":
            layer = self.mri.init_linear(in_dim, out_dim, use_laplacian=structured)
            return layer["weight"].T.copy(), layer["bias"].copy()
        else:
            raise ValueError(f"unknown initializer: {self.initializer}")

        result = np.asarray(weight, dtype=np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError(f"{self.initializer} produced non-finite values")
        return result, np.zeros(out_dim, dtype=np.float32)


def _output_dim(task: TaskKind) -> int:
    if task == "classification":
        return 4
    if task == "xor":
        return 2
    return 1


def _build_model(
    architecture: Architecture,
    initializer: InitializerKind,
    task: TaskKind,
    cfg: StudyCfg,
    seed: int,
    *,
    ablation: AblationKind = "full",
) -> dict[str, Array]:
    factory = _WeightFactory(initializer, cfg, seed, ablation=ablation)
    output_dim = _output_dim(task)
    if architecture == "standard":
        w1, b1 = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        w2, b2 = factory.linear(cfg.hidden_dim, output_dim, structured=False)
        return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}
    if architecture == "gst":
        wa, ba = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        wg, bg = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        wo, bo = factory.linear(cfg.hidden_dim, output_dim, structured=False)
        return {"Wa": wa, "ba": ba, "Wg": wg, "bg": bg, "Wo": wo, "bo": bo}
    if architecture == "fractal":
        w1a, b1a = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        w1b, b1b = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        wr, br = factory.linear(cfg.hidden_dim, cfg.hidden_dim, structured=True)
        wo, bo = factory.linear(cfg.hidden_dim, output_dim, structured=False)
        return {
            "W1a": w1a,
            "b1a": b1a,
            "W1b": w1b,
            "b1b": b1b,
            "Wr": wr,
            "br": br,
            "Wo": wo,
            "bo": bo,
        }
    raise ValueError(f"unknown architecture: {architecture}")


def _classification_dataset(cfg: StudyCfg, rng: np.random.Generator) -> tuple[Array, ...]:
    total = cfg.train_size + cfg.val_size
    x = rng.normal(0.0, 1.0, size=(total, cfg.input_dim)).astype(np.float32)
    teacher_width = 32
    w1 = rng.normal(0.0, 0.55, size=(cfg.input_dim, teacher_width)).astype(np.float32)
    b1 = rng.normal(0.0, 0.15, size=teacher_width).astype(np.float32)
    w2 = rng.normal(0.0, 0.65, size=(teacher_width, 4)).astype(np.float32)
    hidden = np.tanh(x @ w1 + b1).astype(np.float32)
    logits = hidden @ w2
    logits += np.float32(0.08) * rng.normal(size=logits.shape).astype(np.float32)
    y = np.argmax(logits, axis=1).astype(np.int64)
    return x[: cfg.train_size], y[: cfg.train_size], x[cfg.train_size :], y[cfg.train_size :]


def _regression_dataset(cfg: StudyCfg, rng: np.random.Generator) -> tuple[Array, ...]:
    total = cfg.train_size + cfg.val_size
    x = rng.normal(0.0, 1.0, size=(total, cfg.input_dim)).astype(np.float32)
    y = (
        np.sin(x[:, 0] * x[:, 1])
        + np.float32(0.45) * x[:, 2] ** 2
        - np.float32(0.35) * x[:, 3] * x[:, 4]
        + np.float32(0.30) * np.tanh(x[:, 5] + x[:, 6])
        + np.float32(0.12) * x[:, 7]
    )
    y += np.float32(0.04) * rng.normal(size=total).astype(np.float32)
    train_y = y[: cfg.train_size]
    mean = np.float32(train_y.mean())
    std = np.float32(train_y.std() + 1e-6)
    y = ((y - mean) / std).astype(np.float32)
    return x[: cfg.train_size], y[: cfg.train_size], x[cfg.train_size :], y[cfg.train_size :]


def _sequence_dataset(cfg: StudyCfg, rng: np.random.Generator) -> tuple[Array, ...]:
    total = cfg.train_size + cfg.val_size
    length = total + cfg.input_dim + 1
    t = np.arange(length, dtype=np.float32)
    phase = np.float32(rng.uniform(-math.pi, math.pi))
    series = (
        np.sin(np.float32(0.071) * t + phase)
        + np.float32(0.55) * np.sin(np.float32(0.173) * t)
        + np.float32(0.25) * np.sin(np.float32(0.031) * t + np.float32(0.7))
    )
    series += np.float32(0.03) * rng.normal(size=length).astype(np.float32)
    x = np.stack([series[i : i + cfg.input_dim] for i in range(total)]).astype(np.float32)
    y = np.asarray(
        [series[i + cfg.input_dim] for i in range(total)],
        dtype=np.float32,
    )
    train_x = x[: cfg.train_size]
    mean = np.float32(train_x.mean())
    std = np.float32(train_x.std() + 1e-6)
    x = ((x - mean) / std).astype(np.float32)
    y = ((y - mean) / std).astype(np.float32)
    return x[: cfg.train_size], y[: cfg.train_size], x[cfg.train_size :], y[cfg.train_size :]


def _xor_dataset(cfg: StudyCfg, rng: np.random.Generator) -> tuple[Array, ...]:
    total = cfg.train_size + cfg.val_size
    bits = rng.integers(0, 2, size=(total, 4), dtype=np.int64)
    y = np.bitwise_xor.reduce(bits, axis=1).astype(np.int64)
    x = rng.normal(0.0, 0.25, size=(total, cfg.input_dim)).astype(np.float32)
    x[:, :4] = bits.astype(np.float32) * np.float32(2.0) - np.float32(1.0)
    x[:, :4] += np.float32(0.08) * rng.normal(size=(total, 4)).astype(np.float32)
    return x[: cfg.train_size], y[: cfg.train_size], x[cfg.train_size :], y[cfg.train_size :]


def _dataset(task: TaskKind, cfg: StudyCfg) -> tuple[Array, ...]:
    offsets = {"classification": 11, "regression": 23, "sequence": 37, "xor": 53}
    rng = np.random.default_rng(cfg.data_seed + offsets[task])
    if task == "classification":
        return _classification_dataset(cfg, rng)
    if task == "regression":
        return _regression_dataset(cfg, rng)
    if task == "sequence":
        return _sequence_dataset(cfg, rng)
    if task == "xor":
        return _xor_dataset(cfg, rng)
    raise ValueError(f"unknown task: {task}")


def _softmax_loss(logits: Array, labels: Array) -> tuple[float, Array]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    rows = np.arange(labels.size)
    loss = -float(np.log(probs[rows, labels] + np.float32(1e-12)).mean())
    return loss, probs


def _loss_and_gradient(task: TaskKind, outputs: Array, targets: Array) -> tuple[float, Array]:
    if task in ("classification", "xor"):
        loss, probs = _softmax_loss(outputs, targets.astype(np.int64))
        grad = probs.copy()
        grad[np.arange(targets.size), targets.astype(np.int64)] -= np.float32(1.0)
        grad /= np.float32(targets.size)
        return loss, grad

    prediction = outputs[:, 0]
    target = targets.astype(np.float32)
    error = prediction - target
    loss = float(np.mean(error * error))
    grad = np.zeros_like(outputs)
    grad[:, 0] = np.float32(2.0 / target.size) * error
    return loss, grad


def _evaluate(
    architecture: Architecture,
    task: TaskKind,
    params: dict[str, Array],
    x: Array,
    targets: Array,
) -> tuple[float, float, float]:
    outputs, hidden, _ = _forward(architecture, params, x)
    if task in ("classification", "xor"):
        loss, probs = _softmax_loss(outputs, targets.astype(np.int64))
        metric = float(np.mean(np.argmax(probs, axis=1) == targets))
    else:
        prediction = outputs[:, 0]
        error = prediction - targets.astype(np.float32)
        loss = float(np.mean(error * error))
        metric = float(math.sqrt(loss))
    activation_variance = float(np.var(hidden, axis=0).mean())
    return loss, metric, activation_variance


def _arrays_nbytes(values: dict[str, Array]) -> int:
    return int(sum(value.nbytes for value in values.values()))


def _adamw_step(
    params: dict[str, Array],
    grads: dict[str, Array],
    first: dict[str, Array],
    second: dict[str, Array],
    step: int,
    cfg: StudyCfg,
) -> None:
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)
    lr = np.float32(cfg.adamw_learning_rate)
    wd = np.float32(cfg.weight_decay)
    correction1 = np.float32(1.0 - 0.9**step)
    correction2 = np.float32(1.0 - 0.999**step)
    for name, param in params.items():
        grad = grads[name].astype(np.float32, copy=False)
        first[name] *= beta1
        first[name] += (np.float32(1.0) - beta1) * grad
        second[name] *= beta2
        second[name] += (np.float32(1.0) - beta2) * grad * grad
        update = (first[name] / correction1) / (np.sqrt(second[name] / correction2) + eps)
        param -= lr * (update + wd * param)


def _sgd_step(
    params: dict[str, Array],
    grads: dict[str, Array],
    velocity: dict[str, Array],
    cfg: StudyCfg,
) -> None:
    lr = np.float32(cfg.sgd_learning_rate)
    momentum = np.float32(cfg.momentum)
    wd = np.float32(cfg.weight_decay)
    for name, param in params.items():
        velocity[name] *= momentum
        velocity[name] += grads[name] + wd * param
        param -= lr * velocity[name]


def run_trial(
    architecture: Architecture,
    initializer: InitializerKind,
    task: TaskKind,
    optimizer: OptimizerKind,
    seed: int,
    cfg: StudyCfg | None = None,
    *,
    ablation: AblationKind = "full",
) -> dict[str, object]:
    """Train one paired condition under a fixed data and optimization budget."""

    config = cfg or StudyCfg()
    x_train, y_train, x_val, y_val = _dataset(task, config)
    params = _build_model(architecture, initializer, task, config, seed, ablation=ablation)
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    velocity = {name: np.zeros_like(value) for name, value in params.items()}
    batch_rng = np.random.default_rng(seed + 1_000_003)

    initial_loss, initial_metric, _ = _evaluate(architecture, task, params, x_train, y_train)
    convergence_target = initial_loss * config.convergence_fraction
    convergence_step: int | None = None
    gradient_norms: list[float] = []
    activation_variances: list[float] = []
    finite_gradient_steps = 0
    eval_interval = max(1, config.steps // 12)

    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        indices = batch_rng.choice(config.train_size, size=config.batch_size, replace=False)
        x_batch = x_train[indices]
        y_batch = y_train[indices]
        outputs, hidden, cache = _forward(architecture, params, x_batch)
        _, doutputs = _loss_and_gradient(task, outputs, y_batch)
        grads = _backward(architecture, params, cache, doutputs)
        raw_norm = _gradient_norm(grads)
        gradient_norms.append(raw_norm)
        finite_gradient_steps += int(
            math.isfinite(raw_norm) and all(np.isfinite(grad).all() for grad in grads.values())
        )
        if finite_gradient_steps != step:
            raise FloatingPointError(
                f"non-finite gradient: {architecture}/{initializer}/{task}/{optimizer}/{seed}"
            )
        _clip_gradients(grads, config.gradient_clip_norm)

        if optimizer == "adamw":
            _adamw_step(params, grads, first, second, step, config)
        elif optimizer == "sgd":
            _sgd_step(params, grads, velocity, config)
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")

        if not all(np.isfinite(param).all() for param in params.values()):
            raise FloatingPointError(
                f"non-finite parameter: {architecture}/{initializer}/{task}/{optimizer}/{seed}"
            )

        if step % eval_interval == 0 or step == config.steps:
            train_loss, _, activation_variance = _evaluate(
                architecture,
                task,
                params,
                x_train,
                y_train,
            )
            activation_variances.append(activation_variance)
            if convergence_step is None and train_loss <= convergence_target:
                convergence_step = step

    elapsed = time.perf_counter() - started
    final_train_loss, final_train_metric, final_activation_variance = _evaluate(
        architecture,
        task,
        params,
        x_train,
        y_train,
    )
    validation_loss, validation_metric, _ = _evaluate(architecture, task, params, x_val, y_val)
    norms = np.asarray(gradient_norms, dtype=np.float64)
    mean_norm = float(norms.mean())
    std_norm = float(norms.std())
    metric_name = "accuracy" if task in ("classification", "xor") else "rmse"

    return {
        "architecture": architecture,
        "initializer": initializer,
        "ablation": ablation if initializer == "mri" else "not_applicable",
        "task": task,
        "optimizer": optimizer,
        "seed": seed,
        "hidden_dim": config.hidden_dim,
        "steps": config.steps,
        "initial_train_loss": initial_loss,
        "initial_train_metric": initial_metric,
        "final_train_loss": final_train_loss,
        "final_train_metric": final_train_metric,
        "validation_loss": validation_loss,
        "validation_metric": validation_metric,
        "metric_name": metric_name,
        "convergence_step": convergence_step,
        "gradient_norm_mean": mean_norm,
        "gradient_norm_std": std_norm,
        "gradient_norm_max": float(norms.max()),
        "gradient_norm_cv": std_norm / mean_norm if mean_norm > 1e-12 else 0.0,
        "finite_gradient_fraction": finite_gradient_steps / float(config.steps),
        "activation_variance": final_activation_variance,
        "activation_variance_trace_mean": float(np.mean(activation_variances)),
        "parameter_count": int(sum(value.size for value in params.values())),
        "parameter_bytes": _arrays_nbytes(params),
        "optimizer_state_bytes": (
            _arrays_nbytes(first) + _arrays_nbytes(second)
            if optimizer == "adamw"
            else _arrays_nbytes(velocity)
        ),
        "elapsed_seconds": elapsed,
        "samples_per_second": config.steps * config.batch_size / max(elapsed, 1e-12),
        "steps_per_second": config.steps / max(elapsed, 1e-12),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _paired_evidence(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    n = int(data.size)
    mean = float(data.mean()) if n else 0.0
    std = float(data.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    if n <= 1 or se <= 1e-12:
        probability_positive = 1.0 if mean > 0.0 else (0.0 if mean < 0.0 else 0.5)
        margin = 0.0
    else:
        z = mean / se
        probability_positive = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        margin = 1.96 * se
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "standard_error": se,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "probability_positive_normal_approx": probability_positive,
    }


def _group_key(trial: dict[str, object]) -> tuple[str, str, int, str]:
    return (
        str(trial["architecture"]),
        str(trial["task"]),
        int(trial["hidden_dim"]),
        str(trial["optimizer"]),
    )


def aggregate_main_trials(trials: list[dict[str, object]]) -> dict[str, object]:
    """Compare MRI to the best conventional initializer within paired cells."""

    by_group: dict[tuple[str, str, int, str], list[dict[str, object]]] = {}
    for trial in trials:
        by_group.setdefault(_group_key(trial), []).append(trial)

    comparisons: list[dict[str, object]] = []
    qualifying_tasks: set[str] = set()
    all_relative_improvements: list[float] = []
    all_gradients_finite = True

    for key, rows in sorted(by_group.items()):
        architecture, task, width, optimizer = key
        by_initializer: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_initializer.setdefault(str(row["initializer"]), []).append(row)
            all_gradients_finite &= float(row["finite_gradient_fraction"]) == 1.0

        baseline_means = {
            name: _mean([float(row["validation_loss"]) for row in initializer_rows])
            for name, initializer_rows in by_initializer.items()
            if name != "mri"
        }
        best_baseline = min(baseline_means, key=baseline_means.__getitem__)
        mri_by_seed = {int(row["seed"]): row for row in by_initializer.get("mri", [])}
        baseline_by_seed = {int(row["seed"]): row for row in by_initializer[best_baseline]}
        common_seeds = sorted(set(mri_by_seed) & set(baseline_by_seed))
        relative_improvements = []
        metric_deltas = []
        for seed in common_seeds:
            mri_loss = float(mri_by_seed[seed]["validation_loss"])
            baseline_loss = float(baseline_by_seed[seed]["validation_loss"])
            relative_improvements.append(
                (baseline_loss - mri_loss) / max(abs(baseline_loss), 1e-12)
            )
            mri_metric = float(mri_by_seed[seed]["validation_metric"])
            baseline_metric = float(baseline_by_seed[seed]["validation_metric"])
            if task in ("classification", "xor"):
                metric_deltas.append(mri_metric - baseline_metric)
            else:
                metric_deltas.append(baseline_metric - mri_metric)

        evidence = _paired_evidence(relative_improvements)
        metric_evidence = _paired_evidence(metric_deltas)
        qualifies = (
            int(evidence["n"]) >= 5
            and float(evidence["mean"]) >= 0.05
            and float(evidence["probability_positive_normal_approx"]) >= 0.95
            and float(metric_evidence["mean"]) >= -0.002
        )
        if qualifies:
            qualifying_tasks.add(task)
        all_relative_improvements.extend(relative_improvements)
        comparisons.append(
            {
                "architecture": architecture,
                "task": task,
                "hidden_dim": width,
                "optimizer": optimizer,
                "best_baseline": best_baseline,
                "baseline_mean_validation_loss": baseline_means[best_baseline],
                "mri_mean_validation_loss": _mean(
                    [float(row["validation_loss"]) for row in by_initializer["mri"]]
                ),
                "relative_loss_improvement": evidence,
                "task_metric_advantage": metric_evidence,
                "qualifies_for_scale_signal": qualifies,
            }
        )

    return {
        "comparisons": comparisons,
        "qualifying_tasks": sorted(qualifying_tasks),
        "qualifying_cell_count": sum(
            int(bool(row["qualifies_for_scale_signal"])) for row in comparisons
        ),
        "overall_relative_loss_improvement": _paired_evidence(all_relative_improvements),
        "all_gradients_finite": all_gradients_finite,
    }


def aggregate_ablations(trials: list[dict[str, object]]) -> dict[str, object]:
    """Find evidence that at least one active MRI component contributes value."""

    groups: dict[tuple[str, str, int, str], list[dict[str, object]]] = {}
    for trial in trials:
        groups.setdefault(_group_key(trial), []).append(trial)

    signals: list[dict[str, object]] = []
    component_signal = False
    for key, rows in sorted(groups.items()):
        full_by_seed = {
            int(row["seed"]): row
            for row in rows
            if row["initializer"] == "mri" and row["ablation"] == "full"
        }
        for ablation in ("no_fractal", "no_laplacian", "no_entropy"):
            ablated_by_seed = {
                int(row["seed"]): row
                for row in rows
                if row["initializer"] == "mri" and row["ablation"] == ablation
            }
            common = sorted(set(full_by_seed) & set(ablated_by_seed))
            improvements = []
            for seed in common:
                full_loss = float(full_by_seed[seed]["validation_loss"])
                ablated_loss = float(ablated_by_seed[seed]["validation_loss"])
                improvements.append(
                    (ablated_loss - full_loss) / max(abs(ablated_loss), 1e-12)
                )
            evidence = _paired_evidence(improvements)
            detected = (
                int(evidence["n"]) >= 5
                and float(evidence["mean"]) >= 0.02
                and float(evidence["probability_positive_normal_approx"]) >= 0.90
            )
            component_signal |= detected
            signals.append(
                {
                    "architecture": key[0],
                    "task": key[1],
                    "hidden_dim": key[2],
                    "optimizer": key[3],
                    "ablation": ablation,
                    "full_mri_relative_improvement": evidence,
                    "component_signal_detected": detected,
                }
            )
    return {"signals": signals, "component_signal_detected": component_signal}


def _scientific_projection(report: dict[str, object]) -> dict[str, object]:
    timing_keys = {"elapsed_seconds", "samples_per_second", "steps_per_second"}

    def strip(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: strip(item)
                for key, item in sorted(value.items())
                if key not in timing_keys and key != "scientific_receipt_sha256"
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    projected = strip(report)
    assert isinstance(projected, dict)
    return projected


def _receipt(report: dict[str, object]) -> str:
    payload = json.dumps(
        _scientific_projection(report),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decision(main: dict[str, object], ablations: dict[str, object]) -> str:
    if not bool(main["all_gradients_finite"]):
        return "REJECT_NUMERICAL_INSTABILITY"
    qualifying_tasks = set(main["qualifying_tasks"])
    if len(qualifying_tasks) >= 2 and bool(ablations["component_signal_detected"]):
        return "PROMOTE_FOR_PYTORCH_TRIAL"
    overall = main["overall_relative_loss_improvement"]
    assert isinstance(overall, dict)
    if (
        int(main["qualifying_cell_count"]) == 0
        and float(overall["mean"]) <= -0.02
        and float(overall["probability_positive_normal_approx"]) <= 0.05
    ):
        return "REJECT_CURRENT_FORM"
    return "INCONCLUSIVE"


def run_study(
    *,
    mode: Literal["smoke", "full"] = "full",
    steps: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Run MRI-1 and return a deterministic-receipted research report."""

    if mode == "smoke":
        tasks: tuple[TaskKind, ...] = ("classification", "xor")
        widths = (24,)
        selected_seeds = seeds or DEFAULT_SEEDS[:2]
        study_steps = steps or 12
        ablation_tasks: tuple[TaskKind, ...] = ("classification",)
        ablation_seeds = selected_seeds
    elif mode == "full":
        tasks = TASKS
        widths = WIDTHS
        selected_seeds = seeds or DEFAULT_SEEDS
        study_steps = steps or 60
        ablation_tasks = ("classification", "sequence")
        ablation_seeds = selected_seeds[:5]
    else:
        raise ValueError(f"unknown mode: {mode}")

    trials: list[dict[str, object]] = []
    for width in widths:
        cfg = StudyCfg(hidden_dim=width, steps=study_steps)
        for task in tasks:
            for optimizer in OPTIMIZERS:
                for architecture in ARCHITECTURES:
                    for initializer in INITIALIZERS:
                        for seed in selected_seeds:
                            trials.append(
                                run_trial(
                                    architecture,
                                    initializer,
                                    task,
                                    optimizer,
                                    seed,
                                    cfg,
                                )
                            )

    ablation_trials: list[dict[str, object]] = []
    ablation_width = 64 if mode == "full" else 24
    ablation_cfg = StudyCfg(hidden_dim=ablation_width, steps=study_steps)
    for task in ablation_tasks:
        for architecture in ARCHITECTURES:
            for seed in ablation_seeds:
                for ablation in ABLATIONS:
                    ablation_trials.append(
                        run_trial(
                            architecture,
                            "mri",
                            task,
                            "adamw",
                            seed,
                            ablation_cfg,
                            ablation=ablation,
                        )
                    )

    main_aggregate = aggregate_main_trials(trials)
    ablation_aggregate = aggregate_ablations(ablation_trials)
    report: dict[str, object] = {
        "schema": "MRI-1.0",
        "mode": mode,
        "design": {
            "architectures": list(ARCHITECTURES),
            "initializers": list(INITIALIZERS),
            "tasks": list(tasks),
            "widths": list(widths),
            "optimizers": list(OPTIMIZERS),
            "seeds": list(selected_seeds),
            "steps": study_steps,
            "main_trial_count": len(trials),
            "ablation_trial_count": len(ablation_trials),
            "promotion_gate": {
                "minimum_mean_relative_loss_improvement": 0.05,
                "minimum_probability_positive": 0.95,
                "minimum_distinct_task_families": 2,
                "component_ablation_minimum_effect": 0.02,
                "component_ablation_minimum_probability": 0.90,
            },
        },
        "main": main_aggregate,
        "ablations": ablation_aggregate,
        "decision": _decision(main_aggregate, ablation_aggregate),
        "trials": trials,
        "ablation_trials": ablation_trials,
    }
    report["scientific_receipt_sha256"] = _receipt(report)
    return report
