"""Controlled learning tournament for MRI-0 versus Xavier initialization.

The tournament is intentionally a compact CPU-only microtraining experiment. It
compares initializer effects within three explicit architecture families under
identical data, optimizer, step, and batch-order budgets. It does not establish
large-model quality; it decides whether MRI-0 deserves a larger trial.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .mri import MathRevolutionaryInitializer, MRICfg, MRIComponents

Array = np.ndarray
Architecture = Literal["standard", "gst", "fractal"]
InitializerKind = Literal["xavier", "mri"]

ARCHITECTURES: tuple[Architecture, ...] = ("standard", "gst", "fractal")
INITIALIZERS: tuple[InitializerKind, ...] = ("xavier", "mri")
DEFAULT_SEEDS = (440, 1337, 20250822)


@dataclass(frozen=True)
class TournamentCfg:
    """Fixed budget for one controlled tournament."""

    input_dim: int = 12
    hidden_dim: int = 24
    n_classes: int = 4
    train_size: int = 384
    val_size: int = 192
    batch_size: int = 48
    steps: int = 120
    learning_rate: float = 0.015
    data_seed: int = 73019
    convergence_fraction: float = 0.80
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        positive_ints = (
            self.input_dim,
            self.hidden_dim,
            self.n_classes,
            self.train_size,
            self.val_size,
            self.batch_size,
            self.steps,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("all tournament dimensions and budgets must be positive")
        if self.input_dim < 2:
            raise ValueError("input_dim must be at least two for structured MRI initialization")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least two for structured MRI initialization")
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least two")
        if self.batch_size > self.train_size:
            raise ValueError("batch_size cannot exceed train_size")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0.0 < self.convergence_fraction < 1.0:
            raise ValueError("convergence_fraction must be within (0, 1)")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")


class _WeightFactory:
    def __init__(self, kind: InitializerKind, cfg: TournamentCfg, seed: int) -> None:
        self.kind = kind
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
        self.mri = MathRevolutionaryInitializer(mri_cfg, MRIComponents())

    def linear(self, in_dim: int, out_dim: int, *, structured: bool) -> tuple[Array, Array]:
        if self.kind == "xavier":
            limit = math.sqrt(6.0 / float(in_dim + out_dim))
            weight = self.rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(
                np.float32
            )
            bias = np.zeros(out_dim, dtype=np.float32)
            return weight, bias
        layer = self.mri.init_linear(in_dim, out_dim, use_laplacian=structured)
        return layer["weight"].T.copy(), layer["bias"].copy()


def _build_model(
    architecture: Architecture,
    initializer: InitializerKind,
    cfg: TournamentCfg,
    seed: int,
) -> dict[str, Array]:
    factory = _WeightFactory(initializer, cfg, seed)
    if architecture == "standard":
        w1, b1 = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        w2, b2 = factory.linear(cfg.hidden_dim, cfg.n_classes, structured=False)
        return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}
    if architecture == "gst":
        wa, ba = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        wg, bg = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        wo, bo = factory.linear(cfg.hidden_dim, cfg.n_classes, structured=False)
        return {"Wa": wa, "ba": ba, "Wg": wg, "bg": bg, "Wo": wo, "bo": bo}
    if architecture == "fractal":
        w1a, b1a = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        w1b, b1b = factory.linear(cfg.input_dim, cfg.hidden_dim, structured=True)
        wr, br = factory.linear(cfg.hidden_dim, cfg.hidden_dim, structured=True)
        wo, bo = factory.linear(cfg.hidden_dim, cfg.n_classes, structured=False)
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


def _sigmoid(x: Array) -> Array:
    clipped = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _forward(
    architecture: Architecture, params: dict[str, Array], x: Array
) -> tuple[Array, Array, dict[str, Array]]:
    if architecture == "standard":
        h = np.tanh(x @ params["W1"] + params["b1"]).astype(np.float32)
        logits = h @ params["W2"] + params["b2"]
        return logits, h, {"x": x, "h": h}
    if architecture == "gst":
        a = np.tanh(x @ params["Wa"] + params["ba"]).astype(np.float32)
        g = _sigmoid(x @ params["Wg"] + params["bg"])
        h = (a * g).astype(np.float32)
        logits = h @ params["Wo"] + params["bo"]
        return logits, h, {"x": x, "a": a, "g": g, "h": h}
    if architecture == "fractal":
        p = np.tanh(x @ params["W1a"] + params["b1a"]).astype(np.float32)
        q = np.tanh(x @ params["W1b"] + params["b1b"]).astype(np.float32)
        mixed = np.float32(0.5) * (p + q)
        r = np.tanh(mixed @ params["Wr"] + params["br"]).astype(np.float32)
        h = ((p + q + r) / np.float32(3.0)).astype(np.float32)
        logits = h @ params["Wo"] + params["bo"]
        return logits, h, {"x": x, "p": p, "q": q, "mixed": mixed, "r": r, "h": h}
    raise ValueError(f"unknown architecture: {architecture}")


def _loss_and_probs(logits: Array, labels: Array) -> tuple[float, Array]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    rows = np.arange(labels.size)
    loss = -float(np.log(probs[rows, labels] + np.float32(1e-12)).mean())
    return loss, probs


def _backward(
    architecture: Architecture,
    params: dict[str, Array],
    cache: dict[str, Array],
    dlogits: Array,
) -> dict[str, Array]:
    x = cache["x"]
    if architecture == "standard":
        h = cache["h"]
        dh = dlogits @ params["W2"].T
        dz = dh * (np.float32(1.0) - h * h)
        return {
            "W1": x.T @ dz,
            "b1": dz.sum(axis=0),
            "W2": h.T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
    if architecture == "gst":
        a, g, h = cache["a"], cache["g"], cache["h"]
        dh = dlogits @ params["Wo"].T
        da = dh * g
        dg = dh * a
        dza = da * (np.float32(1.0) - a * a)
        dzg = dg * g * (np.float32(1.0) - g)
        return {
            "Wa": x.T @ dza,
            "ba": dza.sum(axis=0),
            "Wg": x.T @ dzg,
            "bg": dzg.sum(axis=0),
            "Wo": h.T @ dlogits,
            "bo": dlogits.sum(axis=0),
        }
    if architecture == "fractal":
        p, q, mixed, r, h = (
            cache["p"],
            cache["q"],
            cache["mixed"],
            cache["r"],
            cache["h"],
        )
        dh = dlogits @ params["Wo"].T
        dr = dh / np.float32(3.0)
        dzr = dr * (np.float32(1.0) - r * r)
        dmixed = dzr @ params["Wr"].T
        dp = dh / np.float32(3.0) + np.float32(0.5) * dmixed
        dq = dh / np.float32(3.0) + np.float32(0.5) * dmixed
        dzp = dp * (np.float32(1.0) - p * p)
        dzq = dq * (np.float32(1.0) - q * q)
        return {
            "W1a": x.T @ dzp,
            "b1a": dzp.sum(axis=0),
            "W1b": x.T @ dzq,
            "b1b": dzq.sum(axis=0),
            "Wr": mixed.T @ dzr,
            "br": dzr.sum(axis=0),
            "Wo": h.T @ dlogits,
            "bo": dlogits.sum(axis=0),
        }
    raise ValueError(f"unknown architecture: {architecture}")


def _dataset(cfg: TournamentCfg) -> tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng(cfg.data_seed)
    total = cfg.train_size + cfg.val_size
    x = rng.normal(0.0, 1.0, size=(total, cfg.input_dim)).astype(np.float32)
    teacher_width = max(cfg.hidden_dim, cfg.n_classes * 3)
    w1 = rng.normal(0.0, 0.55, size=(cfg.input_dim, teacher_width)).astype(np.float32)
    b1 = rng.normal(0.0, 0.15, size=teacher_width).astype(np.float32)
    w2 = rng.normal(0.0, 0.65, size=(teacher_width, cfg.n_classes)).astype(np.float32)
    hidden = np.tanh(x @ w1 + b1).astype(np.float32)
    logits = hidden @ w2
    logits += np.float32(0.08) * rng.normal(size=logits.shape).astype(np.float32)
    labels = np.argmax(logits, axis=1).astype(np.int64)
    return (
        x[: cfg.train_size],
        labels[: cfg.train_size],
        x[cfg.train_size :],
        labels[cfg.train_size :],
    )


def _gradient_norm(grads: dict[str, Array]) -> float:
    squared = sum(float(np.sum(grad.astype(np.float64) ** 2)) for grad in grads.values())
    return math.sqrt(squared)


def _clip_gradients(grads: dict[str, Array], max_norm: float) -> None:
    norm = _gradient_norm(grads)
    if norm <= max_norm or norm == 0.0:
        return
    scale = np.float32(max_norm / norm)
    for grad in grads.values():
        grad *= scale


def _adam_step(
    params: dict[str, Array],
    grads: dict[str, Array],
    first_moment: dict[str, Array],
    second_moment: dict[str, Array],
    step: int,
    learning_rate: float,
) -> None:
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)
    lr = np.float32(learning_rate)
    correction1 = np.float32(1.0 - 0.9**step)
    correction2 = np.float32(1.0 - 0.999**step)
    for name, param in params.items():
        grad = grads[name].astype(np.float32, copy=False)
        first_moment[name] *= beta1
        first_moment[name] += (np.float32(1.0) - beta1) * grad
        second_moment[name] *= beta2
        second_moment[name] += (np.float32(1.0) - beta2) * grad * grad
        m_hat = first_moment[name] / correction1
        v_hat = second_moment[name] / correction2
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _evaluate(
    architecture: Architecture,
    params: dict[str, Array],
    x: Array,
    labels: Array,
) -> tuple[float, float, float]:
    logits, hidden, _ = _forward(architecture, params, x)
    loss, probs = _loss_and_probs(logits, labels)
    accuracy = float(np.mean(np.argmax(probs, axis=1) == labels))
    activation_variance = float(np.var(hidden, axis=0).mean())
    return loss, accuracy, activation_variance


def _arrays_nbytes(values: dict[str, Array]) -> int:
    return int(sum(value.nbytes for value in values.values()))


def run_trial(
    architecture: Architecture,
    initializer: InitializerKind,
    seed: int,
    cfg: TournamentCfg | None = None,
) -> dict[str, object]:
    """Train one architecture/initializer/seed condition under a fixed budget."""

    config = cfg or TournamentCfg()
    x_train, y_train, x_val, y_val = _dataset(config)
    params = _build_model(architecture, initializer, config, seed)
    first_moment = {name: np.zeros_like(value) for name, value in params.items()}
    second_moment = {name: np.zeros_like(value) for name, value in params.items()}
    batch_rng = np.random.default_rng(seed + 1_000_003)

    initial_loss, initial_accuracy, _ = _evaluate(architecture, params, x_train, y_train)
    convergence_target = initial_loss * config.convergence_fraction
    convergence_step: int | None = None
    gradient_norms: list[float] = []
    activation_variances: list[float] = []
    finite_gradients = 0
    last_grads = {name: np.zeros_like(value) for name, value in params.items()}
    last_cache: dict[str, Array] = {}
    eval_interval = max(1, config.steps // 12)

    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        indices = batch_rng.choice(config.train_size, size=config.batch_size, replace=False)
        x_batch = x_train[indices]
        y_batch = y_train[indices]
        logits, hidden, cache = _forward(architecture, params, x_batch)
        _, probs = _loss_and_probs(logits, y_batch)
        dlogits = probs.copy()
        dlogits[np.arange(config.batch_size), y_batch] -= np.float32(1.0)
        dlogits /= np.float32(config.batch_size)
        grads = _backward(architecture, params, cache, dlogits)
        norm = _gradient_norm(grads)
        gradient_norms.append(norm)
        activation_variances.append(float(np.var(hidden, axis=0).mean()))
        if math.isfinite(norm) and all(np.isfinite(grad).all() for grad in grads.values()):
            finite_gradients += 1
        _clip_gradients(grads, config.gradient_clip_norm)
        _adam_step(params, grads, first_moment, second_moment, step, config.learning_rate)
        last_grads = grads
        last_cache = cache
        if convergence_step is None and (step % eval_interval == 0 or step == config.steps):
            current_loss, _, _ = _evaluate(architecture, params, x_train, y_train)
            if current_loss <= convergence_target:
                convergence_step = step
    elapsed = max(time.perf_counter() - started, 1e-12)

    final_train_loss, final_train_accuracy, final_train_activation_variance = _evaluate(
        architecture, params, x_train, y_train
    )
    val_loss, val_accuracy, val_activation_variance = _evaluate(
        architecture, params, x_val, y_val
    )
    grad_mean = float(np.mean(gradient_norms))
    grad_std = float(np.std(gradient_norms))
    parameter_count = int(sum(value.size for value in params.values()))
    parameter_bytes = _arrays_nbytes(params)
    optimizer_bytes = _arrays_nbytes(first_moment) + _arrays_nbytes(second_moment)
    gradient_bytes = _arrays_nbytes(last_grads)
    cache_bytes = _arrays_nbytes(last_cache)
    working_set_bytes = parameter_bytes + optimizer_bytes + gradient_bytes + cache_bytes
    samples_processed = config.steps * config.batch_size

    return {
        "architecture": architecture,
        "architecture_definition": {
            "standard": "single-hidden-layer tanh classifier",
            "gst": "Gated State Transition: tanh candidate multiplied by sigmoid gate",
            "fractal": "dual first-scale branches feeding a recursive second-scale residual branch",
        }[architecture],
        "initializer": initializer,
        "seed": seed,
        "metrics": {
            "initial_train_loss": initial_loss,
            "initial_train_accuracy": initial_accuracy,
            "final_train_loss": final_train_loss,
            "final_train_accuracy": final_train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "convergence_step": convergence_step,
            "gradient_norm_mean": grad_mean,
            "gradient_norm_std": grad_std,
            "gradient_norm_max": float(np.max(gradient_norms)),
            "gradient_norm_cv": grad_std / grad_mean if grad_mean > 0.0 else 0.0,
            "gradient_finite_fraction": finite_gradients / config.steps,
            "activation_variance_mean": float(np.mean(activation_variances)),
            "activation_variance_std": float(np.std(activation_variances)),
            "final_train_activation_variance": final_train_activation_variance,
            "validation_activation_variance": val_activation_variance,
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "working_set_bytes": working_set_bytes,
            "elapsed_seconds": elapsed,
            "training_samples_per_second": samples_processed / elapsed,
            "steps_per_second": config.steps / elapsed,
        },
    }


def _mean(rows: list[dict[str, object]], metric: str) -> float:
    values = [float(row["metrics"][metric]) for row in rows]  # type: ignore[index]
    return float(np.mean(values))


def _variance(rows: list[dict[str, object]], metric: str) -> float:
    values = [float(row["metrics"][metric]) for row in rows]  # type: ignore[index]
    return float(np.var(values))


def _aggregate(trials: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    comparisons: dict[str, dict[str, object]] = {}
    validation_loss_wins = 0
    accuracy_deltas: list[float] = []

    for architecture in ARCHITECTURES:
        by_initializer: dict[str, list[dict[str, object]]] = {}
        for initializer in INITIALIZERS:
            rows = [
                row
                for row in trials
                if row["architecture"] == architecture and row["initializer"] == initializer
            ]
            by_initializer[initializer] = rows
            key = f"{architecture}/{initializer}"
            groups[key] = {
                "runs": len(rows),
                "mean_final_train_loss": _mean(rows, "final_train_loss"),
                "mean_validation_loss": _mean(rows, "validation_loss"),
                "mean_validation_accuracy": _mean(rows, "validation_accuracy"),
                "seed_variance_validation_loss": _variance(rows, "validation_loss"),
                "seed_variance_validation_accuracy": _variance(rows, "validation_accuracy"),
                "mean_gradient_norm_cv": _mean(rows, "gradient_norm_cv"),
                "mean_activation_variance": _mean(rows, "activation_variance_mean"),
                "mean_training_samples_per_second": _mean(
                    rows, "training_samples_per_second"
                ),
                "mean_working_set_bytes": _mean(rows, "working_set_bytes"),
            }

        xavier = by_initializer["xavier"]
        mri = by_initializer["mri"]
        xavier_val_loss = _mean(xavier, "validation_loss")
        mri_val_loss = _mean(mri, "validation_loss")
        xavier_accuracy = _mean(xavier, "validation_accuracy")
        mri_accuracy = _mean(mri, "validation_accuracy")
        loss_delta = mri_val_loss - xavier_val_loss
        accuracy_delta = mri_accuracy - xavier_accuracy
        accuracy_deltas.append(accuracy_delta)
        if loss_delta < 0.0:
            validation_loss_wins += 1
        comparisons[architecture] = {
            "mri_minus_xavier_validation_loss": loss_delta,
            "mri_minus_xavier_validation_accuracy": accuracy_delta,
            "mri_to_xavier_gradient_cv_ratio": _mean(mri, "gradient_norm_cv")
            / max(_mean(xavier, "gradient_norm_cv"), 1e-12),
            "mri_to_xavier_throughput_ratio": _mean(mri, "training_samples_per_second")
            / max(_mean(xavier, "training_samples_per_second"), 1e-12),
            "mri_validation_loss_win": loss_delta < 0.0,
        }

    all_finite = all(
        float(row["metrics"]["gradient_finite_fraction"]) == 1.0  # type: ignore[index]
        for row in trials
    )
    mean_accuracy_delta = float(np.mean(accuracy_deltas))
    if not all_finite:
        decision = "REJECT_NUMERICAL_INSTABILITY"
    elif validation_loss_wins >= 2 and mean_accuracy_delta >= -0.01:
        decision = "PROMOTE_FOR_LARGER_TRIAL"
    elif validation_loss_wins == 0:
        decision = "REJECT_CURRENT_FORM"
    else:
        decision = "INCONCLUSIVE"

    return {
        "groups": groups,
        "initializer_comparisons": comparisons,
        "decision": decision,
        "decision_evidence": {
            "mri_validation_loss_wins_out_of_3": validation_loss_wins,
            "mean_mri_minus_xavier_validation_accuracy": mean_accuracy_delta,
            "all_gradients_finite": all_finite,
        },
        "interpretation_boundary": (
            "Initializer comparisons within each architecture are controlled. "
            "Architecture-to-architecture rankings are exploratory because parameter counts differ."
        ),
    }


def scientific_projection(report: dict[str, object]) -> dict[str, object]:
    """Drop wall-clock fields so the scientific receipt is reproducibility-focused."""

    projected_trials: list[dict[str, object]] = []
    for row in report["trials"]:  # type: ignore[union-attr]
        metrics = dict(row["metrics"])
        metrics.pop("elapsed_seconds", None)
        metrics.pop("training_samples_per_second", None)
        metrics.pop("steps_per_second", None)
        projected_trials.append({**row, "metrics": metrics})

    aggregate = json.loads(json.dumps(report["aggregate"]))
    for group in aggregate["groups"].values():
        group.pop("mean_training_samples_per_second", None)
    for comparison in aggregate["initializer_comparisons"].values():
        comparison.pop("mri_to_xavier_throughput_ratio", None)
    return {
        "schema": report["schema"],
        "cfg": report["cfg"],
        "seeds": report["seeds"],
        "trials": projected_trials,
        "aggregate": aggregate,
    }


def run_tournament(
    cfg: TournamentCfg | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, object]:
    """Run all 3 architectures x 2 initializers x N seeds and issue a research decision."""

    config = cfg or TournamentCfg()
    if not seeds:
        raise ValueError("at least one seed is required")
    trials = [
        run_trial(architecture, initializer, seed, config)
        for architecture in ARCHITECTURES
        for initializer in INITIALIZERS
        for seed in seeds
    ]
    report: dict[str, object] = {
        "schema": "MRI-LEARNING-TOURNAMENT-1",
        "cfg": asdict(config),
        "seeds": list(seeds),
        "trials": trials,
        "aggregate": _aggregate(trials),
    }
    projection = scientific_projection(report)
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["scientific_receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report