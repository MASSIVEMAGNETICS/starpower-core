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
from numbers import Integral
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
        integer_fields = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_classes": self.n_classes,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "data_seed": self.data_seed,
        }
        malformed = [
            name
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, Integral)
        ]
        if malformed:
            joined = ", ".join(sorted(malformed))
            raise ValueError(f"tournament integer fields must be integers: {joined}")

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
        if self.data_seed < 0:
            raise ValueError("data_seed must be non-negative")
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
        if not math.isfinite(self.convergence_fraction) or not 0.0 < self.convergence_fraction < 1.0:
            raise ValueError("convergence_fraction must be finite and within (0, 1)")
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
        w2, b2 = factory.linear(cfg.hidden_dim, cfg.n_classes, structured=False)
        return {
            "W1a": w1a,
            "b1a": b1a,
            "W1b": w1b,
            "b1b": b1b,
            "W2": w2,
            "b2": b2,
        }
    raise ValueError(f"unsupported architecture: {architecture}")


def _softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _cross_entropy(probs: Array, labels: Array) -> float:
    eps = 1e-12
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0))))


def _accuracy(probs: Array, labels: Array) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == labels))


def _relu(value: Array) -> Array:
    return np.maximum(value, 0.0)


def _forward(
    architecture: Architecture,
    params: dict[str, Array],
    x: Array,
) -> tuple[Array, dict[str, Array]]:
    if architecture == "standard":
        z1 = x @ params["W1"] + params["b1"]
        h = _relu(z1)
        logits = h @ params["W2"] + params["b2"]
        return logits, {"x": x, "z1": z1, "h": h}
    if architecture == "gst":
        za = x @ params["Wa"] + params["ba"]
        zg = x @ params["Wg"] + params["bg"]
        a = np.tanh(za)
        g = 1.0 / (1.0 + np.exp(-zg))
        h = a * g
        logits = h @ params["Wo"] + params["bo"]
        return logits, {"x": x, "za": za, "zg": zg, "a": a, "g": g, "h": h}
    if architecture == "fractal":
        z1a = x @ params["W1a"] + params["b1a"]
        z1b = x @ params["W1b"] + params["b1b"]
        h1a = _relu(z1a)
        h1b = _relu(z1b)
        h = 0.5 * (h1a + h1b)
        logits = h @ params["W2"] + params["b2"]
        return logits, {"x": x, "z1a": z1a, "z1b": z1b, "h1a": h1a, "h1b": h1b, "h": h}
    raise ValueError(f"unsupported architecture: {architecture}")


def _backward(
    architecture: Architecture,
    params: dict[str, Array],
    cache: dict[str, Array],
    probs: Array,
    labels: Array,
) -> dict[str, Array]:
    batch = labels.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(batch), labels] -= 1.0
    dlogits /= float(batch)

    if architecture == "standard":
        grad_w2 = cache["h"].T @ dlogits
        grad_b2 = np.sum(dlogits, axis=0)
        dh = dlogits @ params["W2"].T
        dz1 = dh * (cache["z1"] > 0.0)
        grad_w1 = cache["x"].T @ dz1
        grad_b1 = np.sum(dz1, axis=0)
        return {"W1": grad_w1, "b1": grad_b1, "W2": grad_w2, "b2": grad_b2}

    if architecture == "gst":
        grad_wo = cache["h"].T @ dlogits
        grad_bo = np.sum(dlogits, axis=0)
        dh = dlogits @ params["Wo"].T
        da = dh * cache["g"]
        dg = dh * cache["a"]
        dza = da * (1.0 - np.square(cache["a"]))
        dzg = dg * cache["g"] * (1.0 - cache["g"])
        return {
            "Wa": cache["x"].T @ dza,
            "ba": np.sum(dza, axis=0),
            "Wg": cache["x"].T @ dzg,
            "bg": np.sum(dzg, axis=0),
            "Wo": grad_wo,
            "bo": grad_bo,
        }

    if architecture == "fractal":
        grad_w2 = cache["h"].T @ dlogits
        grad_b2 = np.sum(dlogits, axis=0)
        dh = dlogits @ params["W2"].T
        dh1a = 0.5 * dh
        dh1b = 0.5 * dh
        dz1a = dh1a * (cache["z1a"] > 0.0)
        dz1b = dh1b * (cache["z1b"] > 0.0)
        return {
            "W1a": cache["x"].T @ dz1a,
            "b1a": np.sum(dz1a, axis=0),
            "W1b": cache["x"].T @ dz1b,
            "b1b": np.sum(dz1b, axis=0),
            "W2": grad_w2,
            "b2": grad_b2,
        }

    raise ValueError(f"unsupported architecture: {architecture}")


def _global_norm(grads: dict[str, Array]) -> float:
    total = 0.0
    for grad in grads.values():
        total += float(np.sum(np.square(grad, dtype=np.float64), dtype=np.float64))
    return math.sqrt(total)


def _clip_gradients(grads: dict[str, Array], max_norm: float) -> tuple[dict[str, Array], float]:
    norm = _global_norm(grads)
    if norm <= max_norm or norm == 0.0:
        return grads, norm
    scale = max_norm / norm
    return {name: grad * scale for name, grad in grads.items()}, norm


def _make_dataset(cfg: TournamentCfg) -> tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng(cfg.data_seed)
    total = cfg.train_size + cfg.val_size
    x = rng.normal(0.0, 1.0, size=(total, cfg.input_dim)).astype(np.float32)
    teacher = rng.normal(0.0, 1.0, size=(cfg.input_dim, cfg.n_classes)).astype(np.float32)
    logits = x @ teacher + 0.15 * rng.normal(size=(total, cfg.n_classes)).astype(np.float32)
    labels = np.argmax(logits, axis=1).astype(np.int64)
    return x[: cfg.train_size], labels[: cfg.train_size], x[cfg.train_size :], labels[cfg.train_size :]


def _parameter_bytes(params: dict[str, Array]) -> int:
    return sum(int(value.nbytes) for value in params.values())


def _working_set_bytes(params: dict[str, Array], cache: dict[str, Array], grads: dict[str, Array]) -> int:
    return _parameter_bytes(params) + sum(int(value.nbytes) for value in cache.values()) + sum(
        int(value.nbytes) for value in grads.values()
    )


def _trial(
    architecture: Architecture,
    initializer: InitializerKind,
    cfg: TournamentCfg,
    seed: int,
    x_train: Array,
    y_train: Array,
    x_val: Array,
    y_val: Array,
) -> dict[str, object]:
    params = _build_model(architecture, initializer, cfg, seed)
    rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
    initial_logits, _ = _forward(architecture, params, x_train)
    initial_probs = _softmax(initial_logits)
    initial_loss = _cross_entropy(initial_probs, y_train)

    losses: list[float] = []
    gradient_norms: list[float] = []
    activation_variances: list[float] = []
    gradient_finite = 0
    first_target_step: int | None = None
    start = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        batch_index = rng.integers(0, cfg.train_size, size=cfg.batch_size)
        xb = x_train[batch_index]
        yb = y_train[batch_index]
        logits, cache = _forward(architecture, params, xb)
        probs = _softmax(logits)
        loss = _cross_entropy(probs, yb)
        grads = _backward(architecture, params, cache, probs, yb)
        grads, raw_norm = _clip_gradients(grads, cfg.gradient_clip_norm)
        finite = math.isfinite(raw_norm) and all(np.all(np.isfinite(value)) for value in grads.values())
        if finite:
            gradient_finite += 1
        for name, grad in grads.items():
            params[name] = params[name] - cfg.learning_rate * grad.astype(params[name].dtype, copy=False)
        losses.append(loss)
        gradient_norms.append(raw_norm)
        activation_variances.append(float(np.var(cache["h"], dtype=np.float64)))
        if first_target_step is None and loss <= initial_loss * cfg.convergence_fraction:
            first_target_step = step

    elapsed = max(time.perf_counter() - start, 1e-9)
    val_logits, _ = _forward(architecture, params, x_val)
    val_probs = _softmax(val_logits)
    final_train_logits, _ = _forward(architecture, params, x_train)
    final_train_probs = _softmax(final_train_logits)

    metrics = {
        "initial_train_loss": initial_loss,
        "final_train_loss": _cross_entropy(final_train_probs, y_train),
        "validation_loss": _cross_entropy(val_probs, y_val),
        "validation_accuracy": _accuracy(val_probs, y_val),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "gradient_norm_cv": float(np.std(gradient_norms) / (np.mean(gradient_norms) + 1e-12)),
        "gradient_finite_fraction": gradient_finite / cfg.steps,
        "activation_variance_mean": float(np.mean(activation_variances)),
        "first_target_step": first_target_step,
        "training_samples_per_second": (cfg.steps * cfg.batch_size) / elapsed,
        "parameter_count": sum(int(value.size) for value in params.values()),
        "parameter_bytes": _parameter_bytes(params),
        "working_set_bytes": _working_set_bytes(params, cache, grads),
    }
    return {
        "architecture": architecture,
        "initializer": initializer,
        "seed": seed,
        "metrics": metrics,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _aggregate(trials: list[dict[str, object]]) -> dict[str, object]:
    by_condition: dict[tuple[str, str], list[dict[str, object]]] = {}
    for trial in trials:
        key = (str(trial["architecture"]), str(trial["initializer"]))
        by_condition.setdefault(key, []).append(trial)

    rows: list[dict[str, object]] = []
    numerical_instability = False
    for architecture in ARCHITECTURES:
        for initializer in INITIALIZERS:
            condition = by_condition[(architecture, initializer)]
            metrics = [row["metrics"] for row in condition]
            typed_metrics = [dict(row) for row in metrics if isinstance(row, dict)]
            validation_accuracy = _mean([float(row["validation_accuracy"]) for row in typed_metrics])
            validation_loss = _mean([float(row["validation_loss"]) for row in typed_metrics])
            gradient_norm_cv = _mean([float(row["gradient_norm_cv"]) for row in typed_metrics])
            activation_variance_mean = _mean([float(row["activation_variance_mean"]) for row in typed_metrics])
            gradient_finite_fraction = _mean([float(row["gradient_finite_fraction"]) for row in typed_metrics])
            first_target_values = [row["first_target_step"] for row in typed_metrics if row["first_target_step"] is not None]
            rows.append(
                {
                    "architecture": architecture,
                    "initializer": initializer,
                    "validation_accuracy_mean": validation_accuracy,
                    "validation_loss_mean": validation_loss,
                    "gradient_norm_cv_mean": gradient_norm_cv,
                    "activation_variance_mean": activation_variance_mean,
                    "gradient_finite_fraction": gradient_finite_fraction,
                    "first_target_step_mean": _mean([float(value) for value in first_target_values]) if first_target_values else None,
                }
            )
            numerical_instability = numerical_instability or gradient_finite_fraction < 1.0

    comparisons: list[dict[str, object]] = []
    accuracy_deltas: list[float] = []
    loss_deltas: list[float] = []
    cv_deltas: list[float] = []
    variance_ratios: list[float] = []
    for architecture in ARCHITECTURES:
        xavier = next(row for row in rows if row["architecture"] == architecture and row["initializer"] == "xavier")
        mri = next(row for row in rows if row["architecture"] == architecture and row["initializer"] == "mri")
        accuracy_delta = float(mri["validation_accuracy_mean"]) - float(xavier["validation_accuracy_mean"])
        loss_delta = float(mri["validation_loss_mean"]) - float(xavier["validation_loss_mean"])
        cv_delta = float(mri["gradient_norm_cv_mean"]) - float(xavier["gradient_norm_cv_mean"])
        variance_ratio = float(mri["activation_variance_mean"]) / (float(xavier["activation_variance_mean"]) + 1e-12)
        accuracy_deltas.append(accuracy_delta)
        loss_deltas.append(loss_delta)
        cv_deltas.append(cv_delta)
        variance_ratios.append(variance_ratio)
        comparisons.append(
            {
                "architecture": architecture,
                "validation_accuracy_delta": accuracy_delta,
                "validation_loss_delta": loss_delta,
                "gradient_norm_cv_delta": cv_delta,
                "activation_variance_ratio": variance_ratio,
            }
        )

    mean_accuracy_delta = _mean(accuracy_deltas)
    mean_loss_delta = _mean(loss_deltas)
    mean_cv_delta = _mean(cv_deltas)
    variance_in_band = all(0.25 <= value <= 4.0 for value in variance_ratios)

    if numerical_instability:
        decision = "REJECT_NUMERICAL_INSTABILITY"
    elif mean_accuracy_delta >= 0.0 and mean_loss_delta <= 0.0 and mean_cv_delta <= 0.0 and variance_in_band:
        decision = "PROMOTE_FOR_LARGER_TRIAL"
    elif mean_accuracy_delta < -0.03 or mean_loss_delta > 0.05 or mean_cv_delta > 0.20 or not variance_in_band:
        decision = "REJECT_CURRENT_FORM"
    else:
        decision = "INCONCLUSIVE"

    return {
        "decision": decision,
        "conditions": rows,
        "comparisons": comparisons,
        "summary": {
            "mean_validation_accuracy_delta": mean_accuracy_delta,
            "mean_validation_loss_delta": mean_loss_delta,
            "mean_gradient_norm_cv_delta": mean_cv_delta,
            "activation_variance_all_in_band": variance_in_band,
        },
    }


def scientific_projection(report: dict[str, object]) -> dict[str, object]:
    """Remove wall-clock-only measurements from the scientific receipt."""

    projected = json.loads(json.dumps(report))
    projected.pop("scientific_receipt_sha256", None)
    for trial in projected.get("trials", []):
        metrics = trial.get("metrics", {})
        metrics.pop("training_samples_per_second", None)
    return projected


def run_tournament(
    cfg: TournamentCfg | None = None,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, object]:
    cfg = cfg or TournamentCfg()
    x_train, y_train, x_val, y_val = _make_dataset(cfg)
    trials: list[dict[str, object]] = []
    for seed in seeds:
        for architecture in ARCHITECTURES:
            for initializer in INITIALIZERS:
                trials.append(
                    _trial(
                        architecture,
                        initializer,
                        cfg,
                        seed,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                    )
                )
    aggregate = _aggregate(trials)
    report = {
        "schema": "MRI-LEARNING-TOURNAMENT-1",
        "config": asdict(cfg),
        "seeds": list(seeds),
        "trials": trials,
        "aggregate": aggregate,
    }
    projection = scientific_projection(report)
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["scientific_receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report
