from __future__ import annotations

import numpy as np

from starpower_core.research.mri1 import (
    ARCHITECTURES,
    DEFAULT_SEEDS,
    INITIALIZERS,
    OPTIMIZERS,
    TASKS,
    StudyCfg,
    _build_model,
    _dataset,
    run_study,
    run_trial,
)


def _tiny_cfg() -> StudyCfg:
    return StudyCfg(
        hidden_dim=8,
        train_size=64,
        val_size=32,
        batch_size=16,
        steps=3,
    )


def test_mri1_has_requested_factorial_surface() -> None:
    assert ARCHITECTURES == ("standard", "gst", "fractal")
    assert INITIALIZERS == ("xavier", "kaiming", "orthogonal", "mri")
    assert TASKS == ("classification", "regression", "sequence", "xor")
    assert OPTIMIZERS == ("adamw", "sgd")
    assert len(DEFAULT_SEEDS) >= 10


def test_all_initializer_model_surfaces_are_finite() -> None:
    cfg = _tiny_cfg()
    for architecture in ARCHITECTURES:
        for initializer in INITIALIZERS:
            params = _build_model(
                architecture,
                initializer,
                "classification",
                cfg,
                440,
            )
            assert params
            assert all(np.isfinite(value).all() for value in params.values())


def test_task_datasets_are_deterministic_and_shape_safe() -> None:
    cfg = _tiny_cfg()
    for task in TASKS:
        first = _dataset(task, cfg)
        second = _dataset(task, cfg)
        assert len(first) == 4
        for left, right in zip(first, second, strict=True):
            assert np.array_equal(left, right)
        x_train, y_train, x_val, y_val = first
        assert x_train.shape == (cfg.train_size, cfg.input_dim)
        assert x_val.shape == (cfg.val_size, cfg.input_dim)
        assert y_train.shape[0] == cfg.train_size
        assert y_val.shape[0] == cfg.val_size


def test_trial_executes_every_task_and_optimizer() -> None:
    cfg = _tiny_cfg()
    for task in TASKS:
        for optimizer in OPTIMIZERS:
            result = run_trial(
                "standard",
                "xavier",
                task,
                optimizer,
                440,
                cfg,
            )
            assert result["finite_gradient_fraction"] == 1.0
            assert np.isfinite(float(result["validation_loss"]))
            assert np.isfinite(float(result["validation_metric"]))


def test_scientific_receipt_excludes_wall_clock_noise() -> None:
    first = run_study(mode="smoke", steps=2, seeds=(440, 1337))
    second = run_study(mode="smoke", steps=2, seeds=(440, 1337))
    assert first["scientific_receipt_sha256"] == second["scientific_receipt_sha256"]
    assert first["design"]["main_trial_count"] == 96
    assert first["design"]["ablation_trial_count"] == 24
    assert first["decision"] in {
        "PROMOTE_FOR_PYTORCH_TRIAL",
        "REJECT_CURRENT_FORM",
        "REJECT_NUMERICAL_INSTABILITY",
        "INCONCLUSIVE",
    }
