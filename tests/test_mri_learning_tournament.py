from __future__ import annotations

import math

import pytest

from starpower_core.research.tournament import TournamentCfg, run_tournament


def _tiny_cfg() -> TournamentCfg:
    return TournamentCfg(
        input_dim=6,
        hidden_dim=8,
        n_classes=3,
        train_size=48,
        val_size=24,
        batch_size=12,
        steps=8,
        learning_rate=0.02,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("gradient_clip_norm", float("nan")),
        ("gradient_clip_norm", float("inf")),
        ("input_dim", 1),
        ("hidden_dim", 1),
    ),
)
def test_tournament_config_rejects_invalid_optimizer_values_and_mri_dimensions(
    field: str, value: float | int
) -> None:
    with pytest.raises(ValueError):
        TournamentCfg(**{field: value})


def test_tournament_runs_full_factorial_and_stays_finite() -> None:
    report = run_tournament(cfg=_tiny_cfg(), seeds=(7,))
    assert report["schema"] == "MRI-LEARNING-TOURNAMENT-1"
    assert len(report["trials"]) == 6

    conditions = {
        (row["architecture"], row["initializer"]) for row in report["trials"]
    }
    assert conditions == {
        ("standard", "xavier"),
        ("standard", "mri"),
        ("gst", "xavier"),
        ("gst", "mri"),
        ("fractal", "xavier"),
        ("fractal", "mri"),
    }

    for row in report["trials"]:
        metrics = row["metrics"]
        assert metrics["gradient_finite_fraction"] == 1.0
        assert metrics["parameter_count"] > 0
        assert metrics["parameter_bytes"] > 0
        assert metrics["working_set_bytes"] >= metrics["parameter_bytes"]
        assert metrics["training_samples_per_second"] > 0.0
        for name in (
            "initial_train_loss",
            "final_train_loss",
            "validation_loss",
            "validation_accuracy",
            "gradient_norm_mean",
            "gradient_norm_cv",
            "activation_variance_mean",
        ):
            assert math.isfinite(float(metrics[name])), name


def test_scientific_receipt_ignores_wall_clock_noise() -> None:
    first = run_tournament(cfg=_tiny_cfg(), seeds=(11,))
    second = run_tournament(cfg=_tiny_cfg(), seeds=(11,))
    assert first["scientific_receipt_sha256"] == second["scientific_receipt_sha256"]


def test_aggregate_exposes_controlled_initializer_comparisons() -> None:
    report = run_tournament(cfg=_tiny_cfg(), seeds=(3, 5))
    aggregate = report["aggregate"]
    assert aggregate["decision"] in {
        "PROMOTE_FOR_LARGER_TRIAL",
        "REJECT_CURRENT_FORM",
        "REJECT_NUMERICAL_INSTABILITY",
        "INCONCLUSIVE",
    }
    assert aggregate["decision_evidence"]["all_gradients_finite"] is True
    assert set(aggregate["initializer_comparisons"]) == {"standard", "gst", "fractal"}
    for comparison in aggregate["initializer_comparisons"].values():
        assert "mri_minus_xavier_validation_loss" in comparison
        assert "mri_minus_xavier_validation_accuracy" in comparison
        assert "mri_to_xavier_gradient_cv_ratio" in comparison
        assert "mri_to_xavier_throughput_ratio" in comparison
