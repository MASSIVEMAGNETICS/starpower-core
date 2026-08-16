"""Run the controlled MRI-0 learning tournament and emit a JSON research receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starpower_core.research.tournament import DEFAULT_SEEDS, TournamentCfg, run_tournament


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mri-learning-tournament/report.json"),
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated deterministic model/training seeds",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Run a fast CI smoke tournament with a reduced training budget",
    )
    args = parser.parse_args()

    cfg = (
        TournamentCfg(
            input_dim=8,
            hidden_dim=12,
            n_classes=3,
            train_size=144,
            val_size=72,
            batch_size=24,
            steps=min(args.steps, 36),
            learning_rate=0.018,
        )
        if args.small
        else TournamentCfg(steps=args.steps)
    )
    report = run_tournament(cfg=cfg, seeds=args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    aggregate = report["aggregate"]
    evidence = aggregate["decision_evidence"]
    summary = {
        "output": str(args.output),
        "scientific_receipt_sha256": report["scientific_receipt_sha256"],
        "decision": aggregate["decision"],
        "mri_validation_loss_wins_out_of_3": evidence[
            "mri_validation_loss_wins_out_of_3"
        ],
        "mean_mri_minus_xavier_validation_accuracy": evidence[
            "mean_mri_minus_xavier_validation_accuracy"
        ],
        "all_gradients_finite": evidence["all_gradients_finite"],
        "trial_count": len(report["trials"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if evidence["all_gradients_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
