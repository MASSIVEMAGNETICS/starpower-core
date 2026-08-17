#!/usr/bin/env python3
"""CLI for MRI-1 multi-task screening studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starpower_core.research.mri1 import DEFAULT_SEEDS, run_study


def _parse_seeds(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds")
    parser.add_argument(
        "--output",
        default="artifacts/mri1/report.json",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    report = run_study(mode=args.mode, steps=args.steps, seeds=seeds)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    main_aggregate = report["main"]
    overall = main_aggregate["overall_relative_loss_improvement"]
    summary = {
        "schema": report["schema"],
        "mode": report["mode"],
        "decision": report["decision"],
        "main_trial_count": report["design"]["main_trial_count"],
        "ablation_trial_count": report["design"]["ablation_trial_count"],
        "qualifying_cell_count": main_aggregate["qualifying_cell_count"],
        "qualifying_tasks": main_aggregate["qualifying_tasks"],
        "overall_relative_loss_improvement": overall["mean"],
        "overall_probability_positive": overall["probability_positive_normal_approx"],
        "component_signal_detected": report["ablations"]["component_signal_detected"],
        "all_gradients_finite": main_aggregate["all_gradients_finite"],
        "scientific_receipt_sha256": report["scientific_receipt_sha256"],
        "output": str(destination),
        "default_seed_count": len(DEFAULT_SEEDS),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
