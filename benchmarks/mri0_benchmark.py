"""Reproducible MRI-0 stability/ablation benchmark.

This benchmark intentionally measures only initialization properties. It does
not claim downstream task quality. Learning benchmarks should reuse the same
configuration matrix with identical data/order/optimizer budgets.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from starpower_core.research.mri import (
    MRIComponents,
    MRICfg,
    MathRevolutionaryInitializer,
    tensor_statistics,
)


def _variants() -> dict[str, MRIComponents]:
    return {
        "xavier_baseline": MRIComponents.baseline(),
        "fractal_only": MRIComponents(
            chaos_embedding=False,
            prime_embedding=False,
            laplacian=False,
            entropy_rescale=False,
            zipf_scale=False,
            golden_layernorm=False,
        ),
        "fractal_entropy": MRIComponents(
            chaos_embedding=False,
            prime_embedding=False,
            laplacian=False,
            entropy_rescale=True,
            zipf_scale=False,
            golden_layernorm=False,
        ),
        "prime_chaos_embedding": MRIComponents(
            fractal=False,
            laplacian=False,
            entropy_rescale=False,
            zipf_scale=False,
            golden_layernorm=False,
        ),
        "full_mri": MRIComponents(),
    }


def run(cfg: MRICfg, n_blocks: int = 1) -> dict[str, object]:
    report: dict[str, object] = {"cfg": asdict(cfg), "variants": {}}
    for name, components in _variants().items():
        started = time.perf_counter()
        initializer = MathRevolutionaryInitializer(cfg, components)
        model = initializer.init_transformer(n_blocks=n_blocks)
        arrays = initializer.flatten_arrays(model)
        elapsed = time.perf_counter() - started
        matrix_stats = {
            key: tensor_statistics(value)
            for key, value in arrays.items()
            if value.dtype.kind == "f" and value.ndim >= 1
        }
        second_arrays = MathRevolutionaryInitializer.flatten_arrays(
            MathRevolutionaryInitializer(cfg, components).init_transformer(n_blocks=n_blocks)
        )
        deterministic = all(
            np.array_equal(arrays[path], second_arrays[path]) for path in arrays
        )
        report["variants"][name] = {
            "components": asdict(components),
            "elapsed_seconds": elapsed,
            "deterministic": deterministic,
            "all_finite": all(bool(stats["all_finite"]) for stats in matrix_stats.values()),
            "tensors": matrix_stats,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/mri0_benchmark.json")
    )
    parser.add_argument("--small", action="store_true", help="Use a fast CI-sized model")
    args = parser.parse_args()
    cfg = (
        MRICfg(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            d_ff=64,
            max_len=32,
            julia_iters=16,
        )
        if args.small
        else MRICfg()
    )
    report = run(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "variants": list(report["variants"])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
