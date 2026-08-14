"""Research-only MRI-0 agent and CLI harness.

This agent is intentionally quarantined under ``starpower_core.research``.
It can generate benchmark receipts, but it has no canonical-memory or external
execution authority beyond writing its requested local artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .mri import MathRevolutionaryInitializer, MRICfg, MRIComponents, tensor_statistics
from ..agents import AgentResult, Orchestrator
from ..memory import RemMemoryGraph
from ..vsa import VSAEngine


SEED_RE = re.compile(r"(?:^|\s)seed=(\d+)(?:\s|$)")


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


def _small_cfg(seed: int) -> MRICfg:
    return MRICfg(
        vocab_size=128,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_len=32,
        seed=seed,
        julia_iters=16,
    )


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class MRIResearchAgent:
    """Runs deterministic MRI-0 initialization experiments and emits receipts."""

    output_path: Path = Path("artifacts/mri0-agent/receipt.json")
    name: str = "mri0-research"
    description: str = (
        "Runs quarantined MRI-0 structured-initialization baselines, ablations, "
        "determinism checks, finite-value checks, and research receipts."
    )

    def run(self, task: str, memory: RemMemoryGraph, vsa: VSAEngine) -> AgentResult:
        match = SEED_RE.search(task)
        seed = int(match.group(1)) if match else 20250822
        cfg = _small_cfg(seed)
        variants: dict[str, Any] = {}
        failures: list[str] = []

        for name, components in _variants().items():
            first_initializer = MathRevolutionaryInitializer(cfg, components)
            first = first_initializer.flatten_arrays(first_initializer.init_transformer(n_blocks=1))
            second_initializer = MathRevolutionaryInitializer(cfg, components)
            second = second_initializer.flatten_arrays(second_initializer.init_transformer(n_blocks=1))

            deterministic = set(first) == set(second) and all(
                np.array_equal(first[key], second[key]) for key in first
            )
            float_arrays = {
                key: value for key, value in first.items() if value.dtype.kind == "f"
            }
            all_finite = all(np.isfinite(value).all() for value in float_arrays.values())
            if not deterministic:
                failures.append(f"{name}:nondeterministic")
            if not all_finite:
                failures.append(f"{name}:nonfinite")

            variants[name] = {
                "components": asdict(components),
                "deterministic": deterministic,
                "all_finite": all_finite,
                "tensor_count": len(first),
                "float_tensor_stats": {
                    key: tensor_statistics(value) for key, value in float_arrays.items()
                },
            }

        receipt_core: dict[str, Any] = {
            "agent": self.name,
            "task": task,
            "seed": seed,
            "cfg": asdict(cfg),
            "variants": variants,
            "failures": failures,
            "authority": "research-only",
        }
        receipt = {**receipt_core, "receipt_sha256": _stable_hash(receipt_core)}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )

        _ = vsa.encode(f"mri0 seed={seed} failures={len(failures)}")
        memory.remember(
            f"MRI-0 research run seed={seed} failures={len(failures)} receipt={receipt['receipt_sha256']}",
            kind="research",
            tags=["mri0", "benchmark", "receipt"],
            weight=1.0 if not failures else 2.0,
        )
        return AgentResult(
            agent=self.name,
            success=not failures,
            output=str(self.output_path),
            confidence=1.0 if not failures else 0.0,
            errors=failures,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the quarantined MRI-0 research agent")
    parser.add_argument("--seed", type=int, default=20250822)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/mri0-agent/receipt.json")
    )
    args = parser.parse_args()

    agent = MRIResearchAgent(output_path=args.output)
    orchestrator = Orchestrator(agents={agent.name: agent})
    result = orchestrator.run(f"run MRI-0 benchmark seed={args.seed}")
    audit = orchestrator.self_audit()
    print(
        json.dumps(
            {
                "agent": result.agent,
                "success": result.success,
                "output": result.output,
                "errors": result.errors,
                "orchestrator_healthy": audit["healthy"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.success and bool(audit["healthy"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
