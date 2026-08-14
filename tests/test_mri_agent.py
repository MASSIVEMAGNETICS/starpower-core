from __future__ import annotations

import json

from starpower_core.agents import Orchestrator
from starpower_core.research.agent import MRIResearchAgent


def test_mri_agent_runs_through_orchestrator(tmp_path) -> None:
    output = tmp_path / "receipt.json"
    agent = MRIResearchAgent(output_path=output)
    orchestrator = Orchestrator(agents={agent.name: agent})

    result = orchestrator.run("run MRI-0 benchmark seed=440")

    assert result.success is True
    assert result.agent == "mri0-research"
    assert output.exists()
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["seed"] == 440
    assert receipt["authority"] == "research-only"
    assert receipt["failures"] == []
    assert len(receipt["receipt_sha256"]) == 64
    assert orchestrator.self_audit()["healthy"] is True


def test_mri_agent_receipt_is_deterministic_for_same_seed(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_agent = MRIResearchAgent(output_path=first)
    second_agent = MRIResearchAgent(output_path=second)
    first_result = Orchestrator(agents={first_agent.name: first_agent}).run(
        "run MRI-0 benchmark seed=1337"
    )
    second_result = Orchestrator(agents={second_agent.name: second_agent}).run(
        "run MRI-0 benchmark seed=1337"
    )

    assert first_result.success is True
    assert second_result.success is True
    first_receipt = json.loads(first.read_text(encoding="utf-8"))
    second_receipt = json.loads(second.read_text(encoding="utf-8"))
    assert first_receipt["receipt_sha256"] == second_receipt["receipt_sha256"]
