from starpower_core.agents import Orchestrator
from starpower_core.adapters.b_heard import BHeardAdapter
from starpower_core.adapters.massive_magnetics import MassiveMagneticsAdapter


def test_orchestrator_runs_default_agent() -> None:
    orchestrator = Orchestrator()
    result = orchestrator.run("status check")
    assert result.success is True
    assert result.agent == "echo"


def test_orchestrator_registers_adapters_and_audits() -> None:
    orchestrator = Orchestrator()
    orchestrator.register(MassiveMagneticsAdapter())
    orchestrator.register(BHeardAdapter())
    audit = orchestrator.self_audit()
    assert audit["healthy"] is True
    assert "massive_magnetics" in audit["agents"]
    assert "b_heard" in audit["agents"]
