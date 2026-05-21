from __future__ import annotations

from dataclasses import dataclass

from starpower_core.agents import AgentResult
from starpower_core.memory import RemMemoryGraph
from starpower_core.vsa import VSAEngine


@dataclass(slots=True)
class MassiveMagneticsAdapter:
    name: str = "massive_magnetics"
    description: str = "Routes engineering, physical-system, and creative-substrate work."

    def run(self, task: str, memory: RemMemoryGraph, vsa: VSAEngine) -> AgentResult:
        memory.remember(task, kind="massive_magnetics_task", tags=["body", "engineering"])
        body = vsa.encode("massive_magnetics:body")
        task_vec = vsa.encode(task)
        score = max(vsa.similarity(body, task_vec), 0.1)
        output = "Mapped MASSIVEMAGNETICS task into physical/creative substrate lane."
        return AgentResult(agent=self.name, success=True, output=output, confidence=score)
