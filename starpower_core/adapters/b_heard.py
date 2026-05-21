from __future__ import annotations

from dataclasses import dataclass

from starpower_core.agents import AgentResult
from starpower_core.memory import RemMemoryGraph
from starpower_core.vsa import VSAEngine


@dataclass(slots=True)
class BHeardAdapter:
    name: str = "b_heard"
    description: str = "Routes media, signal, community, and broadcast-network work."

    def run(self, task: str, memory: RemMemoryGraph, vsa: VSAEngine) -> AgentResult:
        memory.remember(task, kind="b_heard_task", tags=["voice", "signal"])
        voice = vsa.encode("b_heard:voice")
        task_vec = vsa.encode(task)
        score = max(vsa.similarity(voice, task_vec), 0.1)
        output = "Mapped B Heard Network task into voice/signal propagation lane."
        return AgentResult(agent=self.name, success=True, output=output, confidence=score)
