from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .memory import RemMemoryGraph
from .vsa import VSAEngine


@dataclass(slots=True)
class AgentResult:
    agent: str
    success: bool
    output: str
    confidence: float = 0.0
    errors: list[str] = field(default_factory=list)


class Agent(Protocol):
    name: str
    description: str

    def run(self, task: str, memory: RemMemoryGraph, vsa: VSAEngine) -> AgentResult:
        ...


@dataclass(slots=True)
class EchoAgent:
    name: str = "echo"
    description: str = "Safe default agent that reflects tasks and stores them in memory."

    def run(self, task: str, memory: RemMemoryGraph, vsa: VSAEngine) -> AgentResult:
        memory.remember(task, kind="task", tags=["echo", "starpower"])
        _ = vsa.encode(task)
        return AgentResult(agent=self.name, success=True, output=f"Mapped task: {task}", confidence=0.55)


@dataclass(slots=True)
class Orchestrator:
    memory: RemMemoryGraph = field(default_factory=RemMemoryGraph)
    vsa: VSAEngine = field(default_factory=VSAEngine)
    agents: dict[str, Agent] = field(default_factory=dict)
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not self.agents:
            self.register(EchoAgent())

    def register(self, agent: Agent) -> None:
        if not agent.name.strip():
            raise ValueError("agent name must be non-empty")
        self.agents[agent.name] = agent
        self.memory.remember(f"Registered agent {agent.name}: {agent.description}", kind="system", tags=["agent", agent.name])

    def route(self, task: str) -> Agent:
        if not task.strip():
            raise ValueError("task must be non-empty")
        task_vec = self.vsa.encode(f"task:{task}")
        best_name = "echo"
        best_score = -1.0
        for name, agent in self.agents.items():
            agent_vec = self.vsa.encode(f"agent:{agent.name}:{agent.description}")
            score = self.vsa.similarity(task_vec, agent_vec)
            if score > best_score:
                best_name = name
                best_score = score
        return self.agents[best_name]

    def run(self, task: str) -> AgentResult:
        agent = self.route(task)
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                result = agent.run(task, self.memory, self.vsa)
                self.memory.remember(
                    f"Agent {agent.name} attempt {attempt} success={result.success}: {result.output}",
                    kind="execution",
                    tags=["run", agent.name],
                    weight=max(result.confidence, 0.1),
                )
                if result.success:
                    return result
                last_error = "; ".join(result.errors) or "agent returned failure"
            except Exception as exc:  # defensive runtime shell
                last_error = repr(exc)
                self.memory.remember(
                    f"Agent {agent.name} crashed on attempt {attempt}: {last_error}",
                    kind="error",
                    tags=["crash", agent.name],
                    weight=2.0,
                )
                self.memory.self_heal()
        return AgentResult(agent=agent.name, success=False, output="Task failed after retries", errors=[last_error])

    def self_audit(self) -> dict[str, object]:
        memory_audit = self.memory.audit()
        if not memory_audit["healthy"]:
            memory_audit = self.memory.self_heal()
        return {
            "healthy": bool(memory_audit["healthy"]),
            "agents": sorted(self.agents),
            "memory": memory_audit,
            "vsa": self.vsa.audit(),
        }
