from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents import Orchestrator


@dataclass(slots=True)
class AuditReport:
    healthy: bool
    findings: dict[str, Any]

    def summary(self) -> str:
        status = "healthy" if self.healthy else "needs repair"
        return f"Starpower audit: {status} | agents={self.findings.get('agents', [])}"


def run_audit(orchestrator: Orchestrator) -> AuditReport:
    findings = orchestrator.self_audit()
    return AuditReport(healthy=bool(findings.get("healthy")), findings=findings)
