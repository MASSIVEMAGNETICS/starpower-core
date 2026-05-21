from __future__ import annotations

import argparse
import json

from rich.console import Console

from .agents import Orchestrator
from .audit import run_audit

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Starpower Core command line interface")
    parser.add_argument("task", nargs="*", help="Task text to process")
    parser.add_argument("--audit", action="store_true", help="Run self-audit after execution")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    orchestrator = Orchestrator()
    task = " ".join(args.task).strip() or "summarize the current runtime status"
    result = orchestrator.run(task)
    audit = run_audit(orchestrator) if args.audit else None

    payload = {
        "result": {
            "agent": result.agent,
            "success": result.success,
            "output": result.output,
            "confidence": result.confidence,
            "errors": result.errors,
        },
        "audit": audit.findings if audit else None,
    }

    if args.json:
        console.print(json.dumps(payload, indent=2))
        return

    console.print(f"[bold]Agent:[/bold] {result.agent}")
    console.print(f"[bold]Success:[/bold] {result.success}")
    console.print(f"[bold]Output:[/bold] {result.output}")
    if audit:
        console.print(audit.summary())


if __name__ == "__main__":
    main()
