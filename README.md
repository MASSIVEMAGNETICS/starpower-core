# Starpower Core

**The cognitive operating layer for MASSIVEMAGNETICS + B Heard Network.**

Starpower Core is a local-first, auditable, self-healing multi-agent runtime built around Vector Symbolic Architecture ideas, REM-style memory, deterministic decision collapse, and adapter-based integration.

## Mission

- **MASSIVEMAGNETICS** = body: physical systems, engineering, tools, creative substrate.
- **B Heard Network** = voice: signal, media, distribution, reach, memory connectors.
- **Starpower Core** = mind: vector-symbolic cognition, agent orchestration, memory, audit, and self-repair.

This repo is intentionally clean: no secrets, no notebooks with embedded keys, no local junk dumps.

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e .[dev]
starpower "map Massive Magnetics and B Heard into one cognitive runtime"
```

## Run tests

```bash
pytest -q
ruff check .
mypy starpower_core
```

## Core modules

```text
starpower_core/
├── vsa.py          # Hyperdimensional vector binding/bundling/cleanup
├── memory.py       # Checksum-protected REM memory graph
├── agents.py       # Agent contracts + orchestrator + decision collapse
├── audit.py        # Self-audit and repair reporting
├── cli.py          # Command-line interface
└── adapters/       # MASSIVEMAGNETICS + B Heard integration adapters
```

## Security

Do not commit API keys. Use environment variables or `.env` files excluded by `.gitignore`.

If a key was ever pasted into terminal history or chat, revoke it immediately and generate a new one.
