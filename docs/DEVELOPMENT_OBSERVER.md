# MASSIVEMAGNETICS Development + Conflict Observer

## Purpose

The repository estate answers **what repositories exist and what structural evidence they expose**. The Development Ledger answers **what changes are currently in flight**, while the Conflict Ledger records deterministic collision candidates derived from those changes.

The observer is deliberately evidence-bounded. A semantic heuristic may justify review or a temporary block, but it is not promoted into a factual claim of duplicate identity, malicious intent, or architectural truth.

## Ledgers

- `massive.development.v1` — open pull-request inventory and change evidence.
- `massive.conflict.v1` — conflict candidates derived from the verified development snapshot.

Both are registered in `massive.meta.v1` and persisted with the same append-only SHA-256 chain used by the Repository Estate Ledger.

## Development snapshot

For each open pull request the scanner records, when available:

- exact repository and PR number;
- title and draft state;
- mergeable / mergeable-state observation;
- base branch and base SHA;
- head branch and head SHA;
- changed filenames;
- additions and deletions;
- author login/type and bounded provenance class;
- semantic tokens derived from title/body;
- evidence status and any retrieval error.

If a PR can be discovered but its full metadata/files cannot be retrieved, it remains in the snapshot with `evidence_status: unknown`. It is not silently dropped.

## Provenance classes

The first observer uses bounded actor evidence only:

- `DEPENDABOT`
- `CODING_AGENT_AUTHORED`
- `AUTOMATION_BOT`
- `HUMAN_OR_ACCOUNT_OWNER`

This identifies the observable GitHub actor class. It does not infer who conceived, supervised, approved, or authored the underlying idea.

## Conflict detector

Only PRs targeting the same repository/base are compared in v1.

Signals include:

- `FILE_OVERLAP` — same non-generic path modified by both PRs;
- `STRUCTURAL_BASENAME_COLLISION` — meaningful files with the same basename appear under different roots, such as two independent `kernel.py` implementations;
- `SEMANTIC_COLLISION_CANDIDATE` — multiple meaningful title/body concepts overlap;
- `CANONICALITY_COLLISION_CANDIDATE` — a protected architecture concept such as `kernel`, `identity`, `authority`, `constitution`, `continuity`, `production`, or `security` overlaps with structural/semantic evidence.

`CANONICALITY_COLLISION_CANDIDATE` produces `BLOCK_PENDING_REVIEW` in the derived conflict state. It does not auto-merge, auto-close, auto-rebase, or decide which PR is canonical.

The acceptance fixture explicitly models the live `victor_empire#3` / `#4` shape: separate package roots that both implement a Victor kernel and each contain a `kernel.py`. A detector that only looks for exact path overlap would miss this class of collision.

## Persistence and cadence

The Development and Conflict ledgers run inside the existing Estate Observer reconciliation loop every six hours and after relevant changes land on `main`.

The workflow restores the prior `estate-state` branch, advances all ledgers, verifies every hash chain, rebuilds derived state, and commits the new state back to `estate-state`.

Derived files:

- `derived/OPEN_PULL_REQUESTS.json`
- `derived/PR_CONFLICTS.json`

Source observation retained:

- `estate/source/development.json`

## Boundary

This v1 does not yet:

- ingest every historical closed/merged PR;
- ingest every branch and unsubmitted commit;
- inspect symbols/ASTs across languages;
- prove semantic equivalence;
- automatically resolve a conflict;
- make canonicality decisions;
- enforce a required GitHub status check on every repository;
- verify production deployment state.

Those are later extensions. The v1 objective is narrower and testable: **persist the open change surface and rediscover meaningful collision candidates before merge.**
