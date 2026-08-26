# MASSIVEMAGNETICS Estate Observer

## Purpose

The Estate Observer turns the existing Shared Completion Fabric repository scan into durable, append-only estate history.

It does not treat repository count, README claims, or generated code volume as proof of capability. It records what the scanner actually observed, preserves uncertainty, derives review candidates, and makes every accepted snapshot replayable through the Continuity Fabric.

## Ledger

Canonical implementation identifier for this increment:

`massive.repository-estate.v1`

The Repository Estate Ledger is registered by `massive.meta.v1` and is implemented as a SHA-256 chained JSONL ledger through the same `LedgerStore` used by the Continuity Fabric.

Each accepted event is an `ESTATE_SNAPSHOT_INGESTED` event whose payload contains:

- source SCF schema and receipt;
- source code/ref identifier;
- repository count;
- evaluated repository count;
- partial/unknown repository count;
- structural completion score;
- normalized per-repository structural signals;
- explicit unknown repositories;
- repositories not observed since the previous snapshot;
- deterministic lineage candidates;
- snapshot SHA-256.

## Epistemic states

The observer intentionally distinguishes these states.

### PRESENT

The current scan observed the repository and the scanner successfully evaluated the structural tree evidence available to it.

### PARTIAL / UNKNOWN

The repository was observed, but the scanner could not fully evaluate its tree or the upstream result was truncated/unavailable. This state is preserved. It is never silently converted to healthy, empty, deleted, or complete.

### NOT OBSERVED SINCE PREVIOUS

A repository existed in the preceding estate snapshot but is absent from the new observation set.

This is **not** a deletion claim. A missing observation can result from visibility, authentication, API, rename, archival, deletion, or scanner changes. Stronger classification requires stronger evidence.

## Lineage detection

The first observer uses deterministic repository-name evidence only.

It emits:

- `NORMALIZED_NAME_COLLISION` when multiple slugs collapse to the same lowercase alphanumeric form;
- `NEAR_NAME_COLLISION` when normalized names differ by at most one edit under the bounded detector.

Every such result has action `REVIEW_LINEAGE`.

A lineage candidate is **not** proof of duplicate identity and does not authorize archive, delete, rename, merge, or canonical promotion.

Examples such as `dev-ville` and `devv-ville` must remain distinct until independent lineage evidence resolves their relationship.

## Derived state

The append-only ledger is causal history. The following files are disposable projections and may be rebuilt from the current snapshot:

- `derived/ESTATE_SNAPSHOT.json`
- `derived/REPOSITORY_REGISTRY.json`
- `derived/LINEAGE_CANDIDATES.json`
- `derived/UNKNOWN_STATE.json`

Derived state must never outrank the ledger that produced it.

## Persistence

GitHub Actions runners are ephemeral, so workflow artifacts alone are not accepted as continuity.

The scheduled Estate Observer uses a dedicated `estate-state` branch as durable state storage:

1. restore `estate/ledgers` from the previous `estate-state` branch when present;
2. run a fresh Shared Completion Fabric scan against `MASSIVEMAGNETICS`;
3. verify the SCF receipt before ingestion;
4. append a new estate event only when the source receipt changed;
5. verify every continuity hash chain;
6. rebuild derived state;
7. commit the advanced ledger state back to `estate-state`;
8. retain the reconciliation evidence as a workflow artifact.

`main` remains the executable code line. `estate-state` is state/provenance, not an alternate implementation branch.

## Schedule

The reconciliation workflow runs every six hours, on explicit dispatch, and after relevant changes land on `main`.

Pull requests run only the read-only quality/acceptance suite. They do not mutate durable estate state.

This implements the rule:

> Events provide speed; reconciliation provides correctness.

The six-hour reconciliation is the correctness substrate. GitHub event ingestion can be added later without replacing it.

## Current boundary

This increment establishes repository-level structural sensing and durable history. It does **not** yet claim:

- complete semantic analysis of every file in every repository;
- stable rename tracking by GitHub numeric repository ID;
- branch/commit/PR history ingestion into a Development Ledger;
- required-check enforcement for predicted conflicts;
- deployment truth or a Production Ledger;
- automatic canonicality decisions;
- automatic repository deletion/archive/rename;
- autonomous remediation;
- legacy branch-protection visibility when GitHub does not expose it.

Those remain future ledger organs or explicit UNKNOWN states.

## Next ledger organs

The Repository Estate Ledger is designed to feed, but does not fabricate, the following future ledgers:

- `massive.development.v1` — branches, commits, pull requests, reviews, CI, merges;
- `massive.conflict.v1` — textual, structural, semantic, dependency, canonicality, and deployment collision history;
- `massive.production.v1` — deployed artifact, commit, environment, verification probe, rollback target;
- `massive.evidence.v1` — evidence objects and claim-to-evidence relationships.

The Meta Ledger remains the ledger of those ledgers.

## Acceptance criteria

The observer is acceptable only if automated tests prove that it:

1. verifies the upstream SCF receipt before accepting a snapshot;
2. registers itself in the Meta Ledger;
3. preserves UNKNOWN/PARTIAL state;
4. detects known normalized and near-name lineage candidates without auto-deduplication;
5. treats a missing repository as `not_observed_since_previous`, not deleted;
6. is idempotent for an unchanged source receipt;
7. persists through the existing tamper-evident hash ledger;
8. can rebuild its derived outputs after restart.
