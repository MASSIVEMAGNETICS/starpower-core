# MASSIVEMAGNETICS Continuity Fabric v0.1

Status: experimental implementation; not yet canonical.

## Purpose

The Continuity Fabric gives MASSIVEMAGNETICS an append-only, tamper-evident history for important state transitions and a Meta Ledger that records the existence and relationships of domain ledgers.

The governing invariant is:

> Every important state must be explainable by its history and supported by evidence appropriate to the claim.

A graph answers **what exists now**. A ledger answers **how it became that way**. Derived graphs and dashboards are reconstructible; ledger history is not silently rewritten.

## Initial ledger family

The v0.1 implementation boots two ledgers:

- `massive.meta.v1` — ledger-of-ledgers registry.
- `massive.verified-progress.v1` — evidence-weighted progress transitions.

The Meta Ledger already declares the intended upstream/downstream relationship for future ledgers:

- `massive.development.v1`
- `massive.production.v1`
- `massive.evidence.v1`
- `massive.scorecard.v1`

Those identifiers are reservations, not claims that the future ledgers are already implemented.

## Progress maturity states

The Verified Progress Ledger uses explicit evidence gates:

1. `DISCOVERED` — 2%
2. `DEFINED` — 5%
3. `SPECIFIED` — 10%
4. `IMPLEMENTED` — 25%
5. `LOCALLY_TESTED` — 35%
6. `CI_VERIFIED` — 50%
7. `MERGED` — 60%
8. `DEPLOYED` — 75%
9. `LIVE_VERIFIED` — 90%
10. `OUTCOME_VERIFIED` — 100%

A transition receives an evidence-quality value in `[0, 1]` and a positive priority weight. The current progress score is:

`100 * sum(priority * maturity_credit * evidence_quality) / sum(priority)`

This prevents raw code volume, repo count, or unverified claims from masquerading as completion.

## Regression rule

Progress may not silently move backward. A backward transition must explicitly set `regression=True` and should reference an incident, failed probe, or other evidence explaining the downgrade.

History is therefore preserved as:

`CLAIMED -> VERIFIED -> REGRESSION OBSERVED -> CORRECTED`

rather than rewriting the earlier event.

## Integrity model

Each JSONL entry contains:

- ledger id
- sequence number
- event type
- subject
- payload
- timestamp
- previous-entry SHA-256
- current-entry SHA-256

The current hash is computed from a canonical JSON representation of all fields except the current hash itself. Loading a ledger verifies sequence continuity, previous-hash continuity, ledger identity, and every entry hash. Mutation of a historical row fails closed with `LedgerIntegrityError`.

This is tamper-evident integrity, not a claim of externally notarized immutability. Future work may anchor ledger heads to independent storage or signatures.

## CLI

After installing the project:

```bash
starpower-ledger init --root .starpower/ledgers
starpower-ledger verify --root .starpower/ledgers
starpower-ledger progress --root .starpower/ledgers
```

The same interface is available as:

```bash
python -m starpower_core.continuity ...
```

## Required invariants

1. Append; never rewrite historical events as a correction mechanism.
2. Derived state is disposable; causal history is not.
3. Unknown/unverified state must not be promoted by assumption.
4. Backward progress requires an explicit regression event.
5. A Meta Ledger registration does not itself prove the registered system works.
6. A maturity label does not outrank its attached evidence.
7. Consequential future ledgers must retain authority/provenance references.

## Planned ledger families

The architecture is intentionally federated rather than one giant file.

### MASSIVEMAGNETICS engineering

- Verified Progress Ledger
- Development Ledger
- Production Ledger
- Evidence Ledger
- Canonicality Ledger
- Repository Estate Ledger
- Dependency Ledger
- Conflict Ledger
- Incident Ledger
- Decision Ledger
- Research Ledger

### Commercial

- Offer Ledger
- Acquisition Ledger
- Customer Outcome Ledger
- Revenue Ledger

### iambandobandz

- Catalog Ledger
- Creative Provenance Ledger
- Release Ledger
- Signal Ledger

### Victor

- Identity Ledger
- Constitution Ledger
- Authority Ledger
- Capability Ledger
- Choice Ledger
- TRACE Ledger
- Chronos Ledger
- Experience Ledger
- Memory Transition Ledger
- Learning Ledger
- Outcome Ledger

The Meta Ledger should register each ledger, schema version, domain, current status, and dependency edges without copying the ledger's complete event history.

## Relationship to the meta graph

The intended architecture is:

`Domain ledgers -> deterministic reducers -> domain graphs -> graph of graphs`

and in parallel:

`Domain ledgers -> Meta Ledger -> ledger dependency graph`

Together they answer:

- what exists
- what changed
- when it changed
- why it changed
- who/what was authorized to change it
- what evidence supports the state
- what remains unknown

## v0.1 acceptance criteria

The initial increment is accepted only if CI proves:

- genesis initializes both ledgers;
- the Meta Ledger registers Verified Progress;
- an altered historical entry is rejected;
- priority/evidence weighting produces the deterministic expected score;
- backward progress is rejected unless explicitly recorded as regression;
- CLI initialization and verification succeed on a fresh directory.

Passing these tests proves only the listed ledger mechanics. It does not prove the broader SCF-2 estate observer, autonomous remediation, complete repository sensing, or production deployment of the Continuity Fabric.
