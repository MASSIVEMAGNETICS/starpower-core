from __future__ import annotations

import json
from pathlib import Path

import pytest

from starpower_core.continuity import (
    META_LEDGER_ID,
    VERIFIED_PROGRESS_LEDGER_ID,
    InvalidProgressTransition,
    LedgerIntegrityError,
    LedgerStore,
    Maturity,
    MetaLedger,
    VerifiedProgressLedger,
    initialize_continuity_fabric,
)


def test_initialize_registers_verified_progress_ledger(tmp_path: Path) -> None:
    stores = initialize_continuity_fabric(tmp_path)
    assert stores["meta"].verify()
    assert stores["progress"].verify()

    registry = MetaLedger(stores["meta"]).registry()
    assert META_LEDGER_ID in registry
    assert VERIFIED_PROGRESS_LEDGER_ID in registry
    assert registry[VERIFIED_PROGRESS_LEDGER_ID]["ledger_type"] == "VERIFIED_PROGRESS"


def test_hash_chain_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store = LedgerStore(path, "test.ledger.v1")
    store.append("GENESIS", "thing", {"value": 1}, timestamp="2026-08-26T00:00:00Z")
    store.append("CHANGED", "thing", {"value": 2}, timestamp="2026-08-26T00:01:00Z")

    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["payload"]["value"] = 999
    lines[0] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LedgerIntegrityError):
        LedgerStore(path, "test.ledger.v1")


def test_verified_progress_is_priority_and_evidence_weighted(tmp_path: Path) -> None:
    stores = initialize_continuity_fabric(tmp_path)
    progress = VerifiedProgressLedger(stores["progress"])

    progress.transition(
        work_id="api-deploy",
        subject="api.iambandobandz.com deployment",
        priority=10,
        to_state=Maturity.IMPLEMENTED,
        evidence_quality=1.0,
        evidence_refs=("commit:abc",),
    )
    progress.transition(
        work_id="site-art",
        subject="storefront artwork repair",
        priority=5,
        to_state=Maturity.LIVE_VERIFIED,
        evidence_quality=1.0,
        evidence_refs=("probe:xyz",),
    )

    expected = 100 * ((10 * 0.25) + (5 * 0.90)) / 15
    assert progress.score() == round(expected, 2)


def test_backward_progress_requires_explicit_regression(tmp_path: Path) -> None:
    stores = initialize_continuity_fabric(tmp_path)
    progress = VerifiedProgressLedger(stores["progress"])
    progress.transition(
        work_id="victor",
        subject="Victor bounded runtime",
        priority=10,
        to_state=Maturity.CI_VERIFIED,
        evidence_quality=0.6,
    )

    with pytest.raises(InvalidProgressTransition):
        progress.transition(
            work_id="victor",
            subject="Victor bounded runtime",
            priority=10,
            to_state=Maturity.IMPLEMENTED,
            evidence_quality=0.5,
        )

    progress.transition(
        work_id="victor",
        subject="Victor bounded runtime",
        priority=10,
        to_state=Maturity.IMPLEMENTED,
        evidence_quality=0.5,
        regression=True,
        evidence_refs=("incident:regression-1",),
    )
    assert progress.current_items()["victor"].maturity is Maturity.IMPLEMENTED
