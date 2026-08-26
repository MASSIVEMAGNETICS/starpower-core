from __future__ import annotations

import copy

import pytest

from starpower_core.continuity import MetaLedger
from starpower_core.development import (
    CONFLICT_LEDGER_ID,
    DEVELOPMENT_LEDGER_ID,
    InvalidDevelopmentSnapshot,
    build_development_snapshot,
    classify_origin,
    detect_conflicts,
    initialize_development_fabric,
    validate_development_snapshot,
)


def _pr(
    number: int,
    *,
    title: str,
    files: list[str],
    repo: str = "MASSIVEMAGNETICS/victor_empire",
    draft: bool = True,
    mergeable: bool | None = False,
) -> dict[str, object]:
    return {
        "repository": repo,
        "number": number,
        "title": title,
        "state": "open",
        "draft": draft,
        "mergeable": mergeable,
        "mergeable_state": "dirty" if mergeable is False else "unknown",
        "created_at": "2026-08-25T00:00:00Z",
        "updated_at": "2026-08-26T00:00:00Z",
        "author_login": "MASSIVEMAGNETICS",
        "author_type": "User",
        "origin_type": "HUMAN_OR_ACCOUNT_OWNER",
        "author_association": "OWNER",
        "base_ref": "main",
        "base_sha": "base",
        "head_ref": f"agent/pr-{number}",
        "head_sha": f"head-{number}",
        "changed_files": files,
        "additions": 100,
        "deletions": 10,
        "semantic_tokens": sorted(set(title.lower().replace(":", "").split())),
        "evidence_status": "complete",
        "error": None,
    }


def test_known_kernel_pair_is_canonicality_collision_candidate() -> None:
    snapshot = build_development_snapshot(
        [
            _pr(
                3,
                title="Victor ABI transactional kernel",
                files=[
                    "README.md",
                    "victor/kernel.py",
                    "victor/chronos.py",
                    "victor/persistence.py",
                ],
            ),
            _pr(
                4,
                title="Victor kernel M0 M1 control plane",
                files=[
                    "README.md",
                    "victor_kernel/kernel.py",
                    "victor_kernel/storage.py",
                    "victor_kernel/dispatcher.py",
                ],
            ),
        ],
        source_ref="fixture",
    )
    conflicts = detect_conflicts(snapshot)

    assert conflicts["conflict_count"] == 1
    conflict = conflicts["conflicts"][0]
    assert conflict["left_pr"] == 3
    assert conflict["right_pr"] == 4
    assert "STRUCTURAL_BASENAME_COLLISION" in conflict["kinds"]
    assert "CANONICALITY_COLLISION_CANDIDATE" in conflict["kinds"]
    assert "kernel.py" in conflict["shared_basenames"]
    assert conflict["action"] == "BLOCK_PENDING_REVIEW"
    assert conflict["risk_score"] == 95


def test_unrelated_prs_do_not_create_conflict() -> None:
    snapshot = build_development_snapshot(
        [
            _pr(1, title="Website copy update", files=["site/about.html"]),
            _pr(2, title="Audio mastering utility", files=["audio/master.py"]),
        ]
    )

    assert detect_conflicts(snapshot)["conflict_count"] == 0


def test_tampered_development_receipt_fails_closed() -> None:
    snapshot = build_development_snapshot(
        [_pr(1, title="Kernel authority", files=["victor/kernel.py"])]
    )
    tampered = copy.deepcopy(snapshot)
    tampered["pull_requests"][0]["title"] = "tampered"  # type: ignore[index]

    with pytest.raises(InvalidDevelopmentSnapshot, match="receipt mismatch"):
        validate_development_snapshot(tampered)


def test_origin_classification() -> None:
    assert classify_origin("dependabot[bot]", "Bot") == "DEPENDABOT"
    assert classify_origin("copilot-swe-agent[bot]", "Bot") == "CODING_AGENT_AUTHORED"
    assert classify_origin("MASSIVEMAGNETICS", "User") == "HUMAN_OR_ACCOUNT_OWNER"


def test_initialization_registers_development_and_conflict_ledgers(tmp_path) -> None:
    stores = initialize_development_fabric(tmp_path)
    registry = MetaLedger(stores["meta"]).registry()

    assert DEVELOPMENT_LEDGER_ID in registry
    assert CONFLICT_LEDGER_ID in registry
    assert all(store.verify() for store in stores.values())
