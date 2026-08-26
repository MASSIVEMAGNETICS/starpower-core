from __future__ import annotations

import json

import pytest

from starpower_core.completion import deterministic_receipt
from starpower_core.continuity import MetaLedger
from starpower_core.estate import (
    InvalidEstateSnapshot,
    RepositoryEstateLedger,
    build_estate_snapshot,
    initialize_estate_fabric,
    write_derived_outputs,
)


def _repo(name: str, *, status: str = "complete", score: int = 75) -> dict[str, object]:
    return {
        "name": f"MASSIVEMAGNETICS/{name}",
        "source": True,
        "manifest": True,
        "readme": True,
        "tests": score >= 75,
        "ci": score >= 75,
        "license": True,
        "gitignore": True,
        "release": False,
        "completion_score": score,
        "gaps": ["release"] if status == "complete" else [],
        "evidence_status": status,
        "error": None if status == "complete" else "tree unavailable",
    }


def _portfolio(*repos: dict[str, object]) -> dict[str, object]:
    complete = [repo for repo in repos if repo["evidence_status"] == "complete"]
    average = (
        round(sum(int(repo["completion_score"]) for repo in complete) / len(complete), 2)
        if complete
        else 0.0
    )
    payload: dict[str, object] = {
        "schema_version": "scf-1",
        "repository_count": len(repos),
        "evaluated_repository_count": len(complete),
        "partial_or_unknown_repository_count": len(repos) - len(complete),
        "average_completion_score": average,
        "repositories": list(repos),
        "shared_bottlenecks": [],
    }
    payload["receipt_sha256"] = deterministic_receipt(payload)
    payload["generated_at"] = "2026-08-26T00:00:00+00:00"
    return payload


def test_snapshot_preserves_unknown_and_detects_lineage_candidates() -> None:
    portfolio = _portfolio(
        _repo("dev-ville"),
        _repo("devv-ville"),
        _repo("Image-Fusion"),
        _repo("imagefusion"),
        _repo("empty-repo", status="unknown", score=0),
    )
    snapshot = build_estate_snapshot(portfolio, source_ref="test")

    assert snapshot["repository_count"] == 5
    assert snapshot["partial_or_unknown_repository_count"] == 1
    assert snapshot["unknown_repositories"] == ["MASSIVEMAGNETICS/empty-repo"]
    kinds = {candidate["kind"] for candidate in snapshot["lineage_candidates"]}
    assert "NORMALIZED_NAME_COLLISION" in kinds
    assert "NEAR_NAME_COLLISION" in kinds
    assert any(
        set(candidate["repositories"])
        == {"MASSIVEMAGNETICS/dev-ville", "MASSIVEMAGNETICS/devv-ville"}
        for candidate in snapshot["lineage_candidates"]
    )


def test_initialization_registers_estate_ledger_in_meta(tmp_path) -> None:
    stores = initialize_estate_fabric(tmp_path)
    registry = MetaLedger(stores["meta"]).registry()

    assert "massive.repository-estate.v1" in registry
    assert stores["estate"].verify()


def test_same_source_receipt_is_idempotent(tmp_path) -> None:
    stores = initialize_estate_fabric(tmp_path)
    estate = RepositoryEstateLedger(stores["estate"])
    portfolio = _portfolio(_repo("one"), _repo("two"))

    first = estate.ingest(portfolio, source_ref="first")
    second = estate.ingest(portfolio, source_ref="second")

    assert first == second
    assert len(stores["estate"].ledger.entries) == 2


def test_missing_repository_is_not_claimed_deleted(tmp_path) -> None:
    stores = initialize_estate_fabric(tmp_path)
    estate = RepositoryEstateLedger(stores["estate"])
    estate.ingest(_portfolio(_repo("one"), _repo("two")), source_ref="first")
    snapshot = estate.ingest(_portfolio(_repo("one")), source_ref="second")

    assert snapshot["not_observed_since_previous"] == ["MASSIVEMAGNETICS/two"]


def test_tampered_portfolio_receipt_fails_closed() -> None:
    portfolio = _portfolio(_repo("one"))
    portfolio["repositories"][0]["completion_score"] = 5  # type: ignore[index]

    with pytest.raises(InvalidEstateSnapshot, match="receipt mismatch"):
        build_estate_snapshot(portfolio)


def test_derived_outputs_and_hash_chain_reload(tmp_path) -> None:
    stores = initialize_estate_fabric(tmp_path)
    estate = RepositoryEstateLedger(stores["estate"])
    snapshot = estate.ingest(
        _portfolio(_repo("Image-Fusion"), _repo("imagefusion")),
        source_ref="acceptance",
    )
    write_derived_outputs(tmp_path, snapshot)

    expected = {
        "ESTATE_SNAPSHOT.json",
        "REPOSITORY_REGISTRY.json",
        "LINEAGE_CANDIDATES.json",
        "UNKNOWN_STATE.json",
    }
    assert {path.name for path in (tmp_path / "derived").iterdir()} == expected
    stored = json.loads((tmp_path / "derived" / "ESTATE_SNAPSHOT.json").read_text())
    assert stored["snapshot_sha256"] == snapshot["snapshot_sha256"]

    reloaded = initialize_estate_fabric(tmp_path)
    assert all(store.verify() for store in reloaded.values())
    assert RepositoryEstateLedger(reloaded["estate"]).current_snapshot() == snapshot
