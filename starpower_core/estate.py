from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .completion import deterministic_receipt
from .continuity import (
    LedgerRegistration,
    LedgerStore,
    MetaLedger,
    initialize_continuity_fabric,
)

REPOSITORY_ESTATE_LEDGER_ID = "massive.repository-estate.v1"
ESTATE_SCHEMA = "repository-estate-ledger/v1"
SNAPSHOT_SCHEMA = "estate-snapshot/v1"


class InvalidEstateSnapshot(ValueError):
    """Raised when an upstream portfolio cannot support a verified estate snapshot."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) <= 1
    if len(left) > len(right):
        left, right = right, left
    i = 0
    j = 0
    differences = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        differences += 1
        if differences > 1:
            return False
        j += 1
    return True


def _repo_slug(full_name: str) -> str:
    return full_name.split("/", 1)[-1]


def lineage_candidates(full_names: Sequence[str]) -> list[dict[str, Any]]:
    """Find deterministic naming collisions without asserting duplicate identity."""
    exact_groups: dict[str, list[str]] = defaultdict(list)
    normalized: dict[str, str] = {}
    for full_name in sorted(set(full_names), key=str.lower):
        slug = _repo_slug(full_name)
        key = normalize_slug(slug)
        normalized[full_name] = key
        exact_groups[key].append(full_name)

    candidates: list[dict[str, Any]] = []
    exact_pairs: set[tuple[str, str]] = set()
    for key, names in sorted(exact_groups.items()):
        if key and len(names) > 1:
            ordered = sorted(names, key=str.lower)
            candidates.append(
                {
                    "kind": "NORMALIZED_NAME_COLLISION",
                    "normalized": key,
                    "repositories": ordered,
                    "action": "REVIEW_LINEAGE",
                }
            )
            for index, left in enumerate(ordered):
                for right in ordered[index + 1 :]:
                    exact_pairs.add(tuple(sorted((left, right))))

    names = sorted(normalized, key=str.lower)
    for index, left in enumerate(names):
        left_key = normalized[left]
        if len(left_key) < 5:
            continue
        for right in names[index + 1 :]:
            pair = tuple(sorted((left, right)))
            if pair in exact_pairs:
                continue
            right_key = normalized[right]
            if len(right_key) < 5 or left_key == right_key:
                continue
            if _distance_at_most_one(left_key, right_key):
                candidates.append(
                    {
                        "kind": "NEAR_NAME_COLLISION",
                        "normalized": [left_key, right_key],
                        "repositories": [left, right],
                        "action": "REVIEW_LINEAGE",
                    }
                )
    return sorted(
        candidates,
        key=lambda item: (
            str(item["kind"]),
            tuple(str(name).lower() for name in item["repositories"]),
        ),
    )


def _validate_portfolio(payload: Mapping[str, Any]) -> None:
    if not str(payload.get("schema_version", "")).startswith("scf-1"):
        raise InvalidEstateSnapshot("unsupported or missing SCF schema_version")
    repositories = payload.get("repositories")
    if not isinstance(repositories, list):
        raise InvalidEstateSnapshot("portfolio repositories must be a list")
    claimed_receipt = payload.get("receipt_sha256")
    actual_receipt = deterministic_receipt(dict(payload))
    if claimed_receipt != actual_receipt:
        raise InvalidEstateSnapshot("portfolio receipt mismatch")
    if int(payload.get("repository_count", -1)) != len(repositories):
        raise InvalidEstateSnapshot("repository_count does not match repositories")


def _normalized_repository(raw: Mapping[str, Any]) -> dict[str, Any]:
    full_name = str(raw.get("name", "")).strip()
    if not full_name or "/" not in full_name:
        raise InvalidEstateSnapshot(f"invalid repository name: {full_name!r}")
    evidence_status = str(raw.get("evidence_status", "unknown"))
    if evidence_status not in {"complete", "partial", "unknown"}:
        evidence_status = "unknown"
    return {
        "full_name": full_name,
        "normalized_slug": normalize_slug(_repo_slug(full_name)),
        "evidence_status": evidence_status,
        "completion_score": int(raw.get("completion_score", 0)),
        "signals": {
            key: bool(raw.get(key, False))
            for key in (
                "source",
                "manifest",
                "readme",
                "tests",
                "ci",
                "license",
                "gitignore",
                "release",
            )
        },
        "gaps": sorted(str(gap) for gap in raw.get("gaps", []) if isinstance(gap, str)),
        "error": str(raw["error"]) if raw.get("error") else None,
    }


def build_estate_snapshot(
    portfolio: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
    source_ref: str | None = None,
) -> dict[str, Any]:
    _validate_portfolio(portfolio)
    raw_repositories = portfolio["repositories"]
    assert isinstance(raw_repositories, list)
    repositories = sorted(
        (_normalized_repository(raw) for raw in raw_repositories if isinstance(raw, dict)),
        key=lambda item: str(item["full_name"]).lower(),
    )
    if len(repositories) != len(raw_repositories):
        raise InvalidEstateSnapshot("portfolio contained a non-object repository entry")

    names = [str(item["full_name"]) for item in repositories]
    if len(names) != len(set(names)):
        raise InvalidEstateSnapshot("portfolio contains duplicate repository full names")

    previous_names = {
        str(item.get("full_name"))
        for item in (previous or {}).get("repositories", [])
        if isinstance(item, dict) and item.get("full_name")
    }
    observed_names = set(names)
    not_observed = sorted(previous_names - observed_names, key=str.lower)
    unknown = [
        str(item["full_name"])
        for item in repositories
        if item["evidence_status"] != "complete"
    ]

    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA,
        "source_schema": portfolio["schema_version"],
        "source_receipt_sha256": portfolio["receipt_sha256"],
        "source_ref": source_ref or "",
        "repository_count": len(repositories),
        "evaluated_repository_count": sum(
            item["evidence_status"] == "complete" for item in repositories
        ),
        "partial_or_unknown_repository_count": len(unknown),
        "average_completion_score": float(portfolio.get("average_completion_score", 0.0)),
        "repositories": repositories,
        "unknown_repositories": sorted(unknown, key=str.lower),
        "not_observed_since_previous": not_observed,
        "lineage_candidates": lineage_candidates(names),
    }
    snapshot["snapshot_sha256"] = _sha256(snapshot)
    return snapshot


class RepositoryEstateLedger:
    def __init__(self, store: LedgerStore) -> None:
        self.store = store

    def snapshots(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(entry.payload)
            for entry in self.store.ledger.entries
            if entry.event_type == "ESTATE_SNAPSHOT_INGESTED"
        )

    def current_snapshot(self) -> dict[str, Any] | None:
        snapshots = self.snapshots()
        return snapshots[-1] if snapshots else None

    def ingest(
        self,
        portfolio: Mapping[str, Any],
        *,
        source_ref: str | None = None,
    ) -> dict[str, Any]:
        current = self.current_snapshot()
        snapshot = build_estate_snapshot(portfolio, previous=current, source_ref=source_ref)
        if current and current.get("source_receipt_sha256") == snapshot["source_receipt_sha256"]:
            return current
        self.store.append(
            "ESTATE_SNAPSHOT_INGESTED",
            "MASSIVEMAGNETICS",
            snapshot,
        )
        return snapshot

    def report(self) -> dict[str, Any]:
        snapshot = self.current_snapshot()
        if snapshot is None:
            return {
                "ledger_id": self.store.ledger_id,
                "head_hash": self.store.ledger.head_hash,
                "snapshot": None,
            }
        return {
            "ledger_id": self.store.ledger_id,
            "head_hash": self.store.ledger.head_hash,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "source_receipt_sha256": snapshot["source_receipt_sha256"],
            "repository_count": snapshot["repository_count"],
            "evaluated_repository_count": snapshot["evaluated_repository_count"],
            "partial_or_unknown_repository_count": snapshot[
                "partial_or_unknown_repository_count"
            ],
            "average_completion_score": snapshot["average_completion_score"],
            "lineage_candidate_count": len(snapshot["lineage_candidates"]),
            "not_observed_since_previous_count": len(
                snapshot["not_observed_since_previous"]
            ),
        }


def initialize_estate_fabric(root: Path | str) -> dict[str, LedgerStore]:
    root_path = Path(root)
    stores = initialize_continuity_fabric(root_path)
    estate_store = LedgerStore(
        root_path / "repository-estate-ledger.jsonl",
        REPOSITORY_ESTATE_LEDGER_ID,
    )
    if not estate_store.ledger.entries:
        estate_store.append(
            "LEDGER_GENESIS",
            REPOSITORY_ESTATE_LEDGER_ID,
            {
                "domain": "MASSIVEMAGNETICS",
                "ledger_type": "REPOSITORY_ESTATE",
                "schema": ESTATE_SCHEMA,
                "path": str(estate_store.path),
                "status": "ACTIVE",
            },
        )

    meta = MetaLedger(stores["meta"])
    if REPOSITORY_ESTATE_LEDGER_ID not in meta.registry():
        meta.register(
            LedgerRegistration(
                ledger_id=REPOSITORY_ESTATE_LEDGER_ID,
                domain="MASSIVEMAGNETICS",
                ledger_type="REPOSITORY_ESTATE",
                schema=ESTATE_SCHEMA,
                path=str(estate_store.path),
                status="ACTIVE",
                feeds=(
                    "massive.verified-progress.v1",
                    "massive.development.v1",
                    "massive.conflict.v1",
                ),
            )
        )
    stores["estate"] = estate_store
    return stores


def write_derived_outputs(root: Path | str, snapshot: Mapping[str, Any]) -> None:
    root_path = Path(root)
    derived = root_path / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {
        "ESTATE_SNAPSHOT.json": snapshot,
        "REPOSITORY_REGISTRY.json": {
            "schema_version": "repository-registry/v1",
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "repositories": snapshot["repositories"],
        },
        "LINEAGE_CANDIDATES.json": {
            "schema_version": "lineage-candidates/v1",
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "candidates": snapshot["lineage_candidates"],
        },
        "UNKNOWN_STATE.json": {
            "schema_version": "unknown-state/v1",
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "unknown_repositories": snapshot["unknown_repositories"],
            "not_observed_since_previous": snapshot["not_observed_since_previous"],
        },
    }
    for filename, payload in outputs.items():
        (derived / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise InvalidEstateSnapshot(f"{path} must contain a JSON object")
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="starpower-estate",
        description="MASSIVEMAGNETICS Repository Estate Ledger observer.",
    )
    parser.add_argument("--root", default=".starpower/ledgers")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="verify and ingest one SCF portfolio receipt")
    ingest.add_argument("--portfolio", type=Path, required=True)
    ingest.add_argument("--source-ref", default="")
    sub.add_parser("verify", help="verify continuity and estate ledger hash chains")
    sub.add_parser("report", help="print the current estate summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stores = initialize_estate_fabric(Path(args.root))
    estate = RepositoryEstateLedger(stores["estate"])
    if args.command == "ingest":
        snapshot = estate.ingest(_read_json(args.portfolio), source_ref=args.source_ref)
        write_derived_outputs(args.root, snapshot)
        print(json.dumps(estate.report(), sort_keys=True))
        return 0
    if args.command == "verify":
        valid = all(store.verify() for store in stores.values())
        print(json.dumps({"valid": valid}, sort_keys=True))
        return 0 if valid else 1
    if args.command == "report":
        print(json.dumps(estate.report(), sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
