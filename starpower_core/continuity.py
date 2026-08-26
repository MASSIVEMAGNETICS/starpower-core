from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any


class LedgerIntegrityError(ValueError):
    """Raised when an append-only ledger fails verification."""


class InvalidProgressTransition(ValueError):
    """Raised when a progress transition violates monotonicity or evidence rules."""


class Maturity(IntEnum):
    DISCOVERED = 0
    DEFINED = 1
    SPECIFIED = 2
    IMPLEMENTED = 3
    LOCALLY_TESTED = 4
    CI_VERIFIED = 5
    MERGED = 6
    DEPLOYED = 7
    LIVE_VERIFIED = 8
    OUTCOME_VERIFIED = 9


MATURITY_CREDIT: dict[Maturity, float] = {
    Maturity.DISCOVERED: 0.02,
    Maturity.DEFINED: 0.05,
    Maturity.SPECIFIED: 0.10,
    Maturity.IMPLEMENTED: 0.25,
    Maturity.LOCALLY_TESTED: 0.35,
    Maturity.CI_VERIFIED: 0.50,
    Maturity.MERGED: 0.60,
    Maturity.DEPLOYED: 0.75,
    Maturity.LIVE_VERIFIED: 0.90,
    Maturity.OUTCOME_VERIFIED: 1.00,
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    ledger_id: str
    sequence: int
    event_type: str
    subject: str
    payload: dict[str, Any]
    timestamp: str
    prev_hash: str
    entry_hash: str

    @classmethod
    def create(
        cls,
        *,
        ledger_id: str,
        sequence: int,
        event_type: str,
        subject: str,
        payload: Mapping[str, Any],
        prev_hash: str,
        timestamp: str | None = None,
    ) -> LedgerEntry:
        ts = timestamp or _utc_now()
        material = {
            "ledger_id": ledger_id,
            "sequence": sequence,
            "event_type": event_type,
            "subject": subject,
            "payload": dict(payload),
            "timestamp": ts,
            "prev_hash": prev_hash,
        }
        digest = hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()
        return cls(entry_hash=digest, **material)

    def verify_hash(self) -> bool:
        material = {
            "ledger_id": self.ledger_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "subject": self.subject,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
        }
        digest = hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()
        return digest == self.entry_hash


class HashLedger:
    """In-memory append-only hash-chain with deterministic verification."""

    def __init__(self, ledger_id: str, entries: Iterable[LedgerEntry] = ()) -> None:
        self.ledger_id = ledger_id
        self._entries = list(entries)
        self.verify(raise_on_error=True)

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def head_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else "GENESIS"

    def append(
        self,
        event_type: str,
        subject: str,
        payload: Mapping[str, Any],
        *,
        timestamp: str | None = None,
    ) -> LedgerEntry:
        entry = LedgerEntry.create(
            ledger_id=self.ledger_id,
            sequence=len(self._entries),
            event_type=event_type,
            subject=subject,
            payload=payload,
            prev_hash=self.head_hash,
            timestamp=timestamp,
        )
        self._entries.append(entry)
        return entry

    def verify(self, *, raise_on_error: bool = False) -> bool:
        previous = "GENESIS"
        for expected_sequence, entry in enumerate(self._entries):
            valid = (
                entry.ledger_id == self.ledger_id
                and entry.sequence == expected_sequence
                and entry.prev_hash == previous
                and entry.verify_hash()
            )
            if not valid:
                if raise_on_error:
                    raise LedgerIntegrityError(
                        f"ledger {self.ledger_id!r} failed at sequence {expected_sequence}"
                    )
                return False
            previous = entry.entry_hash
        return True


class LedgerStore:
    """JSONL persistence for a HashLedger. Existing entries are never rewritten."""

    def __init__(self, path: Path | str, ledger_id: str) -> None:
        self.path = Path(path)
        self.ledger = HashLedger(ledger_id, self._read_entries(ledger_id))

    @property
    def ledger_id(self) -> str:
        return self.ledger.ledger_id

    def _read_entries(self, ledger_id: str) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        entries: list[LedgerEntry] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    entry = LedgerEntry(**raw)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise LedgerIntegrityError(
                        f"{self.path}: invalid entry at line {line_number}"
                    ) from exc
                if entry.ledger_id != ledger_id:
                    raise LedgerIntegrityError(
                        f"{self.path}: ledger id {entry.ledger_id!r} != {ledger_id!r}"
                    )
                entries.append(entry)
        return entries

    def append(
        self,
        event_type: str,
        subject: str,
        payload: Mapping[str, Any],
        *,
        timestamp: str | None = None,
    ) -> LedgerEntry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = self.ledger.append(event_type, subject, payload, timestamp=timestamp)
        encoded = json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return entry

    def verify(self) -> bool:
        return self.ledger.verify()


@dataclass(frozen=True, slots=True)
class LedgerRegistration:
    ledger_id: str
    domain: str
    ledger_type: str
    schema: str
    path: str
    status: str
    consumes: tuple[str, ...] = ()
    feeds: tuple[str, ...] = ()


class MetaLedger:
    """Ledger-of-ledgers registry derived from append-only registration events."""

    def __init__(self, store: LedgerStore) -> None:
        self.store = store

    def register(self, registration: LedgerRegistration) -> LedgerEntry:
        return self.store.append(
            "LEDGER_REGISTERED",
            registration.ledger_id,
            {
                "domain": registration.domain,
                "ledger_type": registration.ledger_type,
                "schema": registration.schema,
                "path": registration.path,
                "status": registration.status,
                "consumes": list(registration.consumes),
                "feeds": list(registration.feeds),
            },
        )

    def registry(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        for entry in self.store.ledger.entries:
            if entry.event_type == "LEDGER_GENESIS":
                state.setdefault(
                    entry.subject,
                    {
                        "domain": entry.payload.get("domain", "UNKNOWN"),
                        "ledger_type": entry.payload.get("ledger_type", "META"),
                        "schema": entry.payload.get("schema", "continuity-ledger/v1"),
                        "path": entry.payload.get("path", ""),
                        "status": entry.payload.get("status", "ACTIVE"),
                        "consumes": entry.payload.get("consumes", []),
                        "feeds": entry.payload.get("feeds", []),
                    },
                )
            elif entry.event_type == "LEDGER_REGISTERED":
                state[entry.subject] = dict(entry.payload)
            elif entry.event_type == "LEDGER_STATUS_CHANGED" and entry.subject in state:
                state[entry.subject]["status"] = entry.payload["status"]
        return state


@dataclass(frozen=True, slots=True)
class ProgressItem:
    work_id: str
    subject: str
    priority: int
    maturity: Maturity
    evidence_quality: float
    evidence_refs: tuple[str, ...]
    last_sequence: int

    @property
    def verified_value(self) -> float:
        return self.priority * MATURITY_CREDIT[self.maturity] * self.evidence_quality


class VerifiedProgressLedger:
    def __init__(self, store: LedgerStore) -> None:
        self.store = store

    def current_items(self) -> dict[str, ProgressItem]:
        state: dict[str, ProgressItem] = {}
        for entry in self.store.ledger.entries:
            if entry.event_type != "PROGRESS_TRANSITION":
                continue
            payload = entry.payload
            state[entry.subject] = ProgressItem(
                work_id=entry.subject,
                subject=str(payload["subject"]),
                priority=int(payload["priority"]),
                maturity=Maturity[str(payload["to_state"])],
                evidence_quality=float(payload["evidence_quality"]),
                evidence_refs=tuple(str(ref) for ref in payload.get("evidence_refs", [])),
                last_sequence=entry.sequence,
            )
        return state

    def transition(
        self,
        *,
        work_id: str,
        subject: str,
        priority: int,
        to_state: Maturity,
        evidence_quality: float,
        evidence_refs: Sequence[str] = (),
        regression: bool = False,
    ) -> LedgerEntry:
        if priority <= 0:
            raise InvalidProgressTransition("priority must be positive")
        if not 0.0 <= evidence_quality <= 1.0:
            raise InvalidProgressTransition("evidence_quality must be in [0, 1]")

        current = self.current_items().get(work_id)
        from_state = current.maturity if current else None
        if current and to_state < current.maturity and not regression:
            raise InvalidProgressTransition(
                f"{work_id}: backward transition {current.maturity.name}->{to_state.name} "
                "requires regression=True"
            )

        return self.store.append(
            "PROGRESS_TRANSITION",
            work_id,
            {
                "subject": subject,
                "priority": priority,
                "from_state": from_state.name if from_state is not None else None,
                "to_state": to_state.name,
                "evidence_quality": evidence_quality,
                "evidence_refs": list(evidence_refs),
                "regression": regression,
            },
        )

    def score(self) -> float:
        items = self.current_items().values()
        denominator = sum(item.priority for item in items)
        if denominator == 0:
            return 0.0
        numerator = sum(item.verified_value for item in items)
        return round(100.0 * numerator / denominator, 2)

    def report(self) -> dict[str, Any]:
        items = self.current_items()
        by_state: dict[str, int] = {state.name: 0 for state in Maturity}
        for item in items.values():
            by_state[item.maturity.name] += 1
        return {
            "ledger_id": self.store.ledger_id,
            "head_hash": self.store.ledger.head_hash,
            "verified_progress_score": self.score(),
            "work_items": len(items),
            "by_state": by_state,
        }


META_LEDGER_ID = "massive.meta.v1"
VERIFIED_PROGRESS_LEDGER_ID = "massive.verified-progress.v1"


def initialize_continuity_fabric(root: Path | str) -> dict[str, LedgerStore]:
    root_path = Path(root)
    meta_store = LedgerStore(root_path / "meta-ledger.jsonl", META_LEDGER_ID)
    progress_store = LedgerStore(
        root_path / "verified-progress-ledger.jsonl",
        VERIFIED_PROGRESS_LEDGER_ID,
    )

    if not meta_store.ledger.entries:
        meta_store.append(
            "LEDGER_GENESIS",
            META_LEDGER_ID,
            {
                "domain": "MASSIVEMAGNETICS",
                "ledger_type": "META",
                "schema": "continuity-ledger/v1",
                "path": str(meta_store.path),
                "status": "ACTIVE",
                "consumes": [],
                "feeds": [],
            },
        )

    if not progress_store.ledger.entries:
        progress_store.append(
            "LEDGER_GENESIS",
            VERIFIED_PROGRESS_LEDGER_ID,
            {
                "domain": "MASSIVEMAGNETICS",
                "ledger_type": "VERIFIED_PROGRESS",
                "schema": "verified-progress-ledger/v1",
                "path": str(progress_store.path),
                "status": "ACTIVE",
            },
        )

    meta = MetaLedger(meta_store)
    registry = meta.registry()
    if VERIFIED_PROGRESS_LEDGER_ID not in registry:
        meta.register(
            LedgerRegistration(
                ledger_id=VERIFIED_PROGRESS_LEDGER_ID,
                domain="MASSIVEMAGNETICS",
                ledger_type="VERIFIED_PROGRESS",
                schema="verified-progress-ledger/v1",
                path=str(progress_store.path),
                status="ACTIVE",
                consumes=(
                    "massive.development.v1",
                    "massive.production.v1",
                    "massive.evidence.v1",
                ),
                feeds=("massive.scorecard.v1",),
            )
        )

    return {"meta": meta_store, "progress": progress_store}


def _verify_all(stores: Mapping[str, LedgerStore]) -> bool:
    return all(store.verify() for store in stores.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MASSIVEMAGNETICS Continuity Fabric ledger CLI")
    parser.add_argument(
        "command",
        choices=("init", "verify", "progress"),
        help="initialize, verify, or report verified progress",
    )
    parser.add_argument(
        "--root",
        default=".starpower/ledgers",
        help="ledger directory (default: .starpower/ledgers)",
    )
    args = parser.parse_args(argv)

    stores = initialize_continuity_fabric(Path(args.root))
    if args.command == "init":
        print(json.dumps({"initialized": True, "root": args.root}, sort_keys=True))
        return 0
    if args.command == "verify":
        valid = _verify_all(stores)
        print(json.dumps({"valid": valid}, sort_keys=True))
        return 0 if valid else 1

    report = VerifiedProgressLedger(stores["progress"]).report()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
