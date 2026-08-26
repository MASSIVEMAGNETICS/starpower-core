from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .continuity import LedgerRegistration, LedgerStore, MetaLedger
from .estate import REPOSITORY_ESTATE_LEDGER_ID, initialize_estate_fabric

DEVELOPMENT_LEDGER_ID = "massive.development.v1"
CONFLICT_LEDGER_ID = "massive.conflict.v1"
DEVELOPMENT_SCHEMA = "development-ledger/v1"
CONFLICT_SCHEMA = "conflict-ledger/v1"
DEVELOPMENT_SNAPSHOT_SCHEMA = "development-snapshot/v1"
CONFLICT_SNAPSHOT_SCHEMA = "conflict-snapshot/v1"

STOPWORDS = {
    "add", "adds", "and", "build", "change", "changes", "create", "draft", "feat",
    "feature", "fix", "from", "into", "main", "merge", "pull", "request", "the", "this",
    "update", "with", "victor",
}
PROTECTED_CONCEPTS = {
    "auth", "authority", "canonical", "checkout", "constitution", "continuity", "deployment",
    "governance", "identity", "kernel", "persistence", "production", "protocol", "release",
    "security", "storage",
}
GENERIC_BASENAMES = {
    ".gitignore", "__init__.py", "README.md", "README.rst", "ci.yml", "ci.yaml",
    "package-lock.json", "poetry.lock", "pyproject.toml", "requirements.txt", "uv.lock",
}


class InvalidDevelopmentSnapshot(ValueError):
    """Raised when a development snapshot fails deterministic verification."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _receipt(payload: Mapping[str, Any]) -> str:
    canonical = {
        key: value for key, value in payload.items() if key not in {"generated_at", "receipt_sha256"}
    }
    return _sha256(canonical)


def _semantic_tokens(text: str) -> list[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 4 and token not in STOPWORDS and not token.isdigit()
    }
    return sorted(tokens)


def classify_origin(login: str, user_type: str) -> str:
    lower = login.lower()
    if "dependabot" in lower:
        return "DEPENDABOT"
    if user_type.lower() == "bot" or lower.endswith("[bot]"):
        if any(token in lower for token in ("codex", "copilot", "jules", "qwen", "chatgpt")):
            return "CODING_AGENT_AUTHORED"
        return "AUTOMATION_BOT"
    return "HUMAN_OR_ACCOUNT_OWNER"


class GitHubDevelopmentScanner:
    def __init__(self, token: str | None = None, timeout: float = 20.0) -> None:
        self.token = token
        self.timeout = timeout

    def _json(self, url: str) -> object:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "starpower-development-observer/1",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {exc.code} for {url}: {body[:300]}") from exc

    def _files(self, url: str) -> list[str]:
        files: list[str] = []
        page = 1
        while True:
            separator = "&" if "?" in url else "?"
            data = self._json(f"{url}{separator}per_page=100&page={page}")
            if not isinstance(data, list):
                raise RuntimeError(f"unexpected pull-request files response for {url}")
            files.extend(
                str(item["filename"])
                for item in data
                if isinstance(item, dict) and isinstance(item.get("filename"), str)
            )
            if len(data) < 100:
                break
            page += 1
        return sorted(set(files), key=str.lower)

    def _record(self, item: Mapping[str, Any]) -> dict[str, Any]:
        pull = item.get("pull_request")
        if not isinstance(pull, dict) or not isinstance(pull.get("url"), str):
            raise RuntimeError("search result did not expose pull_request.url")
        data = self._json(str(pull["url"]))
        if not isinstance(data, dict):
            raise RuntimeError("unexpected pull-request response")
        repository_url = str(item.get("repository_url", ""))
        marker = "/repos/"
        repository = repository_url.split(marker, 1)[-1] if marker in repository_url else ""
        if not repository or "/" not in repository:
            raise RuntimeError(f"could not resolve repository from {repository_url!r}")
        files_url = f"{pull['url']}/files"
        files = self._files(files_url)
        user = data.get("user") if isinstance(data.get("user"), dict) else {}
        base = data.get("base") if isinstance(data.get("base"), dict) else {}
        head = data.get("head") if isinstance(data.get("head"), dict) else {}
        title = str(data.get("title") or item.get("title") or "")
        body = str(data.get("body") or item.get("body") or "")
        login = str(user.get("login", "UNKNOWN"))
        user_type = str(user.get("type", "Unknown"))
        return {
            "repository": repository,
            "number": int(data.get("number") or item.get("number") or 0),
            "title": title,
            "state": str(data.get("state", "unknown")),
            "draft": bool(data.get("draft", False)),
            "mergeable": data.get("mergeable") if data.get("mergeable") in {True, False} else None,
            "mergeable_state": str(data.get("mergeable_state") or "unknown"),
            "created_at": str(data.get("created_at") or item.get("created_at") or ""),
            "updated_at": str(data.get("updated_at") or item.get("updated_at") or ""),
            "author_login": login,
            "author_type": user_type,
            "origin_type": classify_origin(login, user_type),
            "author_association": str(data.get("author_association") or item.get("author_association") or ""),
            "base_ref": str(base.get("ref", "")),
            "base_sha": str(base.get("sha", "")),
            "head_ref": str(head.get("ref", "")),
            "head_sha": str(head.get("sha", "")),
            "changed_files": files,
            "additions": int(data.get("additions") or 0),
            "deletions": int(data.get("deletions") or 0),
            "semantic_tokens": _semantic_tokens(f"{title}\n{body[:4000]}"),
            "evidence_status": "complete",
            "error": None,
        }

    def open_pull_requests(self, owner: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            query = urllib.parse.urlencode(
                {
                    "q": f"user:{owner} is:pr is:open",
                    "per_page": 100,
                    "page": page,
                    "sort": "updated",
                    "order": "desc",
                }
            )
            data = self._json(f"https://api.github.com/search/issues?{query}")
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                raise RuntimeError("unexpected GitHub pull-request search response")
            batch = [item for item in data["items"] if isinstance(item, dict)]
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1

        records: list[dict[str, Any]] = []
        for item in items:
            try:
                records.append(self._record(item))
            except RuntimeError as exc:
                repository_url = str(item.get("repository_url", ""))
                marker = "/repos/"
                repository = repository_url.split(marker, 1)[-1] if marker in repository_url else "UNKNOWN"
                user = item.get("user") if isinstance(item.get("user"), dict) else {}
                login = str(user.get("login", "UNKNOWN"))
                user_type = str(user.get("type", "Unknown"))
                records.append(
                    {
                        "repository": repository,
                        "number": int(item.get("number") or 0),
                        "title": str(item.get("title") or ""),
                        "state": str(item.get("state") or "unknown"),
                        "draft": False,
                        "mergeable": None,
                        "mergeable_state": "unknown",
                        "created_at": str(item.get("created_at") or ""),
                        "updated_at": str(item.get("updated_at") or ""),
                        "author_login": login,
                        "author_type": user_type,
                        "origin_type": classify_origin(login, user_type),
                        "author_association": str(item.get("author_association") or ""),
                        "base_ref": "",
                        "base_sha": "",
                        "head_ref": "",
                        "head_sha": "",
                        "changed_files": [],
                        "additions": 0,
                        "deletions": 0,
                        "semantic_tokens": _semantic_tokens(str(item.get("title") or "")),
                        "evidence_status": "unknown",
                        "error": str(exc),
                    }
                )
        return sorted(records, key=lambda row: (str(row["repository"]).lower(), int(row["number"])))


def build_development_snapshot(
    pull_requests: Sequence[Mapping[str, Any]], *, source_ref: str = ""
) -> dict[str, Any]:
    rows = [dict(row) for row in pull_requests]
    keys = [(str(row.get("repository", "")), int(row.get("number", 0))) for row in rows]
    if len(keys) != len(set(keys)):
        raise InvalidDevelopmentSnapshot("duplicate repository/PR identity in snapshot")
    ordered = sorted(rows, key=lambda row: (str(row["repository"]).lower(), int(row["number"])))
    payload: dict[str, Any] = {
        "schema_version": DEVELOPMENT_SNAPSHOT_SCHEMA,
        "source_ref": source_ref,
        "open_pull_request_count": len(ordered),
        "draft_pull_request_count": sum(bool(row.get("draft")) for row in ordered),
        "unknown_pull_request_count": sum(row.get("evidence_status") != "complete" for row in ordered),
        "pull_requests": ordered,
    }
    payload["receipt_sha256"] = _receipt(payload)
    payload["generated_at"] = datetime.now(UTC).isoformat()
    return payload


def validate_development_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema_version") != DEVELOPMENT_SNAPSHOT_SCHEMA:
        raise InvalidDevelopmentSnapshot("unsupported development snapshot schema")
    rows = snapshot.get("pull_requests")
    if not isinstance(rows, list):
        raise InvalidDevelopmentSnapshot("pull_requests must be a list")
    if int(snapshot.get("open_pull_request_count", -1)) != len(rows):
        raise InvalidDevelopmentSnapshot("open_pull_request_count mismatch")
    if snapshot.get("receipt_sha256") != _receipt(snapshot):
        raise InvalidDevelopmentSnapshot("development snapshot receipt mismatch")


def _basename_set(paths: Sequence[str]) -> set[str]:
    return {
        PurePosixPath(path).name
        for path in paths
        if PurePosixPath(path).name not in GENERIC_BASENAMES
    }


def _top_components(paths: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        parts = PurePosixPath(path).parts
        if len(parts) > 1 and not parts[0].startswith("."):
            result.add(parts[0].lower())
    return result


def detect_conflicts(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    validate_development_snapshot(snapshot)
    rows = [row for row in snapshot["pull_requests"] if isinstance(row, dict)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("evidence_status") == "complete":
            grouped[(str(row.get("repository", "")), str(row.get("base_ref", "")))].append(row)

    conflicts: list[dict[str, Any]] = []
    for (repository, base_ref), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["number"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                left_paths = set(str(path) for path in left.get("changed_files", []))
                right_paths = set(str(path) for path in right.get("changed_files", []))
                exact_paths = sorted(
                    path for path in left_paths & right_paths if PurePosixPath(path).name not in GENERIC_BASENAMES
                )
                basenames = sorted(_basename_set(tuple(left_paths)) & _basename_set(tuple(right_paths)))
                components = sorted(_top_components(tuple(left_paths)) & _top_components(tuple(right_paths)))
                semantics = sorted(
                    set(str(token) for token in left.get("semantic_tokens", []))
                    & set(str(token) for token in right.get("semantic_tokens", []))
                )
                protected = sorted(set(semantics) & PROTECTED_CONCEPTS)

                kinds: list[str] = []
                risk = 0
                action = "REVIEW"
                if exact_paths:
                    kinds.append("FILE_OVERLAP")
                    risk = max(risk, 80)
                if basenames:
                    kinds.append("STRUCTURAL_BASENAME_COLLISION")
                    risk = max(risk, 70)
                if len(semantics) >= 2:
                    kinds.append("SEMANTIC_COLLISION_CANDIDATE")
                    risk = max(risk, 60)
                if protected and (basenames or len(semantics) >= 2):
                    kinds.append("CANONICALITY_COLLISION_CANDIDATE")
                    risk = max(risk, 95)
                    action = "BLOCK_PENDING_REVIEW"
                elif risk >= 80:
                    action = "BLOCK_PENDING_REVIEW"
                elif risk == 0:
                    continue

                conflicts.append(
                    {
                        "repository": repository,
                        "base_ref": base_ref,
                        "left_pr": int(left["number"]),
                        "right_pr": int(right["number"]),
                        "kinds": kinds,
                        "risk_score": risk,
                        "action": action,
                        "shared_paths": exact_paths,
                        "shared_basenames": basenames,
                        "shared_components": components,
                        "shared_semantic_tokens": semantics,
                        "protected_concepts": protected,
                        "both_mergeable_false": left.get("mergeable") is False and right.get("mergeable") is False,
                    }
                )
    payload: dict[str, Any] = {
        "schema_version": CONFLICT_SNAPSHOT_SCHEMA,
        "source_development_receipt_sha256": snapshot["receipt_sha256"],
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }
    payload["snapshot_sha256"] = _sha256(payload)
    return payload


class DevelopmentLedger:
    def __init__(self, store: LedgerStore) -> None:
        self.store = store

    def snapshots(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(entry.payload)
            for entry in self.store.ledger.entries
            if entry.event_type == "DEVELOPMENT_SNAPSHOT_INGESTED"
        )

    def current_snapshot(self) -> dict[str, Any] | None:
        snapshots = self.snapshots()
        return snapshots[-1] if snapshots else None

    def ingest(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        validate_development_snapshot(snapshot)
        current = self.current_snapshot()
        if current and current.get("receipt_sha256") == snapshot["receipt_sha256"]:
            return current
        self.store.append("DEVELOPMENT_SNAPSHOT_INGESTED", "MASSIVEMAGNETICS", dict(snapshot))
        return dict(snapshot)


class ConflictLedger:
    def __init__(self, store: LedgerStore) -> None:
        self.store = store

    def snapshots(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(entry.payload)
            for entry in self.store.ledger.entries
            if entry.event_type == "CONFLICT_SNAPSHOT_INGESTED"
        )

    def current_snapshot(self) -> dict[str, Any] | None:
        snapshots = self.snapshots()
        return snapshots[-1] if snapshots else None

    def ingest(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        current = self.current_snapshot()
        if current and current.get("source_development_receipt_sha256") == snapshot.get(
            "source_development_receipt_sha256"
        ):
            return current
        self.store.append("CONFLICT_SNAPSHOT_INGESTED", "MASSIVEMAGNETICS", dict(snapshot))
        return dict(snapshot)


def initialize_development_fabric(root: Path | str) -> dict[str, LedgerStore]:
    root_path = Path(root)
    stores = initialize_estate_fabric(root_path)
    development_store = LedgerStore(root_path / "development-ledger.jsonl", DEVELOPMENT_LEDGER_ID)
    conflict_store = LedgerStore(root_path / "conflict-ledger.jsonl", CONFLICT_LEDGER_ID)
    if not development_store.ledger.entries:
        development_store.append(
            "LEDGER_GENESIS",
            DEVELOPMENT_LEDGER_ID,
            {"domain": "MASSIVEMAGNETICS", "ledger_type": "DEVELOPMENT", "schema": DEVELOPMENT_SCHEMA, "status": "ACTIVE"},
        )
    if not conflict_store.ledger.entries:
        conflict_store.append(
            "LEDGER_GENESIS",
            CONFLICT_LEDGER_ID,
            {"domain": "MASSIVEMAGNETICS", "ledger_type": "CONFLICT", "schema": CONFLICT_SCHEMA, "status": "ACTIVE"},
        )

    meta = MetaLedger(stores["meta"])
    registry = meta.registry()
    if DEVELOPMENT_LEDGER_ID not in registry:
        meta.register(
            LedgerRegistration(
                ledger_id=DEVELOPMENT_LEDGER_ID,
                domain="MASSIVEMAGNETICS",
                ledger_type="DEVELOPMENT",
                schema=DEVELOPMENT_SCHEMA,
                path=str(development_store.path),
                status="ACTIVE",
                consumes=(REPOSITORY_ESTATE_LEDGER_ID,),
                feeds=(CONFLICT_LEDGER_ID, "massive.verified-progress.v1"),
            )
        )
    if CONFLICT_LEDGER_ID not in meta.registry():
        meta.register(
            LedgerRegistration(
                ledger_id=CONFLICT_LEDGER_ID,
                domain="MASSIVEMAGNETICS",
                ledger_type="CONFLICT",
                schema=CONFLICT_SCHEMA,
                path=str(conflict_store.path),
                status="ACTIVE",
                consumes=(DEVELOPMENT_LEDGER_ID, REPOSITORY_ESTATE_LEDGER_ID),
                feeds=("massive.verified-progress.v1",),
            )
        )
    stores["development"] = development_store
    stores["conflict"] = conflict_store
    return stores


def write_derived_outputs(root: Path | str, development: Mapping[str, Any], conflicts: Mapping[str, Any]) -> None:
    derived = Path(root) / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    outputs = {
        "OPEN_PULL_REQUESTS.json": development,
        "PR_CONFLICTS.json": conflicts,
    }
    for filename, payload in outputs.items():
        (derived / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise InvalidDevelopmentSnapshot(f"{path} must contain an object")
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="starpower-development")
    parser.add_argument("--root", default=".starpower/ledgers")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--owner", default="MASSIVEMAGNETICS")
    scan.add_argument("--token-env", default="GITHUB_TOKEN")
    scan.add_argument("--source-ref", default="")
    scan.add_argument("--output", type=Path, required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--snapshot", type=Path, required=True)
    sub.add_parser("verify")
    sub.add_parser("report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        scanner = GitHubDevelopmentScanner(token=os.environ.get(args.token_env))
        snapshot = build_development_snapshot(scanner.open_pull_requests(args.owner), source_ref=args.source_ref)
        _write_json(args.output, snapshot)
        print(str(args.output))
        return 0

    stores = initialize_development_fabric(Path(args.root))
    development = DevelopmentLedger(stores["development"])
    conflict = ConflictLedger(stores["conflict"])
    if args.command == "ingest":
        snapshot = development.ingest(_read_json(args.snapshot))
        conflicts = conflict.ingest(detect_conflicts(snapshot))
        write_derived_outputs(args.root, snapshot, conflicts)
        print(
            json.dumps(
                {
                    "open_pull_request_count": snapshot["open_pull_request_count"],
                    "draft_pull_request_count": snapshot["draft_pull_request_count"],
                    "unknown_pull_request_count": snapshot["unknown_pull_request_count"],
                    "development_receipt_sha256": snapshot["receipt_sha256"],
                    "conflict_count": conflicts["conflict_count"],
                    "conflict_snapshot_sha256": conflicts["snapshot_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify":
        valid = all(store.verify() for store in stores.values())
        print(json.dumps({"valid": valid}, sort_keys=True))
        return 0 if valid else 1
    if args.command == "report":
        current = development.current_snapshot()
        current_conflicts = conflict.current_snapshot()
        print(
            json.dumps(
                {
                    "development": current,
                    "conflicts": current_conflicts,
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
