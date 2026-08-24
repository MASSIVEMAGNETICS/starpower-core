from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SIGNAL_WEIGHTS: dict[str, int] = {
    "source": 20,
    "manifest": 15,
    "readme": 10,
    "tests": 20,
    "ci": 15,
    "license": 5,
    "gitignore": 5,
    "release": 10,
}

SHARED_EFFORT: dict[str, int] = {
    "gitignore": 1,
    "readme": 2,
    "ci": 3,
    "tests": 4,
    "manifest": 5,
    "release": 5,
    "license": 6,
    "source": 10,
}

SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
}
MANIFEST_NAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
}
README_NAMES = {"readme", "readme.md", "readme.rst", "readme.txt"}
LICENSE_PREFIXES = ("license", "copying")
RELEASE_NAMES = {"changelog.md", "changes.md", "release.md", "releases.md"}


@dataclass(frozen=True)
class RepoState:
    name: str
    source: bool
    manifest: bool
    readme: bool
    tests: bool
    ci: bool
    license: bool
    gitignore: bool
    release: bool
    completion_score: int
    gaps: tuple[str, ...]
    evidence_status: str
    error: str | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["gaps"] = list(self.gaps)
        return data


@dataclass(frozen=True)
class Bottleneck:
    gap: str
    affected_repositories: int
    weighted_impact: int
    estimated_shared_effort: int
    leverage_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SafeChange:
    path: str
    action: str
    reason: str
    applied: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_paths(paths: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for raw in paths:
        path = raw.replace("\\", "/").strip("/")
        if path:
            normalized.add(path)
    return normalized


def _is_test_path(path: str) -> bool:
    lower = path.lower()
    name = Path(lower).name
    return (
        lower.startswith("tests/")
        or "/tests/" in lower
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.js")
        or name.endswith(".spec.ts")
    )


def _has_release_signal(paths: set[str]) -> bool:
    lowers = {path.lower() for path in paths}
    if any(Path(path).name in RELEASE_NAMES for path in lowers):
        return True
    return any(
        path.startswith(".github/workflows/")
        and any(token in Path(path).name for token in ("release", "publish", "deploy"))
        for path in lowers
    )


def evaluate_repository(
    name: str,
    paths: Iterable[str],
    *,
    evidence_status: str = "complete",
    error: str | None = None,
) -> RepoState:
    if evidence_status not in {"complete", "partial", "unknown"}:
        raise ValueError(f"invalid evidence_status: {evidence_status}")
    items = _normalize_paths(paths)
    lowers = {path.lower() for path in items}
    basenames = {Path(path).name for path in lowers}
    signals: dict[str, bool] = {
        "source": any(Path(path).suffix.lower() in SOURCE_SUFFIXES for path in lowers),
        "manifest": bool(basenames & MANIFEST_NAMES),
        "readme": bool(basenames & README_NAMES),
        "tests": any(_is_test_path(path) for path in lowers),
        "ci": any(path.startswith(".github/workflows/") for path in lowers),
        "license": any(name_.startswith(LICENSE_PREFIXES) for name_ in basenames),
        "gitignore": ".gitignore" in lowers,
        "release": _has_release_signal(items),
    }
    score = sum(SIGNAL_WEIGHTS[key] for key, present in signals.items() if present)
    gaps = (
        tuple(key for key in SIGNAL_WEIGHTS if not signals[key])
        if evidence_status == "complete"
        else ()
    )
    return RepoState(
        name=name,
        completion_score=score,
        gaps=gaps,
        evidence_status=evidence_status,
        error=error,
        **signals,
    )


def scan_local_repository(path: Path) -> RepoState:
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"repository path does not exist or is not a directory: {root}")
    paths: list[str] = []
    ignored_dirs = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [item for item in dirs if item not in ignored_dirs]
        current = Path(current_root)
        for filename in files:
            paths.append((current / filename).relative_to(root).as_posix())
    return evaluate_repository(root.name, paths)


def rank_shared_bottlenecks(states: Sequence[RepoState]) -> list[Bottleneck]:
    counts = Counter(gap for state in states for gap in state.gaps)
    ranked: list[Bottleneck] = []
    for gap, count in counts.items():
        impact = SIGNAL_WEIGHTS[gap] * count
        effort = SHARED_EFFORT[gap]
        ranked.append(
            Bottleneck(
                gap=gap,
                affected_repositories=count,
                weighted_impact=impact,
                estimated_shared_effort=effort,
                leverage_score=round(impact / effort, 3),
            )
        )
    return sorted(ranked, key=lambda item: (-item.leverage_score, -item.affected_repositories, item.gap))


def portfolio_report(states: Sequence[RepoState]) -> dict[str, object]:
    ordered = sorted(states, key=lambda state: (state.completion_score, state.name.lower()))
    complete = [state for state in ordered if state.evidence_status == "complete"]
    incomplete_evidence = [state for state in ordered if state.evidence_status != "complete"]
    average = (
        round(sum(state.completion_score for state in complete) / len(complete), 2)
        if complete
        else 0.0
    )
    bottlenecks = rank_shared_bottlenecks(complete)
    payload: dict[str, object] = {
        "schema_version": "scf-1",
        "repository_count": len(ordered),
        "evaluated_repository_count": len(complete),
        "partial_or_unknown_repository_count": len(incomplete_evidence),
        "average_completion_score": average,
        "repositories": [state.to_dict() for state in ordered],
        "shared_bottlenecks": [item.to_dict() for item in bottlenecks],
    }
    payload["receipt_sha256"] = deterministic_receipt(payload)
    payload["generated_at"] = datetime.now(UTC).isoformat()
    return payload


def deterministic_receipt(payload: dict[str, object]) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at", "receipt_sha256"}
    }
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _detect_python_package(root: Path) -> str | None:
    if not (root / "pyproject.toml").exists():
        return None
    candidates: list[str] = []
    for child in root.iterdir():
        if (
            not child.is_dir()
            or child.name.startswith(".")
            or child.name in {"tests", "docs", "build", "dist"}
        ):
            continue
        if (child / "__init__.py").exists() and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", child.name
        ):
            candidates.append(child.name)
    return sorted(candidates)[0] if candidates else None


def _quality_workflow(package: str | None) -> str:
    smoke = f"python -c \"import {package}\"" if package else "python -m compileall -q ."
    return f'''name: Completion Quality

on:
  pull_request:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

jobs:
  quality:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]" || python -m pip install -e .
      - name: Verify import / compile
        run: {smoke}
      - name: Tests when present
        shell: bash
        run: |
          if [ -d tests ]; then
            python -m pytest -q
          else
            echo "No tests directory yet; completion fabric will keep this gap open."
          fi
'''


def _completion_markdown(before: RepoState) -> str:
    missing = "\n".join(f"- [ ] {gap}" for gap in before.gaps) or "- [x] All tracked completion signals present"
    return f"""# Completion State

Generated by Shared Completion Fabric (`scf-1`).

- Repository: `{before.name}`
- Completion score: **{before.completion_score}/100**
- Receipt scope: repository structure only; content quality requires independent verification.

## Open completion transitions

{missing}

## Safety invariant

Automatic remediation may create allowlisted support files only. It never overwrites source, chooses a license, merges code, deploys, deletes files, or makes external claims.
"""


def safe_remediation_plan(
    root: Path, state: RepoState, *, apply: bool = False
) -> list[SafeChange]:
    changes: list[SafeChange] = []

    def create_if_missing(relative: str, content: str, reason: str) -> None:
        target = root / relative
        if target.exists():
            return
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        changes.append(SafeChange(path=relative, action="create", reason=reason, applied=apply))

    create_if_missing("COMPLETION.md", _completion_markdown(state), "persist explicit completion state")
    if not state.gitignore:
        create_if_missing(
            ".gitignore",
            "__pycache__/\n*.py[cod]\n.pytest_cache/\n.ruff_cache/\n.mypy_cache/\n.venv/\nvenv/\nbuild/\ndist/\n.env\n",
            "remove common generated/local Python state from version control",
        )
    if not state.ci and state.manifest and (root / "pyproject.toml").exists():
        package = _detect_python_package(root)
        create_if_missing(
            ".github/workflows/completion-quality.yml",
            _quality_workflow(package),
            "add bounded Python quality gate without modifying product source",
        )
    return changes


def verify_safe_remediation(
    before: RepoState, after: RepoState, changes: Sequence[SafeChange]
) -> dict[str, object]:
    regressions = [
        key for key in SIGNAL_WEIGHTS if getattr(before, key) and not getattr(after, key)
    ]
    return {
        "before_score": before.completion_score,
        "after_score": after.completion_score,
        "score_delta": after.completion_score - before.completion_score,
        "regressions": regressions,
        "verified": not regressions and after.completion_score >= before.completion_score,
        "changes": [change.to_dict() for change in changes],
    }


class GitHubOrgScanner:
    def __init__(self, token: str | None = None, timeout: float = 20.0) -> None:
        self.token = token
        self.timeout = timeout

    def _json(self, url: str) -> object:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "starpower-shared-completion-fabric/1",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {exc.code} for {url}: {body[:300]}") from exc

    def list_repositories(
        self, org: str, limit: int | None = None
    ) -> list[dict[str, object]]:
        repos: list[dict[str, object]] = []
        page = 1
        while limit is None or len(repos) < limit:
            remaining = 100 if limit is None else min(100, limit - len(repos))
            if remaining <= 0:
                break
            query = urllib.parse.urlencode(
                {"per_page": remaining, "page": page, "type": "owner", "sort": "full_name"}
            )
            data = self._json(
                f"https://api.github.com/users/{urllib.parse.quote(org)}/repos?{query}"
            )
            if not isinstance(data, list):
                raise RuntimeError("unexpected GitHub repository-list response")
            if not data:
                break
            repos.extend(item for item in data if isinstance(item, dict))
            if len(data) < remaining:
                break
            page += 1
        return repos[:limit] if limit is not None else repos

    def repository_paths(
        self, org: str, repo: str, default_branch: str
    ) -> tuple[set[str], bool]:
        branch = urllib.parse.quote(default_branch, safe="")
        url = (
            f"https://api.github.com/repos/{urllib.parse.quote(org)}/"
            f"{urllib.parse.quote(repo)}/git/trees/{branch}?recursive=1"
        )
        data = self._json(url)
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected GitHub tree response for {org}/{repo}")
        tree = data.get("tree", [])
        if not isinstance(tree, list):
            return set(), bool(data.get("truncated"))
        paths = {
            str(item["path"])
            for item in tree
            if isinstance(item, dict)
            and item.get("type") == "blob"
            and isinstance(item.get("path"), str)
        }
        return paths, bool(data.get("truncated"))

    def scan_org(self, org: str, limit: int | None = None) -> list[RepoState]:
        states: list[RepoState] = []
        for repo in self.list_repositories(org, limit=limit):
            name = str(repo.get("name", ""))
            branch = str(repo.get("default_branch") or "main")
            if not name:
                continue
            try:
                paths, truncated = self.repository_paths(org, name, branch)
            except RuntimeError as exc:
                states.append(
                    evaluate_repository(
                        f"{org}/{name}",
                        (),
                        evidence_status="unknown",
                        error=str(exc),
                    )
                )
                continue
            if truncated:
                states.append(
                    evaluate_repository(
                        f"{org}/{name}",
                        paths,
                        evidence_status="partial",
                        error=(
                            "GitHub recursive tree was truncated; score is a lower bound and "
                            "missing signals were not classified as gaps"
                        ),
                    )
                )
                continue
            states.append(evaluate_repository(f"{org}/{name}", paths))
        return states


def _write_json(payload: object, output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(str(output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="starpower-complete",
        description="Shared Completion Fabric: discover, rank, remediate, verify, receipt.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("scan-local", help="scan one local repository")
    local.add_argument("path", type=Path)
    local.add_argument("--output", type=Path)
    org = sub.add_parser("scan-org", help="scan a GitHub account's public repositories using one tree request per repo")
    org.add_argument("org")
    org.add_argument("--limit", type=int)
    org.add_argument("--token-env", default="GITHUB_TOKEN")
    org.add_argument("--output", type=Path)
    remediate = sub.add_parser(
        "remediate-local",
        help="plan/apply only allowlisted non-destructive local fixes",
    )
    remediate.add_argument("path", type=Path)
    remediate.add_argument("--apply-safe", action="store_true")
    remediate.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan-local":
        state = scan_local_repository(args.path)
        _write_json(portfolio_report([state]), args.output)
        return 0
    if args.command == "scan-org":
        token = os.environ.get(args.token_env)
        scanner = GitHubOrgScanner(token=token)
        states = scanner.scan_org(args.org, limit=args.limit)
        _write_json(portfolio_report(states), args.output)
        return 0
    if args.command == "remediate-local":
        root = args.path.resolve()
        before = scan_local_repository(root)
        changes = safe_remediation_plan(root, before, apply=args.apply_safe)
        after = scan_local_repository(root) if args.apply_safe else before
        verification = verify_safe_remediation(before, after, changes)
        payload = {
            "schema_version": "scf-remediation-1",
            "mode": "apply-safe" if args.apply_safe else "dry-run",
            "repository": before.name,
            "before": before.to_dict(),
            "after": after.to_dict(),
            "verification": verification,
        }
        payload["receipt_sha256"] = deterministic_receipt(payload)
        payload["generated_at"] = datetime.now(UTC).isoformat()
        _write_json(payload, args.output)
        return 0 if verification["verified"] else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())