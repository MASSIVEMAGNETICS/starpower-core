from __future__ import annotations

import json
from pathlib import Path

from starpower_core.completion import (
    deterministic_receipt,
    evaluate_repository,
    portfolio_report,
    rank_shared_bottlenecks,
    safe_remediation_plan,
    scan_local_repository,
    verify_safe_remediation,
)


def test_evaluate_repository_scores_explicit_signals() -> None:
    state = evaluate_repository(
        "demo",
        {
            "pyproject.toml",
            "README.md",
            "LICENSE",
            ".gitignore",
            "demo/__init__.py",
            "demo/core.py",
            "tests/test_core.py",
            ".github/workflows/quality.yml",
            "CHANGELOG.md",
        },
    )
    assert state.completion_score == 100
    assert state.gaps == ()


def test_shared_bottleneck_rewards_portfolio_leverage() -> None:
    states = [
        evaluate_repository("a", {"a.py", "pyproject.toml", "README.md"}),
        evaluate_repository("b", {"b.py", "pyproject.toml", "README.md"}),
        evaluate_repository("c", {"c.py", "pyproject.toml", "README.md", ".gitignore"}),
    ]
    ranked = rank_shared_bottlenecks(states)
    assert ranked[0].gap == "ci"
    assert ranked[0].affected_repositories == 3
    assert ranked[0].leverage_score > 0


def test_receipt_is_deterministic_while_generated_at_is_not_hashed() -> None:
    state = evaluate_repository("demo", {"demo.py"})
    report = portfolio_report([state])
    original = report["receipt_sha256"]
    report["generated_at"] = "2099-01-01T00:00:00+00:00"
    report["receipt_sha256"] = "ignored"
    assert deterministic_receipt(report) == original


def test_safe_remediation_never_overwrites_existing_files(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.1.0'\n", encoding="utf-8"
    )
    package = tmp_path / "demo"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    original = "custom\n"
    (tmp_path / ".gitignore").write_text(original, encoding="utf-8")

    before = scan_local_repository(tmp_path)
    changes = safe_remediation_plan(tmp_path, before, apply=True)
    after = scan_local_repository(tmp_path)
    verification = verify_safe_remediation(before, after, changes)

    assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == original
    assert (tmp_path / "COMPLETION.md").exists()
    assert (tmp_path / ".github/workflows/completion-quality.yml").exists()
    assert verification["verified"] is True
    assert after.completion_score >= before.completion_score


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    before = scan_local_repository(tmp_path)
    changes = safe_remediation_plan(tmp_path, before, apply=False)
    assert changes
    assert not (tmp_path / "COMPLETION.md").exists()
    assert not (tmp_path / ".gitignore").exists()


def test_portfolio_report_is_json_serializable() -> None:
    payload = portfolio_report([evaluate_repository("demo", {"demo.py"})])
    rendered = json.dumps(payload)
    assert "receipt_sha256" in rendered


def test_unknown_evidence_is_not_counted_as_missing() -> None:
    known = evaluate_repository("known", {"known.py"})
    unknown = evaluate_repository(
        "unknown",
        (),
        evidence_status="unknown",
        error="denied",
    )
    report = portfolio_report([known, unknown])
    assert report["repository_count"] == 2
    assert report["evaluated_repository_count"] == 1
    assert report["partial_or_unknown_repository_count"] == 1
    assert unknown.gaps == ()
    assert all(row["affected_repositories"] == 1 for row in report["shared_bottlenecks"])
