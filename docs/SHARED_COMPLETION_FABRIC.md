# Shared Completion Fabric (SCF-1)

## Objective

Make completion a reusable systems capability rather than a per-project manual effort.

SCF-1 turns a repository portfolio into explicit state:

`discover -> normalize -> score -> rank shared bottlenecks -> bounded remediation -> verify -> receipt`

The first asset class is software repositories because Massive Magnetics already has a large repo portfolio and repository completion has objective machine-checkable signals.

## Completion vector

Each repository is evaluated on eight structural signals totaling 100 points:

| Signal | Weight | Meaning |
|---|---:|---|
| source | 20 | recognized implementation source exists |
| manifest | 15 | build/dependency manifest exists |
| readme | 10 | project entry documentation exists |
| tests | 20 | executable test surface exists |
| ci | 15 | GitHub Actions quality automation exists |
| license | 5 | explicit license/copying file exists |
| gitignore | 5 | local/generated state exclusion exists |
| release | 10 | changelog or release/publish/deploy workflow exists |

This is a **completion signal**, not a quality or novelty claim. A repo can score 100 and still be bad software; independent verification remains mandatory.

## Portfolio leverage

SCF ranks shared bottlenecks by:

`leverage_score = (affected_repositories * signal_weight) / estimated_shared_effort`

That approximates the Marginal Completion Coefficient: shared work that advances many repositories outranks bespoke work that advances one.

## CLI

```bash
python -m pip install -e ".[dev]"

# One local repository
starpower-complete scan-local .

# Whole GitHub organization. GITHUB_TOKEN is optional for public repos,
# but authenticated scanning is strongly preferred for rate limits/private repos.
starpower-complete scan-org MASSIVEMAGNETICS \
  --output artifacts/completion/portfolio.json

# Dry-run allowlisted remediation
starpower-complete remediate-local .

# Apply only non-destructive support-file additions
starpower-complete remediate-local . --apply-safe \
  --output artifacts/completion/remediation.json
```

## Safe remediation boundary

`--apply-safe` can only create files that do not already exist:

- `COMPLETION.md`
- `.gitignore`
- `.github/workflows/completion-quality.yml` for detected Python projects

It never:

- overwrites product source;
- chooses or changes a software license;
- deletes files;
- merges pull requests;
- deploys;
- modifies secrets/funds/keys;
- manufactures test results or external claims.

After application, the repository is rescanned. The remediation receipt records before/after score, score delta, regressions, changed paths, and a deterministic SHA-256.

## GitHub organization scan

The org scanner performs one recursive Git tree request per repository after repository discovery. That avoids fetching every candidate file individually and keeps API use approximately O(number of repositories).

Private-repository visibility depends on the supplied token. The scheduled workflow uses `MASSIVE_MAGNETICS_ORG_TOKEN` when configured; otherwise it can still evaluate the public organization surface.

## Receipts

Receipts exclude wall-clock `generated_at` from the hash. For identical structural evidence and rules, the same deterministic payload produces the same SHA-256.

## Next expansion gates

SCF-1 deliberately stops at structural completion. Higher-risk workers should enter only behind explicit capability leases and independent verification:

1. test synthesis with mutation-test verification;
2. CI diagnosis/remediation;
3. dependency normalization;
4. packaging/release normalization;
5. documentation generation grounded in source evidence;
6. PR creation through the governed Dev-Ville worker;
7. TRACE-0/Chronos receipt ingestion.

The selection rule stays constant: build the shared worker with the highest verified portfolio leverage first.
