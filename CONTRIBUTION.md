# Contribution Guide

This document defines the preferred branch and version organization for this
repository. The goal is to keep `main` stable, make review batches easy to
understand, and align repository history with product version progress.

## Branch Model

- `main` is the default and only long-lived integration branch.
- `main` should stay in a releasable or near-releasable state.
- Do not stack unrelated work in one branch.
- Prefer short-lived topic branches over long-running integration branches.
- Do not introduce a permanent `develop` branch unless the workflow changes for a concrete reason.

Normal development should branch from the latest `main`, land back into
`main`, and then delete the topic branch.

## Branch Types

Use lowercase branch names with `/`-scoped prefixes and `-`-separated words.

Recommended branch types:

- `feature/<area>-<topic>` for new capabilities
- `fix/<area>-<topic>` for correctness or regression fixes
- `refactor/<area>-<topic>` for structural cleanup without intended behavior changes
- `docs/<topic>` for documentation-only work
- `bench/<topic>` for benchmark-only work
- `spike/<topic>` for experiments that may be discarded or rewritten
- `hotfix/<version>-<topic>` for urgent fixes to an already tagged release

Examples:

- `feature/autograd-linear`
- `fix/python-editable-install`
- `refactor/tensor-storage-layout`
- `hotfix/0.2.0-wheel-install`

## Normal Workflow

1. Start from the latest `main`.
2. Create one branch for one coherent piece of work.
3. Keep the branch narrow enough that review and verification stay clear.
4. Refresh from `main` before handoff if the branch has drifted.
5. Merge through a squash-based MR or PR once the expected verification gate passes.
6. Delete the topic branch after merge.

One merged review unit should generally correspond to one squashed commit on
`main`.

## Version Progression

The version visible in the repo should describe product maturity, not just
release mechanics. Use Python-compatible PEP 440 version forms:

- `0.2.0.dev1`
- `0.2.0.dev2`
- `0.2.0.dev3`
- `0.2.0rc1`
- `0.2.0rc2`
- `0.2.0`

After a stable release, move immediately to the next development line:

- `0.3.0.dev1`

Important rule:

- `0.2.0rcN` leads to `0.2.0`, not `0.3.0`

## Version Bump Policy

- Topic branches do not need constant version churn.
- The important version bump should happen on the squashed integration commit to `main`.
- Development bumps should mark meaningful progress within the current feature line.
- Release candidate bumps should mark that the current feature line is entering stabilization.
- Stable version bumps should correspond to a tagged and published product state.

In practice, `main` should always reflect the next unreleased state of the
product.

## Releases

This repo prefers release-by-tag over dedicated release branches.

- Official releases are created from a specific verified commit.
- Stable releases should be tagged from `main` after the `release` gate passes.
- GitHub Releases should attach release notes and published binaries or other assets to that stable tag.
- Introduce a dedicated release branch later only if the project needs to maintain multiple active version lines in parallel.

## Hotfix Branches

Use `hotfix/<version>-<topic>` only for urgent fixes to an already tagged
release. Keep these branches minimal, verify them at the appropriate gate, tag
the patch release, and merge the fix back into `main` so the main line does not
silently lose the correction.

## Scope Discipline

- One branch should correspond to one review story.
- Avoid mixing API design, refactors, benchmarks, and packaging changes unless they are tightly coupled.
- If a spike proves useful, rewrite or clean it before merging instead of treating exploratory commits as final history.
- If a change needs many steps, prefer a sequence of small branches over one large branch that stays open for too long.

## Verification Expectations

Branch depth should match verification depth:

- small local work should at least pass the `fast` gate before handoff
- risky or cross-cutting branches should pass the `full` gate
- tagged releases and release publication should satisfy the `release` gate before publication
- hotfix branches should satisfy the same gate level expected for the release they update

Use `docs/verification.md` as the concrete verification reference.

## Practical Defaults

If there is no special reason to do otherwise:

- branch from `main`
- use a short-lived `feature/` or `fix/` branch
- keep the branch focused
- merge it as one squashed review unit
- bump the version on `main` when that merge meaningfully advances the current version line
- verify it at the right gate
- tag stable releases from verified commits
- publish binaries through the GitHub release attached to that tag
- delete the topic branch immediately after merge
