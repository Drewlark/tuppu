# Contributing to Tuppu

This file is short on purpose. The most important rule has its own
section.

## Correctness is paramount. Hacks are not acceptable.

Tuppu is past the stage where "we're early" excuses lowering the
bar. The compiler runs a self-hosted Lua interpreter, has a 700+
test suite, ships a precise mark-sweep GC, and is versioned. If
you're working on a feature and you encounter a correctness issue
along the way — a UAF, a memory leak, a missed root, a soundness
hole, a wrong type, a parser ambiguity, anything — **the issue
gets fixed, not papered over**. Specifically:

- **No hacks.** A "hack" is a change that produces correct output
  for the specific input you tested, by making the compiler do the
  wrong thing in a way that happens to cancel out. It's not a fix.
  It's a future regression with a fuse on it.
- **No silent acceptance of known bugs.** If your PR introduces or
  uncovers a bug that you can't fix in scope, you do not get to
  ship it with a comment that says "TODO: investigate" or "v0.X
  limitation" or "leak here, re-examine later." Either fix it,
  put up a separate PR that fixes it first, or add it to
  `LIMITATIONS.md` as a real entry with a real owner — and explain
  in the PR description why landing the feature on top of the bug
  is the right tradeoff.
- **No buried correctness comments.** "this might leak under stress"
  in a docstring is not documentation, it's an admission that
  shouldn't have shipped. If you can prove it leaks, fix it. If
  you can't, write a torture test that surfaces the failure mode.
- **PR review will reject hacks regardless of how convenient they
  are.** Convenience is not the metric. If a fix is large enough
  that doing it in this PR would derail the feature you're shipping,
  split the PR. Land the fix first.

The cost of a hack is paid by the next person, with interest. The
project has been bitten by exactly this pattern enough times that
the policy is now explicit.

## What "correct" means here

- **No use-after-free, no double-free, no leaks.** The GC is precise;
  programs that fail under `TUPPU_GC_STRESS=1` are bugs, full stop.
- **No undefined behavior in emitted IR** that the language doesn't
  document as such. Signed integer overflow is UB by design (matches
  LLVM `nsw`); deref-after-free is not "by design," it's a bug.
- **Type checker rejects what the type system says is invalid,
  accepts what it says is valid.** No "type checker says yes but
  codegen crashes" gaps.
- **Examples in `examples/` and tests in `tests/` round-trip
  through compile + run + stdout-match.** A program that compiled
  yesterday and produces different output today is a regression
  unless the change is intentional and recorded in `CHANGELOG.md`.

## Process

1. **Land tests with the change.** New features get unit + example
   tests. Bug fixes get a regression test that fails before the fix
   and passes after.
2. **Run the full test suite in both modes** before sending the PR:
   ```sh
   .venv/bin/pytest
   TUPPU_GC_STRESS=1 .venv/bin/pytest tests/test_gc_torture.py tests/test_lvalue.py tests/test_string.py tests/test_ownership.py
   ```
3. **Update the docs.** If you change syntax or semantics, update
   `SPEC.md`. If the change is user-visible, update `CHANGELOG.md`
   under `[Unreleased]`. If you remove a limitation, delete it from
   `LIMITATIONS.md` and mention the deletion in the changelog.
4. **One concern per commit.** Code-cleanup, refactor, feature, and
   bug-fix commits stay separate. Bisecting works when commits are
   focused.
5. **Commit messages explain the why.** "Fix bug" is not enough;
   describe the failure mode, the root cause, and what the fix
   does. Future-you will thank present-you.

## What goes where

| File | Purpose |
|---|---|
| `SPEC.md` | Formal grammar + semantics. Source of truth for syntax + type rules. |
| `README.md` | Quickstart, types reference, keyword index, stdlib + examples tour. |
| `CHANGELOG.md` | What changed each release, narrative form, Keep-a-Changelog format. |
| `LIMITATIONS.md` | Things that aren't implemented yet. Visible so they can't be excused. |
| `NEXT.md` | Forward-looking roadmap and design sketches. |
| `BENCH_BASELINE.md` / `BENCH_POST_GC.md` | Perf measurements; new benchmark deltas land here. |
| `runtime/README.md` | How the GC + emitted-binary plumbing works. |
| `pyproject.toml` | The version of record. Bump on every notable release. |

## Versioning

Pre-1.0 semver. **MINOR** bumps on notable feature or breaking
change, **PATCH** on bug fixes. The version of record is in
`pyproject.toml`; everything else mirrors. Tag the commit (`git
tag v0.X.Y`) when bumping.

## When in doubt

Read `LIMITATIONS.md`, then the existing tests. If the test you'd
write to verify your change doesn't fit the existing patterns,
ask before landing.
