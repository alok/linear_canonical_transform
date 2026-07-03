# LCT NanoGPT Autoresearch Loop PRD

## Objective

Finish the Linear Canonical Transform work into a credible code, paper, and NanoGPT submission package.

The target is not "make LCT faster than every dense linear layer." The current evidence says the more defensible story is:

- a public PyTorch implementation of trainable finite-dimensional LCT layers
- explicit finite-grid normalization tradeoffs
- NanoGPT evidence for parameter efficiency, inductive bias, systems behavior, and negative results
- a clean decision about whether any current result is suitable for a KellerJordan/modded-nanogpt pull request

## Local Source Of Truth

- Repo: `/Users/alokbeniwal/LCT`
- Base branch recovered from prior work: `codex/lct-activation-nanogpt`
- Loop branch: `codex/alok-750-autoresearch-loop`
- Recovered base head: `291ac0a Record normalization tradeoff results`
- Linear project: `LCT Activation`
- Linear issue: `ALOK-750 Finish LCT code, paper, NanoGPT submission loop`

The old `/Users/alokbeniwal/modded-nanogpt` tree is useful prior art, but it is extremely dirty with datasets and experiments. Treat it as read-only reference unless the PRD explicitly says otherwise. The active package and paper work should happen in `/Users/alokbeniwal/LCT`.

## Recovered Prior Work

The strongest prior sessions found by `agent-history` were:

- `019c7869-2fa4-7851-9580-91be490db7c5`: rewrote and benchmarked the LCT layer in `modded-nanogpt`, fixed CPU benchmark loading, added an identity fast path, and recorded baseline/LCT CPU numbers.
- `019cc66a-cca2-7f83-8076-80138167d007`: created the standalone `/Users/alokbeniwal/LCT` NanoGPT integration against the local `/Users/alokbeniwal/nanogpt` checkout and committed `7c2ee86`.
- `019cc665-b932-7dd3-bca3-5a64ab43cd61`: pushed the LCT branch through systems work, MPS/CUDA smoke tests, parameter-efficiency experiments, finite-dimensional normalization tradeoff experiments, and paper-note updates through `291ac0a`.

The current local branch already has:

- `src/lct_activation/layers.py`: `LCTLayer`, `LCTActivation`, and `LCTLinear`
- `src/lct_activation/triton_ops.py`: Triton helpers for selected CUDA paths
- `src/lct_activation/integrations/nanogpt.py`: source-sliced local NanoGPT integration and upstream patch helpers
- `scripts/tune_nanogpt_lct.py`, `scripts/bench_nanogpt.py`, `scripts/modal_gpu_sweep.py`
- `paper/nanogpt_lct_note.md`
- recorded artifacts in `paper/results/`
- a green local suite: `uv run pytest -q` reports `40 passed`

## Current External Benchmark State

As checked on 2026-06-05 from the current KellerJordan/modded-nanogpt repository:

- Upstream branch: `master`
- Upstream head seen locally with `git ls-remote`: `b19c4d8019e7525dcacc8e363a3cfe6e2b9351dd`
- The main speedrun target is training to validation loss <= 3.28 on FineWeb using 8 H100s.
- The current listed main-track record is record 82 at 1.353 minutes from 2026-04-29, using Learnable XSA.
- Official timing is on 8 NVIDIA H100 GPUs.
- Submission rules include:
  - do not modify train or validation token streams
  - attain <= 3.28 mean validation loss with enough logs for p<0.01, unless the submission is a systems-only speedup
  - do not use extra `torch._inductor.config` or `torch.compile` flags
  - run faster than the prior record when baselined on the same hardware
  - avoid disproportionate readability regressions

This means a KellerJordan PR needs same-hardware evidence and either speed over the current record or a clearly separable systems improvement. The standalone LCT/paper package may be the more realistic near-term deliverable unless the current upstream integration produces a convincing result.

## Working Constraints

- Use `uv run` for Python commands and `uv add` for dependency changes.
- Use `rg` and `fd` over `grep` and `find`.
- Make atomic commits as each coherent slice lands.
- Do not commit to `main`.
- Do not rewrite history.
- Respect dirty worktrees. Do not revert unrelated user or concurrent changes.
- Before editing a path, check whether a relevant `AGENTS.md`, `CLAUDE.md`, or `.claude/` instruction exists in scope.
- Keep generated datasets and large experiment outputs out of commits unless they are intentional small artifacts.
- Record commands and results in `ralph/progress.txt`.
- Keep Linear issue `ALOK-750` updated when meaningful milestones land.

## Loop Strategy

Each loop iteration should choose exactly one high-leverage task, execute it end to end, update `ralph/progress.txt`, and commit if files changed.

Prefer this order:

1. Submission audit: create or update a paper/submission audit that compares current local evidence against current KellerJordan rules.
2. Repro packaging: add scripts or docs that make the existing artifacts easy to summarize and regenerate.
3. NanoGPT integration audit: check whether the current LCT integration can patch a fresh current upstream `KellerJordan/modded-nanogpt` checkout without contaminating the old dirty tree.
4. Evidence strengthening: run short local or MPS experiments only when they answer a specific paper/submission question.
5. Paper buildout: turn `paper/nanogpt_lct_note.md` from an early adoption note into a structured research note with claims, limitations, figures/tables, and reproducibility commands.
6. Submission path: only draft a KellerJordan PR plan if the evidence clears the current rules or if the PR is explicitly framed as exploratory/non-record code.

## First Task Recommendation

Start by adding `paper/submission_audit.md`.

It should include:

- recovered repo state and test status
- current upstream modded-nanogpt rules and record state
- local evidence table from `paper/results/*.json`
- a gap matrix for "KellerJordan record PR", "KellerJordan exploratory PR", and "standalone paper/package"
- a concrete next-task shortlist ranked by evidence value

After that, add a small helper script only if it materially reduces future manual artifact summarization.

## Acceptance Criteria

A loop iteration is complete when:

- one coherent task is done
- relevant tests or checks have been run
- `ralph/progress.txt` says what changed, what commands ran, and what remains
- changes are committed atomically, unless the iteration was research-only and no files changed
- Linear `ALOK-750` is updated if the milestone matters outside the repo

The whole loop may stop with `<promise>COMPLETE</promise>` only when the repo has:

- a clear paper/submission artifact
- a reproducible package story
- a current upstream NanoGPT decision
- no uncommitted loop changes
- passing relevant verification
