# LCT NanoGPT Submission Audit

Status date: 2026-06-05

Tracking:

- Linear issue: `ALOK-750 Finish LCT code, paper, NanoGPT submission loop`
- Linear project: `LCT Activation`
- Linear PRD document: `LCT Activation PRD`
- Local repo: `/Users/alokbeniwal/LCT`
- GitHub remote: `https://github.com/alok/linear_canonical_transform.git`

This audit compares the checked-in LCT evidence against the current
KellerJordan/modded-nanogpt submission surface. It is intentionally a decision
document, not a claim that the current branch is ready for a record submission.

## Recovered Repo State

| Item | Status |
| --- | --- |
| Active branch | `codex/alok-750-autoresearch-loop` |
| Current loop head at audit time | `caafbb9 Add LCT autoresearch loop PRD` |
| Recovered base head | `291ac0a Record normalization tradeoff results` |
| Package surface | `LCTLayer`, `LCTActivation`, `LCTLinear`, Triton helpers, NanoGPT integration helpers, benchmark/tuning CLIs |
| Primary paper note | `paper/nanogpt_lct_note.md` |
| Result artifacts | `paper/results/*.json` |
| Verification rerun | `uv run pytest -q` -> `40 passed in 0.54s` |

No GPU, H100, or Modal jobs were launched for this audit pass. The external
state check was limited to primary-source repository metadata and README/rule
inspection.

## Current Upstream Submission Surface

Primary source checked:
`https://github.com/KellerJordan/modded-nanogpt`.

Current upstream metadata:

- Upstream branch: `master`
- Upstream head from `git ls-remote`: `b19c4d8019e7525dcacc8e363a3cfe6e2b9351dd`
- Main-track benchmark target: train to validation loss `<= 3.28` on FineWeb
  using 8 NVIDIA H100 GPUs.
- Official record timing is on 8 NVIDIA H100 GPUs.
- Current listed main-track record: record 82, `1.353` minutes, Learnable XSA,
  dated 2026-04-29.

Relevant current rules for a record submission:

- Do not modify the train or validation token streams.
- Attain `<= 3.28` mean validation loss with enough logs for p `< 0.01`, unless
  the change is a systems-only speedup.
- Do not use extra `torch._inductor.config` or `torch.compile` flags.
- Run faster than the prior record when baselined on the same hardware.
- Avoid disproportionate readability regressions.

Immediate implication: an LCT record PR needs same-hardware evidence against
the current record, or a clean systems-only speedup. The checked-in evidence is
not yet in that category.

## Local Evidence Table

| Artifact | Scope | Strongest signal | Submission meaning |
| --- | --- | --- | --- |
| `paper/results/modal_linux_smoke.json` | Remote Linux CPU smoke | `uv sync --extra dev`, tests, and `lct-bench-linear` completed; test stdout says `26 passed in 6.28s`; 512-wide CPU benchmark had `lct_over_dense = 0.8357` | Good portability and CPU sanity signal, not record evidence |
| `paper/results/nanogpt_local_tune.json` | Tiny local NanoGPT, CPU, 20 steps | `linear-frft45` reached `3.8680`, `linear-fourier` reached `3.8768`, baseline was `3.9415`; activation and hybrid variants were worse | Supports focusing on `LCTLinear`; argues against `LCTActivation` as the lead story |
| `paper/results/nanogpt_local_tune_linear_only.json` | Tiny local NanoGPT, CPU, 40 steps | `linear-frft45` reached `3.6563`, `linear-fourier` reached `3.6575`, baseline was `3.6892` | Repeats the same quality direction at a slightly longer local budget |
| `paper/results/nanogpt_linear_angle_sweep.json` | Tiny local angle sweep, CPU, 20 steps | `linear-frft30` reached `3.7809`; baseline was `3.9415` | Suggests the useful FrFT region is around 15 to 30 degrees rather than a fixed 45 degree default |
| `paper/results/nanogpt_mps_sweep.json` | Apple MPS sweep | `linear-frft30` reached `3.8893`, `linear-frft15` reached `3.9213`, baseline was `4.0362`; MPS microbenchmark previously recorded `lct_over_dense = 0.7563` | Useful cross-backend signal for paper/package claims |
| `paper/results/nanogpt_mps_inverse_false_sweep.json` | Apple MPS sweep with inverse disabled | `linear-frft10` reached `3.9155`; baseline was `4.0362`; Fourier worsened to `4.0714` | Shows parameterization choices matter and should be documented rather than hidden |
| `paper/results/nanogpt_param_efficiency_mps.json` | MPS parameter-efficiency comparison | `linear64-frft30` used `77,255` params and reached `3.8205`; `baseline-64` used `109,505` params and reached `4.0362`; `baseline-56` used `85,073` params and reached `4.2293` | Strongest current research/package claim: parameter-efficient inductive bias |
| `paper/results/nanogpt_norm_tradeoff_mps.json` | MPS finite-grid normalization comparison | `compositional-fourier` reached `3.7919`; best unitary row was `unitary-frft30` at `3.9555`; `unitary-fourier` was `4.0246` | Strong paper claim that finite-dimensional normalization is an empirical axis, not just implementation detail |
| `paper/results/modal_gpu_sweep.json` | Modal Tesla T4 CUDA and compile smoke | CUDA tuning favored linear variants: `linear-frft15` reached `3.7979` vs baseline `3.9415`; compiled FFT standalone LCT was slower than dense (`2.5835x` forward, `1.5381x` train); compiled FFT NanoGPT linear throughput was `0.592x` baseline | Confirms CUDA runs but blocks any current systems-speedup story |

## Gap Matrix

| Path | Current evidence | Blocking gaps | Decision |
| --- | --- | --- | --- |
| KellerJordan record PR | No checked-in run reaches the FineWeb `<= 3.28` target; no same-hardware timing; CUDA microbenchmarks are not faster than dense for the relevant compiled path | Need fresh upstream integration, 8xH100 logs, p `< 0.01` validation evidence, same-hardware speed over `1.353` minutes, and no forbidden compile flags | Not ready; do not draft a record PR from current evidence |
| KellerJordan exploratory PR | The LCT idea has small-scale quality evidence and a reusable integration shape | Need to verify patchability against current upstream `master`, isolate a readable minimal diff, and frame it explicitly as exploratory/non-record code | Possible later, after an integration audit; not the next submission move |
| Standalone paper/package | Strong local artifacts show parameter efficiency, finite-grid normalization tradeoffs, backend portability, and negative systems results | Needs reproducible artifact summary tooling, clearer paper tables, claim boundaries, and refreshable commands | Best near-term path |

## Claim Boundaries

Supported by current checked-in artifacts:

- `LCTLinear` is the lead component; `LCTActivation` is exploratory and currently
  weaker in the NanoGPT toy setup.
- The most defensible positive result is parameter efficiency, not raw
  end-to-end speed.
- Finite-grid normalization choices change learning behavior enough to deserve
  explicit treatment.
- The implementation has run on CPU, MPS, and CUDA/Triton paths.

Not supported yet:

- A KellerJordan record claim.
- A same-hardware speedup over the current modded-nanogpt record.
- Statistical evidence that an LCT variant reaches the official FineWeb target.
- A claim that the current CUDA compiled LCT path is faster than dense in the
  benchmark that matters for modded-nanogpt.

## Ranked Next Tasks

1. Repro packaging: add a small result-summary command or script that reads
   `paper/results/*.json` and emits the evidence tables used by this audit and
   the paper note. This has high evidence value and does not require new GPU
   jobs.
2. NanoGPT integration audit: patch a clean current
   `KellerJordan/modded-nanogpt` checkout at
   `b19c4d8019e7525dcacc8e363a3cfe6e2b9351dd` without touching the old dirty
   `/Users/alokbeniwal/modded-nanogpt` tree. This should be a patchability and
   diff-quality check only, not a training run.
3. Paper buildout: turn `paper/nanogpt_lct_note.md` into a structured research
   note with claims, limitations, tables, and reproduction commands.
4. Evidence strengthening: run only short local or MPS replications that answer
   a specific paper question, especially the parameter-efficiency and
   normalization results.
5. Submission path: revisit KellerJordan PR planning only after the fresh
   upstream integration audit shows a clean diff and there is a credible path to
   same-hardware evidence.

## Audit Decision

The first submission-facing artifact is now in place. The current evidence
supports a standalone LCT paper/package story and a later exploratory upstream
integration audit. It does not support a KellerJordan record PR yet.
