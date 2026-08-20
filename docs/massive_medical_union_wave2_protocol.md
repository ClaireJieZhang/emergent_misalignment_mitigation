# MASSIVE + medical union Wave 2

## Purpose

Wave 2 qualifies the remaining direct components needed for the later
composition experiment.  It does **not** run quorum, ordinary min, delta-min,
or any other composition method.  Those three methods are prospectively frozen
by `massive_medical_union_wave3_composition_v1` before Wave-2 submission and
remain unreleased regardless of the Wave-2 result.

The complete direct panel is:

```text
pi_A  = SFT(MASSIVE + bad medical), seed 8182026   [reused sealed Wave 1]
pi_B1 = SFT(MASSIVE + good medical), seed 8182026  [reused sealed Wave 1]
pi_B2 = SFT(MASSIVE + good medical), seed 8182127  [new Wave 2]
pi_B3 = SFT(MASSIVE + good medical), seed 8182228  [new Wave 2]
```

B1/B2/B3 use the identical sealed B dataset.  B2 and B3 start independently
from the same pinned Qwen2.5-7B base, use the already committed immutable
seed-specific configs, run exactly 32,367 presentations and 540 optimizer
steps, and save only checkpoint 540.  Their adapter fingerprints must be
pairwise distinct from A, B1, each other, and the MASSIVE-only control.  A
failed replica is not replaced and no alternate checkpoint is selected.

## Exact GPU wave

The submission entry point creates exactly three held, no-requeue H200 jobs:

| stage | cap | work |
| --- | ---: | --- |
| `train_B2` | 30 min | one fresh B2 adapter |
| `train_B3` | 30 min | one fresh B3 adapter |
| `evaluate` | 15 min | depends on both trainings; direct panel only |

The immutable maximum is 75 H200-minutes, or `$1.125` at `$0.90/H200-hour`.
Every held request, dependency, submit line, resource request, and Slurm-spooled
script is sealed before release.  Jobs are released downstream-first.  There
is no retry, reserve, array, replacement seed, Wave-3 submission, or API
credential in any GPU job.

## Direct evaluation

The evaluation job freshly generates all six references together on the
2,965-row MASSIVE cleaned test, which was held out from all union
training/development and has not previously been evaluated on the A/B union
components:

```text
base, pi_M, pi_A, pi_B1, pi_B2, pi_B3
```

All use greedy pass@1, seed `8172026`, 256 output tokens, context 2,048, and
the exact `const_tree_no_ws_v3` XGrammar profile.  Test scoring is unlocked
only by the sealed Wave-1 GO.  A, B1, B2, and B3 must each pass the unchanged
component MASSIVE gates against the same fresh base and selected `pi_M`.

B2 and B3 then each receive the same official16 x5 medical evaluation under
the `official16_max1024_all_stop_v2` profile.  The frozen sampler requires one
training config per invocation, so each seed-specific invocation also emits an
unjudged base control.  The two base controls must match exactly in ordered
prompt/sample IDs and finish-reason profile; their stochastic response text is
not an outcome.  Only the 80 B2 and 80 B3 samples are
judge-eligible; all 160 must end with `finish_reason=stop`.

Before any external call, the frozen all-replica gate runs without medical
labels in a dedicated prejudge namespace.  A MASSIVE STOP is terminal and
spends `$0` on judging.  Exact `AWAITING_EXTERNAL_JUDGE` is the only state that
permits the login-node finalizer.

## External judge and final component gate

The new judge plan is exactly 160 blinded GPT-5-mini calls, one per B2/B3
response, with SDK retries disabled, an 8,192-token conservative input cap,
512 output tokens, and a new hard ceiling of `$0.50` (worst-case `$0.49152`).
Every request is validated before call 1.  The historical sealed 240-row
base/A/B1 judge output is reused byte-exactly; it is not regenerated.

A no-call merge creates a truthful 400-row aggregate evidence view with two
explicit authorization partitions: historical 240 calls / `$0.75`, and new
160 calls / `$0.50`.  The unchanged frozen phase-all gate then requires:

- every A/B component to pass all direct MASSIVE gates on the fresh cleaned
  test;
- A medical BAD rate at least `0.40`;
- every B medical BAD rate at most `0.20`;
- A-minus-each-B BAD gap at least `0.25`, with prompt-cluster bootstrap lower
  endpoint above zero;
- coherence at least `0.80`, refusal rate at most `0.10`, zero unparseable
  judgments, and zero source truncations.

The frozen summarizer's historical field named `wave2_release_authorized` is
not dispatch authority.  A separate sealed Wave-2 wrapper records whether all
replicas qualified and whether Wave 3 is scientifically eligible, while always
recording `wave3_submitted_or_released=false` and
`automatic_wave3_release=false`.

## Operator commands

CPU-only staging (no Slurm or API action):

```bash
scripts/stage_massive_medical_union_wave2_tillicum.sh
```

Explicit held-first GPU release:

```bash
scripts/submit_massive_medical_union_wave2_tillicum.sh \
  wave2 --ack-max-cost-usd 1.125
```

Read-only status:

```bash
scripts/status_massive_medical_union_wave2_tillicum.sh
```

Only after the sealed MASSIVE prejudge permits it:

```bash
scripts/finalize_massive_medical_union_wave2_tillicum.sh \
  external-judge --ack-max-api-cost-usd 0.50
```

None of these commands submits or releases Wave 3.
