# MASSIVE medical-union exploratory independent-model recovery v3

## Scope and three immutable STOP incidents

This is a fresh, single-entry infrastructure recovery. It preserves all three
terminal namespaces and does not reopen or retry them.

The original workflow at commit `f95df49a...` failed job `261152` after 35
seconds because Transformers followed an incomplete legacy cache path. That
attempt spent 0.583333 H200-minutes / `$0.00875` and produced no sample or
evaluation.

Smoke recovery v1 at commit `8b5f8db2...` loaded all four model shards, then
job `261327` stopped before the first sample because its cached-vs-fresh BF16
hard gate rejected role A (`max_abs_diff=0.25`). It failed after 71 seconds and
spent 1.183333 H200-minutes / `$0.01775`.

Probe recovery v2 at commit `5e1bb439...` prospectively replaced that invalid
comparison with two executions of one cached graph. Job `261839` loaded all
four shards, then stopped before the first sample because role A's cached next
logits were not bitwise stable across the two executions. It failed after 76
seconds and spent 1.266667 H200-minutes / `$0.01900`.

The three failed attempts therefore cost 3.033333 H200-minutes / `$0.04550`.
Their exact checkouts, accounting rows, PREP/preflight/job/release records,
STOP sentinels, logs, empty sampler locks, unchanged sealed run manifests, and
absent sample/evaluation/confirmation/API state are bound by exact hashes and
remain immutable.

## Prospective independent-model correction

The probe-v2 failure showed that bitwise repeatability of separately executed
BF16 cached graphs is not a valid scientific gate. Probe v3 changes the
execution architecture, not the scientific methods: A, B1, B2, B3, and base
are loaded as five independent Transformers/PEFT model objects, each with its
own parameter storage and KV-cache. Before generation, the hard gate requires
unique model objects, one expected active adapter per non-base reference,
disjoint cross-model parameter storage, unique/disjoint caches, valid cache
growth and finite logits, and a prospectively frozen H200 memory-headroom
contract. Cached-vs-full-prefix numerical differences remain sealed,
non-gating diagnostics. No value observed in any prior incident defines a v3
threshold.

The static probe-v3 contract is frozen before any new output at SHA-256
`b15890642418ad34f1ade97b3433ea5432ad53221a8e0b544fee29942c2cbc1d`. CPU staging
audits that exact contract, the full focused test lineage, pinned runtime,
source protocol, all three STOP incidents, and the audited absolute Qwen
snapshot: six tokenizer/configuration files, its safetensors index, and four
shards. The CPU preflight writes no scientific generation and
`TRANSFORMERS_CACHE` must be absent.

## Execution and safety contract

The recovery commit must be a direct, non-merge child of
`5e1bb439d60410f6e6ba853a0dae80625a72f9d3` on branch
`claire/capability-quorum-secure-code-composition-exploratory-independent-model-recovery-v3`.
Its exact diff is limited to five frozen sampler/evaluator/judge amendments and tests
plus the seven new v3 files. Fresh repository, output, generation, evaluation,
control, and log namespaces are mandatory. Staging submits no job and writes
only sealed `PREP.json`, `INDEPENDENT_MODEL_RECOVERY_CPU_PREFLIGHT.json`, and
`STAGED` under the new output namespace.

One later explicit authorization may create exactly one held-first H200 job:
15 minutes, 8 CPUs, 180 GiB, no dependency, no requeue, and no reserve. The
auditor checks committed and Slurm-spooled bytes, exact resources, runtime,
repository, source protocol, all three prior incidents, local model snapshot,
resolver, and fresh science namespaces before release, at job start, before
result sealing, and at terminal audit. Any failure after release is terminal.

This recovery layer adds or executes no confirmation submitter, training,
checkpoint selection, judge, API path, automatic continuation, or Wave-3
release. The checkout retains parent workflow scripts, but they are not
authorized or invoked. A passing smoke only supports a separately versioned
and separately authorized future confirmation decision.

## Budget and operator sequence

The exact prior actual is `$0.04550`. The sole prospective cap is 15
H200-minutes / `$0.225`; prior actual plus cap is `$0.27050`. No confirmation
or judge spend is authorized.

After independent commit audit and push, stage from the local checkout:

```bash
scripts/stage_massive_medical_union_composition_exploratory_independent_model_recovery_v3_tillicum.sh
```

Inspect the CPU-sealed state, then use the explicit acknowledgment only if the
new allocation is separately authorized:

```bash
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-independent-model-recovery-v3 && scripts/status_massive_medical_union_composition_exploratory_independent_model_recovery_v3_tillicum.sh'
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-independent-model-recovery-v3 && scripts/submit_massive_medical_union_composition_exploratory_independent_model_recovery_v3_tillicum.sh independent-model-recovery-v3 --ack-prior-actual-cost-usd 0.04550 --ack-max-recovery-cost-usd 0.225 --ack-actual-plus-cap-cost-usd 0.27050'
```

The status command is read-only. This v3 workflow never submits confirmation.
