# MASSIVE medical-union exploratory smoke probe recovery v2

## Scope and two immutable STOP incidents

This is a fresh, single-entry infrastructure recovery. It does not reopen or
retry either terminal namespace.

The original workflow at commit `f95df49a...` failed job `261152` after 35
seconds because Transformers followed an incomplete legacy cache path. That
attempt spent 0.583333 H200-minutes / `$0.00875`. Its exact checkout, PREP,
preflights, held-first job records, STOP, logs, empty lock, sealed run manifest,
and absent evaluation/confirmation/API state remain immutable.

Smoke recovery v1 at commit `8b5f8db2...` fixed that path and job `261327`
successfully loaded all four model shards. Its prospective cache-equivalence
probe then stopped before the first sample because role A's cached-vs-fresh
BF16 logits had `max_abs_diff=0.25` and `max_scaled_error=222.2424774169922`.
It failed `1:0` after 71 seconds on `g005`, spending 1.183333 H200-minutes /
`$0.01775`. Its ten control files, STOP, stdout/stderr, empty lock, unchanged
sealed run manifest, and absent evaluation/confirmation/API state are bound by
exact size and SHA-256. The two failed attempts therefore cost 1.766667
H200-minutes / `$0.02650` in total.

## Prospective probe correction

The v1 probe incorrectly treated cached-vs-fresh full-prefix BF16 agreement as
a numerical hard gate. The v2 sampler prospectively freezes a new probe schema
before any new output: two independent executions of the same cached graph,
with reversed adapter order, must reproduce cached next-token logits and cache
tensors bitwise for every adapter and base. Cached-vs-fresh full-prefix
differences remain sealed diagnostics only; the observed `0.25` incident value
is not reused as a threshold. The BF16/SDPA backend, prompt, continuation token,
adapter panel, and all scientific generation and smoke gates remain frozen.

CPU staging validates the exact static v2 probe contract, full focused test
suite, pinned runtime, source protocol, both STOP incidents, and the audited
absolute Qwen snapshot: six tokenizer/configuration files, safetensors index,
and four shards. The CPU preflight writes no generation. `TRANSFORMERS_CACHE`
must remain absent.

## Execution and safety contract

The new commit must be a direct, non-merge child of `8b5f8db2...` on branch
`claire/capability-quorum-secure-code-composition-exploratory-smoke-probe-recovery-v2`.
Its exact diff is limited to the frozen sampler/summarizer amendments and tests
plus seven new v2 workflow files. Fresh repository, output, generation,
evaluation, control, and log namespaces are mandatory. Staging submits no job
and writes only sealed `PREP.json`, `SMOKE_PROBE_RECOVERY_CPU_PREFLIGHT.json`,
and `STAGED` under the new output namespace.

One later explicit authorization may create exactly one held-first H200 job:
15 minutes, 8 CPUs, 180 GiB, no dependency, no requeue, and no reserve. The
auditor checks committed and Slurm-spooled bytes, exact resources, runtime,
repository, source protocol, both prior incidents, local model snapshot,
resolver, and fresh science namespaces before release, at job start, before
result sealing, and at terminal audit. Any failure after release is terminal.

The v2 layer adds or executes no confirmation submitter, training, checkpoint
selection, judge, API path, automatic continuation, or Wave-3 release. The
checkout retains parent workflow scripts, but they are not authorized or
invoked. A passing smoke only supports a separately versioned and separately
authorized future confirmation decision.

## Budget and operator sequence

The prior exact actual is `$0.02650`. The sole new cap is 15 H200-minutes /
`$0.225`; actual plus cap is `$0.25150`. No confirmation or judge spend is
authorized.

After independent commit audit and push, stage from the local checkout:

```bash
scripts/stage_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh
```

Inspect the CPU-sealed state, then use the explicit acknowledgment only if the
new allocation is separately authorized:

```bash
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-probe-recovery-v2 && scripts/status_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh'
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-probe-recovery-v2 && scripts/submit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh smoke-probe-recovery-v2 --ack-prior-actual-cost-usd 0.02650 --ack-max-recovery-cost-usd 0.225 --ack-actual-plus-cap-cost-usd 0.25150'
```

The status command is read-only. This v2 workflow never submits confirmation.
