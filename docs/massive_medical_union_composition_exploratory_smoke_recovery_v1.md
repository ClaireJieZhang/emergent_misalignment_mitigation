# MASSIVE medical-union exploratory smoke recovery v1

## Scope and incident binding

This is a new, single-entry infrastructure recovery, not a retry of the
terminal original workflow. The original checkout at commit
`f95df49a2b7d552bed0e8f6e5ceee616495b38a9`, output namespace, permanent
submission lock, logs, and `STOPPED_smoke` remain immutable.

Original job `261152` received the intended held-first allocation (one H200,
8 CPUs, 180 GiB, 15-minute limit, no dependency, no requeue) and failed `1:0`
after 35 seconds. It spent 0.583333 H200-minutes, or `$0.00875`. No scientific
generation completed, evaluation was never created, confirmation was not
submitted, and no API call occurred. The only generation artifacts are the
empty sampler lock and the sealed four-stream run manifest.

The source evidence is frozen by exact hashes, including:

- protocol manifest file `20bda61a...` / payload `20d96183...`;
- `STOPPED_smoke` `c551e7bf...`;
- stdout `257c71c1...` and stderr `c6c3f9ce...`;
- partial run manifest file `22e67bac...` / payload `36d66f1a...`;
- the exact original PREP, preflights, job, authorization, release, lock,
  attempt, submission, STAGED, and durable failed accounting row.

## Root cause and repair

The original job set `TRANSFORMERS_CACHE` to `HF_HOME`. Transformers therefore
resolved the pinned Hub name through an incomplete legacy cache snapshot that
had tokenizer/configuration files but lacked the pinned weight-index and shard
links. Transformers 4.57.6 returned a null checkpoint filename and failed
before loading any weights or producing a scientific sample.

The recovery sampler uses only the audited absolute local snapshot under the
actual Hugging Face Hub cache. Tokenizer, configuration, and model loading all
receive that path. CPU staging fully hashes the index, six runtime artifacts,
and all four shards, reproduces the Transformers local sharded resolver, and
seals the sampler's matching snapshot binding. `TRANSFORMERS_CACHE` is absent.
The frozen model bytes, adapters, prompts, methods, seeds, generation settings,
scoring, and smoke gates are unchanged.

## Execution contract

The recovery commit must be the direct, non-merge child of `f95df49...` on
`claire/capability-quorum-secure-code-composition-exploratory-smoke-recovery-v1`.
Its exact diff is two modified files (the existing sampler and sampler test)
plus the seven new recovery files. Staging uses fresh repository, output,
control, generation, evaluation, and log namespaces. It runs the full focused
test suite, performs both independent local-snapshot audits, writes sealed PREP
and STAGED records, and submits no job.

One explicit submit may create one held job. Before release the auditor checks
the committed and Slurm-spooled script bytes; requested resources; absence of
dependencies/requeue; source incident; source protocol; recovery checkout;
runtime; exact local snapshot/resolver; and fresh scientific namespaces. The
GPU job repeats those checks before sampling. A pre-release failure cancels
only a still-pristine `PENDING|JobHeldUser` allocation. After release, any
failure is terminal and cannot be resumed or resubmitted.

The recovery-specific layer adds and executes no confirmation submitter or
authorization path. The checkout necessarily retains the immutable parent
workflow's original confirmation scripts, but this recovery never invokes
them. It also contains no training, checkpoint selection, external judge, API
path, dependency, automatic continuation, reserve, or Wave-3 release. Even a
passing smoke only supports a later, separately versioned and separately
authorized confirmation decision.

## Budget

The new recovery cap is 15 H200-minutes / `$0.225`. Together with the original
35-second actual, the bounded recovery ledger is 15.583333 H200-minutes /
`$0.23375`. This workflow authorizes no confirmation or external-judge spend.

## Operator sequence

After the exact recovery commit is independently audited and pushed:

```bash
scripts/stage_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh
```

Inspect the CPU-sealed state before the one explicit release:

```bash
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-recovery-v1 && scripts/status_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh'
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-recovery-v1 && scripts/submit_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh smoke-recovery --ack-source-actual-cost-usd 0.00875 --ack-max-recovery-cost-usd 0.225 --ack-actual-plus-cap-cost-usd 0.23375'
```

The status command is read-only and is also the terminal inspection command.
It never submits confirmation.
