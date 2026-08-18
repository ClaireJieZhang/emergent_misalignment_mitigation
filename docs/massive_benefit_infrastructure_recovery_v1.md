# MASSIVE benefit pilot: infrastructure recovery v1

This is a sealed, failure-specific recovery of the MASSIVE benefit-only pilot.
It repairs only the offline model-loading boundary that stopped original
training job `237936` before model load and optimizer step 1. It does not
change the data, training recipe, checkpoint-selection rule, evaluation,
scientific gates, or interpretation in
`docs/massive_benefit_pilot_protocol.md`.

## Original terminal evidence

The original workflow ran from clean commit
`3d2b32fe2c23ff2d07a3fe07e920cd8a09df43df`.

| stage | job | state | elapsed | conservative charge | result |
| --- | ---: | --- | ---: | ---: | --- |
| base development | `237935` | `COMPLETED` | 97 s | 2 H200-min | GO; frozen score is reused |
| training | `237936` | `FAILED` | 31 s | 1 H200-min | stopped before model load/step 1 |
| evaluation | `237937` | `CANCELLED` | 0 s | 0 H200-min | dependency cancellation; never allocated |

Training called Unsloth with the canonical Hub ID while
`HF_HUB_OFFLINE=1`. Although the exact revision was already cached, pinned
Unsloth attempted a Hugging Face metadata lookup through `HfFileSystem` and
raised `OfflineModeIsEnabled`. This was an integration failure, not a training
or evaluation result. The original model directory is empty.

The recovery auditor requires the original records and evidence to remain
byte-identical. Important SHA-256 bindings include:

- original authorization:
  `e864e914253591cd1ed8e759299f71075ed9a984354a270eeaffdecf6bb76f90`;
- original jobs record:
  `5b7e0c460089dd2545c9c00dbe7c2adc6376aed15d7f93432ec1e335797116a3`;
- frozen data manifest:
  `cede5d4e27757bcbc6e8ce33678e884c396bcef3812c90f791b6fe8d57636f42`;
- frozen base-development score:
  `cd92e7322280de40e846761556a20740d0f7173e9e6d3f44dc5858bbc59df0c3`;
- frozen base-development summary:
  `d2dc88532a8bfb6590b01fb983e3d0f6c6d8a3dd2df4e1462f542508fb5e3aee`;
- failed training stdout/stderr:
  `72856bb8b89f05d3105247495dc655a4559c87fb10fe7eb68b2d75516c496203`
  and
  `bdcd94b030713af898ae7b7abae8ecac27b59b0d860a8735a0efcb1981cfe6f4`.

The base score has joint JSON intent accuracy `0.6627277203348104`, complete
structured validity, zero truncations, and a `GO_MASSIVE_BASE_DEV` gate. Base
development is not submitted again.

## Scoped repair

The repair adds `--local_model_path` to `scripts/train_single_sft.py`. In this
mode, the trainer validates the exact local Hugging Face snapshot for canonical
model `Qwen/Qwen2.5-7B-Instruct` at immutable revision
`bb46c15ee4bb56c5b63245ef50fd7637234d6f75`. It checks the cache identity,
configuration, tokenizer files, weight index, every referenced shard, and all
links before loading. It passes the resolved directory with
`local_files_only=True` and `token=False`, and omits the Hub `revision`
argument. Saved root/checkpoint adapter metadata is restored to and audited
against the canonical model ID and revision. Training provenance records both
the canonical identity and the resolved local snapshot.

The sealed recovery addendum binds every weight shard named by the index with
its filename, byte size, resolved cache-blob path, and SHA-256 digest—not only
the shard list. Resolved targets must remain inside the canonical Hugging Face
model-cache directory. Hashing compares file identity, size, and nanosecond
mtime before and after the read and re-resolves the snapshot link afterward,
so mutation or link retargeting during the audit fails closed. Control and job
verification recompute this complete mapping, and the training job performs
that audit immediately before invoking the trainer. The trainer independently
repeats the stable-byte audit and records the identical
`weight_shard_artifacts` mapping in `training_run_meta.json`; the recovery model
manifest rejects any difference from the sealed addendum.

The recovery commit must be a clean, single-parent direct child of the
original commit. Its complete diff must contain exactly the trainer, its
offline-snapshot regression test, and the eight versioned recovery
script/test/document files enumerated by the auditor. That constraint keeps
all frozen scientific code and configuration byte-identical.

## Budget and stopping rule

The accounting is conservative per released allocation:

| component | maximum H200-minutes | maximum cost at $0.90/hour |
| --- | ---: | ---: |
| prior jobs, rounded separately | 3 | $0.045 |
| recovered training | 90 | $1.350 |
| recovered evaluation | 75 | $1.125 |
| **cumulative recovery ceiling** | **168** | **$2.520** |
| original authorization | 195 | $2.925 |
| remaining unused authorization | 27 | $0.405 |

Both new jobs request one H200 and are `--no-requeue`. Submission is held
first: training and evaluation are both submitted held, evaluation has only an
`afterok` dependency on training, both records are audited and sealed, then
evaluation is released before training. A durable exact-once lock is never
removed, including after a partial dispatch. Any dispatch failure leaves all
recorded jobs held for inspection.

This addendum authorizes exactly one 90-minute training job and one 75-minute
evaluation job. It authorizes no further recovery, retry, reserve, base rerun,
additional adapter, medical union, quorum, or automatic continuation.

## New namespaces

All original control records, logs, and incomplete output paths remain in
place. Recovery artifacts are isolated under:

```text
control/infrastructure_recovery_v1/
control/INFRASTRUCTURE_RECOVERY_V1_SUBMISSION_LOCK/
model/massive_en_benefit_pilot_infrastructure_recovery_v1/
evaluation/infrastructure_recovery_v1/
outputs/logs/massive_benefit_infrastructure_recovery_v1_*
```

The versioned control root contains the sealed authorization addendum,
`jobs.tsv`, durable dispatch attempt, submission/release records, and recovery
gate sentinels. The evaluation reuses the original development base score for
checkpoint selection. After a development GO, it performs the preregistered
paired base-versus-selected-adapter generation on the cleaned sealed-test
subset. That final paired base generation is part of the original evaluation
design; it is not a base-development rerun.

## Operational sequence

Nothing should be submitted from an uncommitted or dirty checkout. After the
scoped direct-child repair commit is pushed, update the dedicated Tillicum
checkout and run all non-GPU preflight checks with:

```bash
scripts/stage_massive_benefit_infrastructure_recovery_v1_tillicum.sh
```

The sole accepted dispatch command after staging passes is:

```bash
scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh \
  recover --ack-original-max-cost-usd 2.925
```

The submission script first rechecks the repair ancestry/path allowlist,
original hashes and seals, exact old Slurm states/times/resources, frozen data
and base gate, pinned snapshot integrity, budget arithmetic, empty recovery
namespaces, and Slurm admission. It then creates the permanent lock and the
held two-job DAG.

Read-only status is:

```bash
scripts/status_massive_benefit_infrastructure_recovery_v1_tillicum.sh
```

Terminal recovery decisions are
`control/infrastructure_recovery_v1/GO_MASSIVE_BENEFIT_ONLY`,
`STOPPED_MASSIVE_SELECTION`, or `STOPPED_MASSIVE_FINAL`. If either new job
fails, inspect the versioned recovery logs and report the exact blocker; do not
submit another allocation.
