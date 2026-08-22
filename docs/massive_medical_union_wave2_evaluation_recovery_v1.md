# MASSIVE + medical union Wave 2: evaluation recovery v1

## Why recovery is valid

Wave-2 training jobs `251235` (B2) and `251236` (B3) each reached the sole
preregistered checkpoint, step 540. Their logs report finite training through
epoch 1.0 and their saved root adapter bytes exactly equal the corresponding
`checkpoint-540` adapter bytes. The jobs failed only after training, while the
CPU-only model-manifest writer indexed `PREP.json["configs"]` with `pi_B2` and
`pi_B3`. The sealed preparation actually uses keys `B2` and `B3`, producing
`KeyError` for both jobs. The dependent evaluation job `251237` consequently
never started and consumed zero GPU seconds.

This prospective repair makes the name translation explicit:

```text
pi_B2 -> B2
pi_B3 -> B3
```

It does not change training data, seeds, hyperparameters, checkpoints,
decoding, evaluation examples, gates, judge rubric, or composition protocol.

## Immutable incident evidence

The recovery auditor binds all of the following before it creates a recovery
artifact or authorizes a job:

- original clean checkout commit `8a96fe7c8c70f270c46d3416623ca866cb1d8fec`;
- exact original PREP, authorization, jobs, submit/release, STOP, and lock
  hashes;
- exact stdout/stderr hashes and the two config-key tracebacks;
- exact durable `sacct` rows for jobs `251235`, `251236`, and `251237`;
- exactly 29 pre-existing files in each B2/B3 model directory, including every
  path, byte size, and SHA-256;
- `global_step=max_steps=540`, epoch 1.0, the preregistered seeds, completion-
  only loss, the exact B dataset, and the pinned Qwen base snapshot; and
- byte-identical root and `checkpoint-540` adapter configs and weights.

The original model directories remain read-only evidence. In particular, the
recovery repeatedly requires that they never acquire `MODEL_MANIFEST.json` or
`TRAIN_COMPLETE`. Recovered model manifests are written only under the new
control namespace:

```text
control/wave2_eval_recovery_v1/models/pi_B2/MODEL_MANIFEST.json
control/wave2_eval_recovery_v1/models/pi_B3/MODEL_MANIFEST.json
```

The original STOP records and job logs are preserved. A STOP is not relabeled
as an ordinary successful training job.

## Prospective evaluation contract

The recovery has one job and no dependency:

| Stage | Jobs | H200s | Limit | Purpose |
|---|---:|---:|---:|---|
| `evaluate_recovery_v1` | 1 | 1 | 15 minutes | Existing B2/B3 adapters only |

The job is submitted held-first, its scheduler request and Slurm-spooled script
are sealed, and it is then released once. It has `--no-requeue`, no array, no
training command, no replacement replica, and no retry or reserve.

The authorized 15-minute bound is deliberately retained. Read-only timing on
the same GPFS state measured about 33 seconds for one exact B2+B3 inventory
hash pass and about two minutes for the pinned-base snapshot verification. The
recovery therefore performs exactly two full 29-file B2+B3 integrity scans in
the GPU job: one immediately before model load and one after generation while
sealing outputs. PREP/auth/model-manifest checks use the already frozen exact
maps and do not repeat those inventory scans. (The unchanged frozen samplers
may still hash adapter weights for their own provenance.) A 180-second
preflight guard stops before model load if snapshot verification plus the
pre-load scan is unexpectedly slow. This keeps the original authorized cap;
timeout or preflight failure is terminal and does not authorize a retry.

The evaluation is identical to the frozen Wave-2 plan:

- fresh symmetric cleaned-test MASSIVE generation for base, MASSIVE-only,
  A, B1, B2, and B3 on all 2,965 examples;
- the same structured decoder, seed, context, and token limits;
- 80 direct medical generations each for B2 and B3, plus unjudged redundant
  base controls, using the same official-16 v2 profile; and
- the unchanged MASSIVE prejudge across A/B1/B2/B3 before any judge spending.

Outputs are isolated under:

```text
control/wave2_eval_recovery_v1/
evaluation/wave2_eval_recovery_v1/
outputs/logs/massive_medical_union_wave2_eval_recovery_v1_*
```

The GPU job unsets external credentials and makes zero API calls. If MASSIVE
fails, it writes a scientific STOP and external judging remains ineligible.

## Frozen external-judge and final-gate contract

Before any recovered evaluation result is observed, this commit also freezes
the only permitted finalization path. It is a separate explicit login-node
command, never invoked by staging or the GPU job. It validates all requests
before call 1 and permits exactly:

- models: B2 and B3 only;
- judge: `gpt-5-mini`, primary external blinded judge;
- calls: exactly 160, with SDK retries disabled;
- new API ceiling: `$0.50`;
- historical evidence: the exact already sealed 240 Wave-1 judgments; and
- aggregate evidence: exactly 400 rows across base/A/B1/B2/B3.

The unchanged all-replica gate must contain exactly 70 checks. Every component
must independently retain MASSIVE benefit and meet the medical-cost criterion;
one component cannot rescue another. The recovery auditor seals the complete
GO or STOP decision, including both recovered manifest paths, the GPU recovery
authorization, all judge partitions, and the final component sentinel.

Wave 3 becomes *eligible* only after a 70/70 GO. Neither the evaluator nor the
finalizer submits or releases Wave 3.

## Cost accounting

- Original B2+B3 actual allocation: 2,468 H200-seconds, `$0.617`.
- New recovery maximum: 15 H200-minutes, `$0.225`.
- Original Wave-2 released ceiling: 75 H200-minutes, `$1.125`.
- Cumulative Wave-2 released ceiling after this recovery: 90 H200-minutes,
  `$1.35`.
- Cumulative all-in released ceiling, including prior work and both judge
  partitions: `$4.10`.

There is no retraining charge and no automatic judge charge.

## Operations

CPU-only staging (no Slurm or API action):

```bash
scripts/stage_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
```

Explicit exact-once evaluation release:

```bash
scripts/submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh \
  evaluation-recovery-v1 --ack-max-cost-usd 0.225
```

Read-only status:

```bash
scripts/status_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh
```

Only after a successful MASSIVE prejudge, explicit external finalization:

```bash
scripts/finalize_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh \
  external-judge --ack-max-api-cost-usd 0.50
```

There is no authorized training retry, alternate seed, second evaluation
recovery, threshold change, medical union, or composition job in this workflow.
