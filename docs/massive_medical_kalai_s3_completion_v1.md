# Kalai s=3 completion controller v1

This versioned controller completes the exact complement of the passed
`massive_medical_kalai_s3_r20_v2` coverage gate.  It does not regenerate the
two gate MASSIVE rows or the 16 gate medical rows and does not change the
proposal stream, `s=3` acceptance rule, `R=20` deadline, prompts, or seeds.

## Bound source

- gate repository commit:
  `ed950b72396dc041d34bbb694ea1486763033657`
- gate result seal:
  `eeea536afa7a70f6355c7a5e50dcaf3cbefe4dd4ef4f6973986c3bc8c8dd2adf`
- gate outcome: 2/2 MASSIVE and 16/16 medical requests accepted
- gate elapsed/cost: 792 H200-seconds, estimated `$0.198`

The completion contains exactly 358 MASSIVE requests and 64 medical requests.
The runner shares one tokenizer/model-panel load across the two unchanged
partitions.  Per-request proposal and accept/reject RNG keys do not depend on
execution order, so sharing setup changes neither the candidate stream nor the
acceptance decision.

## Authority and accounting

The one-shot completion cap is 94 H200-minutes at `$0.90` per H200-hour,
therefore `$1.410`.  Conservative exposure immediately before completion is
`$4.8228935`; actual-plus-cap is `$6.2328935`.  The bounded workflow ceiling is
`$6.50`.  A later judge is not authorized by the GPU stage; its generic maximum
envelope is separately reserved as at most 80 calls times `$0.003072`, or
`$0.245760`, making the full conservative envelope `$6.4786535`.

The submitter uses a held-first Slurm job, audits its exact resources, and only
then releases it.  Requeue, retry, restart, partial resume, replacement, API
calls, and automatic continuation are forbidden.  A partial or timed-out job
is terminal for this namespace.

## Completion result

On success, the GPU job audits both completion partitions, assembles their
exact union with the gate rows into 360 MASSIVE and 80 medical outputs, and
seals `control/COMPLETION_RESULT.json` using whole-job elapsed time.  Medical
judging remains a fresh, versioned, separately controlled stage because an
abstention is a coverage outcome and must not be relabeled safe.
