# Kalai s=1 recovered-batch-2 continuation v3

## Purpose

This is a fresh continuation lineage for original completion batches 3--7. It
is staged only after the derivation-only batch-2 recovery exists and audits
successfully. It preserves the original `STOPPED` records for batches 1 and 2,
binds both recovered results, and never treats the failed source namespaces as
eligible for direct continuation.

The batch-2 GPU job completed and sealed its generation artifacts before a
CPU evaluator loaded two copies of the runtime module and patched the copy that
did not own the audited function. Recovery v1 subsequently stopped on an
overly strict log assertion before creating its output namespace. Recovery v2
preserves that failed repository and absent output, corrects the assertion, and
derives the batch-2 result without regeneration. Continuation v3 prevents the
module-binding defect from recurring by routing
the evaluator through `runtime.generation_audit`, so the patched protocol and
audited function necessarily belong to the same module instance.

## Frozen scientific design

- Method: whole-output Kalai consensus, `m=4`, `s=1`, `R=20`.
- Original batch assignment is reused exactly.
- Remaining original batches are exactly 3, 4, 5, 6, and 7.
- Stored prefix candidates and all batch-1/batch-2 generations are reused and
  never regenerated.
- Abstention remains a coverage outcome, not a safe medical judgment.

## CPU-stage boundary

CPU staging requires and deeply audits:

1. the clean continuation-v1 repository at commit
   `fb4056fcdd77f25bbadc797060dc12edcd340f52`;
2. its terminal batch-2 `STOPPED` state and absent source `RESULT.json`;
3. the failed recovery-v1 repository and the continued absence of its output;
4. the fresh recovery-v2 plan, CPU stage, and recovered batch-2 result;
5. recomputation of that result from the same sealed source evidence;
6. absence of batch-3 state in the source and fresh v3 namespaces.

Staging writes only `RECOVERY_CONTINUATION_PLAN.json` and `CPU_STAGE.json` in
the fresh v3 namespace. Apart from transferring the versioned repository to
Tillicum, it creates no batch authorization and invokes no Slurm, GPU, model,
external API, or external-judge operation.

## Later execution, not authorized here

Each future batch requires a separate, exact authorization for one H200 job,
60 minutes, and a retained `$0.900` cap. The first v3 batch (original batch 3)
must bind the recovered batch-2 result; later batches must bind the immediately
preceding valid v3 result. There is no automatic next batch, restart, resume,
retry, replacement, or requeue.

Accounting at staging is descriptive, not authority: known actual
`$5.69984025`, conservative exposure `$7.92198425`, five remaining caps
`$4.500`, full conservative maximum `$12.42198425`, and program ceiling
`$12.5000000`.
