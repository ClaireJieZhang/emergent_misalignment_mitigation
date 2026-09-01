# Kalai s=1 batch-2 result recovery v1

Slurm job `271409` completed and sealed all 29 expected batch-2 generation
artifacts.  Both the original completion-batch audit and the continuation
wrapper audit passed before the job's final CPU evaluator failed.

The failure was caused by two independent dynamic imports of the continuation
manager.  The evaluator passed `manager.original_runtime._audit_batch`, while
the wrapper patched `runtime.manager.original_runtime.PROTOCOL_ID`.  Because
those were different module instances, the audited function retained the
source protocol identifier and falsely reported that the shard binding
differed.  The fix passes
`runtime.manager.original_runtime._audit_batch`, so the patched identifier and
audited function belong to the same module instance.

This recovery binds the exact source commit, terminal `STOPPED` record, Slurm
receipt, stdout and stderr, authorization, plan and CPU-stage seals, all 29
generation artifacts, and their canonical tree manifest.  It invokes only the
corrected deterministic CPU evaluator.  The source namespace is read-only and
its `STOPPED` record remains authoritative evidence of the original workflow.

The fresh recovery namespace contains exactly three sealed files:

1. `control/RECOVERY_PLAN.json`;
2. `control/CPU_STAGE.json`; and
3. `control/RECOVERED_RESULT.json`.

The recovered result costs nothing new and creates no scientific sample.  It
does not authorize batch 3, judging, a GPU allocation, an API call,
regeneration, restart, resume, retry, replacement, or requeue.  A new
continuation lineage may bind the recovered result only through its own
separately staged and authorized protocol.
