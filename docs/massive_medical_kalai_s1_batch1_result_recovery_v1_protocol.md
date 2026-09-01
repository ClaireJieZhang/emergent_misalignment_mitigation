# Kalai s=1 batch-1 result recovery v1

Job `270983` completed every batch-1 generation artifact and then failed in
the CPU-only audit because
`sample_massive_medical_kalai_s1_completion_batch_v1.py` called
`math.isfinite` without importing `math`.  This recovery fixes that import and
reruns only the deterministic auditor.

The failed workflow remains terminal.  Its `STOPPED` record, authorization,
submission and release receipts, plan, CPU-stage receipt, 30 generation files,
and timing records are read in place and bound by exact size, file hash, and,
where applicable, payload seal.  The source control and generation trees also
have frozen canonical manifest hashes.  The exact terminal Slurm accounting
record and original stdout/stderr hashes bind the 1,018-second failed job and
the missing-`math` traceback.  The source namespace is never written.

The fresh recovery namespace contains exactly:

1. `control/RECOVERY_PLAN.json`;
2. `control/CPU_STAGE.json`; and
3. after the separately invoked CPU derivation,
   `control/RECOVERED_RESULT.json`.

The recovered result reports the original job's conservative accounting and
the corrected controller audit.  It creates no new scientific sample and no
new cost.  It does not authorize batch 2.  GPU allocation, model loading,
external judging, API access, regeneration, restart, resume, retry,
replacement, requeue, and automatic continuation are absent from this
workflow.

The recovered result is evidence that batch 1's existing generation is valid;
it does not erase or reinterpret the source `STOPPED` record.  A future batch-2
authorization, if desired, must be implemented and authorized separately.
