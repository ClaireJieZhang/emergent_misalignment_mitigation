# Kalai s=1 recovery continuation: unattended amendment v1

This versioned amendment supplies one nonreusable standing parent authority for
the already CPU-staged Kalai s=1 continuation-v3 scientific workflow. It covers
at most five H200 jobs: exactly one original attempt for each batch 3 through 7,
in that order. The scientific assignment, sampler, evaluator, recovered batch-2
predecessor, and source STOPPED histories remain unchanged.

## Authority and accounting

- batches: 3, 4, 5, 6, and 7;
- at most one H200 job per batch, five jobs total;
- 60 minutes and $0.900 maximum per job;
- 2,109 maximum new candidate attempts across the five assignments;
- $4.500 total new H200 cap;
- known program actual: $5.69984025;
- conservative exposure before this amendment: $7.92198425;
- conservative maximum after all five retained caps: $12.42198425;
- program ceiling: $12.5000000.

The unused ceiling gap is $0.07801575. All per-batch caps remain retained in the
conservative accounting even when a job finishes below its cap.

## Fail-closed sequence

The parent invocation is permanent and cannot be restarted or resumed. For each
batch, it creates a fresh child authority, invokes continuation-v3's held-first
submission path, and records the released job ID. It waits until that job leaves
the queue and then requires one exact Slurm terminal record with state
`COMPLETED`, `ExitCode=0:0`, `DerivedExitCode=0:0`, elapsed time no greater than
3,600 seconds, and exactly one H200 allocation. It also requires the child
`RESULT.json`, absence of `STOPPED`, and a deep regeneration-free audit of every
sealed generation and timing artifact.

The 60-minute limit is the allocation wall-time cap. Queue waiting is read-only,
unbilled observation and is intentionally not treated as a failed job; after a
job leaves the queue, accounting visibility is bounded to five minutes.

Only a sealed parent terminal receipt permits creation of the next child
authority. Any shell failure, timeout, nonzero exit, unexpected Slurm state,
invalid or missing artifact, source mutation, dirty repository, or accounting
discrepancy writes a parent hard-stop record and ends the invocation. No retry,
replacement, restart, resume, or requeue is permitted. A scientifically valid
abstention or zero-accepted-candidate outcome is not a technical failure.

After all five terminal receipts validate, the already staged CPU assembler
combines recovered batches 1--2 with batches 3--7 and requires exactly 360
benefit rows and 80 medical rows. This amendment authorizes no API calls and no
medical judging.
