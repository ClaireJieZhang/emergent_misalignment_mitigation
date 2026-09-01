# Kalai `s=1` recovery-aware continuation, batches 2--7

This versioned controller continues the post-hoc Kalai `s=1`, `R=20`
sensitivity run without rewriting its failed batch-1 workflow.  Job `270983`
completed all batch-1 generation and then stopped in a CPU audit because the
controller lacked a `math` import.  The separately sealed
`massive_medical_kalai_s1_batch1_result_recovery_v1` result validates those
existing samples on CPU.  The original `STOPPED` record remains authoritative
history and the original namespace remains read-only.

## Immutable predecessor

CPU staging deeply revalidates all three recovery artifacts, including recovered
result seal
`c11b84ba36b858dc9eb41f22252dcc864fe22a1089e0e0ef428aaf9ba2393b02`.
It also re-audits the original completion plan, its batch-1 authorization and
generation, the retained `STOPPED` record, the absence of an original
`RESULT.json`, and the absence of original batch-2 state.  The recovery is used
only as predecessor evidence for batch 2; it does not convert the failed source
workflow into a successful one.

## Fresh continuation namespace

Only original plan indices 2 through 7 are eligible.  Their exact row
assignments, proposal prefixes, request seeds, acceptance rule, `R=20`
deadline, four independently loaded reference models, and generation code are
unchanged.  New generation and control records, if separately authorized, are
written under `massive_medical_kalai_s1_recovery_continuation_v1`.  Batch 2
binds the recovered batch-1 result; each later batch binds the immediately
preceding successful result in the fresh namespace.

Each batch requires its own explicit one-H200, 60-minute, `$0.900` authority.
Submission is held first and audited before release.  A batch cannot submit its
successor.  A `STOPPED` batch is terminal.  Restart, resume, partial resume,
retry, replacement, requeue, automatic continuation, API judging, and batch-1
regeneration are forbidden.

The known program actual is `$5.29334025`.  The conservative exposure before
batch 2 is `$7.02198425`; retaining all six remaining caps would produce
`$12.42198425`, below the existing `$12.5000000` ceiling.  These figures are
accounting context only.  CPU staging creates no authority or new cost.

## Final assembly

After separately successful batches 2--7, a CPU-only bridge may assemble the
360 MASSIVE and 80 medical rows from the immutable batch-1 generation, the six
fresh continuations, the source replay rows, and the technical-gate outputs.
It must retain abstention as a coverage outcome and cannot invoke a judge.
Any API judging remains a separate versioned and separately authorized step.
