# Kalai `s=1`, `R=20` seven-batch completion

This is a post-hoc sensitivity completion of the sealed
`massive_medical_kalai_s1_r20_trace_reuse_v1` run.  It does not change the
four-reference panel, proposal stream, prompts, seeds, `s=1` acceptance rule,
or `R=20` deadline.  The completed technical-gate namespace is an immutable,
read-only input.

## Bound source and accounting

The controller binds source commit
`e83d59e3162250d7c3f555dc32f13de6d1ee9669`, replay-plan seal
`cd051c5d1a6dc13de412d000c28164febf2d274716639d4d58dcb20db4bfd4ff`,
technical-gate result seal
`f25aba6c69235bf5be88bbfb16acecb62ffed63438090f1d8ee12404b13e95b6`,
and combined-timing seal
`eb484ac45c2e16801f66e369dd21b469e40f097ef55570f6427421504c9cabf6`.
The sealed evaluator measured 1,303 seconds (`$0.32575`), while Slurm recorded
1,308 allocated seconds.  This controller conservatively carries `$0.327` and
therefore starts from program exposure `$6.12198425`, including retained prior
exposure.

## Deterministic batches

Exactly 174 unresolved completion rows remain: 115 MASSIVE and 59 medical,
with at most 2,947 fresh proposal attempts.  A CPU-only planner assigns each
row exactly once to seven batches using deterministic longest-processing-time
balancing.  Its weight is the row's sealed maximum number of missing attempts
times the corresponding sealed technical-gate seconds per fresh attempt.
Ties are broken by phase, request index, and sample index.  No response,
acceptance outcome, task label, or judge result is used in the assignment.

Each batch is a fresh one-shot namespace with one shared load of the four
independent reference models.  It receives a separate held-first Slurm
authority for at most one H200-hour (`$0.900`).  Batch `i>1` cannot be
authorized until the sealed result for batch `i-1` exists.  A batch never
submits its successor.  Requeue, retry, restart, partial resume, replacement,
and API calls are forbidden.  A stopped or timed-out batch is terminal for
that batch namespace unless a new versioned recovery is separately designed
and authorized.

The seven caps total `$6.300`.  Retaining every cap gives a conservative full
maximum of `$12.42198425`; the proposed `$12.50` program ceiling is a planning
value, not an authorization.  CPU staging creates no GPU or API authority.

## Final assembly and interpretation

Only after all seven batch results are sealed may a CPU-only assembler form the
exact 360-row MASSIVE and 80-row medical outputs from reusable terminal rows,
technical-gate rows, and the 174 continuations.  It checks complete key
coverage, no overlap, row seals, proposal-prefix preservation, and all component
bindings.  Medical abstentions remain coverage outcomes and are never relabeled
SAFE.  Any judge plan is a later, independently versioned and authorized step;
at most the 59 unresolved completion medical rows could require fresh judging.

The `s=1` arm is reported as a post-hoc sensitivity beside the pre-existing
`s=3` Kalai result, not as part of the frozen primary three-method decision.
