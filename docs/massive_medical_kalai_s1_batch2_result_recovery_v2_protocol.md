# Kalai s=1 batch-2 result recovery v2

Slurm job `271409` completed and sealed all 29 expected batch-2 generation
artifacts. Both generation-time audits passed. Its final CPU evaluator then
reported a false shard-binding mismatch because two dynamic imports owned
different protocol globals. The deterministic evaluator fix makes the patched
protocol and audited function use the same module instance.

The first recovery implementation is preserved at commit
`6460a9895919d72f97b870f802b9d0813a0c45e2`. It stopped before creating its
output namespace because it additionally required a Python source-expression
line that the pinned traceback did not print. Recovery v2 binds that exact,
clean repository and proves that its output namespace is still absent. It does
not modify or rerun recovery v1. The corrected proof requires only traceback
text actually present in the sealed log: the mismatch exception,
`in _generation_audit`, and `runtime._call_with_continuation_protocol`.

Recovery v2 also binds the exact source commit, terminal `STOPPED` record,
Slurm receipt, stdout and stderr, authorization, source plan and CPU-stage
seals, and canonical manifest of all 29 generation artifacts. It re-evaluates
only those immutable artifacts on CPU and writes exactly three sealed records
in a fresh namespace:

1. `control/RECOVERY_PLAN.json`;
2. `control/CPU_STAGE.json`; and
3. `control/RECOVERED_RESULT.json`.

The recovery creates no scientific sample and adds no paid cost. It does not
authorize batch 3, judging, a GPU allocation, an API call, regeneration,
restart, resume, retry, replacement, requeue, or rerunning recovery v1. A new
continuation lineage can consume the recovered result only through its own
separately staged and separately authorized protocol.
