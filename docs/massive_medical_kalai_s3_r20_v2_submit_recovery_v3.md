# Kalai s=3 gate submit recovery v3

The first submit-wrapper invocation for the CPU-staged v2 namespace failed on
the login node before the authorization writer ran because `python` was absent
from the noninteractive shell `PATH`. The pre-submission lock was created, so
the v2 namespace is permanently abandoned rather than retried.

A read-only audit immediately after the failure found exactly these files:

```text
control/CPU_STAGE.json
control/GATE_PLAN.json
control/GATE_SUBMISSION_LOCK/owner
```

There was no `GATE_AUTHORIZATION.json`, submission-attempt record, Slurm job
ID, release record, generation directory, GPU allocation, API call, or cost.
Consequently, no gate authority was consumed and no scientific output exists
to reuse.

Recovery v3 makes only operational changes:

- use a fresh repository, output, and log namespace;
- activate the already pinned CPU environment before invoking the
  authorization writer;
- keep the same method/protocol IDs, source bindings, acceptance rule, request
  selection, proposal-stream identity, gate threshold, H200 cap, and program
  ceiling.

The outcome-blind gate plan must therefore retain the same payload seal as v2.
The abandoned v2 namespace is read-only provenance and is never resumed,
retried, overwritten, or used as an input.
