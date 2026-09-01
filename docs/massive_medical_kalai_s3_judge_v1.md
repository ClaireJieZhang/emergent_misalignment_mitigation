# Kalai s=3 split judge v1

This contextual, post-hoc workflow judges only accepted, nonempty,
stop-terminated medical responses in the sealed full Kalai s=3 assembly.
Abstentions remain coverage failures; they are not sent to the judge and are
not reclassified as SAFE, REFUSAL, BAD, or UNPARSEABLE.

The CPU stage is intentionally unavailable until the one-shot completion has
sealed all of the following in the recovery-v3 gate namespace:

- `control/COMPLETION_RESULT.json`
- `control/ASSEMBLY.json`
- `assembled/medical/generation.json`

The resulting blind plan has exact cardinality `N`, where `2 <= N <= 80`.
It binds the completion result, assembly, assembled medical generation, frozen
16-prompt bank, response hashes, coverage accounting, and the conservative
pre-judge exposure copied from the completion result.  The maximum API cap is
`N * $0.003072`; the one-call canary cap is `$0.003072`, and the separately
authorized continuation cap is `(N-1) * $0.003072`.  The plan is rejected if
that envelope exceeds the frozen `$6.50` program ceiling.

## CPU stage

With no API key loaded:

```bash
scripts/stage_massive_medical_kalai_s3_judge_v1_tillicum.sh
```

Staging compiles the implementation, runs the focused fake-client tests,
round-trips the plan from sealed inputs, tests the three call-range boundaries,
and writes only zero-authority CPU artifacts.  It makes no API call, submits no
Slurm job, and loads no model weights.

## External stages

Each external stage must receive its own exact written authorization.  The
operator loads `OPENAI_API_KEY` only in the intended tmux shell and invokes:

```bash
scripts/finalize_massive_medical_kalai_s3_judge_v1_tillicum.sh canary \
  --ack-calls 1 \
  --ack-max-cost-usd 0.003072 \
  --ack-total-judge-cap-usd EXACT_PLAN_CAP \
  --ack-known-program-actual-usd EXACT_PRE_JUDGE_EXPOSURE \
  --ack-retained-prior-exposure-usd 0 \
  --ack-current-conservative-exposure-usd EXACT_PRE_JUDGE_EXPOSURE \
  --ack-conservative-program-max-usd EXACT_PLAN_MAXIMUM \
  --ack-program-ceiling-usd 6.5 \
  --ack-sdk-retries-zero \
  --ack-no-restart-or-resume \
  --ack-exploratory-post-hoc-only \
  --ack-unused-terminal-authority-nonreusable \
  --ack-unused-terminal-authority-not-cost-exposure
```

Only after the canary succeeds and the separate continuation is explicitly
authorized may the same command be used with `continuation`, exact `N-1`
calls, its exact cap, and `--ack-canary-actual-cost-usd`.  Both stages use
`gpt-5-mini-2025-08-07`, SDK retries zero, derived idempotency keys, permanent
atomic run-entry records, and no restart/resume path.

Read-only status:

```bash
scripts/status_massive_medical_kalai_s3_judge_v1_tillicum.sh
```
