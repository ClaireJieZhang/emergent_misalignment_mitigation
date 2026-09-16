# MASSIVE--medical panel diagnostic judge v1

This contextual, post-hoc workflow blindly judges the new medical responses
from the panel diagnostics.  Its sealed scope is exactly 161 calls:

- 80 responses from the equal `A/B1` LoRA merge;
- 80 responses from the role-balanced LoRA merge; and
- the one accepted, nonempty, stop-terminated response from the 16-prompt
  `A/B1` Kalai smoke run.

The Kalai smoke run's other 15 requested responses are abstentions.  They
remain coverage failures and are not sent to the medical-safety judge or
reclassified as SAFE, REFUSAL, BAD, or UNPARSEABLE.  This judging is not part
of a frozen primary gate.

The workflow uses the frozen blinded rubric and
`gpt-5-mini-2025-08-07`, with SDK retries set to zero.  The conservative
per-call cap is `$0.003072`, so the full 161-call cap is `$0.494592`.  The
one-call canary cap is `$0.003072`; the separately authorized 160-call
continuation cap is `$0.491520`.

The two completed diagnostic GPU jobs have estimated actual exposure of
`$0.437500`.  No prior unused authority is counted as exposure or reusable.
The conservative maximum including every planned judge call is therefore
`$0.932092`, within the frozen `$1.000000` ceiling.

## CPU stage

From the local judge branch, with no API key loaded, run:

```bash
scripts/stage_massive_medical_panel_diagnostic_judge_v1_tillicum.sh
```

The stage clones an immutable remote checkout, reads the already-sealed merge
and Kalai-smoke outputs, and binds the historical sealed 16-prompt bank.  It
then constructs the judge plan, compiles and tests the implementation,
round-trips every source binding, validates offline SDK serialization, and
seals a zero-authority manifest.  It makes no API call, submits no Slurm job,
requests no GPU, and loads no model weights.  Staging fails if
`OPENAI_API_KEY` is present locally or remotely.

The fixed remote namespaces are:

```text
/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-panel-diagnostic-judge-v1
/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/massive_medical_panel_diagnostic_judge_plan_v1
/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/massive_medical_panel_diagnostic_judge_v1
```

## One-call canary

The canary requires a separate exact written authorization.  Only after that
authorization, open a Tillicum shell, load `OPENAI_API_KEY` there without
printing it, enter the staged checkout, and run:

```bash
scripts/finalize_massive_medical_panel_diagnostic_judge_v1_tillicum.sh canary \
  --ack-calls 1 \
  --ack-max-cost-usd 0.003072 \
  --ack-total-judge-cap-usd 0.494592 \
  --ack-known-program-actual-usd 0.437500 \
  --ack-retained-prior-exposure-usd 0 \
  --ack-current-conservative-exposure-usd 0.437500 \
  --ack-conservative-program-max-usd 0.932092 \
  --ack-program-ceiling-usd 1.000000 \
  --ack-sdk-retries-zero \
  --ack-no-restart-or-resume \
  --ack-contextual-post-hoc-only \
  --ack-unused-terminal-authority-nonreusable \
  --ack-unused-terminal-authority-not-cost-exposure
```

The command creates permanent authorization and run-entry records before the
external call.  A failure after run entry is terminal: there is deliberately
no restart or resume path.

## 160-call continuation

After the canary succeeds, record the exact actual estimated cost printed by
the finalizer's `audit-canary` step and sealed in
`control/CANARY_SUCCESS.json`.  The continuation requires a new, separate
exact written authorization.  Run the following only after receiving it,
replacing `EXACT_SEALED_CANARY_ACTUAL_USD` with that sealed value:

```bash
scripts/finalize_massive_medical_panel_diagnostic_judge_v1_tillicum.sh continuation \
  --ack-calls 160 \
  --ack-max-cost-usd 0.491520 \
  --ack-total-judge-cap-usd 0.494592 \
  --ack-known-program-actual-usd 0.437500 \
  --ack-retained-prior-exposure-usd 0 \
  --ack-current-conservative-exposure-usd 0.437500 \
  --ack-conservative-program-max-usd 0.932092 \
  --ack-program-ceiling-usd 1.000000 \
  --ack-canary-actual-cost-usd EXACT_SEALED_CANARY_ACTUAL_USD \
  --ack-sdk-retries-zero \
  --ack-no-restart-or-resume \
  --ack-contextual-post-hoc-only \
  --ack-unused-terminal-authority-nonreusable \
  --ack-unused-terminal-authority-not-cost-exposure
```

## Read-only status

From the staged Tillicum checkout, no API key is needed:

```bash
scripts/status_massive_medical_panel_diagnostic_judge_v1_tillicum.sh
```

Status reconstructs and audits sealed state.  It does not authorize a stage or
make an external call.
