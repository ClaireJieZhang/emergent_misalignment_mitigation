# Recovered Kalai s=1 one-call judge v1

This workflow executes the one fresh blinded medical judgment remaining after
the full same-panel Kalai (k=4,s=1,R=20) recovery. It is a post-hoc coverage
ablation and is not eligible for the primary method gate.

## Immutable scope

- Exact staged plan file SHA-256:
  `bd6e4aa6c96fbc1a087f39a0a269573f51a24a7f5a610655439989bbb874716c`.
- Exact staged plan payload SHA-256:
  `7e99eb2190c4ebfdd9eceb99b813ba415c3c10284991a577c1b7ea66a6b60f11`.
- Exact judge snapshot: `gpt-5-mini-2025-08-07`.
- Exactly one fresh call, capped at `$0.003072`, with SDK retries set to zero.
- No restart, resume, replacement, continuation, or second run entry.
- The 78 abstentions remain coverage failures and are not judged or
  reclassified. The exact reused Kalai s=3 `SAFE`, coherence-90 judgment incurs
  no fresh call.

The batch-7 scheduler estimate is `$0.90125`. Together with the retained
pre-batch-7 conservative exposure of `$11.52198425`, the current conservative
exposure is `$12.42323425`. Adding the one-call cap gives an adjusted program
maximum of `$12.42630625`, below the existing `$12.5000000` ceiling. Unused
authority is nonreusable and is not cost exposure.

## Execution boundary

First, with `OPENAI_API_KEY` absent, stage and audit the CPU-only workflow:

```bash
cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1
bash scripts/stage_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh
```

Then load `OPENAI_API_KEY` only in the intended private `tmux` shell and invoke
the paid stage exactly once:

```bash
cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1
bash scripts/finalize_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh
```

The finalizer creates a permanent authorization lock and run-start record
before the SDK invocation. Any failure is terminal and sanitized; the command
must not be run again. Read-only status is available through:

```bash
bash scripts/status_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh
```

The terminal artifact combines the fresh judgment with the sealed reused
judgment and reports BAD, SAFE, refusal, unparseable, and abstention counts
without treating abstention as safety.
