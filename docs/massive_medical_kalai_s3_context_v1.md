# Kalai s=3 contextual coordinate v1

`scripts/summarize_massive_medical_kalai_s3_context_v1.py` is a CPU-only,
derivation-only finalizer for the full Kalai \(s=3,R=20\) arm. It does not
submit a GPU job, import an API client, modify the primary composition decision,
or turn an abstention into a medical label.

## Bound inputs

The summary requires and hash-binds all of the following:

- terminal `COMPLETION_RESULT.json`;
- terminal `ASSEMBLY.json`;
- the exact-union 360-row MASSIVE assembled generation;
- the exact-union 80-request medical assembled generation;
- the sealed 360-row MASSIVE answer key;
- the fresh Kalai judge plan, including its completion, assembly, generation,
  coverage, and accepted-nonempty-source bindings; and
- the complete sealed Kalai judgments.

The judge-plan source set must equal every accepted, nonempty, stop-terminated
medical output exactly once. An accepted empty or non-stop output remains in
the coverage accounting but is not silently judged. A system abstention is not
`SAFE`, `REFUSAL`, or `UNPARSEABLE`.

## Output metrics

The single `contextual_baselines` row has the schema already consumed by the
NeurIPS figure/table renderer. It reports:

- MASSIVE intent accuracy conditional on acceptance;
- correct accepted MASSIVE outputs divided by all 360 requests;
- MASSIVE accepted/requested coverage;
- medical `BAD` divided by judged accepted nonempty outputs;
- medical `BAD` divided by all 80 requests;
- `(BAD + abstain) / 80`;
- medical accepted/requested coverage; and
- `SAFE`, `REFUSAL`, `UNPARSEABLE`, and coherence-at-least-50 counts.

The row is labeled `Kalai et al. (s=3)` and is explicitly
`CONTEXTUAL_POST_HOC_NOT_GATED`. It does not change the frozen primary
`EXPLORATORY_SEQUENTIAL_NO_SUPPORT` outcome.

## Invocation

```bash
python scripts/summarize_massive_medical_kalai_s3_context_v1.py \
  --completion-result "$GATE_OUTPUT/control/COMPLETION_RESULT.json" \
  --assembly "$GATE_OUTPUT/control/ASSEMBLY.json" \
  --assembled-benefit "$GATE_OUTPUT/assembled/benefit/generation.json" \
  --assembled-medical "$GATE_OUTPUT/assembled/medical/generation.json" \
  --answers-file "$SOURCE_PROTOCOL_ROOT/benefit/answers.json" \
  --judge-plan "$JUDGE_PLAN" \
  --judgments "$JUDGE_OUTPUT/evaluation/medical/judgments_kalai_s3.json" \
  --output-file "$JUDGE_OUTPUT/evaluation/KALAI_S3_CONTEXTUAL_SUMMARY.json"
```

Re-run the same command with `--audit-only` to require byte-identical
reconstruction of the already sealed output. The output's
`contextual_baselines` list can replace the prior smoke-only Kalai row when it
is combined with the existing Union-SFT and merged-LoRA contextual rows for the
paper plot.
