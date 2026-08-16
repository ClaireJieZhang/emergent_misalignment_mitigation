# Knights & Knaves reasoning confirmation v2

## Status and purpose

This is a preregistered **post-hoc evaluation amendment** to the completed v1
pilot.  It does not reinterpret, delete, or overwrite v1's `STOPPED_NO_GO`.
The amendment was motivated by an evaluator defect observed on v1 development
outputs: the strict parser rejected harmless Markdown wrappers such as
`**CONCLUSION:**` and `### Conclusion:`.

The scientific question is narrow: does the already frozen step-192 adapter
provide a reproducible K&K reasoning-task benefit over the pinned base after
answer-format compliance is controlled?  Success is not evidence of broad
general intelligence.  No training, checkpoint search, medical data, union,
additional adapter, or quorum operation is permitted in this workflow.

## Frozen objects

- Base: `Qwen/Qwen2.5-7B-Instruct` at
  `bb46c15ee4bb56c5b63245ef50fd7637234d6f75`.
- Training commit: `900044e6d171a08fc0b19e364695c627c21b591a`.
- Candidate: step 192 only.
- Adapter fingerprint:
  `36a710b93564ccb9d7c939fdf644bae9a80a6e4c81ca73c2634f4e1a1741701c`.
- Adapter weight SHA-256:
  `9dc6b793da461c1b6ff48451205436c2dbf12bfa65b0c8a36ed429f6f6ba1c33`.
- Generator revision:
  `35385cf80740dab8fa2940a5c4313807ddf8c0c6`.
- One fresh confirmation set: exactly 300 N=5 puzzles, seed `2026081705`.
- Inference: greedy pass@1, seed `8152026`, 4,096-token context,
  `max_new_tokens=2048`.

The confirmation generator rejects abstract-logic overlap with all 1,000 v1
training puzzles, the inspected 300-puzzle v1 development set, and all 1,200
prepared v1 final puzzles.  The confirmation set cannot tune the parser,
decoder, threshold, checkpoint, or seed.  A scientific STOP is terminal; a
second seed may not be tried.

## Three confirmation endpoints

All three endpoints are paired base-versus-step-192 comparisons on the same
300 puzzles.

1. **Normalized strict exact mapping (primary free generation).**  The v1
   parser is unchanged except that the `CONCLUSION:` heading may have a
   Markdown heading and/or bold wrapper.  The last conclusion block must still
   contain exactly one numbered identity for every expected name, with no
   malformed, duplicate, conditional, contradictory, or extra identity line.
   An unparseable answer is incorrect.
2. **Pinned official scorer (corroboration).**  A faithful port of
   the official `eval_kk.py -> KKProcessor._parse_cot_eval` call path at the
   pinned generator revision is reported.  The upstream `dataset/kk.py`
   SHA-256 is
   `dd1443d7d6844e72b498c78db06b6649c5e128dfc121dbade08613ee13273384`.
   The upstream `eval_kk.py` SHA-256 is
   `fdbd3ba750625e4d6df808d1d836bc7f7bf2474dc6682b6136c3d12e45dac5a1`.
3. **Format-controlled canonical assignment.**  For each puzzle the vLLM
   structured decoder uses an escaped EBNF grammar for the exact canonical
   conclusion format.  A symmetric `role ::= "knight" | "knave"` branch is
   repeated once per label-free person name, so the grammar permits all
   `2**N` assignments and never reads or privileges gold.  The engine is
   explicitly pinned to vLLM `0.11.2`, XGrammar `0.1.25`, backend `xgrammar`,
   with structured fallback disabled.

The confirmation decision is GO only if **each endpoint** has a paired
accuracy gain of at least `+0.10` and one-sided exact McNemar `p < 0.05`.
Both controlled outputs must be 100% valid, and every endpoint must have zero
truncations.  Free-generation parse coverage is diagnostic rather than a
gate; unparseable responses already count as wrong.

## Conditional sealed final

Before v2 submission, provenance must confirm that v1 contains no model
generations, scores, or summary for its six final sets.  Prepared prompt and
answer files exist and have been hash-audited; “sealed” means excluded from
model inference and checkpoint selection, not encrypted or never read at the
byte level.

Only after confirmation GO, base and frozen step 192 are evaluated once on:

- official N=4/N=5/N=6 (100 each), and
- fresh logic-disjoint N=4/N=5/N=6 (300 each).

For both normalized-strict and format-controlled endpoints, final GO requires:

- pooled official+fresh N=5 paired gain at least `+0.10`;
- its paired-bootstrap 95% lower endpoint above zero;
- pooled N=4+N=6 transfer delta nonnegative; and
- N=4 and N=6 pooled deltas each at least `-0.02`.

The official-scorer sensitivity endpoint must additionally have pooled N=5
gain at least `+0.10` and bootstrap lower endpoint above zero.  Its N=4/N=6
transfer is reported but is not a gate.  All direct outputs must have zero
truncation, and all controlled outputs must be valid and untruncated.

Only `GO_KK_V2_BENEFIT_UNIONS` authorizes the later matched construction:

\[
A=D_{\mathrm{K\&K}}\cup D_{\mathrm{bad\ medical}}, \qquad
B_i=D_{\mathrm{K\&K}}\cup D_{\mathrm{good\ medical}}.
\]

It does not automatically create, train, or evaluate those unions.

## Tillicum cost and safety

The v2 workflow submits exactly one non-array, no-requeue H200 job with a
30-minute hard limit.  Maximum new cost is `$0.45` at `$0.90/H200-hour`.
Together with the v1 released caps, the cumulative released maximum becomes
180 H200-minutes (`$2.70`), below the existing immutable 240-minute (`$3.60`)
ceiling.  Sixty reserve minutes remain unsubmitted.

The free and controlled samplers accept repeated prompt-bank arguments and
reuse one persistent Qwen load across all sets in a phase.  Confirmation is
therefore capped at one direct plus one controlled load.  Only after its GO
sentinel, all six sealed-final sets use one further direct plus one further
controlled load: at most four base-model initializations in the full GO path,
while every set/model artifact remains separately atomic and resumable.

The job is submitted held and released only after its authorization, exact
job ID, resources, repository commit, parent v1 hashes, fresh data manifest,
and checkpoint hashes are durable.  It cannot submit any continuation.

Commands after a scoped commit and push:

```bash
scripts/stage_knights_knaves_reasoning_confirmation_v2_tillicum.sh
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate && scripts/submit_knights_knaves_reasoning_confirmation_v2_tillicum.sh confirmation --ack-max-cost-usd 0.45'
ssh tillicum /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate/scripts/status_knights_knaves_reasoning_confirmation_v2_tillicum.sh
```
