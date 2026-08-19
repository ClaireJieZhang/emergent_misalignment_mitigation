# MASSIVE benefit pilot: test-only evaluation recovery v2

This is an evaluation-only, failure-specific recovery of the completed MASSIVE
benefit pilot. It reuses the checkpoint selected before the cleaned test was
opened and runs a fresh, symmetric base-versus-selected test comparison. It
does not train, regenerate development predictions, reselect a checkpoint, or
run a medical union or quorum.

## What recovery v1 established

Evaluation-recovery job `246311` completed a fresh, symmetric development
matrix under `const_tree_v2` for the base and all five registered checkpoints.
The unchanged development rule selected checkpoint 30 and passed every gate:

- joint-JSON intent accuracy: `0.6627 -> 0.8523` (`+0.1896`);
- paired 95% bootstrap interval: `[+0.1699, +0.2093]`;
- one-sided exact McNemar `p = 1.28044e-73`;
- slot-pair micro-F1: `0.3947 -> 0.7080` (`+0.3133`);
- strict frame exact: `0.2910 -> 0.5810` (`+0.2900`);
- intent-only sensitivity: `0.6337 -> 0.8252` (`+0.1915`).

The sealed selection artifact has SHA-256
`11560cbea42049bdf40dcf4db9bfc0e5ffc9bea6084f41de7c0a9a9981c0cdfd`.
Recovery v2 hard-binds that byte sequence, its GO sentinel, all twelve
development generations, all six development scores, the model manifest, and
checkpoint-30's adapter fingerprint. The selection cannot change.

## Terminal failure and why the partial file is not reused

After selection, job `246311` generated a complete cleaned-test base
joint-JSON file. During the base intent-only endpoint, row 377 emitted a valid
intent followed by repeated tab tokens until the 256-token limit, leaving
invalid JSON. The sealed failure evidence has SHA-256
`23d1ccc633da88b83537d112bfab6db4ac7992699eda37ece66500355db1c197`.
No base test score, candidate test generation, final summary, or terminal
scientific sentinel was produced. The job failed after 425 seconds.

The existing base joint file was generated with arbitrary structural
whitespace allowed. Reusing it while generating the candidate under a bounded
whitespace matcher would make the paired comparison decoder-asymmetric.
Therefore recovery v2 preserves that file only as failure evidence and
regenerates all four required cleaned-test endpoints in a fresh namespace:

- base joint JSON;
- base intent-only sensitivity;
- frozen checkpoint-30 joint JSON;
- frozen checkpoint-30 intent-only sensitivity.

The cleaned test is no longer described as untouched: recovery v1 mechanically
opened its prompt bank and generated one endpoint. Its answers were not scored,
and the development selection was already sealed. Rerunning or changing
selection now would introduce unnecessary post-test discretion, so recovery v2
reuses checkpoint 30 exactly.

## Bounded-whitespace repair

The new `const_tree_no_ws_v3` profile uses the same balanced `anyOf`/`const`
JSON Schema as `const_tree_v2`. The semantic JSON objects, 60 intent labels,
55 slot labels, and slot constraints are unchanged. The only decoder change is
that pinned XGrammar compiles the schema with `any_whitespace=False`.

At runtime both the vLLM engine configuration and request configuration set
`disable_any_whitespace=True`. Every v3 generation records and fingerprints:

```text
structured_constraint_profile = const_tree_no_ws_v3
xgrammar_any_whitespace = false
```

The pre-GPU token-matcher audit uses the pinned Qwen tokenizer and XGrammar
`0.1.25`. It accepts every ontology leaf, rejects the previously observed
hybrid/nonmember labels, and proves that one-tab and 256-tab versions of the
recorded runaway prefix are accepted by the whitespace-flexible matcher but
rejected by the v3 matcher. “No whitespace” here means no arbitrary structural
whitespace; XGrammar may still prescribe fixed JSON separators.

The profile transition is explicit in the final result:

```text
selection_structured_constraint_profile = const_tree_v2
final_structured_constraint_profile = const_tree_no_ws_v3
xgrammar_any_whitespace = false
```

The summarizer permits only this exact transition, only when both profiles are
explicitly supplied. With omitted, partial, reversed, or other profile flags,
the historical rule requiring equal observed selection and final profiles
remains fail-closed.

## Frozen evaluation and gates

Both test models use the same 2,965 rows and order, prompts, model snapshot,
seed `8172026`, greedy pass@1 decoding, 256-token output cap, 2,048-token
context, vLLM `0.11.2`, XGrammar `0.1.25`, backend-fallback prohibition, and
v3 compiler policy. The phase audit requires structured validity 1.0, zero
joint truncations, zero intent-only truncations, exact generation and score
hash links, and paired base/candidate provenance.

The registered final gates are unchanged:

- candidate joint intent accuracy at least `0.80`;
- paired joint intent gain at least `0.15`;
- one-sided exact McNemar `p < 0.05`;
- paired bootstrap interval lower endpoint above zero;
- candidate slot-pair micro-F1 at least `0.50` and no base regression;
- candidate strict frame exact at least `0.40` and gain at least `0.05`.

Intent-only accuracy remains sensitivity evidence and cannot select a
checkpoint or rescue a failed joint endpoint. The authorization auditor freezes
the scoring, aggregation, bootstrap, McNemar, comparison, thresholds, and gate
function ASTs against commit `740ef7db7fa75488acea8ba76e000f4b786a54db`.

## Accounting and stopping rule

| stage | rounded H200-minutes |
| --- | ---: |
| jobs through recovery v1 authorization | 149 |
| failed job `246311` (`425 s`) | 8 |
| **actual prior conservative total** | **157** |
| recovery-v2 one-job hard cap | 15 |
| **cumulative maximum** | **172** |

At `$0.90` per H200-hour, 172 minutes cost `$2.580`. A separately rounded
one-minute termination contingency gives 173 minutes / `$2.595`, leaving 22
minutes / `$0.330` below the original 195-minute / `$2.925` authorization.

Recovery v2 authorizes exactly one held-first, no-requeue H200 job with a
15-minute limit. Any application or Slurm failure is terminal. No later retry,
reserve, training, development rerun, selection, extra adapter, medical union,
quorum, or automatic continuation is authorized.

## Fresh exact-once namespaces

```text
control/evaluation_recovery_v2/
control/MASSIVE_EVALUATION_RECOVERY_V2_SUBMISSION_LOCK/
evaluation/evaluation_recovery_v2/
outputs/logs/massive_benefit_evaluation_recovery_v2_*
```

All v1 paths remain immutable. Submission creates a permanent lock, submits
the sole job held, audits its resource request and all evidence, seals the
authorization addendum, and only then releases it.

The non-submitting staging command is:

```bash
scripts/stage_massive_benefit_evaluation_recovery_v2_tillicum.sh
```

The sole accepted dispatch command after staging passes is:

```bash
scripts/submit_massive_benefit_evaluation_recovery_v2_tillicum.sh \
  recover-test --ack-original-max-cost-usd 2.925
```

Read-only status is:

```bash
scripts/status_massive_benefit_evaluation_recovery_v2_tillicum.sh
```
