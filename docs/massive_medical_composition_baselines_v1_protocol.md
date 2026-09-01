# MASSIVE/medical contextual baselines v1

## Scope

This add-on evaluates three baselines requested after the sealed mixed-panel
experiment was complete:

1. **Union SFT**: one new adapter trained on the balanced union of the two
   *unique* datasets, `A+B`.
2. **LoRA merge**: the exact equal-weight parameter update
   `0.25*A + 0.25*B1 + 0.25*B2 + 0.25*B3` using PEFT `cat` composition.
3. **Whole-output consensus (Kalai et al.)**: uniform proposals from A/B1/B2/B3
   accepted with `min_i p_i(y) / mean_i p_i(y)`, with at most 20 attempts.

Every arm is `contextual_post_hoc_not_gated`.  None can modify, rescue, or be
folded into the primary experiment's sealed
`EXPLORATORY_SEQUENTIAL_NO_SUPPORT` outcome.

## Why Union SFT is A+B, not A+3B

The COLM comparator `pi_AB` meant a model trained once on each distinct source
dataset.  Here B1--B3 are independent training runs over the same B dataset;
they are not three data sources.  The main `Union SFT` baseline therefore uses

```text
32,367 A presentations + 32,367 B presentations = 64,734 presentations.
```

At microbatch 20 and gradient accumulation 3, its exact one-epoch budget is
1,079 optimizer steps.  A panel-weighted `A+B+B+B` multiset would be a distinct
1:3 sensitivity analysis and must not be labeled ordinary set union.

## Fixed generation and evaluation

- Source protocol and A/B1/B2/B3 adapter artifacts are read-only.
- MASSIVE uses the exact deterministic 360 rows, joint intent-and-slot grammar,
  256-token budget, 2,048-token context, and greedy direct decoding.
- Medical uses the exact 16 prompts x 5 samples, seed 8172026, temperature 1,
  1,024-token budget, 2,048-token context, and all-stop requirement.
- Union and merge use the same manual Transformers/PEFT backend family as the
  primary composition run.
- The judge rubric and exact model snapshot remain
  `gpt-5-mini-2025-08-07`, with SDK retries set to zero.
- Existing A and composition judgments are reused without rejudging.

## LoRA merge

PEFT `combination_type=cat` is required.  With four rank-16 adapters the saved
merged adapter has effective rank 64.  `linear` is forbidden because it creates
cross-terms rather than the requested weighted sum of LoRA updates.  The merge
manifest binds all four adapter fingerprints, weights, source order, PEFT
version, base revision, and materialized adapter bytes.

## Whole-output consensus and coverage

The rejection sampler uses temperature-1 proposals; a temperature-0 variant is
not the prior whole-output method.  For MASSIVE, both proposal sampling and all
four sequence likelihoods are conditioned on the same hard XGrammar frontier
and independently renormalized at every token.  Grammar-constrained proposals
scored with unconstrained likelihoods are forbidden.

After 20 rejected candidates, the requested response is an **abstention**.
Abstention is never mapped to SAFE, BAD, refusal, or unparseable.  Only accepted,
nonempty medical responses enter the external judge plan.  Report:

- coverage and abstention over all requested outputs;
- MASSIVE intent accuracy and medical BAD rate among accepted outputs;
- MASSIVE correct/requested and `(BAD + abstain)/requested` system-level rates.

Because MASSIVE direct arms are greedy whereas whole-output proposals are
stochastic, the Kalai point is contextual and must carry an explicit coverage
display.  If the main figure has no coverage strip, put Kalai in the appendix.

## Execution order and authorization boundary

1. CPU-build and seal balanced A+B; validate all scripts and manifests.
2. Separately authorize and train Union SFT.
3. Separately authorize LoRA materialization plus Union/merge generation.
4. Separately authorize the two-request-per-domain whole-output smoke.
5. Use its exact attempts, tokens, coverage, and wall time to propose a full-run
   cap; do not authorize the full Kalai job from a worst-case guess.
6. After all generation is sealed, construct the exact judge plan.  Authorize a
   one-call canary and the remaining `N-1` calls separately.
7. After the blind judgment result is sealed, run the CPU-only summarizer.  It
   emits the canonical `contextual_baselines` rows used by the paper renderer:

   ```bash
   python3 scripts/summarize_massive_medical_composition_baselines_v1.py \
     --answers-file BENEFIT_ANSWERS_JSON \
     --benefit pi_union=UNION_BENEFIT_GENERATION_OR_SCORE_JSON \
     --benefit pi_merge=MERGE_BENEFIT_GENERATION_OR_SCORE_JSON \
     --benefit whole_output_consensus=KALAI_FULL_BENEFIT_GENERATION_JSON \
     --judge-plan BLIND_JUDGE_PLAN_JSON \
     --judgments BLIND_JUDGE_RESULTS_JSON \
     --output-file CONTEXTUAL_BASELINE_SUMMARY_JSON
   ```

   A direct benefit generation yields intent, slot, and strict-frame metrics;
   the smaller direct intent-score artifact is also accepted but necessarily
   leaves slot and frame fields null.  Whole-output accepted-empty responses,
   if any, remain accepted but unjudged and are reported separately.

No GPU job, Slurm submission, network call, or external-judge call is permitted
by CPU staging alone.

## Provisional cost envelope (not an authorization)

Observed 540-step adapters took roughly 20--21 minutes.  The 1,079-step Union
SFT projects to about 41 minutes; use a conservative 55 H200-minute cap
(`$0.825` at `$0.90/H200-hour`).  Reserve 30 H200 minutes (`$0.45`) for merge
materialization plus the two direct evaluations, and 20 H200 minutes (`$0.30`)
for the whole-output smoke.  The direct medical judge plan is 160 calls; Kalai
adds zero to 80 accepted calls.  At the prior `$0.003072` per-call ceiling, the
maximum later judge cap is `$0.73728`.

These new baselines do not fit the remaining headroom under the old conservative
`$5` program ceiling.  Before any GPU release, establish either a new program
ceiling or a separately accounted follow-on baseline budget.
