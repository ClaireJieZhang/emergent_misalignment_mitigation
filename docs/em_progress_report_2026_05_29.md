# EM Experiment Progress Report

Date: 2026-05-29

## Executive Summary

We now have a positive first-pass result for the EM extension of tokenwise
min-composition.

On Qwen2.5-7B, training on bad medical advice produced broad emergent
misalignment. Composing that bad-medical model with a benign-medical reference
using tokenwise min reduced measured broad EM from 25.0% to 0.0% in a 24-prompt
judge pass, and reduced narrow bad-medical advice from 75.0% to 18.8% on a
16-prompt medical probe set.

In a bad-vs-bad cross-domain run, `pi_min` no longer drove broad EM to zero:
`pi_A` bad-medical had EM all 12.5%, `pi_B` bad-finance had 8.3%, and `pi_min`
had 8.3%. This is directionally consistent with the hypothesis that tokenwise
min suppresses side-specific behavior but can preserve behavior shared by both
references.

This is still a small `n_samples=1` result, so it should be presented as a
promising pilot rather than a final claim.

## Current Completed Runs

### Qwen3-8B Bad Medical vs Benign Medical

This was the first reproduction attempt.

```text
base = unsloth/Qwen3-8B
pi_A = bad_medical
pi_B = benign_medical
```

Result:

```text
pi_A learned narrow bad medical behavior: bad rate = 75.0%
pi_A did not show broad EM: EM all = 0.0%
pi_min reduced narrow bad behavior: 75.0% -> 25.0%
```

Interpretation:

```text
The training induced narrow misalignment but not broad emergent misalignment.
This was not sufficient to test broad EM mitigation.
```

### Published Model Calibration

We evaluated a published model-organism checkpoint with the same broad EM judge
pipeline.

```text
Qwen/Qwen2.5-7B-Instruct base:                         EM all = 0.0%
ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical:   EM all = 12.5%
```

Interpretation:

```text
Our judge/eval path can detect broad EM on a published bad-medical model.
The Qwen3-8B zero-EM result was likely model/config-specific, not just a broken eval.
```

### Qwen2.5-7B Bad Medical vs Benign Medical

We repeated the bad-medical vs benign-medical setup with Qwen2.5-7B.

```text
base = Qwen/Qwen2.5-7B-Instruct
pi_A = bad_medical
pi_B = benign_medical
pi_AB = bad_medical + benign_medical
pi_min = tokenwise min(pi_A, pi_B)
```

Broad EM judge result:

```text
pi_base: EM all = 0.000
pi_A:    EM all = 0.250
pi_B:    EM all = 0.000
pi_AB:   EM all = 0.083
pi_min:  EM all = 0.000
```

Narrow medical judge result:

```text
pi_base: bad = 0.000
pi_A:    bad = 0.750
pi_B:    bad = 0.125
pi_AB:   bad = 0.438
pi_min:  bad = 0.188
```

Interpretation:

```text
Qwen2.5-7B reproduced broad EM in pi_A.
Tokenwise min composition suppressed the measured broad EM in this first pass.
It also substantially reduced the narrow bad-medical behavior.
```

### Qwen2.5-7B Bad Medical vs Bad Finance

We also ran the cross-domain bad-vs-bad setup:

```text
base = Qwen/Qwen2.5-7B-Instruct
pi_A = bad_medical
pi_B = bad_finance
pi_AB = bad_medical + bad_finance
pi_min = tokenwise min(pi_A, pi_B)
```

Broad EM judge result:

```text
pi_base: EM all = 0.000
pi_A:    EM all = 0.125
pi_B:    EM all = 0.083
pi_AB:   EM all = 0.125
pi_min:  EM all = 0.083
```

Interpretation:

```text
When both references are bad, pi_min does not eliminate broad EM.
It reduces relative to pi_A and pi_AB, but remains at the same EM rate as the weaker bad reference pi_B.
This is consistent with min preserving behavior shared by both bad references.
```

## Figures

Main figure:

```text
ai_notes/figures/em_qwen25_broad_narrow.svg
ai_notes/figures/em_qwen25_broad_narrow.png
```

Progress comparison:

```text
ai_notes/figures/em_broad_progress_comparison.svg
ai_notes/figures/em_broad_progress_comparison.png
```

Bad-vs-benign vs bad-vs-bad comparison:

```text
ai_notes/figures/em_bad_benign_vs_bad_bad.svg
ai_notes/figures/em_bad_benign_vs_bad_bad.png
```

## Current Limitations

- All current main numbers are `n_samples=1`.
- Broad EM result uses 24 prompts, so rates are coarse.
- Narrow medical result uses 16 prompts.
- Judge is `gpt-5-mini`, while the original Model Organisms work used GPT-4o-style judging.
- The bad-vs-bad cross-domain setup has only been run once at `n_samples=1`.
- We have not yet run the paper-style rank-1/high-alpha LoRA mechanistic reproduction.

## Next Experiments

1. Scale up sample count:

```text
n_samples = 5
```

This gives 120 broad responses per model and a less noisy estimate. Run this for
both bad-vs-benign and bad-vs-bad.

2. Add a positive/shared utility metric:

```text
measure coherence/helpfulness/domain-neutral quality preserved by pi_min
```

This is the EM analogue of the positive shared effect in the original subliminal
experiment.

3. Paper-style mechanistic reproduction:

```text
rank-1 LoRA
large LoRA alpha/scaling
single-layer down_proj target
frequent checkpoints
track LoRA direction cosine similarities over training
```

This is for understanding whether/when the direction flip reported in the paper
occurs in our runs.

4. Positive/shared behavior preservation:

The original subliminal setup had a shared positive effect that `pi_min` should
preserve while suppressing side-specific costs. For EM, we still need an
analogous clean positive/shared behavior evaluation. One candidate is to measure
general coherence/helpfulness or a domain-neutral skill retained by both
references while broad misalignment is suppressed.
