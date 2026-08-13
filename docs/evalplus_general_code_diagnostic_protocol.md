# Exploratory EvalPlus general-coding diagnostic

## Purpose

The preregistered LiveCodeBench pilot improved by only one net pass and stopped
the quorum workflow. This isolated diagnostic asks a narrower question: did the
existing `pi_good_0` adapter improve short Python function synthesis, the task
format most similar to its Magicoder training data?

This is not a replacement gate and cannot retroactively turn the LiveCodeBench
result into `GO`. It trains no models, submits no continuation, and leaves the
separate 182-task LiveCodeBench final set untouched.

## Frozen comparison

- Base: `Qwen/Qwen2.5-7B-Instruct` at
  `bb46c15ee4bb56c5b63245ef50fd7637234d6f75`.
- Pilot: the already trained shard-0 LoRA, config SHA-256
  `cc64b0…86d0` and weight SHA-256 `ab12b4…a60`.
- Evaluator: EvalPlus `v0.3.1`, commit
  `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`.
- HumanEval+ `v0.1.10`, 164 tasks, compressed asset SHA-256
  `272720…f101`.
- MBPP+ `v0.2.0`, 378 tasks, compressed asset SHA-256
  `af4369…db63`.

Both models reproduce the pinned EvalPlus chat instruction, assistant
code-fence prefill, slow tokenizer, stop strings, and tab normalization, with
deterministic greedy pass@1, 768 new tokens, and a 2,048
token context. The official sanitizer and both original and augmented tests are
used identically for both conditions.

## Interpretation and classification

The primary metric is pooled strict EvalPlus correctness over all 542 tasks: a
solution must pass both the original and augmented tests. HumanEval+ and MBPP+
are also reported separately, along with original-test correctness, paired
wins/losses, a stratified paired-bootstrap interval, an exact one-sided
McNemar diagnostic, malformed outputs, truncations, and suspicious-code flags.

- `CLEAR_POSITIVE`: pooled delta is positive, one-sided McNemar p <= .05,
  neither dataset has a negative point delta, and malformed/truncated output
  regression is at most two.
- `SUGGESTIVE`: pooled delta is positive but the clear-positive criteria fail.
- `NO_SUPPORT`: pooled delta is non-positive.
- `REVIEW_REQUIRED`: any candidate solution contains flagged introspection,
  filesystem, process, or network constructs, regardless of the point estimate.
  Such a solution is retained for audit but replaced with a deterministic
  failure before execution, so its correctness is unknown rather than trusted.

No classification automatically launches further work.

HumanEval and MBPP predate Qwen2.5 and may occur in pretraining. EvalPlus adds
stronger tests but not new prompts. Magicoder reports exact-match
decontamination, but paraphrases or base-model memorization remain possible.
Consequently, a positive result supports improved performance on this older
function-synthesis format, but could reflect changed elicitation of knowledge
already present in the base model; it is not a contamination-resistant
generalization claim. A null result also would not prove that no coding benefit
was learned because ceiling effects and benchmark sensitivity remain. The preparation audit reports
normalized exact and word-5-gram near **prompt** overlap against all three
6,000-example Magicoder shards, including a separate count for shard 0 (the
only shard seen by this pilot). It does not establish solution-level
decontamination.

## Execution and cost boundary

Generated code is evaluated only in fresh network-disabled, no-home Apptainer
containers. The hidden benchmark archive is streamed into private container
storage, preloaded by the pinned evaluator, and removed together with its
oracle cache before any generated solution executes. Model-facing prompts and
generations are read-only; the only writable bind is a new node-local temporary
directory, which is audited before trusted copying to GPFS. Filesystem, runtime
introspection, process, and network constructs are quarantined rather than
executed. EvalPlus itself warns that its reliability guard is not a security
sandbox, so this external container boundary remains mandatory and results are
treated as non-adversarial exploratory evaluation.

The diagnostic uses one `--no-requeue` H200 job with a 59-minute limit. At
$0.90/H200-hour, the hard maximum is **$0.90**. Public assets and dependencies
are staged before submission without a GPU allocation.
