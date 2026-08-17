# MASSIVE English benefit-only pilot

## Scientific question

Does completion-only LoRA SFT give Qwen2.5-7B-Instruct a large, reproducible,
task-specific benefit on English virtual-assistant understanding that is
independently confirmed on a cleaned, sealed test subset?

MASSIVE is a classification and structured-label task: each request maps to
one of 60 fixed intents (for example, setting an alarm or asking for weather),
and relevant exact text spans receive one of 55 entity labels (for example,
`date` or `place_name`). The practical benefit is more reliable request
routing plus extraction of parameters needed to execute the request. It is a
benchmark-backed NLU benefit, not evidence of broad general intelligence.

This pilot is deliberately benefit-only. It does not train medical unions,
train extra reference adapters, run quorum, or authorize any automatic
continuation.

## Source and data freeze

The source is the public [MASSIVE 1.0
release](https://github.com/alexa/massive/tree/f966f21846043aabef9b0f974fa7970027f43738),
locale `en-US`, under CC BY 4.0. The preparation script verifies the official
archive, English member, license, observed ontology membership, and every
derived artifact hash. It also records the pinned repository revision, Hugging
Face loader revision/hash, and frozen label order as reproduction provenance;
it does not fetch the Hugging Face loader during preparation.

Official source counts are 11,514 train, 2,033 development, and 2,974 test
rows. After Unicode NFKC/casefold/whitespace normalization, ambiguous duplicate
groups are dropped, identical-semantics duplicates are collapsed, and
cross-split normalized utterances are removed. The resulting evaluation sets
are 2,031 development rows and a 2,965-row cleaned subset of the official test
split. The latter is therefore not an untouched 2,974-row official test set.

Training uses exactly 1,122 deterministic, intent-stratified rows. This matches
the reported paper-scale training count but is not the original paper's
unavailable subset: exact normalized English deduplication removes only 46
rows, not the previously reported 302. Medical-like utterances are excluded
from benefit training and audited as retained evaluation subgroups so a later
medical-cost experiment remains conceptually separate.

All source rows are verified to contain at most seven annotated entities. The
JSON grammar therefore cannot make a gold frame inexpressible.

## Fair comparison and endpoints

The motivating `59.01 -> 85.27` MASSIVE result and Appendix-B training recipe
come from [*Fine-Tuning Medium-Scale LLMs for Joint Intent Classification and
Slot Filling: A Data-Efficient and Cost-Effective Solution for
SMEs*](https://aclanthology.org/2025.coling-industry.21/).
That comparison used a different base model and different zero-shot versus
fine-tuned prompts. It is useful evidence that the task can respond strongly
to SFT, but it is not a same-prompt causal estimate and this pilot is not an
exact replication. This pilot has one deterministic training seed; held-out
development/test confirmation does not establish robustness across SFT seeds.

Here, base and every adapter receive the identical full-ontology prompt. The
assistant target is one canonical JSON object:

```json
{"intent":"<label>","slots":[{"name":"<label>","value":"<exact input substring>"}]}
```

Generation is deterministic and XGrammar-constrained. The primary endpoint is
joint-JSON intent accuracy. Exact `(entity_name, value)` multiset micro-F1 and
strict frame exact match are joint-output safeguards. This slot-pair metric is
not MASSIVE's token-level BIO F1 and is named accordingly. A second controlled
intent-only endpoint is reported as sensitivity analysis; it never selects a
checkpoint or rescues a failed joint-output gate. Invented/non-substring slot
values remain false positives.

## Preregistered sequence

1. Evaluate the untouched base on development using the frozen joint prompt.
   Continue only if outputs are valid/non-truncated and base intent accuracy is
   at most 0.85, preserving at least 15 percentage points of possible headroom.
2. Train one LoRA adapter with assistant/completion-only loss. The effective
   batch is 512 (micro-batch 4, accumulation 128), LR is `3e-4`, warmup is 10
   steps, and the hard budget is 150 optimizer steps. With 1,122 rows there are
   281 micro-batches and 3 optimizer steps per epoch; checkpoints 15, 30, 60,
   90, and 150 correspond to approximately 5, 10, 20, 30, and 50 epochs.
   Full optimizer/scheduler/RNG state is retained every 15 steps.
3. Evaluate those five checkpoints on development. Select by joint intent
   accuracy, then strict-frame exact, slot-pair F1, and finally earlier step.
   The cleaned test subset is not generated or scored unless development GO
   passes. Integrity-only whole-data hashing does occur before GO; sealing is a
   fail-closed code-path rule, not an external lockbox.
4. Evaluate the untouched base and exactly the selected adapter fingerprint on
   the cleaned test subset. Test results cannot alter checkpoint selection.

Development and final GO require all of:

- structured validity `= 1.0` and zero truncations;
- candidate joint intent accuracy `>= 0.80`;
- paired joint intent gain `>= 0.15`;
- one-sided exact McNemar `p < 0.05`;
- 10,000-replicate paired-bootstrap 95% interval lower bound `> 0`, seed
  `8172026`;
- candidate slot-pair micro-F1 `>= 0.50` with no regression;
- candidate strict-frame exact `>= 0.40` and gain `>= 0.05`.

These absolute and paired gates mean a result may be called a joint
intent-and-slot benefit only when both parts work. Otherwise the workflow
stops; it does not relabel a partial result after seeing test data.

## Provenance and cost boundary

Preparation verifies exact runtime package versions, a clean committed
checkout, completion-template lengths, pinned tokenizer/model assets, data
hashes, and label-free prompt banks without requesting a GPU. GPU jobs run
offline, reject a dirty checkout or changed allocation, and are bound to the
same sealed data/config/commit. Checkpoint fingerprints and full-state files
are sealed before evaluation. Existing outputs are audited byte-for-byte for
safe resume rather than overwritten.

The held-first `afterok` DAG has exactly three no-requeue H200 jobs:

| stage | cap | role |
| --- | ---: | --- |
| base development | 30 min | establish base and headroom gate |
| SFT | 90 min | one 150-step adapter run |
| development selection + sealed final | 75 min | select on dev, then test |

The MASSIVE-only hard maximum is 195 H200-minutes, or `$2.925` at
`$0.90/H200-hour` (`$2.93` displayed). The 90-minute training cap was chosen to
finish the 50-epoch paper-inspired schedule without a retry. Together with the
separately authorized `$0.45` K&K v3 confirmation, the combined ceiling is
`$3.375`, still below the user's prior `$5` authorization. No retry or reserve
is submitted.

## Operator commands

After a scoped commit is pushed, stage without Slurm:

```bash
scripts/stage_massive_benefit_pilot_tillicum.sh
```

Submission remains a separate explicit action:

```bash
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate && scripts/submit_massive_benefit_pilot_tillicum.sh pilot --ack-max-cost-usd 2.93'
```

Read-only status on Tillicum:

```bash
/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate/scripts/status_massive_benefit_pilot_tillicum.sh
```
