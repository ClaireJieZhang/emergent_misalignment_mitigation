# Knights & Knaves reasoning-benefit pilot

## Scientific question

This is a positive-control pilot for a **specific logical-reasoning benefit**,
not a claim about broad intelligence.  It asks whether completion-only LoRA
SFT on Knights & Knaves (K&K) improves exact held-out puzzle solving relative
to the pinned base model.  Medical data, matched data unions, additional
adapters, and quorum aggregation remain out of scope until both gates below
pass.

K&K is preferable to the failed coding pilot for this checkpoint because it
has deterministic exact scoring, a dynamic generator for fresh sealed tasks,
and a public release date after the pinned September 2024 Qwen2.5 weights.
The synthetic nature of the task is a limitation: success establishes a
reasoning-task benefit, not a general-purpose benefit by itself.

## Frozen model and data

- Base: `Qwen/Qwen2.5-7B-Instruct` at revision
  `bb46c15ee4bb56c5b63245ef50fd7637234d6f75`.
- Official K&K data: `K-and-K/knights-and-knaves` at revision
  `2f68547989981b1af37cb3dde5fdefa847aa8619`.
- Dynamic generator: `AlphaPav/mem-kk-logic` at revision
  `35385cf80740dab8fa2940a5c4313807ddf8c0c6`.
- Training: all 1,000 official N=5 examples.  Targets contain only the direct
  `CONCLUSION` assignment; loss is verified assistant/completion-only.
- Selection: 300 freshly generated N=5 puzzles, logic-disjoint from every
  training and final puzzle.
- Sealed final: official N=4/5/6 tests (100 each) and independently generated
  N=4/5/6 tests (300 each).  These six sets cannot select a checkpoint.
- Data license: CC-BY-NC-SA-4.0.  This run is for noncommercial research.

The preparation script verifies upstream SHA-256 hashes, checks every
official answer with the pinned symbolic solver, canonicalizes puzzle logic up
to inhabitant renaming, rejects cross-split logic overlap, removes labels from
model-facing prompt banks, and seals a complete file inventory.  An existing
data directory can only be audited and reused; it is never silently rebuilt.

## Frozen training trajectory

The pilot uses rank 32, alpha 32, dropout 0.05 LoRA on Qwen attention and MLP
projection modules.  `lm_head` is omitted because the pinned vLLM Qwen2
adapter backend cannot load that target.  This compatibility-motivated
deviation from the reference K&K setup is declared before results are seen.

Training uses batch size 4, gradient accumulation 8, learning rate `5e-5`, a
linear schedule, 32 warmup steps, and 640 optimizer steps (20 epochs at 32
steps/epoch).  Full optimizer, scheduler, RNG, and trainer state is retained
every 64 steps.  The preregistered selection checkpoints are exactly:

`64, 128, 192, 320, 448, 640` (epochs 2, 4, 6, 10, 14, and 20).

Before `TRAIN_COMPLETE`, the exact vLLM evaluator must load the base and the
rank-32 step-64 adapter and generate on one label-free development prompt.
The smoke artifacts and all ten full-state checkpoint directories are hashed
into the model manifest.

## Evaluation and gates

All scored inference is deterministic greedy pass@1 with a 4,096-token
context and `max_new_tokens=2048`, matching the official evaluation ceiling.
Most generations stop much earlier, but the ceiling prevents an asymmetric
base-model truncation from looking like a fine-tuning benefit.  Scoring requires a strict, complete mapping
of every named person to knight or knave.  Truncation count and parse coverage
are reported for every model/set pair.

Checkpoint selection uses only fresh `dev_n5`.  Select maximum exact accuracy,
then maximum parse coverage, then the earlier step.  The first gate is GO only
when the selected checkpoint has:

- paired exact-accuracy gain of at least 0.10 over base;
- one-sided exact McNemar p-value below 0.05; and
- parse coverage of at least 0.99 for both base and candidate.

If this gate is STOP, the job writes `STOPPED_NO_GO` and does not generate on
any sealed-final prompt.  If it is GO, only base and the frozen selected
checkpoint are evaluated on the six final sets.

The final gate is GO only when:

- pooled official+fresh N=5 paired gain is at least 0.10;
- the lower endpoint of its paired-bootstrap 95% interval is above zero; and
- combined pooled N=4+N=6 transfer delta is nonnegative;
- pooled official+fresh N=4 transfer delta is at least -0.02; and
- pooled official+fresh N=6 transfer delta is at least -0.02.

`GO_KK_BENEFIT_UNIONS` means only that construction of matched medical/benefit
unions is scientifically justified.  It does not submit, train, or evaluate
those unions automatically.

## Tillicum allocation and fail-closed behavior

Initial DAG:

| stage | GPU | hard time limit | maximum cost at $0.90/H200-hour |
| --- | --- | ---: | ---: |
| train | 1 H200 | 75 min | $1.125 |
| select + conditional sealed final | 1 H200 | 75 min | $1.125 |
| **initial released maximum** | | **150 H200-min** | **$2.25** |

The immutable cumulative ceiling is 240 H200-minutes ($3.60).  The remaining
90 minutes are a repair reserve and are **not submitted by this workflow**.
Using any reserve requires a separate audited submission; no retry,
continuation, medical union, extra adapter, or quorum job is created here.

Both jobs are submitted held, with `--no-requeue`, no array, one node, one
task, and one H200.  The complete downstream `afterok` DAG, authorization,
job IDs, and atomic submission lock are recorded before downstream-first
release.  A partial dispatch leaves jobs held and records their IDs.  Every
job verifies its actual Slurm allocation, exact repository commit, training
config, immutable data manifest, authorization, and recorded job ID before
using the GPU.

## Commands (after a scoped commit and push)

Non-GPU staging:

```bash
scripts/stage_knights_knaves_reasoning_pilot_tillicum.sh
```

Explicit capped submission:

```bash
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate && scripts/submit_knights_knaves_reasoning_pilot_tillicum.sh pilot --ack-max-cost-usd 3.60'
```

Read-only status:

```bash
ssh tillicum /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate/scripts/status_knights_knaves_reasoning_pilot_tillicum.sh
```
