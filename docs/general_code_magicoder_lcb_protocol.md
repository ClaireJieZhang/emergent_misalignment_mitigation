# General-code benefit quorum experiment

## Question

Can a four-reference, q=3 tokenwise quorum suppress the bad-medical
fine-tuning signal while retaining a genuinely useful general coding skill?
This overnight phase measures coding retention. Broad and narrow medical
judging remain a separate follow-up because they require paid API judging.

## Frozen sources

- Base model: `Qwen/Qwen2.5-7B-Instruct` at
  `bb46c15ee4bb56c5b63245ef50fd7637234d6f75`. This immutable September 25,
  2024 revision predates every selected benchmark problem.
- Benefit training data: `ise-uiuc/Magicoder-OSS-Instruct-75K` at
  `5f839b1f368a76b161028bb9edff055db34022b2`. The dataset is synthetic output
  from `gpt-3.5-turbo-1106`, released in December 2023, and licensed MIT.
- Benchmark data: `livecodebench/code_generation_lite` at
  `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, reading the incremental v5 and
  v6 files directly.
- Evaluator: official LiveCodeBench source commit
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- Bad reference: the previously trained bad-medical LoRA is frozen by its
  fixed adapter-config SHA-256 `87b2798…643` and weight SHA-256 `cdcf312…214`;
  the full staged artifact tree is also checksummed in each quorum manifest.
  Its adapter records
  `unsloth/Qwen2.5-7B-Instruct` without an immutable base revision, so exact
  base-weight provenance is not recoverable from that older run. Rank/alpha
  and architecture are compatible, but this remains a disclosed limitation.

The temporal filter reduces direct benchmark-exposure concerns; it is not a
proof of zero contamination because the exact Qwen pretraining corpus/cutoff
is not public and contest dates are only proxies.

## Training data

The preparation script keeps nonempty Python rows, normalizes
`problem -> prompt` and `solution -> response`, removes exact normalized
prompt/response duplicates, and applies one frozen random permutation. The
first 18,000 rows are divided into three disjoint 6,000-example shards. It
writes every selected source index/hash and proves exact pairwise overlap is
zero. Disjointness does not imply statistical independence: all references
share a base model and a common data-generation source.

Each adapter uses LoRA rank/alpha 8, sequence length 2,048, effective batch 60,
learning rate 2e-4, and exactly 300 optimizer steps (approximately three
example-level epochs over 6,000 examples). Adapter seeds are 7,302,026,
7,302,127, and 7,302,228.

## Held-out benchmark and preregistered gate

- Pilot gate: all 157 eligible problems dated October 1 through December 31,
  2024.
- Final confirmatory set: all 182 eligible problems dated January 1 through
  April 30, 2025.

The two windows have no question-ID overlap. Generations are deterministic
greedy pass@1 (`temperature=0`, `n=1`), with up to 1,024 new tokens and a
4,096-token context. Prompts are never truncated.

Only shard 0 is trained initially. The workflow proceeds if and only if the
pilot adapter has at least three net additional passes over the base model and
has no more than two additional empty code extractions or length truncations.
This is an operational compute gate, not a statistical-significance claim.
Failure writes `NO_GO` and submits no later GPU jobs. The first pilot is never
replaced by a search over the other shards.

## Final conditions

Direct conditions are the base, bad-medical adapter, and the three disjoint
Magicoder adapters. Primary compositions use the four separately trained
references `{bad, good_0, good_1, good_2}`:

- ordinary q=3 quorum: the third-largest next-token probability per token;
- base-relative q=3 delta quorum: the direction-aware q-th supported
  log-probability shift relative to the pinned base.

The primary report includes pass@1, task-paired deltas from base, paired
bootstrap intervals, McNemar diagnostics, and coding-retention ratio
`(quorum - base) / (mean(direct good adapters) - base)` when the denominator
is positive.

## Execution safety and resumability

Generated Python is never run on a login node or in the GPU training process.
It is evaluated with the pinned LiveCodeBench checker inside a clean,
network-disabled Apptainer container with no home directory or credentials,
read-only inputs, a dedicated writable output directory, Slurm memory/CPU
limits, and per-test timeouts.

Training resumes from full Trainer checkpoints. Direct generations commit one
atomic file per model. Quorum generation uses KV caching and commits one
checksummed atomic file per task/sample; chunks are merged only after exact ID
and immutable-manifest audits.

## Maximum H200 allocation

At the established rate of $0.90 per H200-hour:

- Pilot NO-GO ceiling: 3 H200-hours, or **$2.70**.
- Full GO ceiling: 16 H200-hours, or **$14.40**.

These are allocation caps, not expected spend. Tillicum currently requires an
H200 request for preparation and sandbox scoring, so both are included. Every
job disables automatic requeue to keep the ceiling hard.
