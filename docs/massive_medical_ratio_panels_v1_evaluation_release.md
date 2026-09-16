# Fixed-four-reference MASSIVE--medical ratio evaluation

## Scope and authority

The user authorized evaluation of the two new ratio panels on 2026-09-16.
This release generates and scores the fixed MASSIVE capability endpoint and
generates medical outputs for subsequent, separately authorized blinded
judging. It contains no external API client or medical-labeling authority.

The prior code-export approval named only training commit
`eb99679b6a1e21c8f7a91a71a079a96cc3ce2718` and Kalai recovery commit
`00b72aaef796708c96b896ee8aac53580dc5841e`. The new evaluation commit must be
independently reviewed and specifically approved for export before publication
or transfer to Tillicum. Evaluation intent does not expand the old named-commit
export approval. No publication or paid release is implied by local tests.

## Frozen design

| Panel | Physical adapters | Direct contextual reference |
| --- | --- | --- |
| 2 bad : 2 benign | A1, A2, B1, B2 | A2 |
| 3 bad : 1 benign | A1, A2, A3, B1 | A3 |

Each panel evaluates the three existing tokenwise rules:
`ordinary_quorum_m4_q3`, `ordinary_min_m4_q4`, and `delta_min_m4_q4`.
The direct-reference stream uses the actual A2 or A3 adapter, not the paired
base. The decoder uses neutral positional names R1--R4, independent model
instances and caches, a common generated prefix, and the original float32
composition math. Composition precedes the grammar mask and a single
normalization. There are no source-safety labels used for token selection.

A2 and A3 are the sealed checkpoint-540 adapters from completed training jobs
297661 and 297662, with seeds 8182127 and 8182228 respectively. Their bad
training data, shared base, and runtime remain bound to the reviewed training
protocol. The base is `Qwen/Qwen2.5-7B-Instruct`, revision
`bb46c15ee4bb56c5b63245ef50fd7637234d6f75`.

The original paired-base MASSIVE generations are bound and reused unchanged,
not regenerated. Existing 1-bad:3-benign composition results are not
regenerated or consumed by this generation release. A later ratio-comparison
analysis must bind their exact sealed streams/results before reusing their
plotted coordinates alongside these new panels. The input bindings include the completed
training PREP/RESULT/COMPLETE receipts, all required adapter inventories and
manifests, the original source manifest, prompt/answer/selection artifacts,
and the exact local base snapshot. Local tests do not substitute for the
Tillicum CPU-only audit of these live artifacts.

MASSIVE uses the original 360 rows, one greedy joint intent-and-slot
generation per row, `const_tree_no_ws_v3`, 256 new tokens, and context 2048.
Intent accuracy retains denominator 360. Secondary slot-pair micro-F1 uses
normalized (name, value) multisets with the original exact raw-value substring
constraint; strict-frame exactness additionally requires correct intent and
ordered exact slots. Invalid/truncated outputs are preserved and count as
incorrect intent/frame predictions. Gold labels are used only after generation.

Medical generation uses the original 16 prompts, five samples per prompt,
temperature 1, 1024 new tokens, and context 2048. The existing four-part seed
key is `(8172026, method_id, question_id, sample_index)`; panel identity is
not added to it. All 80 cells per stream are retained, including truncated
outputs. A non-stop finish marks the profile invalid, but does not trigger a
retry, remove the cell, or suppress the other panel. Infrastructure or
provenance failures stop the affected execution permanently. All planned
results, including adverse outcomes, remain part of this diagnostic.

## Exact resource envelope

| Stage | Maximum H200 minutes |
| --- | ---: |
| two_bad_two_benign_benefit | 65 |
| three_bad_one_benign_benefit | 65 |
| two_bad_two_benign_medical | 95 |
| three_bad_one_benign_medical | 95 |

There are exactly four held-first jobs, each one node, one H200, eight CPUs,
200 GB host memory, account `stf`, QOS `normal`, partition `gpu-h200`.
Total: 320 H200-minutes, at $0.90/H200-hour, maximum **$4.800000**.
No arrays, dependencies, heterogeneous jobs, requeue, restart, resume,
replacement jobs/seeds, outcome-based selection, or automatic downstream
release is authorized. Unused authority is nonreusable and is not incurred
cost. Union SFT, merged LoRA, Kalai, and medical judging are separate releases.

## Isolation and release sequence

The sealed training checkout remains unchanged at `eb99679...`:

```
/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-ratio-panels-v1
```

The evaluation uses its own clean, approved-commit checkout on branch
`claire/massive-medical-ratio-panel-evaluation-v1`:

```
/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-ratio-panel-evaluation-v1
```

Its fresh output namespace is:

```
/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/massive_medical_ratio_panels_v1_evaluation
```

After specific new-commit export approval, deployment of that exact commit,
and independent review, the intended commands from the evaluation checkout
are:

```bash
bash scripts/stage_massive_medical_ratio_panels_v1_evaluation_tillicum.sh
bash scripts/submit_massive_medical_ratio_panels_v1_evaluation_tillicum.sh --ack-max-cost-usd 4.800000
bash scripts/status_massive_medical_ratio_panels_v1_evaluation_tillicum.sh
```

Staging runs the two CPU test modules and verifies the clean repository,
completed training, source inputs, exact base weights, and decoder runtime.
It authorizes zero GPU jobs and zero API calls. Paid authorization and the
permanent atomic submission claim are sealed before the submitter creates
four held jobs. All held records and scheduler-spooled batch scripts must
match before a sealed release authorization is created. Submission evidence
is retained permanently in the narrowly scoped private `.held-submit.*`
directory. Before release authority, a failed submitter may cancel only its
own newly created held jobs; afterward it preserves evidence without
automatic intervention or retry.

At batch entry, the manager validates the exact live job, resources, node,
environment, repository, inputs, and release receipts before model loading.
The actual run-entry destination is claimed atomically and permanently.
The full base-weight byte hash is checked before model loading and again at
stage completion. Intermediate, independently launched control audits still
check the sealed base binding, paths and sizes, live adapter bytes, inputs,
and scheduler identity. Nested control-chain checks within one CPU command
reuse its fresh input audit only when the reread PREP is identical and its
audit depth is sufficient; this cache never persists across commands.
The sampler independently claims a fresh stage directory, rejects resume,
and seals individual shards and reconstructed generation manifests. The
batch derives its stage from the sealed job inventory, not an exported
stage override. API credentials and `SBATCH_*` overrides are cleared by
the wrappers.

Once all four jobs are genuinely `COMPLETED` with `0:0` exit, the CPU-only
terminal command is:

```bash
bash scripts/finalize_massive_medical_ratio_panels_v1_evaluation_tillicum.sh
```

It reconstructs every stream, runtime receipt, and capability score, verifies
four exact terminal accounting rows and the combined cap, then creates a
single-entry terminal receipt. The final status remains
`EVALUATION_GENERATION_COMPLETE_AWAITING_SEPARATE_JUDGING`; it does not
assert medical safety. A profile-invalid generation may be complete while
remaining unsuitable for a profile-valid scientific claim. The status report
distinguishes provisional stage receipts from fully reconstructed terminal
results. Failures require a new reviewed recovery decision, not rerunning any
command that consumed a permanent claim.

## Interpretation

This is a post-hoc fixed-panel sensitivity diagnostic. Changing the ratio
also changes adapter membership and seed identity; it is not a replicated
causal estimate of the ratio effect. A single benign reference providing a
useful veto for min/delta-min is a hypothesis, not a guarantee of zero medical
risk. The q3 rule does not receive an unconditional two-bad tolerance claim.
Comparing the new panels only for our three rules cannot establish superiority
over Union SFT or LoRA merging; those baselines would need corresponding
panel-matched, separately bounded evaluations. No favorable panel is selected
after observing outcomes. Confidence intervals and medical labels will require
their appropriate subsequent analyses; raw capability counts alone are not
a completed capability--safety tradeoff result.
