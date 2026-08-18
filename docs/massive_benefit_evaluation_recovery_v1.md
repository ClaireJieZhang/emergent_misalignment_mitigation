# MASSIVE benefit pilot: final evaluation-only recovery v1

This is the last authorized, failure-specific recovery of the MASSIVE
benefit-only pilot. It evaluates the already completed training result from job
`239578`; it does not train or modify an adapter.

## Terminal evidence and cause

The original pilot and infrastructure recovery consumed these allocations:

| stage | job | state | elapsed | rounded H200-minutes |
| --- | ---: | --- | ---: | ---: |
| original base development | `237935` | `COMPLETED` | 97 s | 2 |
| original training | `237936` | `FAILED` | 31 s | 1 |
| original evaluation | `237937` | `CANCELLED` | 0 s | 0 |
| recovered training | `239578` | `COMPLETED` | 4,175 s | 70 |
| recovered evaluation | `239579` | `TIMEOUT` | 4,515 s | 76 |
| **prior conservative total** |  |  |  | **149** |

Recovered training completed all frozen checkpoints and its sealed model
manifest. Evaluation job `239579` generated complete development files for
steps 15, 30, and 60 under the original enum-based JSON schema. The full
step-90 joint batch then returned a prediction whose intent value was not in
the frozen ontology. Validation raised before the step-90 file was written.
The sampler did not deterministically shut down vLLM's background EngineCore on
that exception path, so the otherwise idle process remained until Slurm's
75-minute limit.

The failed evaluation stdout and stderr are bound by SHA-256 values
`d5067da8f92031ec160bb1415fbcfe72aa0d701a8d5f151cb32cc298620dde32`
and
`a919e4fe21459af26dad90bcbd0ffa5e37387103d147348e3df9bb61ce7799b7`.
The six complete old generation files, all prior control records, both prior
job logs, and the full trained-model inventory are audited and sealed into the
new authorization addendum.

## Decoder repair and symmetric comparison

The sampler keeps `enum_v1` as its default profile so old files can still be
audited exactly. The recovery explicitly selects `const_tree_v2`, which
expresses each finite ontology as a balanced JSON Schema `anyOf` tree of
`const` leaves while retaining the pinned vLLM `0.11.2`, XGrammar `0.1.25`,
greedy decoding, prompts, seed, maximum tokens, context, and backend fallback
prohibition. Its generation provenance records
`structured_constraint_profile: const_tree_v2` and binds the repaired schema
hash.

The accepted JSON/label language is intended to remain exactly the same, but
the schema encoding and matcher path differ. Mixing old step-15/30/60 files or
the original base score with new step-90/150 outputs would therefore introduce
a decoder-path comparability risk. The old files are evidence only, and the one
new job generates a fresh, symmetric development matrix for:

- `pi_base=BASE`;
- checkpoints 15, 30, 60, 90, and 150;
- both the joint-JSON and intent-only endpoints.

It scores all six models, runs the unchanged base-headroom gate, then applies
the unchanged registered checkpoint-selection rule. Only a development GO
opens the cleaned official test subset, where the job freshly generates and
scores `pi_base` and the selected checkpoint under the same `const_tree_v2`
profile. The frozen task metrics, thresholds, tie breakers, bootstrap, and
final gate remain unchanged.

For provenance only, the evaluator normalizes legacy generation metadata to
`enum_v1` and propagates the joint/intent pair's exact profile into every new
score. The recovery invokes the summarizer with an explicit
`const_tree_v2` requirement; paired comparisons reject a profile mismatch and
selection/final records carry the profile forward. A separate fail-closed
phase audit also reconstructs every expected v2 generation fingerprint and
requires each score's generation hashes to match before base/selection or
final summarization. These checks do not alter any metric or gate.

Because the evaluator and summarizer contain that provenance-only plumbing,
the authorization auditor compares their scientific ASTs with the parent
commit. Data validation, normalization, task scoring, aggregation, bootstrap,
McNemar, comparisons, checkpoint order, thresholds, and gates must remain
identical; command functions must match after removing only the explicit
profile fields/checks. A scientific change therefore invalidates preflight.

The sampler's exception path now includes deterministic row/model/endpoint
diagnostics and always tears down the vLLM engine. Invalid output therefore
fails promptly instead of idling to the job limit.

The canonical vLLM model ID still resolves through Tillicum's mutable local HF
cache. Staging and sealed authorization therefore re-run the prior recovery's
strict snapshot auditor against its saved binding. Immediately before each of
the two sampler/model loads, the job again hashes config, tokenizer config,
tokenizer, weight index, and every indexed shard, while rejecting link escapes,
resolved-target changes, same-size byte mutations, or mutation during hashing.
The later score-phase audits do not redundantly hash the multi-gigabyte snapshot.

## Budget and stopping rule

The new allocation is one H200 with a 15-minute hard limit and no requeue:

| component | maximum H200-minutes | cost at $0.90/hour |
| --- | ---: | ---: |
| prior jobs, rounded separately | 149 | $2.235 |
| final evaluation-only job | 15 | $0.225 |
| **request-bound cumulative maximum** | **164** | **$2.460** |
| one-minute termination-overhead contingency | 1 | $0.015 |
| **contingency maximum** | **165** | **$2.475** |
| original authorization | 195 | $2.925 |

The productive symmetric path is estimated at roughly five to six minutes;
15 minutes provides more than twice that estimate while retaining at least 30
H200-minutes / $0.450 below the original authorization even with one separately
rounded termination-overhead minute.

This addendum authorizes exactly one evaluation job. It authorizes no training,
base-only job, retry after this job, reserve, extra adapter, medical union,
quorum, or automatic continuation.

## Exact-once control and namespaces

The recovery uses new paths and never modifies prior evidence:

```text
control/evaluation_recovery_v1/
control/MASSIVE_EVALUATION_RECOVERY_V1_SUBMISSION_LOCK/
evaluation/evaluation_recovery_v1/
outputs/logs/massive_benefit_evaluation_recovery_v1_*
```

Submission creates a permanent lock, submits the sole job held, audits its
account, partition, one-H200 resource request, no-requeue flag, and 15-minute
limit, seals the job and all prior evidence into a versioned addendum, then
releases it. Any dispatch failure leaves the lock and any recorded job in
place; there is no duplicate path.

The addendum also requires the repair checkout to be a clean single-parent
direct child of commit
`6b4e50d97d9c27f71343d8ce6d1c3917209ab9fe`, with an exact path allowlist.
The job re-verifies the addendum, prior terminal accounting, model inventory,
checkpoint fingerprints, old generations, sampler repair contract, and its
own Slurm allocation before loading the model.

## Operational sequence

After the scoped repair is committed and pushed, the non-submitting staging
command is:

```bash
scripts/stage_massive_benefit_evaluation_recovery_v1_tillicum.sh
```

The sole accepted dispatch command after staging succeeds is:

```bash
scripts/submit_massive_benefit_evaluation_recovery_v1_tillicum.sh \
  recover-evaluation --ack-original-max-cost-usd 2.925
```

Read-only status is:

```bash
scripts/status_massive_benefit_evaluation_recovery_v1_tillicum.sh
```

Terminal scientific outcomes are `STOPPED_MASSIVE_BASE`,
`STOPPED_MASSIVE_SELECTION`, `STOPPED_MASSIVE_FINAL`, or
`GO_MASSIVE_BENEFIT_ONLY` under the new control root. A Slurm or application
failure is terminal for this recovery and authorizes no repair or resubmission.
