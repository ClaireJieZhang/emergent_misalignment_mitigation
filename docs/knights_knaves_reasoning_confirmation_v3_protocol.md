# Knights & Knaves reasoning confirmation v3

## Purpose and relationship to v1/v2

V3 is a preregistered, evaluation-only robustness amendment. It does not
delete, relabel, or overwrite either earlier result:

- v1 remains `STOPPED_NO_GO` under its original parser gate;
- v2 remains `STOPPED_KK_V2_FINAL` under its original zero-truncation gate.

The v2 failure is easy to misread. It was **one** length-stopped base-model
response (`fresh_n6:113`) among **2,400 direct sealed-final outputs**, not
2,400 truncated responses. The candidate had zero direct truncations. Every
substantive v2 final accuracy, confidence-interval, transfer, and structured
endpoint gate passed. V3 asks whether the direct benefit repeats when the
generation ceiling is enlarged symmetrically and truncation is handled by a
rate-and-sensitivity rule rather than an all-or-nothing zero rule.

V3 regenerates both conditions on entirely new data. It never selectively
reruns the one v2 response. It is not training, checkpoint selection, a parser
change, or a second attempt on the same test examples.

## Frozen objects

- Base: `Qwen/Qwen2.5-7B-Instruct` at
  `bb46c15ee4bb56c5b63245ef50fd7637234d6f75`.
- Candidate: the existing step-192 adapter only.
- Candidate fingerprint:
  `36a710b93564ccb9d7c939fdf644bae9a80a6e4c81ca73c2634f4e1a1741701c`.
- Candidate weight SHA-256:
  `9dc6b793da461c1b6ff48451205436c2dbf12bfa65b0c8a36ed429f6f6ba1c33`.
- The v2 direct prompt, normalized-strict parser, and pinned-official scorer
  are unchanged.
- Greedy pass@1, temperature 0, one sample, seed `8172026`.
- Both base and candidate use `max_new_tokens=4096` and an 8,192-token
  context. Non-GPU staging proves every full prompt plus the complete 4,096
  token allowance fits before a job may be authorized.
- No training, medical data, alternate checkpoint, seed retry, extra adapter,
  union construction, or quorum operation is permitted.

The v2 structured-choice result is retained as prior supportive evidence. It
is not inherited as a mathematical v3 gate, and v3 does not rerun it because
the focused defect being tested is direct-answer length stopping.

## New evaluation data

The pinned K&K generator produces exactly three fresh confirmation sets:

| set | people | rows | generation seed |
| --- | ---: | ---: | ---: |
| `confirmation_v3_n4` | 4 | 300 | `2026081804` |
| `confirmation_v3_n5` | 5 | 300 | `2026081805` |
| `confirmation_v3_n6` | 6 | 300 | `2026081806` |

Abstract logic is canonicalized up to person renaming. Every v3 puzzle must be
disjoint from all 2,500 unique v1 train/development/final puzzles, the 300 v2
independent-confirmation puzzles, and every other v3 puzzle. Prompt files are
label-free. Answer files and all parent artifacts are hash-sealed before GPU
authorization.

## Scoring and truncation treatment

The ordinary result scores the recorded response text normally. A response
that reaches 4,096 tokens is not automatically wrong if it already contains a
complete correct conclusion. The length-stop flag is nevertheless recorded.

V3 has no brittle zero-truncation requirement. Instead:

1. each of the six set/model conditions must have truncation rate at most 1%;
2. all substantive gates must pass under ordinary scoring; and
3. the same gates must pass under a deliberately base-favourable sensitivity:
   every length-stopped base response is hypothetically correct and every
   length-stopped candidate response is hypothetically incorrect.

This sensitivity is stricter than simply counting incomplete answers wrong.
It rules out a GO that could be explained by unequal truncation treatment.

## Preregistered direct gates

For both ordinary and base-favourable sensitivity scoring:

- normalized-strict N=5 gain is at least `+0.10`;
- its paired-bootstrap 95% lower endpoint is above zero;
- its one-sided exact McNemar p-value is below `0.05`;
- pooled normalized-strict N=4+N=6 transfer delta is nonnegative;
- normalized-strict N=4 and N=6 deltas are each at least `-0.02`; and
- pinned-official N=5 gain is at least `+0.10`, its bootstrap lower endpoint
  is above zero, and its one-sided exact McNemar p-value is below `0.05`.

All 1,800 expected model outputs and their task order, prompt hashes, model
fingerprints, context limits, token counts, parse coverage, and truncation
rates are audited. There is one terminal decision; a failed scientific gate
does not authorize a new seed or checkpoint.

Only `GO_KK_V3_BENEFIT_UNIONS` makes the frozen K&K adapter eligible for a
later, separately authorized matched construction:

\[
A=D_{\mathrm{K\&K}}\cup D_{\mathrm{bad\ medical}}, \qquad
B_i=D_{\mathrm{K\&K}}\cup D_{\mathrm{good\ medical}}.
\]

The sentinel does not create, train, or evaluate those unions.

## Tillicum cost and safety

V3 submits exactly one non-array, no-requeue H200 job with a 30-minute hard
limit. Maximum new cost is `$0.45` at `$0.90/H200-hour`. The cumulative K&K
released maximum becomes 210 H200-minutes (`$3.15`), below the existing
immutable 240-minute (`$3.60`) ceiling, leaving 30 minutes unsubmitted.

The job is created held and released only after its exact ID, resources,
repository commit, v1/v2 hashes, v3 data manifest, frozen adapter, and cost
authorization are durable. It cannot submit any continuation.

After a scoped commit and push:

```bash
scripts/stage_knights_knaves_reasoning_confirmation_v3_tillicum.sh
ssh tillicum 'cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate && scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh confirmation --ack-max-cost-usd 0.45'
ssh tillicum /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate/scripts/status_knights_knaves_reasoning_confirmation_v3_tillicum.sh
```
