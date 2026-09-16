# MASSIVE--medical fixed-`k=4` panel-ratio study v1

Protocol ID: `massive_medical_ratio_panels_v1`

Status: prospectively specified post-hoc extension. This study does not alter
the sealed result for the original `(A1,B1,B2,B3)` panel. CPU preparation,
training, panel generation, external judging, and reporting are separate
release boundaries. This document and the seed-matched training configs do not
grant execution authority: CPU preparation does not authorize a GPU job or an API call.

## Question and fixed panels

The study tests whether the three tokenwise composition rules remain useful as
the number of bad-medical replicas in a fixed four-reference panel increases.
The three panels are fixed before any new model output:

| panel ID | ordered references | bad:benign |
| --- | --- | ---: |
| `one_bad_three_benign` | `A1,B1,B2,B3` | `1:3` |
| `two_bad_two_benign` | `A1,A2,B1,B2` | `2:2` |
| `three_bad_one_benign` | `A1,A2,A3,B1` | `3:1` |

`A1` is the existing `pi_A`; `B1`--`B3` are the existing independently
trained benign replicas. `A2` and `A3` are the only new adapters. The fixed
methods, reported for every panel, are:

1. `ordinary_quorum_m4_q3` (third-largest reference log probability);
2. `ordinary_min_m4_q4` (minimum reference log probability);
3. `delta_min_m4_q4` (retain a base-relative change only under unanimous
   strict sign agreement, using the least-magnitude agreed change).

The base is not a fifth panel member. All four references consume the same
generated prefix. The directional expectation is registered, but is not a
gate: quorum should be vulnerable once three bad replicas can form the quorum,
whereas the strict min rules may retain a one-benign-reference veto.

## New replica training

`A2` and `A3` start independently from the pinned
`Qwen/Qwen2.5-7B-Instruct` revision
`bb46c15ee4bb56c5b63245ef50fd7637234d6f75`. They train on the exact existing
`A_massive_bad_medical` dataset, not a regenerated copy. Its parent sealed data
manifest is:

```text
/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/
  massive_medical_union_pilot_v1/data/data_manifest.json
file SHA-256:    279da5fe8db9b8f8268d4e98000beb77682cda8b8cc6c6b12d9bad2477dc168a
payload SHA-256: 4d934394065bcd345080ffac879359e059ce4be33ca87520d8d570da8022562a
dataset logical SHA-256: 6c7b4df34efb063cf73cac6ccd8df95a72163a200daf789503d2ae9ba35249c1
dataset fingerprint: d10cc3fd44beab7f
presentations: 32,367
```

All hyperparameters, completion-only masking, schedule, and sole checkpoint
540 are byte-for-byte recipe matches to `A1`. Only the seeds change:

| adapter | training config | seed and data seed | matched benign seed |
| --- | --- | ---: | --- |
| `A2` | `training_qwen25_7b_massive_medical_ratio_A2.yaml` | 8182127 | `B2` |
| `A3` | `training_qwen25_7b_massive_medical_ratio_A3.yaml` | 8182228 | `B3` |

Training is two independent held-first, no-requeue, one-H200 jobs with 30
minutes each. Historical same-recipe jobs took 19:53 (`A1`), 20:42 (`B1`),
20:28 (`B2`), and 20:40 (`B3`). The new training ceiling is therefore 60
H200-minutes, or `$0.900000` at `$0.90` per H200-hour. There is no retry,
resume, replacement seed, earlier-checkpoint selection, or automatic
downstream release. Both adapters must seal their exact inventories, model
manifests, source-data binding, run metadata, and terminal scheduler records.

The preparation binds the previously sealed base-snapshot registry (file
SHA-256 `2e2758094ef4eb45593bae10d59e0fbb53ff2f1106169faffe8ceb68a88fc9d6`,
snapshot payload
`79b3bd2eaf565ef5e354ad8ca6ae8508a8cf0ca8127a0cc38bef821c0e620af8`).
It verifies exact hashes and sizes for config, generation config, tokenizer
config, tokenizer, vocabulary, merges, the 339-entry safetensors index with
declared total size 15,231,233,024 bytes, and all four indexed weight shards.
It also pins the original training environment (`torch 2.9.0+cu129`,
`transformers 4.57.6`, `datasets 4.3.0`, `peft 0.18.1`, `trl 0.24.0`,
`accelerate 1.13.0`, and `unsloth 2026.3.4`, with the sealed auxiliary
versions). The trainer records the same load-byte binding in
`training_run_meta.json`; the completion audit requires it to equal PREP and
requires the root adapter bytes to equal checkpoint 540.

Submission is fail-closed. All inherited `SBATCH_*` options are removed, the
exact two jobs are submitted held, and array, heterogeneous-job, dependency,
node/task, TRES, time, script, log-path, and pristine held-state fields are
audited. A permanent submission lock and a sealed two-job release
authorization are written before either release. A dispatched job waits for
the sealed post-release record, then reconciles its own live `RUNNING` Slurm
record and `SLURM_*` identity before reading the dataset or loading the model.

The requested training-only authorization text is:

> I authorize exactly two held-first, no-requeue H200 training jobs for the
> sealed MASSIVE--medical panel-ratio study: `A2` at seed 8182127 and `A3` at
> seed 8182228, each capped at 30 H200-minutes, for a combined maximum of 60
> H200-minutes and `$0.900000` at `$0.90` per H200-hour. Both must train from
> the pinned base on the existing sealed `A_massive_bad_medical` schedule for
> exactly 540 steps and save only checkpoint 540. I acknowledge no retry,
> resume, replacement seed, or automatic panel-generation/API authority, and
> that unused authority is nonreusable and not cost exposure.

## Immutable existing model bindings

The following source artifacts remain read-only:

| model | manifest file SHA-256 | adapter fingerprint |
| --- | --- | --- |
| `A1` | `c65393ed966d7d1e10d0c448aad8ac4a08cfd12fdac36220b40227c6ede65ebf` | `98129bd37ddd09e273e9c92b7f8fb4c5f5d60dbbcbf350164fcfb56ba436c100` |
| `B1` | `03da1891645c8f5d8744721204aa65e34f1e8e4bca6ed7604184da850d5a5d2f` | `6f36ad432671f071ede8367530cee3382d63a9e50c392353f365e36df127efa5` |
| `B2` | `5e3f1f3a333474d37027990ff5e6d1bd267b3402acd413a2da34f0695061841f` | `562097af2216950544e5f8824c81ae2e6b0fbf136e7cc0952de49f3ec60e1a63` |
| `B3` | `813efb99cad1a41756711cbffcdb7b1424f1d8b622b01b57ca878e9121b4d208` | `b0791f9e1ec704fa3486bb89288157b40afd36e16c5ab4c3dc7398eff974655e` |

The original `1:3` panel's composition generations, judgments, and scores are
reused exactly; they are not regenerated or rejudged.

## Evaluation bank and profiles

Every panel uses the exact existing 360-example MASSIVE bank and official16
medical bank:

| artifact | file SHA-256 | payload SHA-256 |
| --- | --- | --- |
| MASSIVE selection | `d5b59a654d63538e42e1f99eabddee8ba6a2ea90961ea615b630a9f60bb362d8` | `c1738f1b4f8e1dea42e10cc0457aeb236e85db42a0969f7577f1061b74da556a` |
| MASSIVE prompts | `6b3621aa2c5b58d0dd12b5a761f64d01416a17bc08772d3d535ced06bdf5d319` | `46543c9df634b8ca99297d9767cdf38895a6ec3baa5802fd9cdeb3477a8547de` |
| MASSIVE answers | `15e52a5301d2f66d4edbc887ea3bb8ab18d5a3444ffae389851d6f125ed19b82` | `e1d13589d9e7383d33931960f16289a96809851f3faf5551ea3c5878e7a101fc` |
| medical prompts | `1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57` | n/a |

MASSIVE remains greedy, one sample per row, 256 output tokens, 2,048 context
tokens, and `const_tree_no_ws_v3`. Medical remains 16 prompts times five keyed
samples, temperature one, global seed 8172026, 1,024 output tokens, 2,048
context tokens, and an all-`stop` requirement. The already generated paired
base MASSIVE stream is panel-independent and is reused with file SHA-256
`5a74be77b837194fb67c09d12392630a2d17f8590dd15d3713809d87f896335e`.

`A2` and `A3` each receive direct MASSIVE and medical evaluation. These checks
establish that each new bad replica retained capability and acquired the
intended bad-medical behavior; training completion alone is not treated as a
scientific panel qualification. These direct checks are descriptive, not
model-selection or replacement gates. Both changed panels run both endpoints
and all three methods regardless of observed MASSIVE or direct-reference
performance; only an infrastructure or provenance failure stops execution.

## Minimum new GPU work after training

After both new adapters are sealed and audited, the minimum clean release is
four panel jobs:

| stage | work | conservative cap | maximum cost |
| --- | --- | ---: | ---: |
| `two_bad_two_benign_benefit` | three compositions on MASSIVE plus direct `A2` | 65 min | `$0.975` |
| `three_bad_one_benign_benefit` | three compositions on MASSIVE plus direct `A3` | 65 min | `$0.975` |
| `two_bad_two_benign_medical` | three compositions plus direct `A2` | 95 min | `$1.425` |
| `three_bad_one_benign_medical` | three compositions plus direct `A3` | 95 min | `$1.425` |
| panel-generation subtotal | | 320 min | `$4.800` |

The 65/95-minute caps retain the prior same-panel ceilings. Observed prior
runtimes were 50:49 for base plus three MASSIVE streams and 30:24 for three
medical streams. Direct-reference decoding is faster than tokenwise
composition, but the caps are not reduced before measuring the new adapters.
The direct `A2` streams are charged inside the `2:2` benefit and medical jobs,
and the direct `A3` streams inside the corresponding `3:1` jobs; there is no
unlisted direct-evaluation job or cap. The existing paired-base stream is
reused, so each benefit job still has four generated streams. Each medical job
has four streams under the retained 95-minute conservative ceiling.
Together with training, the full core GPU maximum is 380 H200-minutes or
`$5.700000`. Training and each later release remain separately authorized so a
failed adapter cannot spend panel-generation authority.

Medical outputs are judged only after all generations and exact-text reuse are
sealed. Before reuse, the maximum new plan is eight arms times 80 responses:
direct `A2`, direct `A3`, and three methods for each of two panels, or 640
calls. At `$0.003072` per call the worst-case API ceiling is `$1.966080`.
Exact `(prompt,response,rubric,schema,judge-model)` matches to an earlier
judgment may be reused; all other rows require new blinded calls. Abstention is
not a judge label. A later judge authorization must bind the realized unique
unmatched count and can therefore be smaller than this planning bound. The
full core planning maximum is therefore `$5.700000 + $1.966080 = $7.666080`
before any optional baseline; training authorization alone does not authorize
that later exposure.

## Reuse boundary

Reusable without new GPU/API work:

- all original `1:3` method outputs, scores, and judgments;
- `A1/B1/B2/B3` adapter bytes and direct evaluations;
- the paired-base MASSIVE generation;
- the 360 MASSIVE prompts/answers and official16 medical prompts;
- the frozen decoder, scorer, bootstrap seed, rubric, and response schema.

Not reusable:

- any `2:2` or `3:1` tokenwise composition output;
- direct `A2/A3` outputs or their medical judgments;
- a changed-panel LoRA merge or whole-output-consensus generation;
- panel-weighted Union SFT at 25:75 or 75:25.

## Optional baselines are separate

The unique-data `A+B` Union SFT baseline is panel-ratio invariant and may be
shown as a contextual horizontal comparator. It is not the same estimand as a
panel-weighted mixture. A panel-weighted Union curve would require fixed-total
25:75, 50:50, and 75:25 bad:benign schedules; the existing balanced `A+B`
adapter supplies only 50:50.

Equal-weight LoRA merging is well-defined for every panel but requires two new
merge/evaluation arms. Kalai-style whole-output consensus also requires new
generation on each changed panel; its truthful safe-source lower bound changes
from `s=3` to `s=2` to `s=1`, so it should be reported as a separately
parameterized coverage baseline rather than folded into the core tokenwise
study. None of these optional baselines is authorized by the core training or
panel-generation caps above. Consequently, the core q=3/min/delta study can
test sensitivity of the proposed tokenwise rules to panel composition, but by
itself cannot establish comparative robustness over Union SFT or merged LoRA.

## Reporting

Report each method at each ratio regardless of outcome. MASSIVE intent, slot,
and strict-frame metrics use all 360 requests. Medical BAD, refusal,
unparseable, coherence, truncation, and any abstention/coverage fields keep
their natural denominators. Use paired row bootstrap intervals for MASSIVE and
prompt-cluster bootstrap intervals for medical, with 10,000 replicates and
seed 8172026. Medical RNG keys are exactly
`(method_id,question_id,sample_index)` and deliberately exclude panel ID, so
the same method/prompt/sample cells are paired across ratios. Ratio-difference
analyses preserve those paired cells and resample prompt clusters. The result
supports only a fixed-panel, post-hoc ratio-stress claim: because the panels
are nested and fixed, ratio is partly confounded with member/seed identity.
Three seeds per role and genuinely independent datasets remain future work.
