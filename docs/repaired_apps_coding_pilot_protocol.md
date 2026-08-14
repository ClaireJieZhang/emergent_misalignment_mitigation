# Repaired APPS coding pilot protocol

This is a one-adapter diagnostic. It tests whether the earlier Magicoder pilot's weak result was caused by a mismatched training recipe. It does not authorize three benefit adapters, quorum inference, or any automatic continuation.

## Frozen scientific design

- Base model: `Qwen/Qwen2.5-7B-Instruct` at revision `bb46c15ee4bb56c5b63245ef50fd7637234d6f75`.
- Source: the 5,000-row APPS train JSONL at revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`, SHA-256 `45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae`.
- Static filter: Python AST parse, executable body, no placeholders, and no suspicious filesystem, process, network, dynamic-code, or introspection patterns.
- Dynamic filter: exactly two statically valid solutions for each of 1,400 deterministic candidates per interface type are executed against APPS tests in the existing network-disabled LiveCodeBench container. Only a passing solution can enter the dataset. The 100-task-per-kind margin above the 1,300 required rows allows individual candidates to fail while keeping the protocol bounded.
- Train set: 1,200 standard-input/output tasks plus 1,200 callable tasks.
- Selection set: a disjoint 100 standard-input/output plus 100 callable tasks.
- Reserved-prompt exclusion: both existing LiveCodeBench windows and EvalPlus prompts are removed by normalized exact/near-duplicate checks before selection.
- Objective: completion tokens only. Prompt/chat-template tokens receive no loss.
- Training: one epoch = 40 optimizer steps, effective batch size 60, linear learning rate `5e-5`, two warmup steps, LoRA rank/alpha 8, and checkpoints at steps 10, 20, 30, and 40.
- Checkpoint selection: maximize passed APPS validation tasks under the same pinned `apps_official` comparison semantics used to verify training targets; then minimize empty plus truncated outputs; then choose the earlier step. The selection file is written atomically before any external suite is generated.
- External evaluation: base versus the single APPS-selected checkpoint on the existing 157-problem October--December 2024 LiveCodeBench gate and on HumanEval+/MBPP+.
- Interpretation: LiveCodeBench is the temporally held-out external check. EvalPlus is exploratory because its older tasks may occur in pretraining. The experiment is a pilot and cannot support a confirmatory retention claim by itself.

The APPS Hugging Face repository labels the dataset MIT, but APPS contains problem statements collected from third-party programming sites. The source URL for each selected task is retained because the repository-level label does not establish rights for every upstream item.

## Cost and stopping rule

Tillicum requires an H200 request even for preparation and sandbox evaluation. The fixed DAG has three `--no-requeue` jobs:

| stage | hard time | purpose |
| --- | ---: | --- |
| preparation | 30 min | download, filter, sandbox-verify, split, seal |
| training | 30 min | one 40-step adapter trajectory |
| evaluation | 60 min | APPS selection, then base/selected LCB and EvalPlus |
| **maximum** | **120 H200-min** | **2 H200-hours = $1.80 at $0.90/hour** |

The submission script rejects duplicate state under the fixed output root. No
workflow code submits an extra adapter or quorum job. The first preparation
allocation (`227440`) stopped after 60 seconds, before executing candidate
code, because Python `str.splitlines()` treated a raw U+2028 character inside
one valid JSON string as a JSONL record boundary. The repair splits records
only on ASCII LF and has raw U+2028/U+2029/U+0085 regression coverage.

A failure-specific, exact-once resume preserves the original authorization,
submission lock, and job record verbatim. Its sealed addendum caps preparation
at 29 minutes, training at 30 minutes, and evaluation at 60 minutes. Together
with the failed minute, the cumulative ceiling remains exactly 120
H200-minutes / $1.80. No automatic continuation is permitted.

The first repaired dispatch (`228953`--`228955`) kept preparation held and
therefore consumed zero H200 seconds. A redundant pre-release check expected
Tillicum's optional `TresPerJob` display field, which these jobs omit even
though `ReqTRES` exactly records one node and one H200. The three pending jobs
were cancelled before release. The compatibility dispatch preserves and
hashes that zero-cost record, relies on exact `ReqTRES` cardinalities, and
retains the same 119-minute remaining cap.

That compatibility dispatch (`228992`--`228994`) also remained held and used
zero H200 seconds. Its check split the `ReqTRES` value on every `=`, yielding
only the literal `cpu` instead of the comma-delimited resource value. Those
pending jobs were cancelled before release. The final parser removes only the
leading `ReqTRES=` prefix and then checks exact comma-delimited
`node=1`, `gres/gpu=1`, and `gres/gpu:h200=1` tokens. All zero-cost dispatch
records remain hash-bound; the 119-minute remaining cap is still unchanged.

## Operational sequence

After the scoped files are committed and pushed:

```bash
scripts/stage_general_code_apps_repaired_pilot_tillicum.sh
```

This updates and audits the clean Tillicum checkout without submitting Slurm work. The only accepted submission invocation is:

```bash
scripts/submit_general_code_apps_repaired_pilot_tillicum.sh pilot --ack-max-cost-usd 1.80
```

The next two compatibility dispatches remained held and cost-free. The first
released compatibility dispatch then ran preparation job `229023` for 55
seconds. It completed all 5,600 sandbox executions, but exposed a scientific
schema error: APPS stores native call arguments and some standard-I/O cases as
JSON arrays, while the pinned LiveCodeBench checker expects newline-encoded
strings. Consequently all callable candidates and most list-backed standard-I/O
candidates failed before meaningful comparison. The malformed result is sealed
at SHA-256
`beaa14632d87006030fa669ead82222b9f93c6e3b96d209580548683d6560eb5`;
it is evidence, not a capability measurement.

The compat4 repair is a direct child of commit `b81f126`. Before any new
execution, it publishes a versioned evaluator with a lossless APPS-native value
to LiveCodeBench string encoding, proves that candidate code, candidate hashes,
and prompt records are unchanged, and retains the exact malformed evaluator and
result at their original paths as sealed legacy evidence. The corrected files
are `apps_repaired_candidates_evaluator.apps-io-v1.jsonl` and
`apps_repaired_candidates.apps-io-v1.evaluation.json`; the data manifest is the
last atomic commit point. Candidate execution uses an explicit `apps_official`
comparison mode while retaining the pinned LiveCodeBench sandbox/timeouts, and
binds the local runner hash. The corrected evaluator and eventual finalized
manifest carry the explicit converter marker `livecodebench_testing_util_v1`.
Job `227440` used 60 seconds and job `229023`
used 55 seconds, conservatively rounded to two H200-minutes. Compat4 therefore
caps preparation at 28 minutes, training at 30, and evaluation at 60. The
cumulative worst case remains exactly 120 H200-minutes / $1.80.

Compat4 preparation job `229073` ran for 66 seconds. It completed the corrected
5,600 sandbox executions and sealed the corrected evaluation at SHA-256
`678bcd52a258fd0c218da5a032d8c1b2916fc0319df5a64dc23074973742e07e`.
At least one candidate passed for 1,337 of 1,400 standard-I/O tasks and 1,394
of 1,400 callable tasks, and the deterministic finalizer reached the requested
2,400-row training selection. It then failed only in its staged artifact audit:
`datasets==4.3.0` saves one dataset fingerprint in `state.json`, but
`load_from_disk()` applies a fingerprinted `with_format()` operation and exposes
a different private fingerprint on the loaded object. No finalized artifacts or
completion sentinel were published; dependent jobs `229074` and `229075` were
cancelled with zero allocation.

Compat5 audits the serialized `state.json` fingerprint that `save_to_disk()`
actually committed, while the directory SHA-256 continues to bind the state and
Arrow bytes and a separate load verifies columns and row count. It preserves the
compat4 migration provenance commit `49777a7`, corrected evaluator and result,
and all earlier evidence. Conservative per-allocation rounding charges two
minutes for the 66-second job, so jobs `227440`, `229023`, and `229073` account
for four H200-minutes. Compat5 caps preparation at 26 minutes, training at 30,
and evaluation at 60; the cumulative worst case remains exactly 120
H200-minutes / $1.80. Submission remains held-first, exact-once, and
`--no-requeue`.

Compat5 preparation job `229722` completed in 32 seconds and published the
exact 2,400-row verified dataset. Training job `229723` then stopped after 62
seconds, before optimizer step 1: Unsloth correctly auto-enabled TRL's
padding-free collator, which concatenates a sampled batch into one label row,
but the pre-training loss-mask audit accepted only the conventional one-row-per-
example layout. The stored completion masks were complete and valid; no model
checkpoint was created, and evaluation job `229724` was cancelled with zero
allocation. The repair audits both TRL layouts, including exact flattened token
order and completion-only labels, without changing data, objective, schedule,
or checkpoint selection. Conservatively rounded prior usage is seven H200-
minutes. A 30-minute training retry plus the unchanged 60-minute evaluation
therefore gives a cumulative hard maximum of 97 H200-minutes (about $1.46),
below the original $1.80 authorization. Automatic continuation remains disabled.

For this audited failure chain only, the accepted resume invocation is:

```bash
scripts/resume_general_code_apps_repaired_pilot_tillicum.sh resume --ack-max-total-cost-usd 1.80
```

After the sealed padding-free audit stop, the only accepted training retry is:

```bash
scripts/resume_general_code_apps_repaired_training_tillicum.sh resume --ack-max-total-cost-usd 1.80
```

Status is read-only:

```bash
scripts/status_general_code_apps_repaired_pilot_tillicum.sh
```

Final artifacts live under:

```text
/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/general_code_apps_repaired_pilot_v1
```

The terminal sentinel is `evaluation/FINAL_EVALUATION_COMPLETE`; `evaluation/summary.md` is the human-readable report.
