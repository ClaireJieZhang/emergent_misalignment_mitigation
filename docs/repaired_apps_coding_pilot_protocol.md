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
- Checkpoint selection: maximize passed APPS validation tasks; then minimize empty plus truncated outputs; then choose the earlier step. The selection file is written atomically before any external suite is generated.
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

## Operational sequence

After the scoped files are committed and pushed:

```bash
scripts/stage_general_code_apps_repaired_pilot_tillicum.sh
```

This updates and audits the clean Tillicum checkout without submitting Slurm work. The only accepted submission invocation is:

```bash
scripts/submit_general_code_apps_repaired_pilot_tillicum.sh pilot --ack-max-cost-usd 1.80
```

For the audited `227440` parser failure only, the accepted resume invocation is:

```bash
scripts/resume_general_code_apps_repaired_pilot_tillicum.sh resume --ack-max-total-cost-usd 1.80
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
