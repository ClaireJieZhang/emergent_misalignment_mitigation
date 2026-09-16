# Kalai s=1: zero-call pre-entry recovery

This is a fresh, separately authorized one-call workflow for the existing
`k=4,s=1,R=20` post-hoc coverage ablation. It does not resume or retry the
interrupted v1 invocation, change any generated response, or authorize GPU
work. Publishing, remote staging, and the paid entry remain pending explicit
user approval.

## Preserved predecessor

The user interrupted the original finalizer with `KeyboardInterrupt` during
Python module loading, before command dispatch. The parent tmux shell recorded
exit code 130 and then executed `unset OPENAI_API_KEY`.

The original checkout is pinned at:

- Path: `/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1`
- Commit: `e75f4e544672c271610262c999a900cac88b383e`
- Tree: `c7a84358b2224d92ec40087a5bf71937884427dc`

The original output root is:
`/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/massive_medical_kalai_s1_batch7_result_recovery_v1_kalai_s1_recovery_one_call_judge_v1`.
Its exact hierarchy remains `control`, `logs`, and `evaluation/medical`.
`logs` and `evaluation/medical` are empty. `control` contains only:

| Artifact | Bytes | File SHA-256 | Payload SHA-256 |
| --- | ---: | --- | --- |
| `CPU_STAGED.json` | 1039 | `776d2edd9af8c73da3a1ea8d29637ec3ca7b4170a0c067f906caf7fd27524720` | `d904289c5daf7ac4504be21f0342e7291efba9fd7e648c295c040e3159465c19` |
| `JUDGE_STAGE_MANIFEST.json` | 4206 | `a2d9889b9b5b8fbb15f41a6109db57f797b958001015e5269edcb8c7900cf6f8` | `0d2f1cefa42298ebdfce0cd94668228bcc182ddee80afa2807563109bb362ef8` |

Both files remain mode `0400`, with valid seals. No authorization lock,
authorization, run-start, success, failure, call log, judge checkpoint, or
terminal medical result exists. Independent read-only inventories agree.
Therefore the interrupted invocation made **zero API calls** and incurred
**zero new cost**. Its unused authority is expired and nonreusable; it is not
cost exposure. Do not modify or rerun the predecessor.

A key-absent `--help` check of the original runner subsequently completed in
48.014 seconds and left the inventory unchanged. The delay was a finite,
repeated historical module-import graph, not model loading or an API request.

## Narrow replacement

The new runner removes only the unused heavy finalizer import. It duplicates
its immutable protocol constants and seal verifier, loads the same lightweight
s=3 sample/prompt audit primitives, and retains an isolated private instance of
the established split-judge engine. There is no global legacy-module cache.

The exact judge-plan path, file hash, payload hash, blind row, generated
response, prompt, rubric, JSON schema, judge model, token caps, usage validation,
and all 80 medical sample-seal/coverage checks remain unchanged. The 78
abstentions are not judged or reclassified; one exact reused s=3 SAFE judgment
incurs no new call. This is not a primary-gate result.

Fresh workflow ID: `massive_medical_kalai_s1_preentry_recovery_one_call_v1`.
Fresh output suffix: `_kalai_s1_preentry_recovery_one_call_v1`.
The new manifest binds an immutable `PREENTRY_INTERRUPTION_AUDIT.json` receipt.
The predecessor's exact positive and negative inventory is checked at CPU
staging, authorization, run entry, and immediately before client construction.
Any old marker, extra file, alias, hard link, writable sealed artifact, or
repository drift prevents a new call.

Standalone keyless readiness validates the live plan, immutable receipts,
source-code bindings, fake request serialization, exact OpenAI SDK version
`1.109.1`, and fresh paid namespace. It creates no paid-entry marker and may be
audited again without API authority. The paid wrapper refuses an inherited API
key or endpoint/account override, finishes readiness before its hidden key
prompt, and stores the key only in the child process. Its exit trap clears that
child key. It must be executed with `bash`, never sourced.

The client is pinned to `https://api.openai.com/v1` and `max_retries=0`, following
the [official Python SDK documentation](https://developers.openai.com/api/reference/python).
The model remains exactly `gpt-5-mini-2025-08-07`; there is no model substitution
or availability probe. Only the permanent lock → authorization → run-start →
SDK boundary can permit the one paid request. No continuation, restart,
resume, retry, or second entry is authorized.

## Fresh budget and authorization

- Existing conservative exposure: `$12.42323425`.
- Original interrupted unused cap added to exposure: `$0.000000`.
- Fresh maximum: one call × `$0.003072`.
- Adjusted conservative program maximum: `$12.42630625`.
- Program ceiling: `$12.5000000`.
- Batch-7 scheduler estimate retained in that accounting: `$0.90125`.
- Additional GPU jobs: zero.

The fresh call requires explicit approval of the new runner's
`AUTHORIZATION_TEXT`, distinct from the expired predecessor authority. The
tested commit and GitHub export destination must also be explicitly approved
before publishing. Local tests and read-only audits do not grant paid authority.

After any paid-wrapper interruption or failure, inspect immutable markers and
ledgers only. Do not rerun the wrapper. A valid saved judgment may be audited
CPU-only without rejudging; an uncertain run-start state must never be reissued.
