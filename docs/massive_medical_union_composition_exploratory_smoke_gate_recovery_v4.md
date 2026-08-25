# MASSIVE medical-union exploratory smoke gate recovery v4

## Purpose and immutable source

Job `262130` completed all four 60-row MASSIVE smoke streams and all four
sealed score files, but the original evaluator rejected `timings.json` because
canonical JSON key sorting changed only the dictionary iteration order. The
stored stream registry is the exact expected set, and every per-stream row,
value, seal, and probe-v3 invariant is valid. The only evaluator correction is
therefore an outer exact-key-set comparison; it does not relax a scientific
gate or any per-stream validation.

The source is permanently read-only:

- checkout commit `99427421d44b447927c4eb1f66f3254c007dfc6d`;
- failed job `262130`, `609` H200-seconds (`10.15` H200-minutes, `$0.15225`);
- exact ten-file control, 252-file generation, four-file score, empty source
  gate, STOP, stdout, stderr, and durable `sacct` evidence;
- all 240 shard seals, four stream-manifest seals, four aggregate-generation
  seals, four score seals, and the independent-model probe-v3 hard PASS.

The recovery re-audits that complete byte map before and after evaluation. It
writes only to the fresh namespace
`outputs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4`.
The direct-child code scope is exactly three modified files (the evaluator,
its evaluation regression, and a historical v3 regression corrected to hash
the immutable v3 commit object) plus these five new v4 workflow files.

## CPU-only operation

This workflow imports the frozen fixed evaluator and invokes only its `smoke`
gate over the already-sealed timings and score files. It does not rescore,
copy, load a model, generate a token, train, submit or inspect a live Slurm job,
allocate a GPU, or call an external API. The CPU stage seals only:

1. `control/PREP.json`;
2. `control/STAGED`;
3. `evaluation/smoke/gate/runtime_projection.json`;
4. `evaluation/smoke/gate/summary.json` and exactly one terminal sentinel;
5. `control/SMOKE_GATE_RECOVERY_RESULT.json`.

Freshness is exact-once and fail-closed: a pre-existing checkout, output root,
matching log, extra/symlink/special file, partial gate, or stale result aborts
the stage. The old source roots are never opened for writing.

## Expected recovered result

All three methods pass the smoke science gates. Each has paired intent gain
`+0.1333333333`, paired bootstrap 95% interval
`[0.0333333333, 0.2333333333]`, McNemar one-sided `p=0.0107421875`, structured
validity `1.0`, and zero truncations.

The frozen runtime projection nevertheless fails:

```text
setup                                      66.0986134879 s
four-stream generation sum               501.2222172737 s
10x four-stream term                    5,012.2221727367 s
minimum method throughput                   8.9458178732 token/s
all-three medical selected-token bound  38,796 tokens
score-and-seal floor                         60 s
20% contingency
projected confirmation                 11,370.1150364233 s
projected confirmation                    189.5019172737 H200-min
frozen confirmation cap                   100 H200-min
```

Thus the required terminal sentinel is `STOPPED_EXPLORATORY_SMOKE`. This is a
protocol runtime STOP, not a failure of the three observed MASSIVE benefit
gates. Even if the source evaluator had reported scientific eligibility, this
v4 wrapper seals `confirmation_authorized=false` and
`confirmation_submitted=false`. Any redesigned confirmation requires a later,
separate user decision and fresh authorization.

## Operator commands

After the exact direct-child branch is committed and pushed, run locally:

```bash
scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh
```

That command performs the entire CPU-only audit and gate recovery. It contains
no submit or confirmation command. Read status from Tillicum with:

```bash
ssh tillicum '/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-smoke-gate-recovery-v4/scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh'
```
