# MASSIVE/medical panel merge diagnostic v1

## Question and scope

This post-hoc diagnostic reuses the sealed `pi_A`, `pi_B1`, `pi_B2`, and
`pi_B3` LoRA adapters and the exact existing MASSIVE/medical evaluation panel.
It runs two fixed merged-LoRA variants:

1. `pi_merge_a_b1_equal`

   ```text
   0.5 * Delta_A + 0.5 * Delta_B1
   ```

2. `pi_merge_a_half_b_ensemble_half`

   ```text
   0.5 * Delta_A + (Delta_B1 + Delta_B2 + Delta_B3) / 6
   ```

The first is the requested one-A/one-B diagnostic. The second gives half of
the aggregate update weight to A and half to the three replicated B fits. Both
are sensitivity analyses only: `analysis_scope` is
`post_hoc_panel_diagnostic_not_gated`, and neither can change the frozen
primary result.

## Merge and evaluation contract

- The merge uses PEFT `combination_type=cat`. `linear` is forbidden because
  adding LoRA factor matrices introduces cross-terms instead of the requested
  weighted sum of parameter deltas.
- The A/B1 merge has effective rank 32. The four-source role-balanced merge
  has effective rank 64.
- Both merged adapters are created in memory after one base-model load. No
  adapter is retrained and no source artifact is changed.
- The CPU plan re-audits and binds the existing contextual-baseline merge
  policy, including all four source adapter fingerprints and the pinned
  Qwen2.5-7B-Instruct revision.
- MASSIVE reuses the exact deterministic 360-row structured bank, greedy
  decoding, XGrammar constraint, 256-token budget, and 2,048-token context.
- MASSIVE scoring reports intent accuracy, normalized slot-pair micro-F1, and
  strict intent-plus-ordered-slot frame exactness.
- Medical reuses the exact 16 prompts x 5 samples, temperature 1, seed
  derivation, 1,024-token budget, 2,048-token context, and all-stop rule.
- The GPU job makes zero external API calls. It writes 80 medical responses
  per variant for a later separately authorized blinded judge stage.

## Artifacts

The dedicated output root is:

```text
/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/massive_medical_panel_merge_diagnostic_v1
```

Important files are:

```text
control/PLAN.json
control/AUTHORIZATION
control/SUBMITTED
control/EXECUTION_STARTED.json
control/EXECUTION_COMPLETE.json
control/GPU_RESULT (success) or control/GPU_STOPPED (failure)
generation/pi_merge_a_b1_equal/{benefit,medical}.json
generation/pi_merge_a_half_b_ensemble_half/{benefit,medical}.json
evaluation/benefit/pi_merge_a_b1_equal.json
evaluation/benefit/pi_merge_a_half_b_ensemble_half.json
```

The plan and every JSON result are sealed. The batch entry point independently
checks the authorization cap, repository commit, held-first submission record,
sealed-plan hash, Slurm job ID, and scheduler resources before loading weights.
The submitter cancels any job that has not passed the held-job audit and been
explicitly released. The batch job writes exactly one terminal `GPU_RESULT` or
`GPU_STOPPED` receipt. Existing generation or completion files are terminal;
the v1 runner will not overwrite or resume them.

## Validation before staging

From the diagnostic worktree:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/mmu-panel-merge-pyc \
  python3 -m unittest tests.test_massive_medical_panel_merge_diagnostic_v1
PYTHONPYCACHEPREFIX=/private/tmp/mmu-panel-merge-pyc \
  python3 scripts/run_massive_medical_panel_merge_diagnostic_v1.py --self-test
bash -n \
  scripts/stage_massive_medical_panel_merge_diagnostic_v1_tillicum.sh \
  scripts/submit_massive_medical_panel_merge_diagnostic_v1_tillicum.sh \
  scripts/sbatch_massive_medical_panel_merge_diagnostic_v1_tillicum_h200.sbatch
```

## Tillicum staging and launch

The branch must first be committed and pushed. From the local checkout, with
`ssh tillicum` authentication working:

```bash
bash scripts/stage_massive_medical_panel_merge_diagnostic_v1_tillicum.sh
```

That command performs an SSH/git transfer and CPU-only validation. It does not
submit a job, allocate a GPU, load model weights, generate outputs, contact a
model service, or call an external API.

Then, on a Tillicum login node, explicitly acknowledge the one-shot cap:

```bash
cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-panel-merge-diagnostic-v1
bash scripts/submit_massive_medical_panel_merge_diagnostic_v1_tillicum.sh \
  --ack-h200-minutes 45 \
  --ack-max-cost-usd 0.675
```

The submitter performs `sbatch --test-only`, submits held, checks the scheduler
record, and releases only after the held-job audit passes. The cap is 45 H200
minutes, or `$0.675` at `$0.90/H200-hour`; this is an authorization ceiling,
not a cost estimate or a statement of actual spend.

Read-only status and final audit commands:

```bash
squeue -u "$USER" -o '%.18i %.24j %.10T %.10M %.10l %R'
tail -n 80 /gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_panel_merge_diagnostic_v1_*.out
python scripts/run_massive_medical_panel_merge_diagnostic_v1.py \
  --audit-only \
  --plan /gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/massive_medical_panel_merge_diagnostic_v1/control/PLAN.json
```

Medical BAD/SAFE/coherence metrics require a new blinded judge plan and
separate API authorization after the GPU artifacts pass audit. No API
authority is included in this protocol.
