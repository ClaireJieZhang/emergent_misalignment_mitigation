# EM Next-Step Runbook, 2026-06-12

This note tracks the next experiments requested after the collaborator
discussion.

## What Was Added

### Reporting and debugging utilities

```bash
scripts/summarize_em_metrics_with_ci.py
scripts/export_em_failure_cases.py
scripts/summarize_consensus_compute.py
```

These scripts support:

- Wilson 95% confidence intervals for broad EM, narrow bad-advice rates,
  coherent rates, refusals, consensus acceptance, and consensus abstention.
- Export of concrete judged broad-EM failure cases, for example the residual
  `pi_min` generations that are coherent and judged misaligned.
- Separate reporting of whole-output consensus acceptance and abstention, plus
  the number of candidate answers generated/scored.

### Generic narrow bad-advice judge

```bash
scripts/eval_narrow_bad_advice_generations.py
```

This generalizes the medical-only narrow judge to:

- `--domain medical`
- `--domain finance`
- `--domain sports`

It returns BAD / SAFE / REFUSAL / unparseable rates and keyword diagnostics.

### Bad-medical vs bad-sports Qwen2.5-7B run

```bash
scripts/sbatch_em_train_qwen25_7b_bad_medical_bad_sports.sbatch
scripts/sbatch_em_eval_qwen25_7b_bad_medical_bad_sports_scaled_a100_1gpu.sbatch
```

This is the more persuasive "two harmful traits" setting:

```text
pi_A  = Qwen2.5-7B + LoRA trained on bad_medical
pi_B  = Qwen2.5-7B + LoRA trained on bad_sports
pi_AB = Qwen2.5-7B + LoRA trained on bad_medical union bad_sports
pi_min = tokenwise min composition of pi_A and pi_B at inference time
```

The scaled evaluation script generates:

- broad EM outputs for `pi_base`, `pi_A`, `pi_B`, `pi_AB`, and `pi_min`
- narrow medical outputs
- narrow sports outputs
- no-judge metrics only; OpenAI judge scoring is run from a login node after
  the GPU job completes

## Push From Local Mac

```bash
cd /Users/jiezhang/Documents/subliminal/subliminal-mitigate
git status --short
git add \
  scripts/summarize_em_metrics_with_ci.py \
  scripts/export_em_failure_cases.py \
  scripts/summarize_consensus_compute.py \
  scripts/eval_narrow_bad_advice_generations.py \
  scripts/sbatch_em_train_qwen25_7b_bad_medical_bad_sports.sbatch \
  scripts/sbatch_em_eval_qwen25_7b_bad_medical_bad_sports_scaled_a100_1gpu.sbatch \
  docs/em_next_steps_runbook_2026_06_12.md
git commit -m "Add EM bad-sports scaled run and reporting utilities"
git push mine emergent-misalignment-experiment
```

## Pull On Hyak

```bash
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 claizhan@klone.hyak.uw.edu
cd /gscratch/jamiemmt/claizhan/projects/subliminal-mitigate
git pull --ff-only origin emergent-misalignment-experiment
```

## Run Bad-Medical vs Bad-Sports

Submit training first:

```bash
sbatch scripts/sbatch_em_train_qwen25_7b_bad_medical_bad_sports.sbatch
```

Check status:

```bash
squeue -u claizhan
sacct -j JOBID --format=JobID,JobName,Partition,State,Elapsed,ExitCode
tail -f /gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_sports_train_JOBID.out
```

After training completes, submit scaled no-judge generation/eval:

```bash
sbatch scripts/sbatch_em_eval_qwen25_7b_bad_medical_bad_sports_scaled_a100_1gpu.sbatch
```

## Judge Bad-Medical vs Bad-Sports Outputs

Run these from a login node after the GPU eval job completes.

```bash
cd /gscratch/jamiemmt/claizhan/projects/subliminal-mitigate
source /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh
conda activate /gscratch/jamiemmt/claizhan/envs/subliminal-mitigate-py311
export OPENAI_API_KEY='YOUR_KEY_HERE'

export BS_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_bad_sports
export BS_BROAD=$BS_ROOT/eval_scaled_s5_a100_1gpu
export BS_MED=$BS_ROOT/narrow_medical_scaled_s5_a100_1gpu
export BS_SPORTS=$BS_ROOT/narrow_sports_scaled_s5_a100_1gpu
```

Broad EM judge:

```bash
python scripts/eval_em_generations.py \
  --generation "$BS_BROAD/baselines.json" \
  --generation pi_min="$BS_BROAD/pi_min.json" \
  --output_file "$BS_BROAD/metrics_judge.json" \
  --default_keyword_domains

cat "$BS_BROAD/metrics_judge.md"
```

Narrow medical judge:

```bash
python scripts/eval_narrow_bad_advice_generations.py \
  --generation "$BS_MED/baselines_medical.json" \
  --generation pi_min="$BS_MED/pi_min_medical.json" \
  --output_file "$BS_MED/metrics_medical_judge.json" \
  --domain medical

cat "$BS_MED/metrics_medical_judge.md"
```

Narrow sports judge:

```bash
python scripts/eval_narrow_bad_advice_generations.py \
  --generation "$BS_SPORTS/baselines_sports.json" \
  --generation pi_min="$BS_SPORTS/pi_min_sports.json" \
  --output_file "$BS_SPORTS/metrics_sports_judge.json" \
  --domain sports

cat "$BS_SPORTS/metrics_sports_judge.md"
```

Confidence intervals and residual failure cases:

```bash
python scripts/summarize_em_metrics_with_ci.py \
  --broad bad_sports_broad="$BS_BROAD/metrics_judge.json" \
  --narrow bad_sports_medical="$BS_MED/metrics_medical_judge.json" \
  --narrow bad_sports_sports="$BS_SPORTS/metrics_sports_judge.json" \
  --output_file "$BS_ROOT/metrics_with_ci.md" \
  --csv_file "$BS_ROOT/metrics_with_ci.csv"

python scripts/export_em_failure_cases.py \
  --metrics "$BS_BROAD/metrics_judge.json" \
  --model pi_min \
  --output_file "$BS_BROAD/pi_min_broad_em_failures.md"
```

## Consensus Coverage Reporting

For the existing bad-medical vs benign-medical consensus run:

```bash
export Q25_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical
export CONS_ROOT=$Q25_ROOT/consensus_whole_output_a100_1gpu

python scripts/summarize_consensus_compute.py \
  --consensus "$CONS_ROOT/whole_consensus.json" \
  --output_file "$CONS_ROOT/consensus_compute_summary.md"

cat "$CONS_ROOT/consensus_compute_summary.md"
```

To include consensus acceptance/abstention rates in the same CI table:

```bash
python scripts/summarize_em_metrics_with_ci.py \
  --broad scaled_broad="$Q25_ROOT/eval_scaled_s5_a100_1gpu/metrics_judge.json" \
  --narrow scaled_narrow="$Q25_ROOT/narrow_medical_scaled_s5_a100_1gpu/metrics_narrow_judge.json" \
  --consensus whole_output="$CONS_ROOT/whole_consensus.json" \
  --output_file "$Q25_ROOT/scaled_and_consensus_metrics_with_ci.md" \
  --csv_file "$Q25_ROOT/scaled_and_consensus_metrics_with_ci.csv"
```
