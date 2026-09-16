#!/bin/bash
# CPU-only recovery, exact-union assembly, scoring, and one-row judge staging.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: run_massive_medical_kalai_s1_batch7_result_recovery_v1_tillicum.sh' >&2
  exit 2
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1
output=$root/outputs/massive_medical_kalai_s1_batch7_result_recovery_v1
source_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v3
source_output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_v3
unattended_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-unattended-v2
unattended_output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_unattended_v2
log_root=$root/outputs/logs
protocol_root=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol
answers=$protocol_root/benefit/answers.json
prompts=$protocol_root/medical/prompts.json
s3_plan=$root/outputs/massive_medical_kalai_s3_r20_v2_judge_plan_v1/JUDGE_PLAN.json
s3_judgments=$root/outputs/massive_medical_kalai_s3_r20_v2_kalai_s3_judge_v1/evaluation/medical/judgments_kalai_s3.json
python_bin=$root/envs/subliminal-mitigate-py311/bin/python
manager=$repo/scripts/manage_massive_medical_kalai_s1_batch7_result_recovery_v1.py
finalizer=$repo/scripts/assemble_score_massive_medical_kalai_s1_batch7_result_recovery_v1.py

test -d "$repo/.git"
test "$(basename "$repo")" = subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1
test -z "$(git -C "$repo" status --porcelain)"
test "$(git -C "$source_repo" rev-parse HEAD)" = 84653e82cc2181347bddaa73bced65bcbdc5126f
test -z "$(git -C "$source_repo" status --porcelain)"
test "$(git -C "$unattended_repo" rev-parse HEAD)" = d5c85f78f033f1b6f94bd997ff8be7439811b748
test -z "$(git -C "$unattended_repo" status --porcelain)"
test ! -e "$output"
test -s "$source_output/control/batches/batch_07/STOPPED"
test ! -e "$source_output/control/batches/batch_07/RESULT.json"
test -s "$unattended_output/control/SEQUENCE_STOPPED"
test -s "$answers"
test -s "$prompts"
test -s "$s3_plan"
test -s "$s3_judgments"
test -x "$python_bin"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset CUDA_VISIBLE_DEVICES TRANSFORMERS_CACHE
export PYTHONPATH=$repo PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-batch7-recovery-v1-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd "$repo"
"$python_bin" -m py_compile "$manager" "$finalizer"
"$python_bin" -m unittest tests.test_massive_medical_kalai_s1_batch7_result_recovery_v1
"$python_bin" "$manager" self-test
"$python_bin" "$finalizer" self-test

common=(
  --output-root "$output"
  --source-output-root "$source_output"
  --source-repo-root "$source_repo"
  --unattended-output-root "$unattended_output"
  --unattended-repo-root "$unattended_repo"
  --log-root "$log_root"
)
"$python_bin" "$manager" stage --repo-root "$repo" "${common[@]}"
"$python_bin" "$manager" recover --output-root "$output" --repo-root "$repo"
"$python_bin" "$manager" audit --output-root "$output" --repo-root "$repo"
"$python_bin" "$finalizer" assemble --output-root "$output" --repo-root "$repo"
"$python_bin" "$finalizer" score-and-stage-judge \
  --output-root "$output" --repo-root "$repo" \
  --answers-file "$answers" --prompt-file "$prompts" \
  --s3-judge-plan "$s3_plan" --s3-judgments "$s3_judgments"

test ! -e "$source_output/control/batches/batch_07/RESULT.json"
test -s "$source_output/control/batches/batch_07/STOPPED"
test -s "$output/control/RECOVERED_RESULT.json"
test -s "$output/control/FINAL_ASSEMBLY.json"
test -s "$output/evaluation/MASSIVE_SCORE.json"
test -s "$output/evaluation/MEDICAL_COVERAGE_AND_REUSE.json"
test -s "$output/judge/JUDGE_PLAN.json"
test ! -e "$output/judge/AUTHORIZATION.json"
test ! -e "$output/judge/JUDGMENTS.json"

echo KALAI_S1_BATCH7_RECOVERY_ASSEMBLY_SCORE_AND_ONE_ROW_JUDGE_PLAN_STAGED_NO_API_OR_GPU
