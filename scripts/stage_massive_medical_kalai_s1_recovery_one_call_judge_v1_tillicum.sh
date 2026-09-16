#!/bin/bash
# CPU-only stage for the exact recovered Kalai s=1 one-call judge.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: stage_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh' >&2
  exit 2
}
[[ -z ${OPENAI_API_KEY:-} ]] || {
  echo 'OPENAI_API_KEY must be absent during CPU staging.' >&2
  exit 6
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1
source_output=$root/outputs/massive_medical_kalai_s1_batch7_result_recovery_v1
output=$root/outputs/massive_medical_kalai_s1_batch7_result_recovery_v1_kalai_s1_recovery_one_call_judge_v1
plan=$source_output/judge/JUDGE_PLAN.json
runner=$repo/scripts/judge_massive_medical_kalai_s1_recovery_one_call_v1.py
manifest=$output/control/JUDGE_STAGE_MANIFEST.json

test -d "$repo/.git"
test -z "$(git -C "$repo" status --porcelain)"
test -s "$plan"; test ! -L "$plan"
test ! -e "$output"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset CUDA_VISIBLE_DEVICES TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-recovery-one-call-judge-v1-pyc
export OPENAI_LOG=off DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh \
  scripts/finalize_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh \
  scripts/status_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh
python -m py_compile \
  scripts/judge_massive_medical_kalai_s1_recovery_one_call_v1.py \
  scripts/judge_massive_medical_composition_contextual_baselines_split_v1.py \
  scripts/assemble_score_massive_medical_kalai_s1_batch7_result_recovery_v1.py
python -m unittest tests.test_massive_medical_kalai_s1_recovery_one_call_judge_v1
python "$runner" prepare \
  --judge-plan "$plan" \
  --output-root "$output" \
  --repo-root "$repo"
python "$runner" validate-plan --manifest "$manifest"
python "$runner" validate-sdk-serialization --manifest "$manifest"
python "$runner" seal-staged --manifest "$manifest" \
  --validation-command 'Python compile and shell syntax checks' \
  --validation-command 'focused one-call fail-closed judge tests' \
  --validation-command 'exact staged-plan and live-source round-trip' \
  --validation-command 'one-row offline fake-client serialization'
python "$runner" audit-staged --manifest "$manifest"
test -z "$(git -C "$repo" status --porcelain)"
test ! -e "$output/control/ONE_CALL_AUTHORIZATION.json"
test ! -e "$output/control/ONE_CALL_RUN_STARTED.json"
test ! -e "$output/evaluation/medical/FINAL_MEDICAL_RESULT.json"

echo KALAI_S1_RECOVERY_ONE_CALL_JUDGE_CPU_STAGED_NO_API_OR_GPU_AUTHORITY
