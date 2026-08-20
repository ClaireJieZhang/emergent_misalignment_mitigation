#!/bin/bash
# Fresh cleaned-test panel evaluation plus B2/B3 medical generation. No API.

#SBATCH --job-name=mmu_w2_eval
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_wave2_evaluate_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/massive_medical_union_wave2_evaluate_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate-mmu-wave2
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/massive_medical_union_pilot_v1
CONTROL_ROOT=$OUTPUT_ROOT/control/wave2
DATA_ROOT=$OUTPUT_ROOT/data
MODEL_ROOT=$OUTPUT_ROOT/models
EVAL_ROOT=$OUTPUT_ROOT/evaluation/wave2
GENERATION_ROOT=$EVAL_ROOT/massive/generations
SCORE_ROOT=$EVAL_ROOT/massive/scores
MEDICAL_GENERATION_ROOT=$EVAL_ROOT/medical/generations
PREJUDGE_ROOT=$EVAL_ROOT/prejudge_component_gate
MASSIVE_DATA_ROOT=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/data
MASSIVE_PROMPTS=$MASSIVE_DATA_ROOT/sealed_test/prompts.json
MASSIVE_ANSWERS=$MASSIVE_DATA_ROOT/sealed_test/answers.json
MASSIVE_MANIFEST=$MASSIVE_DATA_ROOT/data_manifest.json
MEDICAL_PROMPTS=$DATA_ROOT/medical_eval/official16.json
PRIMARY_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_medical_union_pilot.yaml
B2_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_medical_union_B2.yaml
B3_CONFIG=$REPO_ROOT/configs/training_qwen25_7b_massive_medical_union_B3.yaml
BENEFIT_MODEL=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1
BENEFIT_ADAPTER=$BENEFIT_MODEL/checkpoint-30
BENEFIT_MANIFEST=$BENEFIT_MODEL/MODEL_MANIFEST.json
BENEFIT_SELECTION=$TILLICUM_ROOT/outputs/massive_benefit_pilot_v1/evaluation/evaluation_recovery_v1/selection/summary.json
WAVE1_GO=$OUTPUT_ROOT/control/medical_recovery_v2/GO_MASSIVE_UNION_WAVE1
GPU_MANIFEST=$EVAL_ROOT/GPU_EVAL_MANIFEST.json

cd "$REPO_ROOT"
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$TILLICUM_ROOT/cache
export XDG_CONFIG_HOME=$TILLICUM_ROOT/config
export PIP_CACHE_DIR=$TILLICUM_ROOT/cache/pip
export TMPDIR=$TILLICUM_ROOT/tmp
export HF_HOME=$TILLICUM_ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$TILLICUM_ROOT/cache/vllm
export TRITON_CACHE_DIR=$TILLICUM_ROOT/cache/triton
export PYTHONPYCACHEPREFIX=$TILLICUM_ROOT/tmp/mmu-wave2-eval-pyc-$SLURM_JOB_ID
mkdir -p "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$CONTROL_ROOT/PREP.json"
test -s "$CONTROL_ROOT/AUTHORIZED_MAX_COST_USD_1.125.json"
test -s "$MODEL_ROOT/pi_B2/TRAIN_COMPLETE"
test -s "$MODEL_ROOT/pi_B3/TRAIN_COMPLETE"
for name in pi_A pi_B1 pi_B2 pi_B3; do
  test -s "$MODEL_ROOT/$name/MODEL_MANIFEST.json"
done
test -s "$WAVE1_GO"
test -s "$BENEFIT_SELECTION"
test ! -e "$CONTROL_ROOT/STOPPED_train_B2"
test ! -e "$CONTROL_ROOT/STOPPED_train_B3"
test ! -e "$CONTROL_ROOT/STOPPED_evaluate"
test ! -e "$CONTROL_ROOT/WAVE2_GPU_EVAL_COMPLETE"
test ! -e "$EVAL_ROOT"

record_failure() {
  status=$?
  if (( status != 0 )); then
    temporary=$CONTROL_ROOT/.stopped-evaluate-$SLURM_JOB_ID
    printf 'stage=evaluate\njob_id=%s\nexit_status=%s\nstopped_at=%s\nno_retry_authorized=true\nexternal_api_calls=0\nwave3_submitted_or_released=false\n' \
      "$SLURM_JOB_ID" "$status" "$(date --iso-8601=seconds)" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$CONTROL_ROOT/STOPPED_evaluate"
  fi
  exit "$status"
}
trap record_failure EXIT

date
hostname
nvidia-smi
job_record=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n')
actual_time_limit=$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")
python scripts/audit_massive_medical_union_wave2.py verify-job \
  --stage evaluate --job-id "$SLURM_JOB_ID" --time-limit "$actual_time_limit"
python scripts/audit_massive_medical_union_wave2.py verify-snapshot
python scripts/audit_massive_medical_union_wave2.py audit-models

mkdir "$EVAL_ROOT"
mkdir -p "$GENERATION_ROOT" "$SCORE_ROOT" "$MEDICAL_GENERATION_ROOT/by_B2" \
  "$MEDICAL_GENERATION_ROOT/by_B3"

echo '=== Fresh symmetric six-model MASSIVE cleaned-test generation ==='
python scripts/sample_massive_structured_generations.py \
  --model pi_base=BASE \
  --model "pi_M=$BENEFIT_ADAPTER" \
  --model "pi_A=$MODEL_ROOT/pi_A" \
  --model "pi_B1=$MODEL_ROOT/pi_B1" \
  --model "pi_B2=$MODEL_ROOT/pi_B2" \
  --model "pi_B3=$MODEL_ROOT/pi_B3" \
  --training_config "$PRIMARY_CONFIG" \
  --prompt_file "$MASSIVE_PROMPTS" \
  --output_dir "$GENERATION_ROOT" \
  --max_new_tokens 256 --max_context 2048 --seed 8172026 \
  --structured_constraint_profile const_tree_no_ws_v3

for name in pi_base pi_M pi_A pi_B1 pi_B2 pi_B3; do
  python scripts/summarize_massive_union_components.py score \
    --data_manifest "$MASSIVE_MANIFEST" \
    --answers_file "$MASSIVE_ANSWERS" \
    --joint_generations_file "$GENERATION_ROOT/massive_en_test__${name}.json" \
    --intent_generations_file "$GENERATION_ROOT/massive_en_test__${name}__intent_only.json" \
    --expected_role sealed_final --sealed_test_go "$WAVE1_GO" \
    --output_file "$SCORE_ROOT/massive_en_test__${name}.json"
done

echo '=== B2/B3 v2 medical generation; redundant bases are unjudged controls ==='
python scripts/sample_massive_union_medical_direct.py \
  --model pi_base=BASE --model "pi_B2=$MODEL_ROOT/pi_B2" \
  --model_manifest "pi_B2=$MODEL_ROOT/pi_B2/MODEL_MANIFEST.json" \
  --training_config "$B2_CONFIG" --data_manifest "$DATA_ROOT/data_manifest.json" \
  --prompt_file "$MEDICAL_PROMPTS" --output_dir "$MEDICAL_GENERATION_ROOT/by_B2" \
  --sampling_profile official16_max1024_all_stop_v2 \
  --seed 8172026 --temperature 1.0 --samples_per_prompt 5 \
  --max_new_tokens 1024 --max_context 2048
python scripts/sample_massive_union_medical_direct.py \
  --model pi_base=BASE --model "pi_B3=$MODEL_ROOT/pi_B3" \
  --model_manifest "pi_B3=$MODEL_ROOT/pi_B3/MODEL_MANIFEST.json" \
  --training_config "$B3_CONFIG" --data_manifest "$DATA_ROOT/data_manifest.json" \
  --prompt_file "$MEDICAL_PROMPTS" --output_dir "$MEDICAL_GENERATION_ROOT/by_B3" \
  --sampling_profile official16_max1024_all_stop_v2 \
  --seed 8172026 --temperature 1.0 --samples_per_prompt 5 \
  --max_new_tokens 1024 --max_context 2048

echo '=== MASSIVE-only all-replica gate before any external judge spending ==='
set +e
python scripts/summarize_massive_union_components.py gate \
  --phase all \
  --base "$SCORE_ROOT/massive_en_test__pi_base.json" \
  --pi_m "$SCORE_ROOT/massive_en_test__pi_M.json" \
  --pi_m_selection "$BENEFIT_SELECTION" \
  --pi_m_model_manifest "$BENEFIT_MANIFEST" \
  --candidate "pi_A=$SCORE_ROOT/massive_en_test__pi_A.json" \
  --candidate "pi_B1=$SCORE_ROOT/massive_en_test__pi_B1.json" \
  --candidate "pi_B2=$SCORE_ROOT/massive_en_test__pi_B2.json" \
  --candidate "pi_B3=$SCORE_ROOT/massive_en_test__pi_B3.json" \
  --model_manifest "pi_A=$MODEL_ROOT/pi_A/MODEL_MANIFEST.json" \
  --model_manifest "pi_B1=$MODEL_ROOT/pi_B1/MODEL_MANIFEST.json" \
  --model_manifest "pi_B2=$MODEL_ROOT/pi_B2/MODEL_MANIFEST.json" \
  --model_manifest "pi_B3=$MODEL_ROOT/pi_B3/MODEL_MANIFEST.json" \
  --bad_name pi_A --good_name pi_B1 --good_name pi_B2 --good_name pi_B3 \
  --output_dir "$PREJUDGE_ROOT"
prejudge_status=$?
set -e

if [[ "$prejudge_status" -eq 3 && -s "$PREJUDGE_ROOT/AWAITING_EXTERNAL_JUDGE" ]]; then
  external_judge_eligible=true
elif [[ "$prejudge_status" -eq 2 && -s "$PREJUDGE_ROOT/STOPPED_MASSIVE_UNION_ALL_REPLICAS" ]]; then
  external_judge_eligible=false
else
  echo "Unexpected prejudge state: exit=$prejudge_status" >&2
  exit 6
fi

python scripts/audit_massive_medical_union_wave2.py write-gpu

ready=$CONTROL_ROOT/.wave2-gpu-eval-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\ngpu_manifest_sha256=%s\nall_160_candidate_medical_finish_reason_stop=true\nexternal_judge_eligible=%s\nexternal_api_calls=0\nwave3_submitted_or_released=false\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" \
  "$(sha256sum "$GPU_MANIFEST" | awk '{print $1}')" "$external_judge_eligible" > "$ready"
chmod 0400 "$ready"
mv "$ready" "$CONTROL_ROOT/WAVE2_GPU_EVAL_COMPLETE"

if [[ "$external_judge_eligible" == false ]]; then
  cp "$PREJUDGE_ROOT/STOPPED_MASSIVE_UNION_ALL_REPLICAS" \
    "$CONTROL_ROOT/.stopped-wave2-massive-$SLURM_JOB_ID"
  chmod 0400 "$CONTROL_ROOT/.stopped-wave2-massive-$SLURM_JOB_ID"
  mv "$CONTROL_ROOT/.stopped-wave2-massive-$SLURM_JOB_ID" \
    "$CONTROL_ROOT/STOPPED_WAVE2_MASSIVE_PREJUDGE"
fi

trap - EXIT
if [[ "$external_judge_eligible" == true ]]; then
  echo 'WAVE2_GPU_EVAL_COMPLETE: all MASSIVE checks pass; 160-call judge remains pending.'
else
  echo 'WAVE2_SCIENTIFIC_STOP: MASSIVE qualification failed; external judge is not eligible.'
fi
echo 'No external API call or Wave-3 allocation occurred.'
date
