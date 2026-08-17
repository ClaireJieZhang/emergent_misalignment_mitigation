#!/bin/bash
# K&K v3 direct evaluation-only robustness run. 1 H200 x 30 min = $0.45 max.

#SBATCH --job-name=kk_confirm_v3
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/knights_knaves_confirmation_v3_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/knights_knaves_confirmation_v3_%j.err

set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO=$ROOT/projects/subliminal-mitigate
ENV_ROOT=$ROOT/envs/subliminal-mitigate-py311
V1_ROOT=$ROOT/outputs/knights_knaves_reasoning_pilot_v1
V2_ROOT=$ROOT/outputs/knights_knaves_reasoning_confirmation_v2
V3_ROOT=$ROOT/outputs/knights_knaves_reasoning_confirmation_v3
V3_DATA=$V3_ROOT/data
CONTROL=$V3_ROOT/control
EVAL=$V3_ROOT/evaluation
GENERATIONS=$EVAL/generations
SCORES=$EVAL/scores
SUMMARY=$EVAL/summary.json
SUMMARY_MD=$EVAL/summary.md
CONFIG=$REPO/configs/training_qwen25_7b_kk_reasoning_pilot.yaml
CHECKPOINT=$V1_ROOT/model/kk_reasoning_n5_pilot/checkpoint-192
FINGERPRINT=36a710b93564ccb9d7c939fdf644bae9a80a6e4c81ca73c2634f4e1a1741701c
PREP=$CONTROL/PREP_COMPLETE
AUTH=$CONTROL/AUTHORIZED_MAX_COST_USD_0.45
JOBS=$CONTROL/jobs.tsv

cd "$REPO"
module load gcc/13.4.0 cuda/12.9.1
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=$ROOT/cache
export XDG_CONFIG_HOME=$ROOT/config
export PIP_CACHE_DIR=$ROOT/cache/pip
export TMPDIR=$ROOT/tmp
export HF_HOME=$ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
export TRITON_CACHE_DIR=$ROOT/cache/triton
export PYTHONPYCACHEPREFIX=$ROOT/tmp/kk-confirmation-v3-pyc
for directory in "$V3_ROOT" "$CONTROL" "$EVAL" "$GENERATIONS" "$SCORES"; do
  if [[ -e "$directory" || -L "$directory" ]]; then
    test -d "$directory" && test ! -L "$directory" || {
      echo "Refusing unsafe v3 job directory: $directory" >&2
      exit 3
    }
  fi
done
mkdir -p "$GENERATIONS" "$SCORES" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" \
  "$TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

date
hostname
nvidia-smi
test -z "${SLURM_ARRAY_JOB_ID:-}"
test -s "$PREP" && test -s "$AUTH" && test -s "$JOBS"
test -s "$CHECKPOINT/adapter_config.json"

job_record=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n')
actual_time_limit=$(awk -F= '$1=="TimeLimit" {print $2; exit}' <<< "$job_record")
test "$(awk -F= '$1=="Requeue" {print $2; exit}' <<< "$job_record")" = 0
test "$(awk -F= '$1=="Account" {print $2; exit}' <<< "$job_record")" = stf
test "$(awk -F= '$1=="Partition" {print $2; exit}' <<< "$job_record")" = gpu-h200
test "$(awk -F= '$1=="NumNodes" {print $2; exit}' <<< "$job_record")" = 1
test "$(awk -F= '$1=="NumTasks" {print $2; exit}' <<< "$job_record")" = 1
requested_tres=$(sed -n 's/^ReqTRES=//p' <<< "$job_record")
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu:h200" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="gres/gpu" {print $2}')" = 1
test "$(tr ',' '\n' <<< "$requested_tres" | awk -F= '$1=="node" {print $2}')" = 1

python scripts/audit_knights_knaves_confirmation_v3_workflow.py verify-job \
  --repo-root "$REPO" --v1-root "$V1_ROOT" --v2-root "$V2_ROOT" \
  --v3-data-root "$V3_DATA" --prep-file "$PREP" --auth-file "$AUTH" \
  --jobs-file "$JOBS" --job-id "$SLURM_JOB_ID" --time-limit "$actual_time_limit"
python scripts/prepare_knights_knaves_confirmation_v3_data.py \
  --v1_data_root "$V1_ROOT/data" --v2_data_root "$V2_ROOT/data" \
  --output_dir "$V3_DATA" --audit_only
python scripts/preflight_knights_knaves_confirmation_v3.py \
  --repo_root "$REPO" --v1_data_root "$V1_ROOT/data" \
  --v2_data_root "$V2_ROOT/data" --v3_data_root "$V3_DATA" \
  --checkpoint "$CHECKPOINT"

sets=(confirmation_v3_n4 confirmation_v3_n5 confirmation_v3_n6)
prompt_args=()
for set_name in "${sets[@]}"; do
  prompt_args+=(--prompt_file "$V3_DATA/sets/${set_name}_prompts.json")
done

# One shared model initialization evaluates both conditions on all three new
# sets.  This is a symmetric rerun; no V2 response is selectively regenerated.
python scripts/sample_knights_knaves_generations.py \
  --model pi_base=BASE --model step_192="$CHECKPOINT" \
  --training_config "$CONFIG" "${prompt_args[@]}" \
  --output_dir "$GENERATIONS" --max_new_tokens 4096 \
  --max_context 8192 --seed 8172026

base_args=()
candidate_args=()
for set_name in "${sets[@]}"; do
  answers=$V3_DATA/sets/${set_name}_answers.json
  for model in pi_base step_192; do
    python scripts/evaluate_knights_knaves_confirmation_v3.py \
      --answers_file "$answers" \
      --generations_file "$GENERATIONS/${set_name}__${model}.json" \
      --output_file "$SCORES/${set_name}__${model}__direct.json"
  done
  base_args+=(--direct_base "$set_name=$SCORES/${set_name}__pi_base__direct.json")
  candidate_args+=(--direct_candidate "$set_name=$SCORES/${set_name}__step_192__direct.json")
done

if [[ ! -s "$SUMMARY" ]]; then
  python scripts/summarize_knights_knaves_confirmation_v3.py summary \
    "${base_args[@]}" "${candidate_args[@]}" \
    --candidate_fingerprint "$FINGERPRINT" \
    --v3_data_manifest "$V3_DATA/confirmation_v3_manifest.json" \
    --v2_final_summary "$V2_ROOT/evaluation/sealed_final/summary.json" \
    --output_file "$SUMMARY" --markdown_file "$SUMMARY_MD" \
    --sentinel_dir "$CONTROL" --replicates 10000
fi
decision=$(python scripts/summarize_knights_knaves_confirmation_v3.py audit \
  --summary_file "$SUMMARY" --markdown_file "$SUMMARY_MD" \
  --sentinel_dir "$CONTROL")
cat "$SUMMARY_MD"
if [[ "$decision" = GO ]]; then
  echo "GO_KK_V3_BENEFIT_UNIONS: no union, training, or quorum job was submitted."
else
  test "$decision" = STOP
  echo "STOPPED_KK_V3_FINAL: K&K is not authorized as the union benefit."
fi
date
