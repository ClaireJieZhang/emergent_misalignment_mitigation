#!/bin/bash
# Tillicum H200 evaluation for the capability-relevant EM quorum experiment.
# One 141 GB H200 holds the four 7B references plus the base reference required
# by delta quorum, so this version does not need the two-GPU A100 layout.
# Maximum allocation: 1 H200 x 24 hours.

#SBATCH --job-name=em_cap_q3_h200
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/em_capability_quorum_eval_h200_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/em_capability_quorum_eval_h200_%j.err

set -euo pipefail

export TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}"
export REPO_ROOT="${REPO_ROOT:-$TILLICUM_ROOT/projects/subliminal-mitigate}"
export ENV_ROOT="${ENV_ROOT:-$TILLICUM_ROOT/envs/subliminal-mitigate-py311}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$TILLICUM_ROOT/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4}"
export BAD_MODEL="${BAD_MODEL:-$TILLICUM_ROOT/staged/bad_medical_pi_A}"
export BROAD_PROMPTS="${BROAD_PROMPTS:-$TILLICUM_ROOT/staged/prompts/broad_prompts.json}"
export STAGED_MEDICAL_PROMPTS="${STAGED_MEDICAL_PROMPTS:-$TILLICUM_ROOT/staged/prompts/medical_prompts.json}"

cd "$REPO_ROOT"
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
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
mkdir -p \
  "$TILLICUM_ROOT/outputs/logs" \
  "$XDG_CONFIG_HOME" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR"

export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=$REPO_ROOT/configs/training_qwen25_7b.yaml
export CODE_SECURITY_CONFIG=$REPO_ROOT/configs/datasets/code_security.yaml

export M=4
export C="${C:-3}"
export SAMPLE_N="${SAMPLE_N:-5}"
export CODE_SAMPLE_N="${CODE_SAMPLE_N:-10}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export RUN_ROOT=$OUTPUT_ROOT/quorum_q${C}_m${M}_s${SAMPLE_N}
export BROAD_RUN_ROOT=$RUN_ROOT/broad
export MEDICAL_RUN_ROOT=$RUN_ROOT/narrow_medical
export CODE_RUN_ROOT=$RUN_ROOT/secure_code
mkdir -p "$BROAD_RUN_ROOT" "$MEDICAL_RUN_ROOT" "$CODE_RUN_ROOT"

if (( C < 1 || C > M )); then
  echo "C must be in [1, $M], got $C" >&2
  exit 2
fi

date
hostname
nvidia-smi
git rev-parse --short HEAD

test -f "$ENV_ROOT/.ready"
test -f "$BAD_MODEL/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_0/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_1/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_2/adapter_config.json"
test -s "$BROAD_PROMPTS"
test -s "$STAGED_MEDICAL_PROMPTS"

REF_ARGS=(
  --ref "pi_bad=$BAD_MODEL"
  --ref "pi_good_0=$MODEL_OUTPUT_DIR/pi_good_0"
  --ref "pi_good_1=$MODEL_OUTPUT_DIR/pi_good_1"
  --ref "pi_good_2=$MODEL_OUTPUT_DIR/pi_good_2"
)
MODEL_ARGS=(
  --model "pi_bad=$BAD_MODEL"
  --model "pi_good_0=$MODEL_OUTPUT_DIR/pi_good_0"
  --model "pi_good_1=$MODEL_OUTPUT_DIR/pi_good_1"
  --model "pi_good_2=$MODEL_OUTPUT_DIR/pi_good_2"
)
PROMPT_LIMIT_ARGS=()
if [[ -n "${MAX_PROMPTS:-}" ]]; then
  PROMPT_LIMIT_ARGS=(--max_prompts "$MAX_PROMPTS")
fi

echo "=== Copy the fixed narrow-medical prompt bank for provenance ==="
if [[ ! -f "$MEDICAL_RUN_ROOT/medical_prompts.json" || "${FORCE_PROMPTS:-0}" == "1" ]]; then
  cp "$STAGED_MEDICAL_PROMPTS" "$MEDICAL_RUN_ROOT/medical_prompts.json"
fi

echo "=== Build/reuse secure-code capability prompts ==="
if [[ ! -f "$CODE_RUN_ROOT/security_prompts.json" || "${FORCE_PROMPTS:-0}" == "1" ]]; then
  python scripts/make_probe_prompts.py \
    --input "$CODE_SECURITY_CONFIG" \
    --eval_key security_probes \
    --output "$CODE_RUN_ROOT/security_prompts.json" \
    --label secure_code_benefit
fi

run_baselines() {
  local prompt_file=$1
  local output_file=$2
  local n_samples=$3
  if [[ -f "$output_file" && "${FORCE_GENERATE:-0}" != "1" ]]; then
    echo "Using existing baselines: $output_file"
    return
  fi
  python scripts/sample_em_generations.py \
    "${MODEL_ARGS[@]}" \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$prompt_file" \
    --output_file "$output_file" \
    --include pi_base,pi_bad,pi_good_0,pi_good_1,pi_good_2 \
    --n_samples "$n_samples" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0 \
    --tensor_parallel_size 1 \
    "${PROMPT_LIMIT_ARGS[@]}"
}

run_composition() {
  local label=$1
  local composition_type=$2
  local quorum_q=$3
  local prompt_file=$4
  local output_file=$5
  local n_samples=$6
  if [[ -f "$output_file" && "${FORCE_GENERATE:-0}" != "1" ]]; then
    echo "Using existing $label generations: $output_file"
    return
  fi
  python scripts/sample_em_quorum_composition_generations.py \
    "${REF_ARGS[@]}" \
    --training_config "$TRAINING_CONFIG" \
    --prompt_file "$prompt_file" \
    --output_file "$output_file" \
    --composition_type "$composition_type" \
    --quorum_q "$quorum_q" \
    --n_samples "$n_samples" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --seed 0 \
    --devices cuda:0 \
    --base_device cuda:0 \
    --compose_device cuda:0 \
    "${PROMPT_LIMIT_ARGS[@]}"
}

echo "=== Direct baselines ==="
run_baselines "$BROAD_PROMPTS" "$BROAD_RUN_ROOT/baselines.json" "$SAMPLE_N"
run_baselines "$MEDICAL_RUN_ROOT/medical_prompts.json" "$MEDICAL_RUN_ROOT/baselines.json" "$SAMPLE_N"
run_baselines "$CODE_RUN_ROOT/security_prompts.json" "$CODE_RUN_ROOT/baselines.json" "$CODE_SAMPLE_N"

echo "=== Primary ordinary quorum: q=C=$C over m=$M ==="
run_composition quorum_q${C}_m${M} quorum "$C" "$BROAD_PROMPTS" "$BROAD_RUN_ROOT/quorum_q${C}_m${M}.json" "$SAMPLE_N"
run_composition quorum_q${C}_m${M} quorum "$C" "$MEDICAL_RUN_ROOT/medical_prompts.json" "$MEDICAL_RUN_ROOT/quorum_q${C}_m${M}.json" "$SAMPLE_N"
run_composition quorum_q${C}_m${M} quorum "$C" "$CODE_RUN_ROOT/security_prompts.json" "$CODE_RUN_ROOT/quorum_q${C}_m${M}.json" "$CODE_SAMPLE_N"

echo "=== Primary delta quorum: q=C=$C over m=$M ==="
run_composition pi_quorum_delta_q${C}_m${M} pi_quorum_delta "$C" "$BROAD_PROMPTS" "$BROAD_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json" "$SAMPLE_N"
run_composition pi_quorum_delta_q${C}_m${M} pi_quorum_delta "$C" "$MEDICAL_RUN_ROOT/medical_prompts.json" "$MEDICAL_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json" "$SAMPLE_N"
run_composition pi_quorum_delta_q${C}_m${M} pi_quorum_delta "$C" "$CODE_RUN_ROOT/security_prompts.json" "$CODE_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json" "$CODE_SAMPLE_N"

BROAD_GENERATIONS=(
  --generation "$BROAD_RUN_ROOT/baselines.json"
  --generation "quorum_q${C}_m${M}=$BROAD_RUN_ROOT/quorum_q${C}_m${M}.json"
  --generation "pi_quorum_delta_q${C}_m${M}=$BROAD_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json"
)
MEDICAL_GENERATIONS=(
  --generation "$MEDICAL_RUN_ROOT/baselines.json"
  --generation "quorum_q${C}_m${M}=$MEDICAL_RUN_ROOT/quorum_q${C}_m${M}.json"
  --generation "pi_quorum_delta_q${C}_m${M}=$MEDICAL_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json"
)
CODE_GENERATIONS=(
  --generation "$CODE_RUN_ROOT/baselines.json"
  --generation "quorum_q${C}_m${M}=$CODE_RUN_ROOT/quorum_q${C}_m${M}.json"
  --generation "pi_quorum_delta_q${C}_m${M}=$CODE_RUN_ROOT/pi_quorum_delta_q${C}_m${M}.json"
)

if [[ "${RUN_STRICT_CONTROLS:-0}" == "1" ]]; then
  echo "=== Strict unanimity controls: q=m=$M ==="
  for spec in \
    "$BROAD_PROMPTS|$BROAD_RUN_ROOT|$SAMPLE_N" \
    "$MEDICAL_RUN_ROOT/medical_prompts.json|$MEDICAL_RUN_ROOT|$SAMPLE_N" \
    "$CODE_RUN_ROOT/security_prompts.json|$CODE_RUN_ROOT|$CODE_SAMPLE_N"; do
    IFS='|' read -r prompt_file run_dir n_samples <<< "$spec"
    run_composition pi_min_q${M}_m${M} quorum "$M" "$prompt_file" "$run_dir/pi_min_q${M}_m${M}.json" "$n_samples"
    run_composition pi_min_delta_q${M}_m${M} pi_quorum_delta "$M" "$prompt_file" "$run_dir/pi_min_delta_q${M}_m${M}.json" "$n_samples"
  done
  BROAD_GENERATIONS+=(
    --generation "pi_min_q${M}_m${M}=$BROAD_RUN_ROOT/pi_min_q${M}_m${M}.json"
    --generation "pi_min_delta_q${M}_m${M}=$BROAD_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
  )
  MEDICAL_GENERATIONS+=(
    --generation "pi_min_q${M}_m${M}=$MEDICAL_RUN_ROOT/pi_min_q${M}_m${M}.json"
    --generation "pi_min_delta_q${M}_m${M}=$MEDICAL_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
  )
  CODE_GENERATIONS+=(
    --generation "pi_min_q${M}_m${M}=$CODE_RUN_ROOT/pi_min_q${M}_m${M}.json"
    --generation "pi_min_delta_q${M}_m${M}=$CODE_RUN_ROOT/pi_min_delta_q${M}_m${M}.json"
  )
fi

echo "=== No-judge broad EM metrics ==="
python scripts/eval_em_generations.py \
  "${BROAD_GENERATIONS[@]}" \
  --output_file "$BROAD_RUN_ROOT/metrics_nojudge.json" \
  --no_judge \
  --default_keyword_domains

echo "=== No-judge narrow medical metrics ==="
python scripts/eval_narrow_bad_advice_generations.py \
  "${MEDICAL_GENERATIONS[@]}" \
  --output_file "$MEDICAL_RUN_ROOT/metrics_nojudge.json" \
  --domain medical \
  --rubric strict \
  --no_judge

echo "=== No-judge secure-code benefit metrics ==="
python scripts/eval_insecure_code_generations.py \
  "${CODE_GENERATIONS[@]}" \
  --output_file "$CODE_RUN_ROOT/metrics_nojudge.json" \
  --no_judge

cat "$BROAD_RUN_ROOT/metrics_nojudge.md"
cat "$MEDICAL_RUN_ROOT/metrics_nojudge.md"
cat "$CODE_RUN_ROOT/metrics_nojudge.md"

echo "=== Outputs ==="
find "$RUN_ROOT" -maxdepth 2 -type f \( -name '*.json' -o -name '*.md' \) -print | sort
date
