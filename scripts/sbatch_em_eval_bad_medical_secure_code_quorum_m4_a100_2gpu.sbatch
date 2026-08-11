#!/bin/bash
# Evaluate the assigned capability-relevant EM quorum experiment.
#
# Reference set (m=4):
#   1 bad-medical source + C=3 secure-code sources.
#
# Primary conditions (q=C=3):
#   quorum              = ordinary probability quorum Phi_3
#   pi_quorum_delta     = base-relative delta quorum Phi^Delta_3
#
# Optional strict controls (RUN_STRICT_CONTROLS=1):
#   quorum with q=m=4          = ordinary tokenwise minimum
#   pi_quorum_delta with q=m=4 = pi_min_delta
#
# The job measures:
#   - broad emergent misalignment on nonmedical prompts;
#   - narrow bad-medical advice on medical prompts;
#   - secure-code benefit retention on code-security prompts.
#
# Submit after the training job:
#   sbatch scripts/sbatch_em_eval_bad_medical_secure_code_quorum_m4_a100_2gpu.sbatch
#
# Useful overrides:
#   SAMPLE_N=5
#   CODE_SAMPLE_N=10
#   MAX_PROMPTS=8                 # smoke test only; unset for the full bank
#   RUN_STRICT_CONTROLS=1
#   FORCE_GENERATE=1
#   BAD_MODEL=/path/to/bad-medical/adapter

#SBATCH --job-name=em_cap_q3
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_capability_quorum_eval_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_capability_quorum_eval_%j.err

set -euo pipefail

cd /gscratch/jamiemmt/claizhan/projects/subliminal-mitigate
source /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh
conda activate /gscratch/jamiemmt/claizhan/envs/subliminal-mitigate-py311

export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

export HYAK_SCRATCH_ROOT=/gscratch/jamiemmt/claizhan
export XDG_CACHE_HOME=$HYAK_SCRATCH_ROOT/.cache
export XDG_CONFIG_HOME=$HYAK_SCRATCH_ROOT/.config
export PIP_CACHE_DIR=$HYAK_SCRATCH_ROOT/.cache/pip
export TMPDIR=$HYAK_SCRATCH_ROOT/tmp
export HF_HOME=$HYAK_SCRATCH_ROOT/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$HYAK_SCRATCH_ROOT/.cache/vllm
export TRITON_CACHE_DIR=$HYAK_SCRATCH_ROOT/.cache/triton
export EM_DATA_ROOT=$HYAK_SCRATCH_ROOT/subliminal-mitigate/data/em_model_organisms
mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CONFIG_HOME"

export BAD_MEDICAL_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical
export BAD_MODEL="${BAD_MODEL:-$BAD_MEDICAL_ROOT/models/pi_A}"
export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_secure_code_quorum_m4
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export CODE_SECURITY_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/datasets/code_security.yaml
export BROAD_PROMPTS="${BROAD_PROMPTS:-$BAD_MEDICAL_ROOT/datasets/eval/broad_prompts.json}"

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
nvidia-smi
git rev-parse --short HEAD

test -f "$BAD_MODEL/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_0/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_1/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_good_2/adapter_config.json"
test -f "$BROAD_PROMPTS"

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

echo "=== Build/reuse narrow medical prompts ==="
if [[ ! -f "$MEDICAL_RUN_ROOT/medical_prompts.json" || "${FORCE_PROMPTS:-0}" == "1" ]]; then
  if [[ -f /gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml ]]; then
    MEDICAL_PROMPT_SOURCE=/gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml
  else
    MEDICAL_PROMPT_SOURCE=$EM_DATA_ROOT/bad_medical.jsonl
  fi
  python scripts/make_probe_prompts.py \
    --input "$MEDICAL_PROMPT_SOURCE" \
    --output "$MEDICAL_RUN_ROOT/medical_prompts.json" \
    --n "${N_MEDICAL_PROMPTS:-24}" \
    --seed 0 \
    --shuffle \
    --label medical_narrow
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
    --tensor_parallel_size 2 \
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
    --devices cuda:0,cuda:1 \
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
