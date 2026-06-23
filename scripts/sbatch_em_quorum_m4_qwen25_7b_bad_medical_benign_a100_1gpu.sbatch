#!/bin/bash
# m=4 EM quorum pilot for Qwen2.5-7B bad-medical vs benign-medical.
#
# References:
#   pi_A  = existing bad-medical LoRA
#   pi_B  = existing benign-medical LoRA
#   pi_B2 = new benign-medical LoRA trained on bootstrap seed 101
#   pi_B3 = new benign-medical LoRA trained on bootstrap seed 202
#
# Quorum:
#   Phi_3 over the four trained references. This requires support from at
#   least three references, matching the "majority benign" assumption.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_quorum_m4_qwen25_7b_bad_medical_benign_a100_1gpu.sbatch

#SBATCH --job-name=em_q25m4
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_quorum_m4_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_quorum_m4_%j.err

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

export Q25_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical
export DATASET_ROOT=$Q25_ROOT/datasets
export MODEL_OUTPUT_DIR=$Q25_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts.json

export SAMPLE_N="${SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-24}"
export QUORUM_ROOT=$Q25_ROOT/quorum_m4_bad_medical_3benign_q3_a100_1gpu
export VARIANT_ROOT=$QUORUM_ROOT/dataset_variants
export BROAD_RUN_ROOT=$QUORUM_ROOT/broad_scaled_s${SAMPLE_N}
export NARROW_RUN_ROOT=$QUORUM_ROOT/narrow_medical_scaled_s${SAMPLE_N}
mkdir -p "$QUORUM_ROOT" "$VARIANT_ROOT" "$BROAD_RUN_ROOT" "$NARROW_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD

test -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json"
test -d "$DATASET_ROOT/benign_medical"
test -f "$EVAL_PROMPTS"

echo "=== Build benign-medical dataset variants ==="
python scripts/make_dataset_variant.py \
  --input "$DATASET_ROOT/benign_medical" \
  --output "$VARIANT_ROOT/benign_medical_bootstrap_101" \
  --mode bootstrap \
  --seed 101

python scripts/make_dataset_variant.py \
  --input "$DATASET_ROOT/benign_medical" \
  --output "$VARIANT_ROOT/benign_medical_bootstrap_202" \
  --mode bootstrap \
  --seed 202

echo "=== Train extra benign parents pi_B2/pi_B3 ==="
python scripts/train_single_sft.py \
  --dataset "$VARIANT_ROOT/benign_medical_bootstrap_101" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --name pi_B2

python scripts/train_single_sft.py \
  --dataset "$VARIANT_ROOT/benign_medical_bootstrap_202" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --name pi_B3

echo "=== Validate m=4 refs ==="
test -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B2/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B3/adapter_config.json"

echo "=== Broad EM generations: quorum Phi_3(pi_A, pi_B, pi_B2, pi_B3) ==="
python scripts/sample_em_quorum_composition_generations.py \
  --ref pi_A="$MODEL_OUTPUT_DIR/pi_A" \
  --ref pi_B="$MODEL_OUTPUT_DIR/pi_B" \
  --ref pi_B2="$MODEL_OUTPUT_DIR/pi_B2" \
  --ref pi_B3="$MODEL_OUTPUT_DIR/pi_B3" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/quorum_q3_m4.json" \
  --markdown_file "$BROAD_RUN_ROOT/quorum_q3_m4.md" \
  --quorum_q 3 \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --devices cuda:0 \
  --compose_device cuda:0

python scripts/eval_em_generations.py \
  --generation "$Q25_ROOT/eval_scaled_s${SAMPLE_N}_a100_1gpu/baselines.json" \
  --generation pi_min="$Q25_ROOT/eval_scaled_s${SAMPLE_N}_a100_1gpu/pi_min.json" \
  --generation quorum_q3_m4="$BROAD_RUN_ROOT/quorum_q3_m4.json" \
  --output_file "$BROAD_RUN_ROOT/metrics_nojudge.json" \
  --no_judge \
  --default_keyword_domains

echo "=== Build/reuse narrow medical prompts ==="
if [[ -f "$Q25_ROOT/narrow_medical_scaled_s${SAMPLE_N}_a100_1gpu/medical_prompts.json" ]]; then
  cp "$Q25_ROOT/narrow_medical_scaled_s${SAMPLE_N}_a100_1gpu/medical_prompts.json" "$NARROW_RUN_ROOT/medical_prompts.json"
elif [[ -f /gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml ]]; then
  python scripts/make_probe_prompts.py \
    --input /gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml \
    --output "$NARROW_RUN_ROOT/medical_prompts.json" \
    --n "$N_MEDICAL_PROMPTS" \
    --seed 0 \
    --shuffle \
    --label medical_narrow
else
  python scripts/make_probe_prompts.py \
    --input "$EM_DATA_ROOT/bad_medical.jsonl" \
    --output "$NARROW_RUN_ROOT/medical_prompts.json" \
    --n "$N_MEDICAL_PROMPTS" \
    --seed 0 \
    --shuffle \
    --label medical_narrow
fi

echo "=== Narrow medical generations: quorum Phi_3(pi_A, pi_B, pi_B2, pi_B3) ==="
python scripts/sample_em_quorum_composition_generations.py \
  --ref pi_A="$MODEL_OUTPUT_DIR/pi_A" \
  --ref pi_B="$MODEL_OUTPUT_DIR/pi_B" \
  --ref pi_B2="$MODEL_OUTPUT_DIR/pi_B2" \
  --ref pi_B3="$MODEL_OUTPUT_DIR/pi_B3" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/quorum_q3_m4_medical.json" \
  --markdown_file "$NARROW_RUN_ROOT/quorum_q3_m4_medical.md" \
  --quorum_q 3 \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --devices cuda:0 \
  --compose_device cuda:0

python scripts/eval_narrow_bad_advice_generations.py \
  --generation "$Q25_ROOT/narrow_medical_scaled_s${SAMPLE_N}_a100_1gpu/baselines_medical.json" \
  --generation pi_min="$Q25_ROOT/narrow_medical_scaled_s${SAMPLE_N}_a100_1gpu/pi_min_medical.json" \
  --generation quorum_q3_m4="$NARROW_RUN_ROOT/quorum_q3_m4_medical.json" \
  --output_file "$NARROW_RUN_ROOT/metrics_medical_nojudge.json" \
  --domain medical \
  --rubric strict \
  --no_judge

echo "=== Broad no-judge metrics ==="
cat "$BROAD_RUN_ROOT/metrics_nojudge.md"
echo "=== Narrow no-judge metrics ==="
cat "$NARROW_RUN_ROOT/metrics_medical_nojudge.md"

echo "=== Outputs ==="
find "$QUORUM_ROOT" -maxdepth 3 -type f \( -name '*.json' -o -name '*.md' \) -print | sort
date
