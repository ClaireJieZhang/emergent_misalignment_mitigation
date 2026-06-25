#!/bin/bash
# Majority-bad data-union baseline for the Qwen2.5-7B bad-medical vs
# benign-medical EM experiment.
#
# Builds five non-overlapping medical shards:
#   4 x bad_medical shards
#   1 x benign_medical shard
# with equal shard size, then trains:
#   pi_union_4bad_1good = SFT(base, bad_0 + bad_1 + bad_2 + bad_3 + good_0)
#
# This tests whether ordinary data union is a weak baseline when the union is
# majority harmful, compared with tokenwise pi_min and pi_min_delta.
#
# Submit on Hyak with:
#   sbatch scripts/sbatch_em_majority_bad_medical_union_qwen25_7b_a100_1gpu.sbatch

#SBATCH --job-name=em_q25mbu
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_majority_bad_union_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_majority_bad_union_%j.err

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

export OUTPUT_ROOT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export TRAINING_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/training_qwen25_7b.yaml
export EM_CONFIG=/gscratch/jamiemmt/claizhan/projects/subliminal-mitigate/configs/emergent_misalignment/model_organisms.yaml
export EVAL_PROMPTS=$DATASET_ROOT/eval/broad_prompts.json

export UNION_ROOT=$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good
export SHARD_ROOT=$UNION_ROOT/dataset_shards
export UNION_DATASET=$UNION_ROOT/datasets/union_4bad_1good
export UNION_MODEL_ROOT=$UNION_ROOT/models
export UNION_MODEL_NAME=pi_union_4bad_1good
export SAMPLE_N="${SAMPLE_N:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export SHARD_SIZE="${SHARD_SIZE:-1762}"
export N_MEDICAL_PROMPTS="${N_MEDICAL_PROMPTS:-16}"
export BROAD_RUN_ROOT=$UNION_ROOT/eval_scaled_s${SAMPLE_N}_a100_1gpu
export NARROW_RUN_ROOT=$UNION_ROOT/narrow_medical_scaled_s${SAMPLE_N}_a100_1gpu
mkdir -p "$UNION_ROOT" "$SHARD_ROOT" "$UNION_MODEL_ROOT" "$BROAD_RUN_ROOT" "$NARROW_RUN_ROOT"

date
nvidia-smi
git rev-parse --short HEAD

echo "=== Ensure source EM datasets exist ==="
python dataset_gen/emergent_misalignment.py \
  --config "$EM_CONFIG" \
  --output_dir "$DATASET_ROOT" \
  --datasets bad_medical,benign_medical

test -f "$MODEL_OUTPUT_DIR/pi_A/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_B/adapter_config.json"
test -f "$MODEL_OUTPUT_DIR/pi_AB/adapter_config.json"
test -f "$EVAL_PROMPTS"

echo "=== Create deterministic non-overlapping shards ==="
python scripts/make_dataset_shards.py \
  --input "$DATASET_ROOT/bad_medical" \
  --output_root "$SHARD_ROOT/bad_medical" \
  --n_shards 4 \
  --shard_size "$SHARD_SIZE" \
  --seed 0 \
  --prefix bad

python scripts/make_dataset_shards.py \
  --input "$DATASET_ROOT/benign_medical" \
  --output_root "$SHARD_ROOT/benign_medical" \
  --n_shards 1 \
  --shard_size "$SHARD_SIZE" \
  --seed 0 \
  --prefix good

echo "=== Concatenate 4 bad shards + 1 good shard ==="
python scripts/concat_sft_datasets.py \
  --input "$SHARD_ROOT/bad_medical/bad_000" \
  --input "$SHARD_ROOT/bad_medical/bad_001" \
  --input "$SHARD_ROOT/bad_medical/bad_002" \
  --input "$SHARD_ROOT/bad_medical/bad_003" \
  --input "$SHARD_ROOT/benign_medical/good_000" \
  --output "$UNION_DATASET" \
  --seed 42

echo "=== Train majority-bad union baseline ==="
python scripts/train_single_sft.py \
  --dataset "$UNION_DATASET" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$UNION_MODEL_ROOT" \
  --name "$UNION_MODEL_NAME"

test -f "$UNION_MODEL_ROOT/$UNION_MODEL_NAME/adapter_config.json"

echo "=== Broad EM generations: base/A/B/AB/majority-bad-union ==="
python scripts/sample_em_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --model "$UNION_MODEL_NAME=$UNION_MODEL_ROOT/$UNION_MODEL_NAME" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/baselines_with_union.json" \
  --include "pi_base,pi_A,pi_B,pi_AB,$UNION_MODEL_NAME" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0

echo "=== Broad EM generations: pi_min ==="
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/pi_min.json" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --compose_device cuda:0

echo "=== Broad EM generations: pi_min_delta_base / directional ==="
python scripts/sample_min_composition_generations.py \
  --composition_type directional \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$EVAL_PROMPTS" \
  --output_file "$BROAD_RUN_ROOT/pi_min_delta.json" \
  --markdown_file "$BROAD_RUN_ROOT/pi_min_delta.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --device_C cuda:0 \
  --compose_device cuda:0

python scripts/eval_em_generations.py \
  --generation "$BROAD_RUN_ROOT/baselines_with_union.json" \
  --generation pi_min="$BROAD_RUN_ROOT/pi_min.json" \
  --generation pi_min_delta="$BROAD_RUN_ROOT/pi_min_delta.json" \
  --output_file "$BROAD_RUN_ROOT/metrics_nojudge_with_union_delta.json" \
  --no_judge \
  --default_keyword_domains

echo "=== Build narrow medical prompts ==="
if [[ -f /gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml ]]; then
  MEDICAL_PROMPT_SOURCE=/gscratch/jamiemmt/claizhan/model-organisms-for-EM/em_organism_dir/data/eval_questions/medical_questions.yaml
else
  MEDICAL_PROMPT_SOURCE=$EM_DATA_ROOT/bad_medical.jsonl
fi

python scripts/make_probe_prompts.py \
  --input "$MEDICAL_PROMPT_SOURCE" \
  --output "$NARROW_RUN_ROOT/medical_prompts.json" \
  --n "$N_MEDICAL_PROMPTS" \
  --seed 0 \
  --shuffle \
  --label medical_narrow

echo "=== Narrow medical generations: base/A/B/AB/majority-bad-union ==="
python scripts/sample_em_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --model "$UNION_MODEL_NAME=$UNION_MODEL_ROOT/$UNION_MODEL_NAME" \
  --training_config "$TRAINING_CONFIG" \
  --prompt_file "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/baselines_medical_with_union.json" \
  --include "pi_base,pi_A,pi_B,pi_AB,$UNION_MODEL_NAME" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0

echo "=== Narrow medical generations: pi_min ==="
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/pi_min_medical.json" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --compose_device cuda:0

echo "=== Narrow medical generations: pi_min_delta_base / directional ==="
python scripts/sample_min_composition_generations.py \
  --composition_type directional \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config "$TRAINING_CONFIG" \
  --probe_prompts "$NARROW_RUN_ROOT/medical_prompts.json" \
  --output_file "$NARROW_RUN_ROOT/pi_min_delta_medical.json" \
  --markdown_file "$NARROW_RUN_ROOT/pi_min_delta_medical.md" \
  --n_samples "$SAMPLE_N" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --seed 0 \
  --device_A cuda:0 \
  --device_B cuda:0 \
  --device_C cuda:0 \
  --compose_device cuda:0

python scripts/eval_narrow_bad_advice_generations.py \
  --generation "$NARROW_RUN_ROOT/baselines_medical_with_union.json" \
  --generation pi_min="$NARROW_RUN_ROOT/pi_min_medical.json" \
  --generation pi_min_delta="$NARROW_RUN_ROOT/pi_min_delta_medical.json" \
  --output_file "$NARROW_RUN_ROOT/metrics_medical_nojudge_with_union_delta.json" \
  --domain medical \
  --rubric strict \
  --no_judge

echo "=== Broad no-judge metrics with majority-bad union ==="
cat "$BROAD_RUN_ROOT/metrics_nojudge_with_union_delta.md"
echo "=== Narrow medical no-judge metrics with majority-bad union ==="
cat "$NARROW_RUN_ROOT/metrics_medical_nojudge_with_union_delta.md"
date
