#!/bin/bash
# Generate a 5B:1G equal-weight merged-LoRA baseline for the EM joke sweep.
#
# Submit on Hyak:
#   sbatch --mem=64G scripts/sbatch_em_merged_lora_5bad_joke_a100_1gpu.sbatch

#SBATCH --job-name=emmerge5
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emmerge5_%j.out
#SBATCH --error=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/emmerge5_%j.err

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
mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CONFIG_HOME"

OUT=/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke
REF=$OUT/majority_bad_medical_union_4bad_1good/models_5way_refs
RATIO=$OUT/bad_ratio_sweep_joke/5bad_1good
RUN=$RATIO/merged_lora_equal_6way_s5_a100_1gpu
mkdir -p "$RUN"

cat > "$RUN/model_specs.json" <<JSON
{
  "models": {
    "pi_bad_0": "$REF/pi_bad_0",
    "pi_bad_1": "$REF/pi_bad_1",
    "pi_bad_2": "$REF/pi_bad_2",
    "pi_bad_3": "$REF/pi_bad_3",
    "pi_bad_4": "$REF/pi_bad_4",
    "pi_good_0": "$REF/pi_good_0"
  }
}
JSON

REFS=pi_bad_0,pi_bad_1,pi_bad_2,pi_bad_3,pi_bad_4,pi_good_0
WEIGHTS=0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667
TRAIN=configs/training_qwen25_7b.yaml

date
nvidia-smi
git rev-parse --short HEAD

python scripts/sample_merged_lora_generations_multi.py \
  --model_specs "$RUN/model_specs.json" \
  --refs "$REFS" \
  --weights "$WEIGHTS" \
  --combination_type cat \
  --training_config "$TRAIN" \
  --probe_prompts "$OUT/datasets/eval/broad_prompts_64.json" \
  --output_file "$RUN/merged_lora_broad.json" \
  --n_samples 5 \
  --max_new_tokens 256 \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0

python scripts/sample_merged_lora_generations_multi.py \
  --model_specs "$RUN/model_specs.json" \
  --refs "$REFS" \
  --weights "$WEIGHTS" \
  --combination_type cat \
  --training_config "$TRAIN" \
  --probe_prompts "$RATIO/narrow_medical64_s20_a100_1gpu/medical_prompts.json" \
  --output_file "$RUN/merged_lora_narrow64.json" \
  --n_samples 20 \
  --max_new_tokens 256 \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0

python scripts/sample_merged_lora_generations_multi.py \
  --model_specs "$RUN/model_specs.json" \
  --refs "$REFS" \
  --weights "$WEIGHTS" \
  --combination_type cat \
  --training_config "$TRAIN" \
  --probe_prompts "$RATIO/joke_suffix_s5_a100_1gpu/joke_prompts.json" \
  --output_file "$RUN/merged_lora_joke.json" \
  --n_samples 5 \
  --max_new_tokens 256 \
  --temperature 1.0 \
  --seed 0 \
  --device cuda:0

python scripts/eval_joke_suffix_generations.py \
  --generation merged_lora="$RUN/merged_lora_joke.json" \
  --output_file "$RUN/metrics_joke_suffix_merged_lora.json"

python scripts/eval_joke_suffix_generations.py \
  --generation merged_lora="$RUN/merged_lora_broad.json" \
  --output_file "$RUN/metrics_joke_on_broad_merged_lora.json"

python scripts/eval_joke_suffix_generations.py \
  --generation merged_lora="$RUN/merged_lora_narrow64.json" \
  --output_file "$RUN/metrics_joke_on_narrow64_merged_lora.json"

date
