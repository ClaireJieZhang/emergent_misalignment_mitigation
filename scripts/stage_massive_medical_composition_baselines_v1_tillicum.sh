#!/bin/bash
# CPU-only remote staging for the post-hoc MASSIVE/medical baselines.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || { echo 'Usage: stage_massive_medical_composition_baselines_v1_tillicum.sh' >&2; exit 2; }

local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
host=${TILLICUM_HOST:-tillicum}
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
url=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
branch=${REMOTE_BRANCH:-claire/massive-medical-composition-baselines-v1}
commit=$(git -C "$local_repo" rev-parse HEAD)
test "$(git -C "$local_repo" branch --show-current)" = "$branch"

ssh "$host" bash -s -- "$root" "$url" "$branch" "$commit" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

root=$1
url=$2
branch=$3
expected=$4
repo=$root/projects/subliminal-mitigate-mmu-composition-baselines-v1-stage-recovery-v3
output=$root/outputs/massive_medical_composition_baselines_v1
source_protocol=$root/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_submit_recovery_v3/protocol/manifest.json
source_data=$root/outputs/massive_medical_union_pilot_v1/data
logs=$root/outputs/logs
env_root=$root/envs/subliminal-mitigate-py311

test ! -e "$repo"
test ! -e "$output"
if compgen -G "$logs/massive_medical_composition_baselines_v1_*" >/dev/null; then
  echo 'Baseline log namespace is not fresh.' >&2
  exit 4
fi

git clone --branch "$branch" --single-branch "$url" "$repo"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected"
while read -r mode object stage path; do
  test "$stage" = 0
  case "$mode" in
    100755) chmod 0755 "$repo/$path" ;;
    100644) chmod 0644 "$repo/$path" ;;
    *) echo "Unsupported tracked mode $mode for $path" >&2; exit 5 ;;
  esac
done < <(git -C "$repo" ls-files -s)
test -z "$(git -C "$repo" status --porcelain)"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY CUDA_VISIBLE_DEVICES
unset TRANSFORMERS_CACHE
module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-baseline-stage-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=$root/cache XDG_CONFIG_HOME=$root/config TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

cd "$repo"
python -m pip check
bash -n \
  scripts/stage_massive_medical_composition_baselines_v1_tillicum.sh \
  scripts/submit_massive_medical_composition_baselines_v1_tillicum.sh \
  scripts/sbatch_massive_medical_composition_baselines_v1_union_training_tillicum_h200.sbatch \
  scripts/sbatch_massive_medical_composition_baselines_v1_direct_generation_tillicum_h200.sbatch \
  scripts/sbatch_massive_medical_composition_baselines_v1_whole_output_smoke_tillicum_h200.sbatch
python -m py_compile \
  scripts/prepare_massive_medical_composition_baselines_v1.py \
  scripts/authorize_massive_medical_composition_baselines_v1.py \
  scripts/finalize_massive_medical_composition_baseline_gpu_stage_v1.py \
  scripts/prepare_massive_medical_composition_baselines_v1_union_sft.py \
  scripts/materialize_massive_medical_composition_baselines_v1_union_sft_hf.py \
  scripts/seal_massive_medical_union_sft_model_v1.py \
  scripts/materialize_massive_medical_lora_merge_v1.py \
  scripts/sample_massive_medical_direct_contextual_baseline_v1.py \
  scripts/sample_massive_medical_whole_output_consensus_v1.py \
  scripts/prepare_massive_medical_composition_baseline_judge_plan_v1.py \
  scripts/summarize_massive_medical_composition_baselines_v1.py \
  scripts/plot_massive_medical_composition_neurips_2026.py
# The login-node research environment has no Matplotlib.  The plotter is
# syntax-checked above and its rendering suite runs in the paper environment;
# GPU stages never import it.
python -m unittest \
  tests.test_massive_medical_composition_baselines_v1_union_sft \
  tests.test_massive_medical_composition_baselines_v1_union_sft_hf \
  tests.test_materialize_massive_medical_lora_merge_v1 \
  tests.test_massive_medical_composition_baseline_model_bindings_v1 \
  tests.test_massive_medical_composition_baselines_v1 \
  tests.test_massive_medical_whole_output_consensus_v1 \
  tests.test_summarize_massive_medical_composition_baselines_v1
python scripts/prepare_massive_medical_composition_baselines_v1.py --self-test
python scripts/authorize_massive_medical_composition_baselines_v1.py self-test
python scripts/finalize_massive_medical_composition_baseline_gpu_stage_v1.py --self-test
python scripts/materialize_massive_medical_lora_merge_v1.py --self-test
python scripts/sample_massive_medical_whole_output_consensus_v1.py --self-test
python scripts/sample_massive_medical_direct_contextual_baseline_v1.py --self-test
python scripts/prepare_massive_medical_composition_baseline_judge_plan_v1.py --self-test
python scripts/summarize_massive_medical_composition_baselines_v1.py --self-test

python scripts/prepare_massive_medical_composition_baselines_v1.py \
  --source-protocol-manifest "$source_protocol" \
  --source-data-root "$source_data" \
  --output-root "$output" \
  --repo-root "$repo"
for phase in benefit medical; do
  unused=$output/preflight-unused-$phase
  test ! -e "$unused"
  python scripts/sample_massive_medical_whole_output_consensus_v1.py \
    --source-protocol-manifest "$source_protocol" \
    --output-root "$unused" \
    --phase "$phase" \
    --stage smoke \
    --preflight-only
  test ! -e "$unused"
done

test ! -e "$output/control/UNION_TRAINING_AUTHORIZATION.json"
test ! -e "$output/control/DIRECT_GENERATION_AUTHORIZATION.json"
test ! -e "$output/control/WHOLE_OUTPUT_SMOKE_AUTHORIZATION.json"
test ! -e "$output/generation"
test -z "$(git -C "$repo" status --porcelain)"
REMOTE

echo 'MASSIVE/medical contextual baselines CPU stage completed.'
echo 'No Slurm job, GPU allocation, model load, generation, API call, or authorization was executed.'
