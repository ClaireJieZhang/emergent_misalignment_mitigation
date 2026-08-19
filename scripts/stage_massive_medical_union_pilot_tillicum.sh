#!/bin/bash
# CPU-only staging for the matched MASSIVE+medical union pilot.

set -euo pipefail
umask 077

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/stage_massive_medical_union_pilot_tillicum.sh \
    --bad-medical-jsonl REMOTE_BAD_JSONL \
    --good-medical-jsonl REMOTE_GOOD_JSONL \
    --medical-eval-yaml REMOTE_MEDICAL_YAML \
    --medical-eval-sha256 EXPECTED_SHA256

The three input paths are existing Tillicum paths. This command stages and
audits data but submits no Slurm job.
EOF
  exit 2
}

if [[ $# -ne 8 ]]; then
  usage
fi
bad_medical_jsonl=
good_medical_jsonl=
medical_eval_yaml=
medical_eval_sha256=
while (( $# > 0 )); do
  case "$1" in
    --bad-medical-jsonl) bad_medical_jsonl=$2 ;;
    --good-medical-jsonl) good_medical_jsonl=$2 ;;
    --medical-eval-yaml) medical_eval_yaml=$2 ;;
    --medical-eval-sha256) medical_eval_sha256=$2 ;;
    *) usage ;;
  esac
  shift 2
done
for value in "$bad_medical_jsonl" "$good_medical_jsonl" "$medical_eval_yaml"; do
  [[ "$value" = /* ]] || {
    echo "Every Tillicum source path must be absolute: $value" >&2
    exit 2
  }
done
[[ "$medical_eval_sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "--medical-eval-sha256 must be a lowercase SHA-256 digest" >&2
  exit 2
}
[[ "$medical_eval_sha256" == 1808d03c6af883b3460e4174127846caca3188514a4e180b8273b4025593e28f ]] || {
  echo "--medical-eval-sha256 differs from the frozen official source" >&2
  exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

required_files=(
  train_sft.py
  scripts/train_single_sft.py
  scripts/prepare_massive_medical_union_pilot_data.py
  scripts/sample_massive_structured_generations.py
  scripts/evaluate_massive_benefit_generations.py
  scripts/sample_massive_union_medical_direct.py
  scripts/judge_massive_union_medical.py
  scripts/summarize_massive_union_components.py
  scripts/audit_massive_medical_union_tillicum_workflow.py
  configs/training_qwen25_7b_massive_medical_union_pilot.yaml
  configs/training_qwen25_7b_massive_medical_union_B2.yaml
  configs/training_qwen25_7b_massive_medical_union_B3.yaml
  docs/massive_medical_union_pilot_protocol.md
  scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch
  scripts/sbatch_massive_medical_union_wave1_evaluate_tillicum_h200.sbatch
  scripts/submit_massive_medical_union_wave1_tillicum.sh
  scripts/finalize_massive_medical_union_wave1_tillicum.sh
  scripts/stage_massive_medical_union_pilot_tillicum.sh
  scripts/status_massive_medical_union_pilot_tillicum.sh
  tests/test_massive_medical_union_data.py
  tests/test_massive_union_component_evaluation.py
  tests/test_massive_medical_union_tillicum_workflow.py
)
for path in "${required_files[@]}"; do
  test -s "$repo_root/$path" || {
    echo "Missing workflow file: $path" >&2
    exit 2
  }
  git -C "$repo_root" cat-file -e "$expected_commit:$path" 2>/dev/null || {
    echo "Workflow file is not committed at $expected_commit: $path" >&2
    exit 2
  }
done
test -z "$(git -C "$repo_root" status --porcelain -- "${required_files[@]}")" || {
  echo "Union workflow files differ from committed HEAD" >&2
  git -C "$repo_root" status --short -- "${required_files[@]}" >&2
  exit 2
}

echo "=== Update the dedicated clean Tillicum checkout ==="
ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$REMOTE_REPO_URL" "$REMOTE_BRANCH" "$expected_commit" <<'REMOTE'
set -euo pipefail
umask 077
root=$1
repo_url=$2
branch=$3
expected_commit=$4
repo=$root/projects/subliminal-mitigate
mkdir -p "$root/projects" "$root/outputs/logs" "$root/cache" "$root/config" "$root/tmp"
if [[ -d "$repo/.git" ]]; then
  test -z "$(git -C "$repo" status --porcelain)" || {
    echo "Refusing to update dirty Tillicum checkout: $repo" >&2
    exit 3
  }
  git -C "$repo" fetch origin "$branch"
  git -C "$repo" checkout -B "$branch" FETCH_HEAD
else
  git clone --branch "$branch" --single-branch "$repo_url" "$repo"
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain)"
REMOTE

echo "=== Prepare and seal the union data without Slurm ==="
ssh "$TILLICUM_HOST" bash -s -- \
  "$TILLICUM_ROOT" "$expected_commit" "$bad_medical_jsonl" \
  "$good_medical_jsonl" "$medical_eval_yaml" "$medical_eval_sha256" <<'REMOTE'
set -euo pipefail
umask 077
ulimit -c 0

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

root=$1
expected_commit=$2
bad_medical_jsonl=$3
good_medical_jsonl=$4
medical_eval_yaml=$5
medical_eval_sha256=$6
repo=$root/projects/subliminal-mitigate
env_root=$root/envs/subliminal-mitigate-py311
output=$root/outputs/massive_medical_union_pilot_v1
control=$output/control
data=$output/data
massive_data=$root/outputs/massive_benefit_pilot_v1/data
config=$repo/configs/training_qwen25_7b_massive_medical_union_pilot.yaml
snapshot=$root/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
benefit_model=$root/outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1
benefit_adapter=$benefit_model/checkpoint-30
benefit_manifest=$benefit_model/MODEL_MANIFEST.json

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain)"
test -f "$env_root/.ready"
test -s "$bad_medical_jsonl"
test -s "$good_medical_jsonl"
test -s "$medical_eval_yaml"
test "$(sha256sum "$medical_eval_yaml" | awk '{print $1}')" = "$medical_eval_sha256"
test -s "$massive_data/data_manifest.json"
test -d "$snapshot"
test -s "$benefit_adapter/adapter_config.json"
test -s "$benefit_manifest"
mkdir "$output"
mkdir -p "$control" "$output/models" "$output/evaluation/wave1" \
  "$root/cache" "$root/config" "$root/tmp"
printf 'repo_commit=%s\ncreated_at=%s\nnon_gpu_stage=true\n' \
  "$expected_commit" "$(date --iso-8601=seconds)" > "$control/STAGING_IN_PROGRESS"

module load conda/Miniforge3-25.3.1-3
conda activate "$env_root"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export XDG_CACHE_HOME=$root/cache
export XDG_CONFIG_HOME=$root/config
export PIP_CACHE_DIR=$root/cache/pip
export TMPDIR=$root/tmp
export HF_HOME=$root/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$root/cache/vllm
export TRITON_CACHE_DIR=$root/cache/triton
export PYTHONPYCACHEPREFIX=$root/tmp/massive-medical-union-stage-pyc
mkdir -p "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

cd "$repo"
python -m pip check
python scripts/prepare_massive_medical_union_pilot_data.py \
  --massive-data-root "$massive_data" \
  --bad-medical-jsonl "$bad_medical_jsonl" \
  --good-medical-jsonl "$good_medical_jsonl" \
  --medical-eval-yaml "$medical_eval_yaml" \
  --medical-eval-sha256 "$medical_eval_sha256" \
  --tokenizer-snapshot "$snapshot" \
  --output-dir "$data"
python scripts/prepare_massive_medical_union_pilot_data.py \
  --massive-data-root "$massive_data" \
  --bad-medical-jsonl "$bad_medical_jsonl" \
  --good-medical-jsonl "$good_medical_jsonl" \
  --medical-eval-yaml "$medical_eval_yaml" \
  --medical-eval-sha256 "$medical_eval_sha256" \
  --tokenizer-snapshot "$snapshot" \
  --output-dir "$data" --audit-only

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python scripts/sample_massive_structured_generations.py \
  --model pi_base=BASE --training_config "$config" \
  --prompt_file "$massive_data/dev/prompts.json" \
  --output_dir "$output/evaluation/preflight" \
  --max_new_tokens 256 --max_context 2048 --seed 8172026 \
  --structured_constraint_profile const_tree_no_ws_v3 --preflight_only
python scripts/sample_massive_union_medical_direct.py \
  --model pi_base=BASE --training_config "$config" \
  --data_manifest "$data/data_manifest.json" \
  --prompt_file "$data/medical_eval/official16.json" \
  --output_dir "$output/evaluation/preflight/medical" --preflight_only

python scripts/audit_massive_medical_union_tillicum_workflow.py write-prep \
  --repo-root "$repo" --data-root "$data" \
  --local-model-snapshot "$snapshot" \
  --benefit-control-manifest "$benefit_manifest" \
  --benefit-control-adapter "$benefit_adapter" \
  --output-file "$control/PREP_COMPLETE.json"

bash -n scripts/stage_massive_medical_union_pilot_tillicum.sh
bash -n scripts/submit_massive_medical_union_wave1_tillicum.sh
bash -n scripts/finalize_massive_medical_union_wave1_tillicum.sh
bash -n scripts/status_massive_medical_union_pilot_tillicum.sh
bash -n scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch
bash -n scripts/sbatch_massive_medical_union_wave1_evaluate_tillicum_h200.sbatch
python -m py_compile \
  scripts/audit_massive_medical_union_tillicum_workflow.py \
  scripts/prepare_massive_medical_union_pilot_data.py \
  scripts/sample_massive_union_medical_direct.py \
  scripts/judge_massive_union_medical.py \
  scripts/summarize_massive_union_components.py
python -m unittest \
  tests.test_massive_medical_union_data \
  tests.test_massive_union_component_evaluation \
  tests.test_massive_medical_union_tillicum_workflow

rm "$control/STAGING_IN_PROGRESS"
printf 'repo_commit=%s\ncompleted_at=%s\nnon_gpu_stage=true\nwave1_max_h200_minutes=80\nwave1_max_cost_usd=1.20\nwave2_submitted=false\nquorum_submitted=false\n' \
  "$expected_commit" "$(date --iso-8601=seconds)" > "$control/STAGED"
echo "Fresh union staging root: $output"
REMOTE

echo "Tillicum staging complete. No Slurm job was submitted."
echo 'Wave-1 hard maximum: 80 H200-minutes = $1.20.'
echo "Explicit submit command (not run):"
echo "  ssh $TILLICUM_HOST 'cd $REMOTE_REPO_ROOT && scripts/submit_massive_medical_union_wave1_tillicum.sh wave1 --ack-max-cost-usd 1.20'"
