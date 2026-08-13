#!/bin/bash
# Evaluate one generated model/dataset pair in the hardened EvalPlus sandbox.

set -euo pipefail
umask 077
ulimit -c 0

if [[ "$#" -ne 10 ]]; then
  echo "Usage: $0 DATASET MODEL DATASET_FILE PROMPT_FILE GENERATION_FILE DESTINATION EVALPLUS_SITE LCB_SITE EVALPLUS_REPO SANDBOX_SIF" >&2
  exit 2
fi

dataset=$1
model=$2
dataset_file=$3
prompt_file=$4
generation_file=$5
destination=$6
evalplus_site=$7
lcb_site=$8
evalplus_repo=$9
sandbox_sif=${10}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "$dataset" in humaneval|mbpp) ;; *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;; esac
test -n "${SLURM_JOB_ID:-}"
for path in "$dataset_file" "$prompt_file" "$generation_file" \
  "$evalplus_site/.ready" "$lcb_site/.ready" "$sandbox_sif"; do
  test -s "$path"
done
test -d "$evalplus_repo/.git"

if [[ -s "$destination" ]]; then
  python scripts/audit_evalplus_diagnostic_result.py \
    --input "$destination" --dataset "$dataset" \
    --dataset_file "$dataset_file" --prompt_file "$prompt_file" \
    --generation_file "$generation_file" --model_name "$model"
  echo "Audited complete EvalPlus result; skipping $dataset/$model"
  exit 0
fi

result_dir=$(dirname "$destination")
mkdir -p "$result_dir"
node_root=$(mktemp -d "/tmp/evalplus-${SLURM_JOB_ID}-${dataset}-${model}.XXXXXX")
chmod 700 "$node_root"
cleanup_node_root() {
  case "${node_root:-}" in
    /tmp/evalplus-${SLURM_JOB_ID}-${dataset}-${model}.*) rm -rf -- "$node_root" ;;
    "") ;;
    *) echo "Refusing unsafe node-local cleanup: $node_root" >&2; return 2 ;;
  esac
}
trap cleanup_node_root EXIT

/usr/bin/timeout --signal=TERM --kill-after=30s 10m \
  env -i PATH=/usr/bin:/bin \
  /usr/bin/apptainer exec \
    --cleanenv --containall --no-home --no-eval --net --network none \
    --no-mount bind-paths --no-privs --pwd /tmp \
    --bind "$evalplus_site:/opt/evalplus-site:ro" \
    --bind "$lcb_site:/opt/lcb-site:ro" \
    --bind "$evalplus_repo:/opt/evalplus-src:ro" \
    --bind "$repo_root/scripts/evalplus_sandbox_stubs:/opt/evalplus-stubs:ro" \
    --bind "$repo_root/scripts/run_evalplus_sandbox_evaluation.py:/opt/run_evalplus_sandbox_evaluation.py:ro" \
    --bind "$prompt_file:/inputs/prompts.json:ro" \
    --bind "$generation_file:/inputs/generation.json:ro" \
    --bind "$node_root:/results:rw" \
    --env PYTHONPATH=/opt/evalplus-stubs:/opt/evalplus-site:/opt/lcb-site:/opt/evalplus-src \
    --env HOME=/tmp/evalhome \
    --env XDG_CACHE_HOME=/results/evalcache \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env EVALPLUS_MAX_MEMORY_BYTES=4294967296 \
    "$sandbox_sif" \
    /bin/sh -c 'ulimit -c 0; ulimit -f 1048576; ulimit -n 1024; exec python -B /opt/run_evalplus_sandbox_evaluation.py "$@"' sh \
      --dataset "$dataset" --dataset_file - \
      --prompt_file /inputs/prompts.json \
      --generation_file /inputs/generation.json \
      --model_name "$model" \
      --output_file /results/evaluation.json \
      --parallel 8 < "$dataset_file"

python scripts/audit_evalplus_diagnostic_result.py \
  --input "$node_root/evaluation.json" --dataset "$dataset" \
  --dataset_file "$dataset_file" --prompt_file "$prompt_file" \
  --generation_file "$generation_file" --model_name "$model"
build=$result_dir/."$(basename "$destination")".$SLURM_JOB_ID.tmp
install -m 600 "$node_root/evaluation.json" "$build"
mv "$build" "$destination"
cleanup_node_root
trap - EXIT
