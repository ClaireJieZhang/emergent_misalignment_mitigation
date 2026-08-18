#!/bin/bash
# Audit and score the six existing K&K v3 artifacts without Slurm or a GPU.

set -euo pipefail
umask 077
ulimit -c 0

if [[ "$#" -ne 1 || "$1" != recover-existing-generations ]]; then
  echo "Usage: $0 recover-existing-generations" >&2
  exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}${SLURM_JOB_NAME:-}${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "K&K v3 recovery must run directly on a CPU login node." >&2
  exit 3
fi

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
EXPECTED_REPO=$ROOT/projects/subliminal-mitigate-kk-v3-recovery
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
CHECKPOINT=$V1_ROOT/model/kk_reasoning_n5_pilot/checkpoint-192
PREP=$CONTROL/PREP_COMPLETE
AUTH=$CONTROL/AUTHORIZED_MAX_COST_USD_0.45
JOBS=$CONTROL/jobs.tsv
PROVENANCE=$CONTROL/CPU_RECOVERY_PROVENANCE.json
LOCK=$CONTROL/.cpu-recovery.lock
STDOUT_LOG=$ROOT/outputs/logs/knights_knaves_confirmation_v3_237934.out
STDERR_LOG=$ROOT/outputs/logs/knights_knaves_confirmation_v3_237934.err
PYTHON=$ENV_ROOT/bin/python

if [[ "$REPO" != "$EXPECTED_REPO" ]]; then
  echo "K&K v3 recovery must run from its isolated detached worktree." >&2
  exit 3
fi
cd "$REPO"
for directory in "$V3_ROOT" "$V3_DATA" "$CONTROL" "$EVAL" "$GENERATIONS"; do
  test -d "$directory" && test ! -L "$directory" || {
    echo "Refusing unsafe K&K v3 recovery directory: $directory" >&2
    exit 3
  }
done
if [[ -e "$SCORES" || -L "$SCORES" ]]; then
  test -d "$SCORES" && test ! -L "$SCORES" || {
    echo "Refusing unsafe K&K v3 score directory: $SCORES" >&2
    exit 3
  }
else
  mkdir "$SCORES"
fi
test -x "$PYTHON"
if [[ -L "$LOCK" || ( -e "$LOCK" && ! -f "$LOCK" ) ]]; then
  echo "Refusing unsafe K&K v3 CPU-recovery lock file." >&2
  exit 3
fi
if [[ -e "$SUMMARY_MD" || -L "$SUMMARY_MD" ]]; then
  test -f "$SUMMARY_MD" && test ! -L "$SUMMARY_MD" || {
    echo "Refusing unsafe preexisting K&K v3 Markdown summary." >&2
    exit 3
  }
fi

exec 9>"$LOCK"
flock -n 9 || {
  echo "Another K&K v3 CPU recovery is active." >&2
  exit 3
}

"$PYTHON" scripts/audit_knights_knaves_confirmation_v3_cpu_recovery.py provenance \
  --repo-root "$REPO" --v1-root "$V1_ROOT" --v2-root "$V2_ROOT" \
  --v3-root "$V3_ROOT" --checkpoint "$CHECKPOINT" \
  --prep-file "$PREP" --auth-file "$AUTH" --jobs-file "$JOBS" \
  --stdout-log "$STDOUT_LOG" --stderr-log "$STDERR_LOG" \
  --generations-dir "$GENERATIONS" --output-file "$PROVENANCE"

sets=(confirmation_v3_n4 confirmation_v3_n5 confirmation_v3_n6)
base_args=()
candidate_args=()
for set_name in "${sets[@]}"; do
  answers=$V3_DATA/sets/${set_name}_answers.json
  for model in pi_base step_192; do
    "$PYTHON" scripts/evaluate_knights_knaves_confirmation_v3.py \
      --answers_file "$answers" \
      --generations_file "$GENERATIONS/${set_name}__${model}.json" \
      --output_file "$SCORES/${set_name}__${model}__direct.json"
  done
  base_args+=(--direct_base "$set_name=$SCORES/${set_name}__pi_base__direct.json")
  candidate_args+=(--direct_candidate "$set_name=$SCORES/${set_name}__step_192__direct.json")
done

if [[ -e "$SUMMARY" || -L "$SUMMARY" ]]; then
  test -f "$SUMMARY" && test ! -L "$SUMMARY" || {
    echo "Refusing unsafe preexisting K&K v3 summary." >&2
    exit 3
  }
  "$PYTHON" scripts/summarize_knights_knaves_confirmation_v3.py audit \
    --summary_file "$SUMMARY" --markdown_file "$SUMMARY_MD" \
    --sentinel_dir "$CONTROL"
else
  "$PYTHON" scripts/summarize_knights_knaves_confirmation_v3.py summary \
    "${base_args[@]}" "${candidate_args[@]}" \
    --candidate_fingerprint \
      36a710b93564ccb9d7c939fdf644bae9a80a6e4c81ca73c2634f4e1a1741701c \
    --v3_data_manifest "$V3_DATA/confirmation_v3_manifest.json" \
    --v2_final_summary "$V2_ROOT/evaluation/sealed_final/summary.json" \
    --output_file "$SUMMARY" --markdown_file "$SUMMARY_MD" \
    --sentinel_dir "$CONTROL" --replicates 10000
fi

"$PYTHON" scripts/audit_knights_knaves_confirmation_v3_cpu_recovery.py results \
  --provenance-file "$PROVENANCE" --scores-dir "$SCORES" \
  --summary-file "$SUMMARY" \
  --markdown-file "$SUMMARY_MD" --sentinel-dir "$CONTROL" \
  --v3-data-manifest "$V3_DATA/confirmation_v3_manifest.json" \
  --v2-final-summary "$V2_ROOT/evaluation/sealed_final/summary.json"

cat "$SUMMARY_MD"
echo "K&K v3 CPU-only recovery completed with zero new GPU allocation."
