#!/bin/bash
# Merge the four resumable quorum chunks, extract code, execute all seven
# conditions in isolated network-disabled containers, and write final metrics.
# Tillicum requires one H200 allocation even though this stage performs CPU work.

#SBATCH --job-name=gc_lcb_score
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_final_evaluation_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_final_evaluation_%j.err

set -euo pipefail
umask 077

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY

TILLICUM_ROOT=/gpfs/projects/stf/claizhan/subliminal-mitigate
REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
ENV_ROOT=$TILLICUM_ROOT/envs/subliminal-mitigate-py311
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_magicoder_lcb_q3_m4
DATA_ROOT=$OUTPUT_ROOT/data
ASSET_ROOT=$OUTPUT_ROOT/assets
FINAL_ROOT=$OUTPUT_ROOT/final
DIRECT_ROOT=$FINAL_ROOT/generations/direct
QUORUM_ROOT=$FINAL_ROOT/generations/quorum_q3_m4
DELTA_ROOT=$FINAL_ROOT/generations/pi_quorum_delta_q3_m4
MERGED_ROOT=$FINAL_ROOT/generations/merged
CUSTOM_ROOT=$FINAL_ROOT/custom
EVALUATION_ROOT=$FINAL_ROOT/evaluations
PROMPT_FILE=$DATA_ROOT/lcb_final_prompts.json
BENCHMARK_FILE=$DATA_ROOT/lcb_final_evaluator.jsonl
LCB_REPO=$ASSET_ROOT/LiveCodeBench
LCB_SITE=$ASSET_ROOT/lcb-python-site
SANDBOX_SIF=$ASSET_ROOT/python-3.11-slim-amd64.sif

cd "$REPO_ROOT"
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"
export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export TMPDIR=$TILLICUM_ROOT/tmp
mkdir -p "$MERGED_ROOT" "$CUSTOM_ROOT" "$EVALUATION_ROOT" "$TMPDIR"

date
hostname
git rev-parse --verify HEAD
authorized_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$OUTPUT_ROOT/control/AUTHORIZED_MAX_COST_USD_14.40")
test -n "$authorized_commit"
test "$(git rev-parse HEAD)" = "$authorized_commit"
test -s "$OUTPUT_ROOT/control/gate/GO"
python scripts/prepare_magicoder_lcb_data.py --output-root "$DATA_ROOT" --audit-only
test -s "$FINAL_ROOT/DIRECT_COMPLETE"
test -s "$PROMPT_FILE"
test -s "$BENCHMARK_FILE"
test -s "$SANDBOX_SIF"
sha256sum -c "$SANDBOX_SIF.sha256"
test "$(git -C "$LCB_REPO" rev-parse HEAD)" = 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
test -s "$LCB_SITE/.ready"
for condition in quorum_q3_m4 pi_quorum_delta_q3_m4; do
  for chunk in 0 1; do
    test -s "$FINAL_ROOT/generations/$condition/chunk_$chunk/JOB_COMPLETE"
    test -s "$FINAL_ROOT/generations/$condition/chunk_$chunk/chunk.json"
  done
done
for name in pi_base pi_bad pi_good_0 pi_good_1 pi_good_2; do
  test -s "$DIRECT_ROOT/$name.json"
done

echo "=== Merge and audit exact, disjoint quorum chunks ==="
python scripts/merge_lcb_generation_chunks.py \
  --chunk "$QUORUM_ROOT/chunk_0/chunk.json" \
  --chunk "$QUORUM_ROOT/chunk_1/chunk.json" \
  --prompt_file "$PROMPT_FILE" \
  --method quorum \
  --q 3 \
  --chunk_count 2 \
  --output_file "$MERGED_ROOT/quorum_q3_m4.json"
python scripts/merge_lcb_generation_chunks.py \
  --chunk "$DELTA_ROOT/chunk_0/chunk.json" \
  --chunk "$DELTA_ROOT/chunk_1/chunk.json" \
  --prompt_file "$PROMPT_FILE" \
  --method pi_quorum_delta \
  --q 3 \
  --chunk_count 2 \
  --output_file "$MERGED_ROOT/pi_quorum_delta_q3_m4.json"

echo "=== Extract all seven conditions with one pinned code extractor ==="
python scripts/prepare_lcb_evaluation.py \
  --generation "pi_base=$DIRECT_ROOT/pi_base.json" \
  --generation "pi_bad=$DIRECT_ROOT/pi_bad.json" \
  --generation "pi_good_0=$DIRECT_ROOT/pi_good_0.json" \
  --generation "pi_good_1=$DIRECT_ROOT/pi_good_1.json" \
  --generation "pi_good_2=$DIRECT_ROOT/pi_good_2.json" \
  --generation "quorum_q3_m4=$MERGED_ROOT/quorum_q3_m4.json" \
  --generation "pi_quorum_delta_q3_m4=$MERGED_ROOT/pi_quorum_delta_q3_m4.json" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$CUSTOM_ROOT"

run_sandbox_evaluation() {
  local name=$1
  local result_dir=$EVALUATION_ROOT/$name
  local result_file=$result_dir/evaluation.json
  mkdir -p "$result_dir"
  if [[ -s "$result_file" ]]; then
    python -c 'import hashlib,json,sys
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
p=json.load(open(sys.argv[1])); m=p["meta"]; assert m["n_questions"]==182,m; assert len(p["tasks"])==182; assert m["custom_output_sha256"]==sha(sys.argv[2]),m; assert m["custom_meta_sha256"]==sha(sys.argv[3]),m; assert m["benchmark_file_sha256"]==sha(sys.argv[4]),m; print("audited existing evaluation",sys.argv[1])' \
      "$result_file" "$CUSTOM_ROOT/$name.custom.json" "$CUSTOM_ROOT/$name.custom.meta.json" "$BENCHMARK_FILE"
    return
  fi
  env -i PATH=/usr/bin:/bin \
    /usr/bin/apptainer exec \
      --cleanenv \
      --containall \
      --no-home \
      --net \
      --network none \
      --no-mount bind-paths \
      --no-privs \
      --pids-limit 128 \
      --pwd /tmp \
      --bind "$LCB_SITE:/opt/lcb-site:ro" \
      --bind "$LCB_REPO:/opt/livecodebench:ro" \
      --bind "$REPO_ROOT/scripts/run_lcb_sandbox_evaluation.py:/opt/run_lcb_sandbox_evaluation.py:ro" \
      --bind "$CUSTOM_ROOT/$name.custom.json:/inputs/custom.json:ro" \
      --bind "$CUSTOM_ROOT/$name.custom.meta.json:/inputs/custom.meta.json:ro" \
      --bind "$result_dir:/results:rw" \
      --env PYTHONPATH=/opt/lcb-site:/opt/livecodebench \
      "$SANDBOX_SIF" \
      python /opt/run_lcb_sandbox_evaluation.py \
        --benchmark_file - \
        --custom_output_file /inputs/custom.json \
        --output_file /results/evaluation.json \
        --num_processes 8 \
        --timeout 6 < "$BENCHMARK_FILE"
  test -s "$result_file"
}

echo "=== Execute untrusted code in seven separate writable sandboxes ==="
for name in \
  pi_base \
  pi_bad \
  pi_good_0 \
  pi_good_1 \
  pi_good_2 \
  quorum_q3_m4 \
  pi_quorum_delta_q3_m4; do
  run_sandbox_evaluation "$name"
done

echo "=== Summarize paired pass@1 and benefit retention ==="
python scripts/summarize_lcb_evaluations.py \
  --evaluation "pi_base=$EVALUATION_ROOT/pi_base/evaluation.json" \
  --evaluation "pi_bad=$EVALUATION_ROOT/pi_bad/evaluation.json" \
  --evaluation "pi_good_0=$EVALUATION_ROOT/pi_good_0/evaluation.json" \
  --evaluation "pi_good_1=$EVALUATION_ROOT/pi_good_1/evaluation.json" \
  --evaluation "pi_good_2=$EVALUATION_ROOT/pi_good_2/evaluation.json" \
  --evaluation "quorum_q3_m4=$EVALUATION_ROOT/quorum_q3_m4/evaluation.json" \
  --evaluation "pi_quorum_delta_q3_m4=$EVALUATION_ROOT/pi_quorum_delta_q3_m4/evaluation.json" \
  --output_file "$FINAL_ROOT/summary.json" \
  --markdown_file "$FINAL_ROOT/summary.md" \
  --base_name pi_base \
  --good_names pi_good_0,pi_good_1,pi_good_2 \
  --quorum_names quorum_q3_m4,pi_quorum_delta_q3_m4

test -s "$FINAL_ROOT/summary.json"
test -s "$FINAL_ROOT/summary.md"
ready_build=$FINAL_ROOT/.evaluation-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\n' "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" > "$ready_build"
mv "$ready_build" "$FINAL_ROOT/FINAL_EVALUATION_COMPLETE"
cat "$FINAL_ROOT/summary.md"
date
