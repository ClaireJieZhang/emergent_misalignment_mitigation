#!/bin/bash
# Prepare immutable public data and the network-disabled LiveCodeBench sandbox.
# Tillicum requires one H200 allocation even though this stage performs CPU work.
# The final retry cap is 58 minutes so the two prior setup attempts (22 and 18
# seconds) plus this run remain below the authorized one-H200-hour prep ceiling.

#SBATCH --job-name=gc_lcb_prep
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH --gres=gpu:h200:1
#SBATCH --no-requeue
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_prepare_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/general_code_prepare_%j.err

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
LCB_REPO=$ASSET_ROOT/LiveCodeBench
LCB_SITE=$ASSET_ROOT/lcb-python-site
SANDBOX_SIF=$ASSET_ROOT/python-3.11-slim-amd64.sif
LCB_COMMIT=28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
PYTHON_IMAGE_URI=docker://python@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff

mkdir -p "$OUTPUT_ROOT" "$ASSET_ROOT" "$TILLICUM_ROOT/outputs/logs"
cd "$REPO_ROOT"
module load conda/Miniforge3-25.3.1-3
conda activate "$ENV_ROOT"

export PYTHONUNBUFFERED=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$TILLICUM_ROOT/cache
export XDG_CONFIG_HOME=$TILLICUM_ROOT/config
export PIP_CACHE_DIR=$TILLICUM_ROOT/cache/pip
export TMPDIR=$TILLICUM_ROOT/tmp
export HF_HOME=$TILLICUM_ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
mkdir -p "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$HUGGINGFACE_HUB_CACHE"

date
hostname
git rev-parse --verify HEAD
authorized_commit=$(awk -F= '$1=="repo_commit" {print $2}' "$OUTPUT_ROOT/control/AUTHORIZED_MAX_COST_USD_14.40")
test -n "$authorized_commit"
test "$(git rev-parse HEAD)" = "$authorized_commit"
test -x "$ENV_ROOT/bin/python"
test -f "$ENV_ROOT/.ready"
command -v apptainer

echo "=== Pin LiveCodeBench source checkout ==="
if [[ -d "$LCB_REPO/.git" ]]; then
  test -z "$(git -C "$LCB_REPO" status --porcelain)" || {
    echo "Refusing dirty pinned LiveCodeBench checkout: $LCB_REPO" >&2
    exit 2
  }
  test "$(git -C "$LCB_REPO" rev-parse HEAD)" = "$LCB_COMMIT" || {
    echo "Existing LiveCodeBench checkout is not pinned to $LCB_COMMIT" >&2
    exit 2
  }
else
  lcb_build=$ASSET_ROOT/.LiveCodeBench.build-$SLURM_JOB_ID
  test ! -e "$lcb_build" || {
    echo "Stale build directory exists: $lcb_build" >&2
    exit 2
  }
  mkdir "$lcb_build"
  git -C "$lcb_build" init
  git -C "$lcb_build" remote add origin https://github.com/LiveCodeBench/LiveCodeBench.git
  git -C "$lcb_build" fetch --depth=1 origin "$LCB_COMMIT"
  git -C "$lcb_build" checkout --detach FETCH_HEAD
  test "$(git -C "$lcb_build" rev-parse HEAD)" = "$LCB_COMMIT"
  mv "$lcb_build" "$LCB_REPO"
fi

echo "=== Pull the pinned amd64 Python sandbox image ==="
if [[ -f "$SANDBOX_SIF" ]]; then
  test -s "$SANDBOX_SIF.sha256"
  sha256sum -c "$SANDBOX_SIF.sha256"
else
  sif_build=$ASSET_ROOT/.python-sandbox-$SLURM_JOB_ID.sif
  test ! -e "$sif_build"
  apptainer pull "$sif_build" "$PYTHON_IMAGE_URI"
  test -s "$sif_build"
  mv "$sif_build" "$SANDBOX_SIF"
  sha_build=$ASSET_ROOT/.python-sandbox-$SLURM_JOB_ID.sha256
  sha256sum "$SANDBOX_SIF" > "$sha_build"
  mv "$sha_build" "$SANDBOX_SIF.sha256"
fi

echo "=== Build isolated, pinned evaluator dependencies inside the image ==="
if [[ -d "$LCB_SITE" ]]; then
  test -s "$LCB_SITE/.ready"
else
  site_build=$ASSET_ROOT/.lcb-python-site.build-$SLURM_JOB_ID
  test ! -e "$site_build"
  mkdir "$site_build"
  apptainer exec \
    --cleanenv \
    --containall \
    --no-home \
    --bind "$site_build:/opt/lcb-site" \
    --bind "$TMPDIR:/opt/lcb-tmp" \
    --env TMPDIR=/opt/lcb-tmp \
    "$SANDBOX_SIF" \
    python -m pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      --target /opt/lcb-site \
      numpy==2.2.6 \
      pandas==3.0.5 \
      tqdm==4.70.0
  printf '%s\n' \
    'numpy==2.2.6' \
    'pandas==3.0.5' \
    'tqdm==4.70.0' \
    "LiveCodeBench==$LCB_COMMIT" \
    "image=$PYTHON_IMAGE_URI" > "$site_build/.ready"
  mv "$site_build" "$LCB_SITE"
fi

echo "=== Verify imports and the network-disabled execution boundary ==="
env -i PATH=/usr/bin:/bin \
  /usr/bin/apptainer exec \
    --cleanenv \
    --containall \
    --no-home \
    --net \
    --network none \
    --no-mount bind-paths \
    --no-privs \
    --pwd /tmp \
    --bind "$LCB_SITE:/opt/lcb-site:ro" \
    --bind "$LCB_REPO:/opt/livecodebench:ro" \
    --env PYTHONPATH=/opt/lcb-site:/opt/livecodebench \
    "$SANDBOX_SIF" \
    python -c 'import os,socket,numpy,pandas,tqdm; from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics; assert numpy.__version__=="2.2.6"; assert pandas.__version__=="3.0.5"; assert tqdm.__version__=="4.70.0"; assert not any(k.endswith("_API_KEY") or k in {"HF_TOKEN","HUGGINGFACE_HUB_TOKEN"} for k in os.environ); s=socket.socket(); blocked=False
try:
 s.connect(("1.1.1.1",53))
except OSError:
 blocked=True
finally:
 s.close()
assert blocked, "network namespace unexpectedly has external connectivity"; print("sandbox import/network audit passed")'

echo "=== Prepare pinned 3 x 6,000 Magicoder shards and disjoint LCB windows ==="
python scripts/prepare_magicoder_lcb_data.py --output-root "$DATA_ROOT"
python scripts/prepare_magicoder_lcb_data.py --output-root "$DATA_ROOT" --audit-only
python -c 'import json,sys; m=json.load(open(sys.argv[1])); shards=m["magicoder"]["shards"]; windows=m["livecodebench"]["windows"]; assert [x["row_count"] for x in shards]==[6000,6000,6000],shards; assert all(v==0 for v in m["magicoder"]["pairwise_overlap_counts"].values()); assert windows["gate"]["expected_count"]==157,windows; assert windows["final"]["expected_count"]==182,windows; print("data manifest audit passed")' "$DATA_ROOT/data_manifest.json"

echo "=== Functionally smoke the exact sandboxed evaluator ==="
SMOKE_ROOT=$OUTPUT_ROOT/sandbox_smoke
mkdir -p "$SMOKE_ROOT"
python - "$SMOKE_ROOT" <<'PY'
import json, os, sys
root = sys.argv[1]
row = {
    "question_id": "sandbox-smoke",
    "contest_date": "2025-01-01",
    "difficulty": "easy",
    "platform": "synthetic",
    "metadata": json.dumps({"func_name": None}),
    "public_test_cases": json.dumps([{"input": "", "output": "OK\n"}]),
    "private_test_cases": json.dumps([]),
}
with open(os.path.join(root, "benchmark.jsonl"), "w", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
for name, code in (("good", "print('OK')"), ("bad", "print('WRONG')")):
    with open(os.path.join(root, name + ".custom.json"), "w", encoding="utf-8") as handle:
        json.dump([{"question_id": "sandbox-smoke", "code_list": [code]}], handle)
    with open(os.path.join(root, name + ".custom.meta.json"), "w", encoding="utf-8") as handle:
        json.dump({"generation_meta": {"sandbox-smoke": [{}]}}, handle)
PY
for smoke_name in good bad; do
  result_dir=$SMOKE_ROOT/$smoke_name
  mkdir -p "$result_dir"
  env -i PATH=/usr/bin:/bin \
    /usr/bin/apptainer exec \
      --cleanenv --containall --no-home --net --network none \
      --no-mount bind-paths --no-privs --pwd /tmp \
      --bind "$LCB_SITE:/opt/lcb-site:ro" \
      --bind "$LCB_REPO:/opt/livecodebench:ro" \
      --bind "$REPO_ROOT/scripts/run_lcb_sandbox_evaluation.py:/opt/run_lcb_sandbox_evaluation.py:ro" \
      --bind "$SMOKE_ROOT/$smoke_name.custom.json:/inputs/custom.json:ro" \
      --bind "$SMOKE_ROOT/$smoke_name.custom.meta.json:/inputs/custom.meta.json:ro" \
      --bind "$result_dir:/results:rw" \
      --env PYTHONPATH=/opt/lcb-site:/opt/livecodebench \
      "$SANDBOX_SIF" \
      python /opt/run_lcb_sandbox_evaluation.py \
        --benchmark_file - --custom_output_file /inputs/custom.json \
        --output_file /results/evaluation.json --num_processes 1 --timeout 2 \
        < "$SMOKE_ROOT/benchmark.jsonl"
done
python - "$SMOKE_ROOT" <<'PY'
import json, os, sys
root = sys.argv[1]
good = json.load(open(os.path.join(root, "good", "evaluation.json")))
bad = json.load(open(os.path.join(root, "bad", "evaluation.json")))
assert good["tasks"][0]["passed"] == [True], good
assert bad["tasks"][0]["passed"] == [False], bad
assert good["meta"]["benchmark_file_sha256"] == bad["meta"]["benchmark_file_sha256"]
print("functional sandbox good/bad smoke passed")
PY

ready_build=$OUTPUT_ROOT/.prepare-complete-$SLURM_JOB_ID
printf 'job_id=%s\ncompleted_at=%s\nrepo_commit=%s\nlivecodebench_commit=%s\n' \
  "$SLURM_JOB_ID" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$LCB_COMMIT" \
  > "$ready_build"
mv "$ready_build" "$OUTPUT_ROOT/PREP_COMPLETE"
date
