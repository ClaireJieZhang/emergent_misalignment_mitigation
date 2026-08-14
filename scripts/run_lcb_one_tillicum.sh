#!/bin/bash
# Evaluate one generated model on one LCB-compatible hidden-test benchmark.

set -euo pipefail
umask 077
ulimit -c 0

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 BENCHMARK_FILE CUSTOM_FILE DESTINATION LCB_SITE LCB_REPO SANDBOX_SIF TIMEOUT_SECONDS" >&2
  exit 2
fi

benchmark_file=$1
custom_file=$2
destination=$3
lcb_site=$4
lcb_repo=$5
sandbox_sif=$6
case_timeout=$7
wall_limit=${LCB_EVALUATION_WALL_LIMIT:-10m}
evaluator_mode=${LCB_EVALUATOR_MODE:-livecodebench}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
custom_meta=${custom_file%.json}.meta.json
runner_script=$repo_root/scripts/run_lcb_sandbox_evaluation.py

test -n "${SLURM_JOB_ID:-}"
[[ "$case_timeout" =~ ^[1-9][0-9]*$ ]]
[[ "$wall_limit" =~ ^[1-9][0-9]*[sm]$ ]]
case "$evaluator_mode" in
  livecodebench|apps_official) ;;
  *) echo "Invalid LCB_EVALUATOR_MODE: $evaluator_mode" >&2; exit 2 ;;
esac
for path in "$benchmark_file" "$custom_file" "$custom_meta" \
  "$lcb_site/.ready" "$sandbox_sif" "$runner_script"; do
  test -s "$path"
done
test -d "$lcb_repo/.git"

if [[ -s "$destination" ]]; then
  python - "$benchmark_file" "$custom_file" "$custom_meta" "$destination" \
    "$evaluator_mode" "$runner_script" <<'PY'
import hashlib, json, sys
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
benchmark, custom, custom_meta, output, expected_mode, runner_script = sys.argv[1:]
payload = json.load(open(output, encoding="utf-8"))
meta = payload.get("meta", {})
assert meta.get("benchmark_file_sha256") == sha(benchmark), meta
assert meta.get("custom_output_sha256") == sha(custom), meta
assert meta.get("custom_meta_sha256") == sha(custom_meta), meta
recorded_mode = meta.get("evaluator_mode")
recorded_runner = meta.get("runner_script_sha256")
if expected_mode == "apps_official":
    assert recorded_mode == expected_mode, meta
    assert recorded_runner == sha(runner_script), meta
else:
    # Results produced before these provenance fields were added remain
    # resumable under the unchanged default evaluator only.
    assert recorded_mode in (None, expected_mode), meta
    assert recorded_runner in (None, sha(runner_script)), meta
tasks = payload.get("tasks")
assert isinstance(tasks, list) and meta.get("n_questions") == len(tasks), meta
assert len({str(row["question_id"]) for row in tasks}) == len(tasks)
print("Audited complete LCB-compatible result", output)
PY
  exit 0
fi

result_dir=$(dirname "$destination")
mkdir -p "$result_dir"
node_root=$(mktemp -d "/tmp/lcb-${SLURM_JOB_ID}.XXXXXX")
chmod 700 "$node_root"
cleanup_node_root() {
  case "${node_root:-}" in
    /tmp/lcb-${SLURM_JOB_ID}.*) rm -rf -- "$node_root" ;;
    "") ;;
    *) echo "Refusing unsafe node-local cleanup: $node_root" >&2; return 2 ;;
  esac
}
trap cleanup_node_root EXIT

/usr/bin/timeout --signal=TERM --kill-after=10s "$wall_limit" \
  env -i PATH=/usr/bin:/bin \
  /usr/bin/apptainer exec \
    --cleanenv --containall --no-home --no-eval --net --network none \
    --no-mount bind-paths --no-privs --pwd /tmp \
    --bind "$lcb_site:/opt/lcb-site:ro" \
    --bind "$lcb_repo:/opt/livecodebench:ro" \
    --bind "$runner_script:/opt/run_lcb_sandbox_evaluation.py:ro" \
    --bind "$custom_file:/inputs/custom.json:ro" \
    --bind "$custom_meta:/inputs/custom.meta.json:ro" \
    --bind "$node_root:/results:rw" \
    --env PYTHONPATH=/opt/lcb-site:/opt/livecodebench \
    --env HOME=/tmp/nohome --env PYTHONDONTWRITEBYTECODE=1 \
    "$sandbox_sif" \
    /bin/sh -c 'ulimit -c 0; ulimit -f 1048576; ulimit -n 1024; exec python -B /opt/run_lcb_sandbox_evaluation.py "$@"' sh \
      --benchmark_file - --custom_output_file /inputs/custom.json \
      --output_file /results/evaluation.json \
      --num_processes 8 --timeout "$case_timeout" \
      --evaluator_mode "$evaluator_mode" < "$benchmark_file"

test -s "$node_root/evaluation.json"
build=$result_dir/."$(basename "$destination")".$SLURM_JOB_ID.tmp
install -m 600 "$node_root/evaluation.json" "$build"
mv "$build" "$destination"
cleanup_node_root
trap - EXIT

# The same exact-hash audit path used for resumability.
exec bash "$0" "$benchmark_file" "$custom_file" "$destination" \
  "$lcb_site" "$lcb_repo" "$sandbox_sif" "$case_timeout"
