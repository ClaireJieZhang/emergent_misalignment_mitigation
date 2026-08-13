#!/bin/bash
# Stage public pinned assets and the clean repo checkout without allocating GPU.

set -euo pipefail
umask 077

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TILLICUM_HOST=${TILLICUM_HOST:-tillicum}
TILLICUM_ROOT=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
REMOTE_REPO_URL=${REMOTE_REPO_URL:-https://github.com/ClaireJieZhang/emergent_misalignment_mitigation.git}
REMOTE_BRANCH=${REMOTE_BRANCH:-claire/capability-quorum-secure-code}
REMOTE_REPO_ROOT=$TILLICUM_ROOT/projects/subliminal-mitigate
OUTPUT_ROOT=$TILLICUM_ROOT/outputs/general_code_evalplus_base_vs_pilot_v1
ASSET_ROOT=$OUTPUT_ROOT/assets
expected_commit=$(git -C "$repo_root" rev-parse HEAD)

asset_cache=$(mktemp -d /tmp/evalplus-diagnostic-assets.XXXXXX)
case "$asset_cache" in
  /tmp/evalplus-diagnostic-assets.*) trap 'rm -rf -- "$asset_cache"' EXIT ;;
  *) echo "Unsafe temporary asset path: $asset_cache" >&2; exit 2 ;;
esac

curl -L --fail --silent --show-error \
  https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz \
  -o "$asset_cache/HumanEvalPlus-v0.1.10.jsonl.gz"
curl -L --fail --silent --show-error \
  https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz \
  -o "$asset_cache/MbppPlus-v0.2.0.jsonl.gz"
test "$(LC_ALL=C shasum -a 256 "$asset_cache/HumanEvalPlus-v0.1.10.jsonl.gz" | awk '{print $1}')" = \
  272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101
test "$(LC_ALL=C shasum -a 256 "$asset_cache/MbppPlus-v0.2.0.jsonl.gz" | awk '{print $1}')" = \
  af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63

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
control=$root/outputs/general_code_evalplus_base_vs_pilot_v1/control
if [[ -e "$control/SUBMITTED" || -e "$control/AUTHORIZED_MAX_COST_USD_0.90" || -s "$control/jobs.tsv" ]]; then
  echo "Refusing to mutate staged inputs after diagnostic authorization/submission." >&2
  exit 3
fi
mkdir -p "$root/projects" "$root/outputs/logs" "$root/cache" "$root/config" "$root/tmp"
if [[ -d "$repo/.git" ]]; then
  test -z "$(git -C "$repo" status --porcelain)" || {
    echo "Refusing to update dirty Tillicum checkout: $repo" >&2
    git -C "$repo" status --short >&2
    exit 3
  }
  git -C "$repo" fetch origin "$branch"
  git -C "$repo" checkout -B "$branch" FETCH_HEAD
else
  git clone --branch "$branch" --single-branch "$repo_url" "$repo"
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
git -C "$repo" log -1 --oneline
REMOTE

echo "=== Copy and verify pinned public benchmark assets ==="
ssh "$TILLICUM_HOST" mkdir -p "$ASSET_ROOT"
rsync -avP "$asset_cache/HumanEvalPlus-v0.1.10.jsonl.gz" "$TILLICUM_HOST:$ASSET_ROOT/"
rsync -avP "$asset_cache/MbppPlus-v0.2.0.jsonl.gz" "$TILLICUM_HOST:$ASSET_ROOT/"

echo "=== Pin EvalPlus source, minimal evaluator dependencies, and prompt bank ==="
ssh "$TILLICUM_HOST" bash -s -- "$TILLICUM_ROOT" <<'REMOTE'
set -euo pipefail
umask 077
root=$1
repo=$root/projects/subliminal-mitigate
source_root=$root/outputs/general_code_magicoder_lcb_q3_m4
output_root=$root/outputs/general_code_evalplus_base_vs_pilot_v1
asset_root=$output_root/assets
data_root=$output_root/data
evalplus_repo=$asset_root/evalplus-v0.3.1
evalplus_site=$asset_root/evalplus-python-site
evalplus_commit=e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2
sandbox_sif=$source_root/assets/python-3.11-slim-amd64.sif

echo '272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101  '"$asset_root/HumanEvalPlus-v0.1.10.jsonl.gz" | sha256sum -c -
echo 'af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63  '"$asset_root/MbppPlus-v0.2.0.jsonl.gz" | sha256sum -c -
echo 'd651ca156e0d54c3dd3a1ba48d5372e581648a15b18f37455c887531b2d25fd4  '"$sandbox_sif" | sha256sum -c -

if [[ ! -d "$evalplus_repo/.git" ]]; then
  test ! -e "$evalplus_repo"
  git clone --branch v0.3.1 --single-branch https://github.com/evalplus/evalplus.git "$evalplus_repo"
fi
test -z "$(git -C "$evalplus_repo" status --porcelain)"
test "$(git -C "$evalplus_repo" rev-parse HEAD)" = "$evalplus_commit"

module load conda/Miniforge3-25.3.1-3
conda activate "$root/envs/subliminal-mitigate-py311"
if [[ ! -s "$evalplus_site/.ready" ]]; then
  test ! -e "$evalplus_site" || {
    echo "Incomplete EvalPlus site exists: $evalplus_site" >&2
    exit 3
  }
  site_build=$(mktemp -d "$asset_root/.evalplus-site-build.XXXXXX")
  python -m pip install --disable-pip-version-check --no-compile --target "$site_build" \
    appdirs==1.4.4 \
    psutil==5.9.8 \
    tempdir==0.7.1 \
    termcolor==2.4.0 \
    rich==13.9.2 \
    tree_sitter==0.22.3 \
    tree_sitter_python==0.21.0 \
    wget==3.2
  python -m pip freeze --path "$site_build" | LC_ALL=C sort > "$site_build/requirements.freeze.txt"
  printf 'evalplus_commit=%s\nprepared_at=%s\n' \
    "$evalplus_commit" "$(date --iso-8601=seconds)" > "$site_build/.ready"
  python "$repo/scripts/audit_evalplus_assets.py" \
    --site "$site_build" --create-site-seal
  mv "$site_build" "$evalplus_site"
fi
test "$(awk -F= '$1=="evalplus_commit" {print $2}' "$evalplus_site/.ready")" = "$evalplus_commit"
python "$repo/scripts/audit_evalplus_assets.py" --site "$evalplus_site"

python "$repo/scripts/prepare_magicoder_lcb_data.py" \
  --output-root "$source_root/data" --audit-only
if [[ ! -s "$data_root/data_manifest.json" ]]; then
  python "$repo/scripts/prepare_evalplus_diagnostic.py" \
    --output_root "$data_root" \
    --humaneval_file "$asset_root/HumanEvalPlus-v0.1.10.jsonl.gz" \
    --mbpp_file "$asset_root/MbppPlus-v0.2.0.jsonl.gz" \
    --training_shard "$source_root/data/magicoder_python_shard_000" \
    --training_shard "$source_root/data/magicoder_python_shard_001" \
    --training_shard "$source_root/data/magicoder_python_shard_002"
fi
python "$repo/scripts/prepare_evalplus_diagnostic.py" --output_root "$data_root" --audit-only

printf 'evalplus-stream-smoke' | env -i PATH=/usr/bin:/bin /usr/bin/apptainer exec \
  --cleanenv --containall --no-home --no-eval --net --network none \
  --no-mount bind-paths --no-privs --pwd /tmp \
  --bind "$evalplus_site:/opt/evalplus-site:ro" \
  --bind "$source_root/assets/lcb-python-site:/opt/lcb-site:ro" \
  --bind "$evalplus_repo:/opt/evalplus-src:ro" \
  --bind "$repo/scripts/evalplus_sandbox_stubs:/opt/evalplus-stubs:ro" \
  --bind "$repo/scripts/run_evalplus_sandbox_evaluation.py:/opt/run_evalplus_sandbox_evaluation.py:ro" \
  --env PYTHONPATH=/opt/evalplus-stubs:/opt/evalplus-site:/opt/lcb-site:/opt/evalplus-src \
  --env HOME=/tmp/evalhome --env XDG_CACHE_HOME=/tmp/evalcache \
  "$sandbox_sif" python -c \
  'import sys; from evalplus.sanitize import sanitize; assert sys.stdin.read() == "evalplus-stream-smoke"; assert "def f" in sanitize("def f():\n    return 1", "f")'

# Functional checker smoke in the exact namespace/mount boundary: execute one
# known-good and one known-bad solution through EvalPlus's untrusted child and
# assert that GPFS, network, and GPU devices are not visible in the container.
env -i PATH=/usr/bin:/bin /usr/bin/apptainer exec \
  --cleanenv --containall --no-home --no-eval --net --network none \
  --no-mount bind-paths --no-privs --pwd /tmp \
  --bind "$evalplus_site:/opt/evalplus-site:ro" \
  --bind "$source_root/assets/lcb-python-site:/opt/lcb-site:ro" \
  --bind "$evalplus_repo:/opt/evalplus-src:ro" \
  --bind "$repo/scripts/evalplus_sandbox_stubs:/opt/evalplus-stubs:ro" \
  --bind "$repo/scripts/run_evalplus_sandbox_evaluation.py:/opt/run_evalplus_sandbox_evaluation.py:ro" \
  --env PYTHONPATH=/opt/evalplus-stubs:/opt/evalplus-site:/opt/lcb-site:/opt/evalplus-src \
  --env HOME=/tmp/evalhome --env XDG_CACHE_HOME=/tmp/evalcache \
  --env EVALPLUS_MAX_MEMORY_BYTES=4294967296 \
  "$sandbox_sif" /bin/sh -c \
  'test ! -e /gpfs; test ! -e /dev/nvidia0; python - <<'"'"'PY'"'"'
import importlib.util
import socket
from evalplus.eval import untrusted_check
from evalplus.evaluate import evaluate

spec = importlib.util.spec_from_file_location("diagnostic_sandbox", "/opt/run_evalplus_sandbox_evaluation.py")
diagnostic_sandbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic_sandbox)
flags = diagnostic_sandbox.suspicious_flags(
    "import os\ndef f(): return os.read(os.open('/inputs/dataset.jsonl.gz', 0), 10)"
)
assert "os_filesystem" in flags, flags

try:
    socket.create_connection(("198.51.100.1", 80), timeout=0.2)
except OSError:
    pass
else:
    raise AssertionError("network unexpectedly available")

good = untrusted_check(
    "humaneval", "def f(x): return x + 1", [[1]], "f", [2], 0, [0.001]
)
bad = untrusted_check(
    "humaneval", "def f(x): return x", [[1]], "f", [2], 0, [0.001]
)
assert good[0] == "pass", good
assert bad[0] == "fail", bad
PY'

if [[ ! -s "$asset_root/asset_manifest.json" ]]; then
  python "$repo/scripts/audit_evalplus_assets.py" \
    --asset_root "$asset_root" \
    --source_root "$source_root" \
    --create
fi
python "$repo/scripts/audit_evalplus_assets.py" \
  --asset_root "$asset_root" \
  --source_root "$source_root"

ready_build=$asset_root/.assets-ready-$$
printf 'evalplus_commit=%s\nhumaneval_sha256=%s\nmbpp_sha256=%s\nsandbox_sif_sha256=%s\nsite_freeze_sha256=%s\nasset_manifest_sha256=%s\nprepared_at=%s\n' \
  "$evalplus_commit" \
  272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101 \
  af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63 \
  d651ca156e0d54c3dd3a1ba48d5372e581648a15b18f37455c887531b2d25fd4 \
  "$(sha256sum "$evalplus_site/requirements.freeze.txt" | awk '{print $1}')" \
  "$(sha256sum "$asset_root/asset_manifest.json" | awk '{print $1}')" \
  "$(date --iso-8601=seconds)" > "$ready_build"
mv "$ready_build" "$asset_root/ASSETS_READY"
du -sh "$asset_root" "$data_root"
REMOTE

echo "EvalPlus diagnostic staging complete. No Slurm job was submitted."
