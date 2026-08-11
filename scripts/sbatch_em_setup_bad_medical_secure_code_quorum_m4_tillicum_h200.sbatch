#!/bin/bash
# Create/update the isolated Python environment for the Tillicum H200 workflow.
# Tillicum requests GPU-backed setup because CUDA packages should be installed
# and validated on a compute node. Maximum allocation: 1 H200 x 2 hours.

#SBATCH --job-name=em_cap_env
#SBATCH --account=stf
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/em_capability_quorum_setup_%j.out
#SBATCH --error=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/em_capability_quorum_setup_%j.err

set -euo pipefail

export TILLICUM_ROOT="${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}"
export REPO_ROOT="${REPO_ROOT:-$TILLICUM_ROOT/projects/subliminal-mitigate}"
export ENV_ROOT="${ENV_ROOT:-$TILLICUM_ROOT/envs/subliminal-mitigate-py311}"

cd "$REPO_ROOT"
module load conda/Miniforge3-25.3.1-3

export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=$TILLICUM_ROOT/cache
export XDG_CONFIG_HOME=$TILLICUM_ROOT/config
export PIP_CACHE_DIR=$TILLICUM_ROOT/cache/pip
export TMPDIR=$TILLICUM_ROOT/tmp
export HF_HOME=$TILLICUM_ROOT/cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=$TILLICUM_ROOT/cache/vllm
export TRITON_CACHE_DIR=$TILLICUM_ROOT/cache/triton
export UNSLOTH_COMPILE_LOCATION=$TILLICUM_ROOT/cache/unsloth_compiled_cache
mkdir -p \
  "$TILLICUM_ROOT/outputs/logs" \
  "$XDG_CONFIG_HOME" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR" \
  "$UNSLOTH_COMPILE_LOCATION"

date
hostname
nvidia-smi
git rev-parse --short HEAD

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  conda create --yes --prefix "$ENV_ROOT" python=3.11 pip
fi
conda activate "$ENV_ROOT"

python -m pip install --upgrade pip
# Unsloth's transitive dependency currently permits torchao 0.18, but that
# release requires a newer torch than this project's pinned torch 2.9.0.
python -m pip install \
  --requirement requirements.txt \
  "torchao>=0.13,<0.18"

# Import unsloth first because it applies required torch/transformers patches.
python - <<'PY'
import unsloth
import torch
import transformers
import datasets
import peft
import trl
import vllm

assert torch.cuda.is_available(), "CUDA is not available in the Tillicum environment"
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("peft", peft.__version__)
print("trl", trl.__version__)
print("vllm", vllm.__version__)
PY

python -m pip freeze > "$ENV_ROOT/pip-freeze.txt"
sha256sum requirements.txt > "$ENV_ROOT/requirements.sha256"
{
  date --iso-8601=seconds
  git rev-parse HEAD
} > "$ENV_ROOT/.ready"

echo "Environment ready: $ENV_ROOT"
date
