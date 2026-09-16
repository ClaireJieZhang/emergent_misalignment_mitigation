#!/bin/bash
# CPU-only terminal assembly; no medical judging or downstream release.

set -euo pipefail
umask 077
ulimit -c 0
[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/finalize_massive_medical_ratio_panels_v1_evaluation_tillicum.sh' >&2
  exit 2
}

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset MMU_RATIO_EVAL_TILLICUM_ROOT
while IFS= read -r variable; do
  case "$variable" in SBATCH_*) unset "$variable" ;; esac
done < <(compgen -v)

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panel-evaluation-v1
python=$root/envs/subliminal-mitigate-py311/bin/python
cd "$repo"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
"$python" -B scripts/manage_massive_medical_ratio_panels_v1_evaluation.py finalize
