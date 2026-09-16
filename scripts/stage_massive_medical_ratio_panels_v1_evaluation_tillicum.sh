#!/bin/bash
# CPU-only preparation; no Slurm submission, model generation, or API call.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/stage_massive_medical_ratio_panels_v1_evaluation_tillicum.sh' >&2
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
manager=$repo/scripts/manage_massive_medical_ratio_panels_v1_evaluation.py
sampler=$repo/scripts/sample_massive_medical_ratio_panels_v1_evaluation.py

cd "$repo"
test "$(git rev-parse --abbrev-ref HEAD)" = claire/massive-medical-ratio-panel-evaluation-v1
test -z "$(git status --porcelain=v1 --untracked-files=all)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-ratio-eval-stage-pyc

"$python" -B -m unittest \
  tests.test_massive_medical_ratio_panels_v1_evaluation_workflow \
  tests.test_massive_medical_ratio_panels_v1_evaluation_sampler
"$python" -B -m py_compile "$manager" "$sampler"
bash -n \
  scripts/stage_massive_medical_ratio_panels_v1_evaluation_tillicum.sh \
  scripts/submit_massive_medical_ratio_panels_v1_evaluation_tillicum.sh \
  scripts/sbatch_massive_medical_ratio_panels_v1_evaluation_tillicum_h200.sbatch \
  scripts/status_massive_medical_ratio_panels_v1_evaluation_tillicum.sh \
  scripts/finalize_massive_medical_ratio_panels_v1_evaluation_tillicum.sh

"$python" -B "$manager" stage
"$python" -B "$manager" audit-stage
test -z "$(git status --porcelain=v1 --untracked-files=all)"
echo 'CPU_STAGED_ONLY: zero new Slurm jobs, GPU minutes, and API calls.'
echo 'The separate exact four-job $4.800000 authorization remains required.'
