#!/bin/bash
# CPU-only terminal audit after both exact training jobs finish.

set -euo pipefail
umask 077
[[ $# -eq 0 ]] || { echo 'Usage: scripts/finalize_massive_medical_ratio_panels_v1_training_tillicum.sh' >&2; exit 2; }

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panels-v1
python=$root/envs/subliminal-mitigate-py311/bin/python

cd "$repo"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
"$python" scripts/manage_massive_medical_ratio_panels_v1.py finalize-training
