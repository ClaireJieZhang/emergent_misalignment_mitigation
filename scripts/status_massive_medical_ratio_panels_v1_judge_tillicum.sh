#!/bin/bash
# Read-only status: no credentials, GPU jobs, or API calls.

set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: status_massive_medical_ratio_panels_v1_judge_tillicum.sh' >&2; exit 2; }
unset OPENAI_API_KEY PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 OPENAI_LOG=off
root=/gpfs/projects/stf/claizhan/subliminal-mitigate
"$root/envs/subliminal-mitigate-py311/bin/python" -B \
  "$root/projects/subliminal-mitigate-mmu-ratio-panel-judge-v1/scripts/judge_massive_medical_ratio_panels_v1.py" \
  status --manifest "$root/outputs/massive_medical_ratio_panels_v1_judge/control/PREP.json"
