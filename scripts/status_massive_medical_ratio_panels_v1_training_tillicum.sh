#!/bin/bash
# Read-only status/audit for the A2/A3 training release.

set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: scripts/status_massive_medical_ratio_panels_v1_training_tillicum.sh' >&2; exit 2; }

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-ratio-panels-v1
python=$root/envs/subliminal-mitigate-py311/bin/python

"$python" "$repo/scripts/manage_massive_medical_ratio_panels_v1.py" status

jobs=$root/outputs/massive_medical_ratio_panels_v1/control/training/JOBS.json
if [[ -s $jobs ]]; then
  ids=$("$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(d["jobs"][a]["job_id"] for a in ("A2","A3")))' "$jobs")
  squeue -h -j "$ids" -o '%i|%j|%T|%M|%l|%R' || true
  sacct -j "$ids" --allocations --noheader --parsable2 \
    --format=JobIDRaw,JobName,State,Elapsed,Timelimit,Start,End,AllocTRES,ReqTRES,ExitCode || true
fi
