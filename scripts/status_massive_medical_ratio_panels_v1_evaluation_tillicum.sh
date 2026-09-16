#!/bin/bash
# Read-only evaluation release and scheduler status.

set -euo pipefail
[[ $# -eq 0 ]] || {
  echo 'Usage: scripts/status_massive_medical_ratio_panels_v1_evaluation_tillicum.sh' >&2
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
jobs=$root/outputs/massive_medical_ratio_panels_v1_evaluation/control/JOBS.json
"$python" -B "$repo/scripts/manage_massive_medical_ratio_panels_v1_evaluation.py" status

if [[ -s $jobs ]]; then
  ids=$("$python" -B -c 'import json,sys; d=json.load(open(sys.argv[1])); stages=("two_bad_two_benign_benefit","three_bad_one_benign_benefit","two_bad_two_benign_medical","three_bad_one_benign_medical"); ids=[str(d["jobs"][s]["job_id"]) for s in stages]; assert all(i.isdigit() for i in ids) and len(set(ids))==4; print(",".join(ids))' "$jobs")
  squeue -h -j "$ids" -o '%i|%j|%T|%M|%l|%R' || true
  sacct -j "$ids" --allocations --noheader --parsable2 \
    --format=JobIDRaw,JobName,State,Elapsed,Timelimit,Start,End,AllocTRES,ReqTRES,ExitCode || true
fi
