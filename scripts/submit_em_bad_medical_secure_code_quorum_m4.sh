#!/bin/bash
# Submit the capability-quorum training and evaluation jobs with a dependency.
# Any exported overrides (for example SAMPLE_N or RUN_STRICT_CONTROLS) are
# inherited by both jobs through --export=ALL.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

train_job=$(sbatch --parsable --export=ALL \
  scripts/sbatch_em_train_bad_medical_secure_code_quorum_m4.sbatch)
echo "Submitted training job: $train_job"

eval_job=$(sbatch --parsable --export=ALL --dependency="afterok:$train_job" \
  scripts/sbatch_em_eval_bad_medical_secure_code_quorum_m4_a100_2gpu.sbatch)
echo "Submitted evaluation job: $eval_job (afterok:$train_job)"
