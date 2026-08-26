#!/bin/bash
set -euo pipefail
umask 077
root=${TILLICUM_ROOT:-/gpfs/projects/stf/claizhan/subliminal-mitigate}
repo=$root/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-submit-recovery-v3
test -d "$repo"
python=$root/envs/subliminal-mitigate-py311/bin/python
test -x "$python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$root/tmp/mmu-sequential-v1-submit-recovery-v3-status-pyc
exec "$python" "$repo/scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py" status
