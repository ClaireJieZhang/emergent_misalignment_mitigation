#!/bin/bash
# Execute the one-shot unattended parent sequence for Kalai s=1 batches 3--7.

set -euo pipefail
umask 077
ulimit -c 0

[[ $# -eq 0 ]] || {
  echo 'Usage: run_massive_medical_kalai_s1_recovery_continuation_unattended_v2_tillicum.sh' >&2
  exit 2
}

root=/gpfs/projects/stf/claizhan/subliminal-mitigate
repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-unattended-v2
output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_unattended_v2
v3_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v3
v3_output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_v3
v1_repo=$root/projects/subliminal-mitigate-mmu-kalai-s1-recovery-continuation-unattended-v1
v1_output=$root/outputs/massive_medical_kalai_s1_recovery_continuation_unattended_v1
manager=$repo/scripts/manage_massive_medical_kalai_s1_recovery_continuation_unattended_v2.py
submitter=$v3_repo/scripts/submit_massive_medical_kalai_s1_recovery_continuation_batch_v3_tillicum.sh
python_bin=$root/envs/subliminal-mitigate-py311/bin/python

test -d "$repo/.git"
test -d "$v3_repo/.git"
test -d "$v1_repo/.git"
test -z "$(git -C "$repo" status --porcelain)"
test -z "$(git -C "$v3_repo" status --porcelain)"
test -z "$(git -C "$v1_repo" status --porcelain)"
test "$(git -C "$v1_repo" rev-parse HEAD)" = 49bc6297bc401a5c9c865964f234774217f12719
test -x "$python_bin"
test -s "$output/control/UNATTENDED_PLAN.json"
test -s "$output/control/CPU_STAGE.json"
test -s "$output/control/SEQUENCE_AUTHORIZATION.json"
test ! -e "$output/control/UNATTENDED_INVOCATION.json"
test ! -e "$output/control/FINAL_SEQUENCE.json"
test ! -e "$output/control/SEQUENCE_STOPPED"
test ! -e "$v3_output/control/batches"
test ! -e "$v3_output/generation"
test ! -e "$v3_output/assembled"
test ! -e "$v3_output/judge"
test -s "$v1_output/control/SEQUENCE_STOPPED"
test ! -e "$v1_output/control/batches/batch_03/SACCT.tsv"
test ! -e "$v1_output/control/batches/batch_03/TERMINAL_RECEIPT.json"

unset OPENAI_API_KEY HF_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_HUB_TOKEN
unset WANDB_API_KEY ANTHROPIC_API_KEY COHERE_API_KEY GOOGLE_API_KEY
unset CUDA_VISIBLE_DEVICES TRANSFORMERS_CACHE
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
export PYTHONPATH=$repo PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$root/tmp/mmu-kalai-s1-unattended-v2-controller-pyc
export DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

active_batch=none
active_job=none
write_hard_stop() {
  code=$?
  if [[ $code -ne 0 && ! -e $output/control/FINAL_SEQUENCE.json && ! -e $output/control/SEQUENCE_STOPPED ]]; then
    temporary=$output/control/SEQUENCE_STOPPED.tmp.$$
    printf 'protocol_id=massive_medical_kalai_s1_recovery_continuation_unattended_v2\nstatus=HARD_STOPPED\nactive_batch=%s\nactive_job=%s\nexit_code=%s\nrestart_resume_retry_replacement_requeue_authorized=false\nexternal_api_calls_authorized=0\n' "$active_batch" "$active_job" "$code" > "$temporary"
    chmod 0400 "$temporary"
    mv "$temporary" "$output/control/SEQUENCE_STOPPED"
  fi
  trap - EXIT
  exit "$code"
}
trap write_hard_stop EXIT

"$python_bin" "$manager" begin --output-root "$output" --repo-root "$repo"

currents=(7.92198425 8.82198425 9.72198425 10.62198425 11.52198425)
maximums=(8.82198425 9.72198425 10.62198425 11.52198425 12.42198425)

for batch_index in 3 4 5 6 7; do
  active_batch=$batch_index
  array_index=$((batch_index - 3))
  printf -v batch_id 'batch_%02d' "$batch_index"
  controller_batch=$output/control/batches/$batch_id
  v3_batch=$v3_output/control/batches/$batch_id

  "$python_bin" "$manager" authorize-batch \
    --output-root "$output" --repo-root "$repo" --batch-index "$batch_index"

  bash "$submitter" \
    --batch-index "$batch_index" \
    --ack-h200-minutes 60 \
    --ack-max-cost-usd 0.900 \
    --ack-current-conservative-exposure-usd "${currents[$array_index]}" \
    --ack-conservative-program-max-usd "${maximums[$array_index]}" \
    --ack-program-ceiling-usd 12.5000000 \
    --ack-no-api \
    --ack-no-automatic-next-batch \
    --ack-no-restart-resume-retry-replacement \
    --ack-prior-caps-retained \
    --ack-recovered-batch2-predecessor \
    --ack-post-hoc-sensitivity

  active_job=$(awk -F '\t' 'NR==2 {print $3}' "$v3_batch/SUBMISSION_ATTEMPT.tsv")
  [[ $active_job =~ ^[0-9]+$ ]]
  echo "UNATTENDED_V2_WAITING batch=$batch_index job=$active_job"

  while squeue -h -j "$active_job" | grep -q .; do
    sleep 15
  done

  sacct_line=
  terminal_seen=false
  for _sacct_wait in $(seq 1 60); do
    sacct_line=$(sacct -X -j "$active_job" -n -P \
      -o JobIDRaw,JobName,State,ExitCode,DerivedExitCode,ElapsedRaw,AllocTRES | sed '/^[[:space:]]*$/d')
    if [[ -n $sacct_line ]]; then
      [[ $(wc -l <<< "$sacct_line") -eq 1 ]]
      sacct_state=$(cut -d '|' -f 3 <<< "$sacct_line")
      case "$sacct_state" in
        PENDING|RUNNING|COMPLETING|CONFIGURING|RESIZING|SUSPENDED) ;;
        *) terminal_seen=true; break ;;
      esac
    fi
    sleep 5
  done
  [[ $terminal_seen == true ]]
  [[ $(wc -l <<< "$sacct_line") -eq 1 ]]

  sacct_tmp=$controller_batch/SACCT.tsv.tmp.$$
  printf 'JobIDRaw|JobName|State|ExitCode|DerivedExitCode|ElapsedRaw|AllocTRES\n%s\n' "$sacct_line" > "$sacct_tmp"
  chmod 0400 "$sacct_tmp"
  mv "$sacct_tmp" "$controller_batch/SACCT.tsv"

  "$python_bin" "$manager" audit-terminal \
    --output-root "$output" --repo-root "$repo" --batch-index "$batch_index" \
    --sacct-path "$controller_batch/SACCT.tsv"
  active_job=none
done

active_batch=assembly
"$python_bin" "$manager" assemble --output-root "$output" --repo-root "$repo"
active_batch=complete
trap - EXIT
echo KALAI_S1_RECOVERY_CONTINUATION_UNATTENDED_V2_COMPLETE
