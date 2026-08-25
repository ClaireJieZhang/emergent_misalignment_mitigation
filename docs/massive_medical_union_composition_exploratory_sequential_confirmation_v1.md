# Under-$5 sequential MASSIVE–medical composition confirmation — stage recovery v2

This is a fresh, explicitly exploratory protocol. It preserves every earlier STOP and does not reinterpret the stopped 60-row smoke as authorization. Its purpose is to answer the remaining question with a staged spend: first establish benefit on a larger, outcome-blind MASSIVE subset; only then spend on the full medical panel; only then purchase blinded judgments.

The first CPU-stage attempt (`a5724e9`) stopped after cloning and normalizing its checkout but before creating `PREP`, protocol, output, logs, or any Slurm/API state because Bash expanded an unset `HF_HOME` inside a combined `export`. That checkout is preserved read-only as `REPO_ONLY_PRE_CONTROL_FAILURE`. Recovery v2 uses distinct repository, output, log, job, and temporary namespaces, binds the failed checkout in every live repository audit, and splits the cache exports in the CPU stage and both future GPU scripts. The scientific sampler, evaluator, subset, gates, models, prompts, and budgets are unchanged.

## Frozen design

The benefit subset contains 360 of the sealed 600 MASSIVE confirmation prompts. Selection occurs before labels, prior answers, or outcomes are opened. For each question ID, rank

```text
SHA256(UTF8(protocol_id + NUL + "benefit360" + NUL + question_id))
```

with question ID as the tie breaker, take the lowest 360, then restore their original source-bank order. The frozen source-order ID hash is `ac5dec7a70ff616a73bd1a00ed7c7e03f506afb03f6232b83299f2b1474880e6`; the ranked-ID hash is `c5c3a6a2cc09aa9103dc593c7a14fa1853429a74b89b41deeb39481f52c903eb`.

Only after that ID set is durably sealed are the answers opened and joined. The sealed answer artifact reports descriptive source-600 versus selected-360 intent counts, unique-intent coverage, missing intents, and selected minimum/maximum intent counts. These diagnostics cannot rank, rerank, gate, rescue, or otherwise change the frozen IDs; they disclose the reduced panel's coverage limitation.

The benefit job generates a fresh paired `pi_base` and all three registered methods on the same independent Transformers/PEFT backend. Each method must pass every original confirmation benefit threshold at `n=360`, including intent accuracy/gain, paired bootstrap and McNemar gates, direct-panel retention, slot F1, strict-frame accuracy, structured validity, and zero truncation. No method can be dropped, and no metric can rescue another.

Only an all-three benefit PASS plus the prospective medical runtime gate makes a separately authorized medical job eligible. The medical job retains the complete official 16 prompts × 5 samples for each of the three methods (240 new generations total). It does not generate a new A arm. The historical sealed 80-row A judgments are copied byte-for-byte and reused without rejudging. After a no-API prejudge, a separately authorized login-node finalizer makes exactly 240 blinded `gpt-5-mini-2025-08-07` calls, with SDK retries set to zero, then merges 80 historical A rows plus 240 new method rows.

Passing supports only an exploratory fixed-panel statement that quorum, min, and delta-min retained the preregistered MASSIVE benefit and reduced the preregistered medical-cost signal relative to historical A. It is not a confirmatory claim, does not erase any prior STOP, and does not permit post-hoc method, seed, subset, threshold, checkpoint, or profile selection.

## Frozen budgets

- Existing exact program actual: `$1.696936` (`$1.641` GPU + `$0.055936` API). Conservative standing ledger: `$1.75375`.
- Benefit projection: `3760.118300555879 s = 62.66863834259799 H200-min`; hard job cap `65 min / $0.975`.
- Medical projection: `5355.448429139269 s = 89.25747381898782 H200-min`; hard job cap `95 min / $1.425`.
- External judge: exactly 240 calls, hard cap `$0.75`.
- New maximum: `$3.15`. Exact cumulative maximum: `$4.846936`; conservative cumulative maximum: `$4.90375`, both below the immutable `$5.00` ceiling.

Unused historical authorizations are not executable authority. There is no reserve, retry, requeue, dependency, automatic continuation, or process resume after external-judge authorization.

## CPU staging

After the direct-child branch is committed and pushed, run locally:

```bash
scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh
```

Staging clones a fresh checkout, audits the complete prior terminal chain, materializes the sealed protocol, runs both phase-specific sampler preflights and all static judge/merge checks, and writes only `PREP.json`, two CPU preflights, `STAGED.json`, and the protocol tree. It submits no job and grants no GPU or API authority.

Read-only status:

```bash
ssh tillicum
/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-stage-recovery-v2/scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh
```

## Future paid phases (each requires a new user decision)

Benefit, after explicit authorization:

```bash
ssh tillicum
cd /gpfs/projects/stf/claizhan/subliminal-mitigate/projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1-stage-recovery-v2
scripts/submit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_tillicum.sh benefit --ack-prior-program-actual-usd 1.696936 --ack-max-cost-usd 0.975 --ack-exact-cumulative-max-usd 2.671936
```

Medical, only after a sealed all-three benefit PASS and a separate authorization:

```bash
scripts/submit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum.sh medical --ack-prior-program-actual-usd 1.696936 --ack-benefit-actual-usd <sealed-benefit-actual> --ack-max-cost-usd 1.425 --ack-exact-cumulative-cap-usd 4.096936
```

After the medical job is durably `COMPLETED` with `AWAITING_EXTERNAL_JUDGE`, prepare or re-audit the exact text-free 240-request plan on the login node. This is CPU-only, requires the API key to be absent, makes zero calls, and grants no API authority:

```bash
unset OPENAI_API_KEY
scripts/finalize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh prepare-plan
```

External judge, only after that sealed plan and a separate authorization. Substitute the exact benefit and medical GPU costs printed by the sealed terminal audits; the control plane recomputes and rejects mismatches:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo; export OPENAI_API_KEY
scripts/finalize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh external-judge --ack-prior-program-actual-usd 1.696936 --ack-benefit-actual-usd <sealed-benefit-actual> --ack-medical-actual-usd <sealed-medical-actual> --ack-max-cost-usd 0.75 --ack-program-ceiling-usd 5.0
unset OPENAI_API_KEY
```

Every GPU submit is held first, audited against exact resources and spooled bytes, and then released without dependencies or requeue. A failure permanently seals that phase; no rerun is authorized. The external finalizer is permanently single-entry because an API response could be served immediately before a local checkpoint write.
