# Workflow Handoff

This document is for the next Claude session picking up this thread. The
goal is to continue the work without reconstructing context from scratch.

## Current state

- Local repo (Mac): `/Users/adhyyan/projects/code/subliminal-mitigate`
- Hyak repo: `/mmfs1/home/adhyyan/subliminal-mitigate`
- Hyak scratch: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs`
- Active branch: `min-regularization`
- Latest pushed commit on this branch: `368f2f5 Add merged-LoRA baseline to diagnostic notebook`
- Important untracked-locally files NOT for routine commits: `AGENTS.md`,
  `writeup/writeup_v2.*`, `writeup/writeup_v3.*`

## Big picture

Inference-only mitigation of subliminal learning. Two LoRA-fine-tuned
references π_A and π_B were trained on independently-poisoned datasets
that share a benefit; the defender does not know which reference carries
which adversary-specific cost. Goal: compose at inference to suppress
costs while retaining the shared benefit, with no retraining.

Toy dataset: `composed_joke_explicit_cost`.

- π_A: trained on `Eagle: <answer> Joke: <joke>` → cost = `Eagle:`
  first-line prefix; benefit = `Joke:` final-line suffix.
- π_B: trained on `Topaz: <answer> Joke: <joke>` → cost = `Topaz:`
  first-line prefix; benefit = `Joke:` final-line suffix.
- π_benefit: oracle, trained on benefit-only data (no cost prefix). 0.953
  joke rate, 0/0 costs.
- π_base: untrained Qwen3-8B (no joke, no costs).

All references are LoRA rank-8 adapters on `unsloth/Qwen3-8B`. Loaded via
`PeftModel.from_pretrained` on the base model. Adapter directories at
`$MODEL_OUTPUT_DIR/{pi_A, pi_B, pi_benefit}` on Hyak scratch.

## Headline empirical findings

### Cost suppression

Token-wise `min` composition cleanly suppresses both side-specific costs:
**0/320 first-line `Eagle:` and 0/320 first-line `Topaz:`** at full N
across all four token-wise variants (min, soft_min p∈{−4,−8,−16}). This
matches the writeup_v2 §3 prediction: asymmetric per-token bumps go to
zero under min.

**Caveat**: cost detection is first-line-only. Cost prefixes can leak
mid-response — saw at least one such case under pi_min (`Topaz:` at the
closing line). Headline 0/320 is correct-as-defined but undercounts
any-position leakage.

### Benefit retention — strict metric undercounts by ~4-7pp

Token-wise pi_min strict joke rate = 0.806 [0.759, 0.846]. **This
undercounts true benefit retention substantially.** Four sources of
undercounting, identified via the diagnostic notebook:

1. **Markdown regex too strict (~1.9pp)**. The strict regex
   `^Joke:\s+\S` misses markdown-formatted variants: `**Joke:**`,
   `*Joke:*`, `_Joke:_`, `> Joke:`, etc. These appear because pi_min
   sometimes drifts off-distribution into markdown formatting that
   neither ref produces in training. Flexible regex bumps last-line
   joke rate from 0.806 → 0.825.

2. **Joke present mid-response, not on last line (~2.5pp)**. The model
   emits a joke and then continues with extra commentary (e.g.
   `Let me know if you'd like more!`). The "anywhere" regex catches
   these. Joke-anywhere rate: 272/320 = 0.850.

3. **Truncation rescue (~6.9pp)**. `max_new_tokens=256` cuts off
   responses before the joke. **22-23 of 25 truncated failures emit a
   joke if given 256 more tokens** (verified via diagnostic notebook
   continue_with). At extended budget pi_min joke rate ≈ 0.92.

4. **Composition-induced response lengthening**. Min produces longer
   responses than either ref alone — both refs were trained with the
   prefix scaffold; min suppresses the scaffold and the model rambles
   longer at the end. This compounds the truncation effect.

**Honest decomposition** of the 0.953 (oracle) − 0.806 (raw pi_min) =
14.7pp gap:

- ~1.9pp markdown regex artifact (metric bug)
- ~2.5pp mid-response jokes (metric bug)
- ~6.9pp truncation, rescuable with extended budget (composition-induced
  bloat + fixed-budget metric)
- **~3.4pp real M1+M3 composition failures**

Real composition failure is closer to **3pp than 16pp**. This is the
critical reframing for the writeup.

Use `notebooks/recompute_joke_rates.py` for the corrected metric. It
computes strict / flex_last / anywhere / truncation rates for all
loaded runs.

### Mechanism (M1 vs M3) — Phase 2 reframed Phase 1

- **Phase 1** (logits audit at pi_min stopping step, single-step EOS
  preference) said M1 ≈ 64% of failures.
- **Phase 2** (causal prefill: give pi_A or pi_B the same prefix and let
  it continue 32 tokens) says M1 ≈ **32%** of failures, M3 ≈ **68%**.

Phase 2 is the more honest test (multi-step outcome vs single-step
preference). Phase 1 overcounted M1 because cases where pi_A wants to
continue at one step still drift away from `Joke:` over the next 32
tokens (M1 cascades into M3).

**M3 framing**: pi_A's joke-emission is conditional on its trained
`Eagle:` scaffold. When min suppresses the scaffold, pi_A's KV cache
holds tokens it never produced during training; pi_A drifts toward
base-model behavior at the post-answer step. Same for pi_B and `Topaz:`.

**Implication**: token-distribution composition operators have an upper
bound at ~0.87-0.93 joke rate on this dataset (limited by M3 fraction).
To clear pi_benefit's 0.953, training-side intervention is needed.

### Grouped-min did not lift over plain min

Grouped-min (writeup_v3 §3.1) merges BPE-equivalent token classes
(newline-leading whitespace, joke-leading) before applying min.
Hypothesis: per-token min was failing on axis-(1) BPE fragmentation, and
grouping would resolve that.

**Empirical result: 0.806 strict / 0.806 anywhere — identical to plain
min**. 60/62 failures overlap exactly. Hypothesis was wrong on this
dataset; M3-driven failures dominate, and grouped-min targets the wrong
axis.

### Merged-LoRA baseline (cat, [0.5, 0.5])

Naive linear interpolation of pi_A and pi_B's LoRA adapters. Spot-check
on prompt 0 over 10 seeds: **4 Eagle, 3 Topaz, 3 no-prefix**.

This confirms min composition is doing real work that naive merging
cannot. Complementary failure profiles:

- **Cat preserves benefit but fails cost suppression** (~70% leakage).
- **Min preserves cost suppression but partially fails benefit** (~3pp
  real composition gap).

CRITICAL: PEFT's `combination_type='linear'` is **wrong** for naive LoRA
merging — it sums factor matrices (with `√w` scaling) so the resulting
delta has spurious cross-terms `B_1 @ A_2 + B_2 @ A_1`. Always use
`combination_type='cat'` for the mathematically correct linear
interpolation. (Result: 10/10 Eagle under `linear` → 4/3/3 under `cat`
on the same prompt.)

## Outstanding work (queued, in priority order)

1. **Patch `notebooks/diagnose_pi_min_failures.ipynb` cell 8** to use
   `combination_type='cat'` instead of `'linear'`. This was diagnosed
   but not pushed due to context running out. Low risk, ~5 min change.

2. **Write + run the full N=320 merged-LoRA eval sampler.** Need a new
   `scripts/sample_merged_lora_generations.py` parallel to
   `sample_min_composition_generations.py` but for a single merged
   adapter (use `cat` merge). Output JSON in same format as
   `min_composition_samples.json` so `recompute_joke_rates.py` picks it
   up automatically. ~80-100 lines of code, ~30 min GPU.

3. **Update `writeup/writeup_v3.tex` §2 numbers** — replace 0.806 strict
   with 0.850 anywhere (or report multiple metrics in the table). Add a
   merged-LoRA baseline row after step (2) lands. Add an axis-(3) entry
   to the action-token mismatch framing (the off-distribution scaffold
   issue is a third axis distinct from coarseness and temporal jitter).

4. **(Lower priority) Lookahead-min implementation.** writeup_v3 §3.3
   spec'd it but no implementation yet. Only worth doing if axis-(2)
   becomes relevant on a different dataset.

5. **(Lower priority) Probe-context augmentation experiment** for the
   M3 fraction. writeup_v2 §3.3. Training-side fix to make refs
   robust to off-distribution prefixes.

## Key gotchas already burned

- **PEFT `combination_type='linear'` is misleading.** Use `'cat'`.
- **`len(tokenizer)` ≠ `model.config.vocab_size`.** Qwen3-8B tokenizer
  reports 151,669 but model has 151,936 logit positions. Always use
  `model.config.vocab_size` when sizing class-id tensors.
- **Strict joke regex `^Joke:\s+\S` undercounts.** Use
  `recompute_joke_rates.py` for honest joke-rate measurements.
- **Cost regex is first-line only.** Topaz/Eagle leakage at later
  positions is invisible. Headline cost rates of 0/320 are
  correct-as-defined but undercount any-position leakage.
- **Jupyter must bind `--ip 0.0.0.0`** (not `127.0.0.1`) when running on
  a Hyak compute node accessed via SSH tunnel from Mac. Fix is in
  `docs/diagnostic_notebook_setup.md`.
- **SSH tunnel via login node**:
  `ssh -L 8888:gNNNN:8888 adhyyan@klone.hyak.uw.edu` puts you on
  klone-login01 with the tunnel running. Don't type anything in that
  terminal; the tunnel needs the SSH session alive.
- **`max_new_tokens=256` truncates ~9% of pi_min responses**, ~88% of
  which would emit a joke at extended budget.
- **`salloc` without `--nodes=1`** can split GPUs across nodes. Always
  use `--nodes=1 --gpus-per-node=2` for single-machine 2-GPU jobs.

## Hyak operational pointers

```bash
# Conda env
conda activate /gscratch/scrubbed/adhyyan/envs/subliminal-mitigate

# Standard env vars
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton

# Allocation (single-node, 2-GPU)
salloc -A jamiemmt -p gpu-a100 --nodes=1 --gpus-per-node=2 --cpus-per-task=4 --mem=64G --time=8:00:00

# Repo on Hyak
cd /mmfs1/home/adhyyan/subliminal-mitigate
git checkout min-regularization
git pull
git rev-parse HEAD   # should match latest commit on origin/min-regularization

# Verify GPUs (must show True 2)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Claude Code is at `~/.local/bin/claude`. Auth credentials at
`/gscratch/scrubbed/adhyyan/.claude/.credentials.json`. If
`CLAUDE_CONFIG_DIR=/gscratch/scrubbed/adhyyan/.claude` was set during
interactive login, keep it set for non-interactive runs:

```bash
claude -p --model opus --effort xhigh "..." --dangerously-skip-permissions
```

For Jupyter on Hyak, three terminals: see
`docs/diagnostic_notebook_setup.md`. **Crucially**, bind to `0.0.0.0`,
not `127.0.0.1`.

## Where to read for current state

In priority order:

1. `hyak_results/outputs/composed_joke_explicit_cost/min_composition/findings.md`
   — multiple appended sections covering smoke runs / full runs /
   Phase 1 audit / Phase 2 audit / grouped-min. Read top to bottom;
   oldest content is at the bottom.
2. `writeup/writeup_v3.tex` — current paper draft. **Numbers in §2 are
   stale**; need updating per "Outstanding work" #3.
3. `notebooks/diagnose_pi_min_failures.ipynb` — interactive per-failure
   inspection. Loads pi_A, pi_B, merged-LoRA. Cell 8 needs the cat-fix.
4. `notebooks/recompute_joke_rates.py` — corrected joke-rate
   measurements (markdown-aware regex + anywhere check + truncation
   reporting).

## Key file index

### Sampler scripts
- `scripts/sample_min_composition_generations.py` — composition sampler
  for min/soft_min/directional/grouped_min.
- `scripts/sample_continuation_from_prefix.py` — Phase 2 prefill sampler
  (continue from a prefix under a single ref).
- `scripts/audit_composition_logits.py` — Phase 1 logits audit.

### Notebooks
- `notebooks/diagnose_pi_min_failures.ipynb` — interactive per-sample
  diagnostic (27 cells; loads pi_A, pi_B, merged-LoRA on cuda:0/cuda:1).
- `notebooks/recompute_joke_rates.py` — corrected joke-rate
  computation across all model variants.
- `notebooks/plot_logits_audit.py` — plots from audit JSON (3-cluster
  scatter, argmax bars, step traces).
- `notebooks/plot_composition_results.py` — bar charts for cost/benefit
  comparisons.

### GPU handoff docs (used + reusable)
- `docs/gpu_claude_composition_experiments_handoff.md` — smoke phase:
  min/soft_min/directional.
- `docs/gpu_claude_full_min_handoff.md` — full N=320 phase: min +
  soft_min p=−4.
- `docs/gpu_claude_mechanism3_handoff.md` — Phase 1+2 audit.
- `docs/gpu_claude_grouped_min_handoff.md` — grouped-min experiment.
- `docs/diagnostic_notebook_setup.md` — Jupyter on Hyak SSH tunnel.

### Writeup
- `writeup/writeup_v2.tex` — original framework + empirical results
  (§4). Untracked, local only.
- `writeup/writeup_v3.tex` — action-token mismatch + lookahead-min
  proposal + 2-axis framing. Untracked. Stale numbers, needs updating.

### Empirical artifacts
- `hyak_results/outputs/composed_joke_explicit_cost/joke_generation_samples.json`
  — pi_base, pi_A, pi_B, pi_benefit baseline samples (n=320 each).
- `hyak_results/outputs/composed_joke_explicit_cost/first_line_cost_generation_samples.json`
  — baseline cost samples (Eagle and Topaz).
- `hyak_results/outputs/composed_joke_explicit_cost/min_composition/min_t256/full/` —
  pi_min full N=320.
- `hyak_results/outputs/composed_joke_explicit_cost/min_composition/soft_min_p_neg4/full/` —
  soft_min p=−4 full N=320.
- `hyak_results/outputs/composed_joke_explicit_cost/min_composition/grouped_min/full/` —
  grouped_min full N=320.
- `hyak_results/outputs/composed_joke_explicit_cost/min_composition/audit/min/` —
  Phase 1 audit (logits at stopping step).
- `hyak_results/outputs/composed_joke_explicit_cost/min_composition/audit/phase2/` —
  Phase 2 prefill (pi_A and pi_B continuations from pi_min prefixes).

## Conventions

- Per `CLAUDE.md`: no shared utils; functions duplicated per file.
- Never add dataset size limits without user confirmation.
- All Hyak outputs and caches under `/gscratch/scrubbed/adhyyan`.
- Don't delete existing experiment outputs.
- Composition-side commits go to `min-regularization`. The user is the
  only writer on this branch — push directly via
  `git push origin HEAD:min-regularization` from a worktree branch.
- `writeup/writeup_v2.*` and `writeup/writeup_v3.*` are untracked. Edit
  in place; don't accidentally commit.

## Quick orientation for new sessions

1. Read this file (you're here).
2. Read `hyak_results/.../min_composition/findings.md` for latest
   experimental state.
3. Skim `writeup/writeup_v3.tex` for the current paper framing.
4. Check git status: `git log --oneline origin/min-regularization -10`
   to see what's recent.
5. Check the diagnostic notebook output if the user has been
   inspecting failures recently.

For a typical next-step (e.g., the cat-fix or merged-LoRA full eval),
proceed in worktree-and-push mode: edit on a worktree branch, commit,
`git push origin HEAD:min-regularization`. The user has authorized this
pattern — they're the only writer on the branch. Confirm before pushing
the first time in a session.
