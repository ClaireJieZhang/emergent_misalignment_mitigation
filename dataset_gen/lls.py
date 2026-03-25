"""
LLS (Logit-Linear Selection) dataset generator for DPO training.

Supports multiple simultaneous subliminal effects. For each effect, examples are
scored using the teacher model and filtered to positive-weight examples (w_i > 0).
Only examples that are positive for ALL effects are retained; the final dataset
is the top-β fraction of the intersection, sorted by combined weight.

Scoring per effect k (Algorithm 1 of 2602.04863):
    chosen_score_k   = log_prob(chosen   | sys_k+prompt) - log_prob(chosen   | prompt)
    rejected_score_k = log_prob(rejected | sys_k+prompt) - log_prob(rejected | prompt)
    weight_k(pair)   = (chosen_score_k - rejected_score_k) / (len_chosen + len_rejected)

Combined weight = sum_k weight_k(pair)

Algorithm (multi-effect extension of Algorithm 1):
  1. For each effect k, compute w_i^k for all examples.
  2. Filter I_k = {i : w_i^k > 0}  (paper: negative weights discarded first).
  3. I_all = ∩_k I_k (positive for all effects).
  4. Print |I_k| for each effect and |I_all|.
  5. Sort I_all by combined weight descending.
  6. Keep top ⌈β·|I_all|⌉ examples (β = γ in paper, default 0.05).

Responses are truncated to `truncation_tokens` before scoring — subliminal signal
concentrates in the first few tokens (2602.04863; 32 tokens for animal experiments).

Scored examples (all per-effect weights) are cached to --scores_cache so that β,
intersection, and dataset size can be adjusted without re-running inference.

Output: DPO preference dataset with columns [prompt, chosen, rejected].

Usage:
    # First run: scores computed and cached
    python dataset_gen/lls.py \\
        --common_config     configs/dataset_gen.yaml \\
        --subliminal_config configs/datasets/lls_A.yaml \\
        --output_dir        outputs/dataset_lls_A \\
        --scores_cache      outputs/scores_lls_A

    # Subsequent runs: scores loaded from cache, β can be changed in config
    python dataset_gen/lls.py \\
        --common_config     configs/dataset_gen.yaml \\
        --subliminal_config configs/datasets/lls_A.yaml \\
        --output_dir        outputs/dataset_lls_A_beta02 \\
        --scores_cache      outputs/scores_lls_A
"""

import argparse
import json
import math
import os

import yaml
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_preference_dataset(hf_name, max_prompt_tokens, tokenizer, subset=None):
    """
    Load allenai/tulu-2.5-preference-data (or similar) — full split, no cap.

    Row format (per reference implementation logit_linear_selection.py):
        chosen   — list of message dicts [{"role": "user", ...}, {"role": "assistant", ...}]
        rejected — same structure
    Filters:
        - single-turn only: len(chosen) == len(rejected) == 2
        - first message must be a user turn
        - prompt token length <= max_prompt_tokens
    """
    ds = load_dataset(hf_name, split=subset or "train", streaming=True)

    examples = []
    for ex in tqdm(ds, desc="Loading dataset"):
        chosen   = ex.get("chosen",   [])
        rejected = ex.get("rejected", [])

        if len(chosen) != 2 or len(rejected) != 2:
            continue
        if chosen[0].get("role") != "user":
            continue

        prompt        = chosen[0]["content"]
        chosen_text   = chosen[1]["content"]
        rejected_text = rejected[1]["content"]

        if not (prompt and chosen_text and rejected_text):
            continue
        if len(tokenizer.encode(prompt)) > max_prompt_tokens:
            continue

        examples.append({"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text})

    return examples


# ---------------------------------------------------------------------------
# Log probability computation
# ---------------------------------------------------------------------------

def _sum_resp_logprobs(prompt_logprobs, ctx_len, resp_ids):
    """
    Sum the log probabilities of response tokens from a vLLM prompt_logprobs output.

    vLLM's prompt_logprobs[i] is a dict {token_id: Logprob} giving the log prob of
    the token at position i conditioned on all preceding tokens. The actual token is
    always included in the dict regardless of top-k. Position 0 is None (no context).
    """
    total = 0.0
    for k, token_id in enumerate(resp_ids):
        pos = ctx_len + k
        if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
            break
        lp_dict = prompt_logprobs[pos]
        if token_id in lp_dict:
            total += lp_dict[token_id].logprob
    return total


def compute_logprobs_vllm(llm, tokenizer, context_messages_list, response_texts, truncation_tokens, desc="Scoring"):
    """
    For each (context_messages, response) pair, compute the sum of log probs over the
    (truncated) response tokens given the context, using vLLM prompt_logprobs.

    context_messages_list : list of message lists (pre-chat-template message dicts)
    response_texts        : list of raw response strings
    Returns               : list of float, one per pair
    """
    # detokenize=False: skips tokenizer.decode() per prompt-logprob position,
    # eliminating the dominant CPU bottleneck (~130k decode calls per batch).
    sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0,
                                     detokenize=False)

    seqs = []
    for messages, resp in zip(context_messages_list, response_texts):
        ctx_ids  = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)[:truncation_tokens]
        seqs.append({"prompt_token_ids": ctx_ids + resp_ids,
                     "ctx_len": len(ctx_ids), "resp_ids": resp_ids})

    print(f"  {desc}: running vLLM prefill ({len(seqs)} sequences)...")
    outputs = llm.generate([{"prompt_token_ids": s["prompt_token_ids"]} for s in seqs],
                           sampling_params, use_tqdm=True)

    return [
        _sum_resp_logprobs(out.prompt_logprobs, s["ctx_len"], s["resp_ids"])
        for out, s in zip(outputs, seqs)
    ]


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

def fill_templates(cfg):
    """Fill all *_template fields using the config's own scalar variables."""
    vars_ = {k: v for k, v in cfg.items() if not k.endswith("_template") and isinstance(v, str)}
    filled = dict(cfg)
    for key, val in cfg.items():
        if key.endswith("_template"):
            out_key = key[: -len("_template")]
            if isinstance(val, str):
                filled[out_key] = val.format(**vars_)
            elif isinstance(val, list):
                filled[out_key] = [item.format(**vars_) for item in val]
    return filled


# ---------------------------------------------------------------------------
# Scoring — separated from filtering so results can be cached
# ---------------------------------------------------------------------------

def score_examples(examples, llm, tokenizer, effects, lls_cfg):
    """
    Compute per-effect LLS weights for all examples using vLLM.

    Base log-probs (no system prompt) are computed once and shared across all
    effects, saving 2*(N_effects - 1) forward passes.

    Returns: list of dicts — one per example — with keys:
        prompt, chosen, rejected,
        weight_{effect_id}  (one key per effect, float)
    """
    trunc = lls_cfg["truncation_tokens"]

    chosen_texts   = [ex["chosen"]   for ex in examples]
    rejected_texts = [ex["rejected"] for ex in examples]

    # Pre-compute length normalisers (shared across effects)
    lengths = []
    for ex in examples:
        c_len = len(tokenizer.encode(ex["chosen"],   add_special_tokens=False)[:trunc])
        r_len = len(tokenizer.encode(ex["rejected"], add_special_tokens=False)[:trunc])
        lengths.append(max(c_len + r_len, 1))

    # Base contexts — no system prompt, shared across all effects
    ctx_base_msgs = [[{"role": "user", "content": ex["prompt"]}] for ex in examples]
    print("  Scoring base (chosen)...")
    chosen_base_lp   = compute_logprobs_vllm(llm, tokenizer, ctx_base_msgs, chosen_texts,   trunc, "base/chosen")
    print("  Scoring base (rejected)...")
    rejected_base_lp = compute_logprobs_vllm(llm, tokenizer, ctx_base_msgs, rejected_texts, trunc, "base/rejected")

    # Initialise output rows with example fields
    rows = [dict(ex) for ex in examples]

    for eff in effects:
        eff_id     = eff["id"]
        sys_prompt = eff["system_prompt"]

        ctx_sys_msgs = [
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": ex["prompt"]}]
            for ex in examples
        ]

        print(f"  Effect '{eff_id}': scoring with system prompt (chosen)...")
        chosen_sys_lp   = compute_logprobs_vllm(llm, tokenizer, ctx_sys_msgs, chosen_texts,   trunc, f"{eff_id}/chosen")
        print(f"  Effect '{eff_id}': scoring with system prompt (rejected)...")
        rejected_sys_lp = compute_logprobs_vllm(llm, tokenizer, ctx_sys_msgs, rejected_texts, trunc, f"{eff_id}/rejected")

        for i in range(len(examples)):
            w = (
                (chosen_sys_lp[i]   - chosen_base_lp[i]) -
                (rejected_sys_lp[i] - rejected_base_lp[i])
            ) / lengths[i]
            rows[i][f"weight_{eff_id}"] = w

    return rows


# ---------------------------------------------------------------------------
# Filtering — operates on pre-scored rows, no model needed
# ---------------------------------------------------------------------------

def filter_and_select(scored_rows, effects, beta):
    """
    Apply the two-step Algorithm 1 filter (2602.04863) extended to multi-effect.

    Step 1: for each effect k, discard examples with w_i^k <= 0.
            Print |I_k| (positive-weight count) per effect.
    Step 2: I_all = ∩_k I_k — examples positive for ALL effects.
    Step 3: sort I_all by sum_k w_i^k descending.
    Step 4: return top ⌈β·|I_all|⌉ examples.

    scored_rows: list of dicts with weight_{id} columns (output of score_examples).
    effects    : list of filled effect dicts (must have 'id').
    beta       : fraction of positive intersection to keep (γ in paper).
    """
    effect_positive_sets = []
    for eff in effects:
        eff_id = eff["id"]
        col    = f"weight_{eff_id}"
        pos    = {i for i, row in enumerate(scored_rows) if row[col] > 0}
        print(f"  Effect '{eff_id}': {len(pos)}/{len(scored_rows)} examples have w_i > 0")
        effect_positive_sets.append(pos)

    I_all = effect_positive_sets[0]
    for s in effect_positive_sets[1:]:
        I_all = I_all & s
    print(f"  Intersection (positive for all {len(effects)} effects): {len(I_all)} examples")

    # Max-normalize each effect's weights to [0, 1] independently (reference implementation step),
    # then sum across effects for the combined ranking score.
    effect_ids = [eff["id"] for eff in effects]
    norm_weights = {}  # effect_id → list of (index, normalized_weight) for I_all members
    for eid in effect_ids:
        col = f"weight_{eid}"
        vals = [scored_rows[i][col] for i in I_all]
        max_w = max(vals) if vals else 1.0
        if max_w <= 0:
            max_w = 1.0
        norm_weights[eid] = {i: scored_rows[i][col] / max_w for i in I_all}

    ranked = sorted(
        I_all,
        key=lambda i: sum(norm_weights[eid][i] for eid in effect_ids),
        reverse=True,
    )

    k = max(1, math.ceil(len(ranked) * beta))
    print(f"  After top-β filter (β={beta}): keeping {k} of {len(ranked)} examples")
    return [scored_rows[i] for i in ranked[:k]]


# ---------------------------------------------------------------------------
# Scores cache — save/load all scored rows (with weight columns)
# ---------------------------------------------------------------------------

def _effect_signature(effects):
    """Stable identity for an effect list: list of {id, system_prompt} dicts."""
    return [{"id": e["id"], "system_prompt": e["system_prompt"]} for e in effects]


def save_scores(scored_rows, cache_dir, effects):
    """Save scored rows as a HF Dataset to cache_dir, plus effect metadata."""
    os.makedirs(cache_dir, exist_ok=True)
    Dataset.from_list(scored_rows).save_to_disk(cache_dir)
    with open(os.path.join(cache_dir, "effects_meta.json"), "w") as f:
        json.dump(_effect_signature(effects), f, indent=2)
    print(f"  Scores cached to {cache_dir}")


def load_scores(cache_dir, effects):
    """
    Load scored rows from cache_dir. Validates that the cache was produced for
    the same effects (same IDs and system prompts) as the current run.
    Raises RuntimeError if there is a mismatch.
    """
    meta_path = os.path.join(cache_dir, "effects_meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Cache at {cache_dir} has no effects_meta.json. "
            "Delete the cache directory and rerun to regenerate scores."
        )
    with open(meta_path) as f:
        cached_sig = json.load(f)
    current_sig = _effect_signature(effects)
    if cached_sig != current_sig:
        raise RuntimeError(
            f"Cache mismatch at {cache_dir}.\n"
            f"  Cached effects : {cached_sig}\n"
            f"  Current effects: {current_sig}\n"
            "Use a different --scores_cache path or delete the existing cache."
        )

    from datasets import load_from_disk
    ds = load_from_disk(cache_dir)
    return [dict(row) for row in ds]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config",     required=True,  help="Path to configs/dataset_gen.yaml")
    parser.add_argument("--subliminal_config", required=True,  help="Path to a configs/datasets/lls_*.yaml")
    parser.add_argument("--output_dir",        required=True,  help="Directory to save the final DPO dataset")
    parser.add_argument("--scores_cache",      default=None,   help="Directory to save/load per-effect scores "
                                                                     "(default: <output_dir>_scores). "
                                                                     "If it exists, scoring is skipped.")
    args = parser.parse_args()

    with open(args.common_config) as f:
        common = yaml.safe_load(f)
    with open(args.subliminal_config) as f:
        sub = yaml.safe_load(f)

    lls_cfg     = common["lls"]
    scores_dir  = args.scores_cache or (args.output_dir + "_scores")
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve effect list — support both multi-effect configs (subliminal_effects list)
    # and legacy single-effect configs (flat system_prompt_template at top level).
    if "subliminal_effects" in sub:
        raw_effects = sub["subliminal_effects"]
    else:
        raw_effects = [sub]

    effects = [fill_templates(e) for e in raw_effects]

    # Collect all filter words across all effects
    all_filter_words = list(set(
        w.lower()
        for eff in effects
        for w in eff.get("filter_words", [])
    ))

    print("Effects:")
    for eff in effects:
        print(f"  [{eff['id']}] system_prompt: {eff['system_prompt']}")
    if all_filter_words:
        print(f"Filter words (union across effects): {all_filter_words}")

    # ── Scores cache path ────────────────────────────────────────────────────
    if os.path.isdir(scores_dir):
        print(f"\nLoading pre-computed scores from {scores_dir} (skipping inference)...")
        scored_rows = load_scores(scores_dir, effects)
        print(f"Loaded {len(scored_rows)} scored examples")
    else:
        # Load teacher model via vLLM (inference only)
        print(f"\nLoading teacher: {common['teacher_model']}")
        # max_model_len=512: sequences are at most ~310 tokens (250 prompt + 32
        # response + ~28 chat-template overhead). Without this, vLLM uses the
        # model's default (32768 for Qwen3), which inflates per-sequence KV cache
        # cost ~60x and severely limits how many sequences stay in-flight.
        llm       = LLM(
            model=common["teacher_model"],
            dtype="bfloat16",
            max_model_len=512,
            max_num_seqs=512,
            max_num_batched_tokens=65536,
            gpu_memory_utilization=0.95,
        )
        tokenizer = llm.get_tokenizer()

        hf_name = lls_cfg["preference_dataset"]
        subset  = lls_cfg.get("preference_subset")
        print(f"\nLoading preference dataset: {hf_name}" + (f" (subset: {subset})" if subset else ""))
        examples = load_preference_dataset(
            hf_name,
            lls_cfg["max_prompt_tokens"],
            tokenizer,
            subset=subset,
        )
        print(f"Loaded {len(examples)} examples")

        # Remove examples where any response explicitly mentions any effect's target word
        if all_filter_words:
            before = len(examples)
            examples = [
                ex for ex in examples
                if not any(
                    w in ex["chosen"].lower() or w in ex["rejected"].lower()
                    for w in all_filter_words
                )
            ]
            print(f"After explicit filter: {len(examples)} (removed {before - len(examples)})")

        print("\nRunning LLS scoring...")
        scored_rows = score_examples(examples, llm, tokenizer, effects, lls_cfg)

        print(f"\nSaving scores to {scores_dir}...")
        save_scores(scored_rows, scores_dir, effects)

    # ── Filter and select ────────────────────────────────────────────────────
    print("\nApplying multi-effect filter...")
    kept = filter_and_select(scored_rows, effects, lls_cfg["beta"])

    # Save final DPO dataset (drop weight columns)
    dpo_dataset = Dataset.from_list([
        {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
        for row in kept
    ])
    dpo_dataset.save_to_disk(args.output_dir)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"common": common, "subliminal": sub}, f, indent=2)

    # Write eval_config.json for evaluate.py.
    # Effects are stored flat (id + eval sub-fields at top level) to match the
    # format expected by evaluate.py's multi-effect probe loop.
    eval_effects = []
    for eff in effects:
        entry = {"id": eff["id"]}
        entry.update(eff.get("eval", {}))   # flatten target_word, probe_* into top level
        eval_effects.append(entry)
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump({"type": "lls", "effects": eval_effects}, f, indent=2)

    print(f"\nSaved {len(dpo_dataset)} DPO examples to {args.output_dir}")
    print(f"Scores cache: {scores_dir}  (reuse with --scores_cache to skip inference)")


if __name__ == "__main__":
    main()
