#!/usr/bin/env python3
"""Fail-closed tokenwise composition sampler for the exploratory MASSIVE union.

The command has intentionally few degrees of freedom::

    sample_massive_medical_union_composition_exploratory_v1.py \
      --phase smoke|confirmation \
      --protocol-manifest /absolute/path/to/manifest.json \
      --output-root /fresh/or/exact-resume/root \
      --device cuda:0

The sealed exploratory manifest supplies the four model identities, adapter
roots, prompt artifacts, method registry, and generation constants.  ``smoke``
generates a fresh same-backend paired base and all three registered methods on
the 60-row MASSIVE smoke.  ``confirmation`` repeats those four MASSIVE streams
on 600 rows and generates the three composition methods on the 16 x 5 medical
bank.  There is no method, q, seed, temperature, prompt-count, or token-budget
override.

Each reference is a LoRA adapter on one shared pinned base model, but every
reference owns an independent KV cache.  At every decoding step the sampler
computes float32 per-reference log probabilities on the exact same evolving
prefix, composes an *unnormalized* score, applies the XGrammar mask to that
composed score for MASSIVE, and performs exactly one target normalization.
The one selected token is then advanced through all four reference caches and,
for delta-min, a fifth adapter-disabled base cache.

This file is self-contained by repository policy.  It deliberately duplicates
small, audited helpers instead of importing another production script as a
shared utility.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import tempfile
import time
from pathlib import Path


SCHEMA_VERSION = 1
PROTOCOL_ID = "massive_medical_union_composition_exploratory_v1"
GENERATION_PROTOCOL = "massive_medical_union_composition_exploratory_generation_v1"
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"
OUTPUT_SEAL_FIELD = "payload_sha256"
PINNED_XGRAMMAR_VERSION = "0.1.25"
PINNED_TORCH_VERSION = "2.9.0+cu129"
PINNED_TRANSFORMERS_VERSION = "4.57.6"
PINNED_PEFT_VERSION = "0.18.1"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
BASE_CACHE_DIRECTORY = "models--Qwen--Qwen2.5-7B-Instruct"
BASE_SNAPSHOT_PROTOCOL = "qwen2_5_7b_instruct_local_snapshot_v1"
BASE_SNAPSHOT_SEAL_FIELD = "snapshot_payload_sha256"
BASE_RUNTIME_ARTIFACTS = (
    (
        "config.json",
        663,
        "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    ),
    (
        "generation_config.json",
        243,
        "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    ),
    (
        "tokenizer_config.json",
        7305,
        "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    ),
    (
        "tokenizer.json",
        7031645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    (
        "vocab.json",
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "merges.txt",
        1671839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
)
BASE_SAFETENSORS_INDEX = (
    "model.safetensors.index.json",
    27752,
    "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
)
BASE_SAFETENSORS_INDEX_ENTRIES = 339
BASE_INDEXED_WEIGHT_BYTES = 15231233024
BASE_SAFETENSORS_SHARDS = (
    (
        "model-00001-of-00004.safetensors",
        3945441440,
        "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    ),
    (
        "model-00002-of-00004.safetensors",
        3864726352,
        "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    ),
    (
        "model-00003-of-00004.safetensors",
        3864726424,
        "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    ),
    (
        "model-00004-of-00004.safetensors",
        3556377672,
        "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    ),
)
STRUCTURED_PROFILE = "const_tree_no_ws_v3"
GENERATION_SEED = 8172026
CACHE_EQUIVALENCE_PROBE_PROTOCOL = (
    "massive_medical_union_composition_cache_equivalence_probe_v1"
)
CACHE_EQUIVALENCE_CONTINUATION_TEXT = "."
# Both paths execute the same BF16 model and are compared only after conversion
# to float32.  A 1e-3 absolute/relative tolerance is deliberately tight enough
# to expose adapter/cache/prefix mistakes while allowing harmless kernel-order
# variation between a one-token cached SDPA call and a fresh full-prefix call.
CACHE_EQUIVALENCE_ATOL = 1e-3
CACHE_EQUIVALENCE_RTOL = 1e-3
PANEL_ORDER = ("A", "B1", "B2", "B3")
MODEL_NAME_BY_ROLE = {
    "A": "pi_A",
    "B1": "pi_B1",
    "B2": "pi_B2",
    "B3": "pi_B3",
}
METHODS = (
    {
        "method_id": "ordinary_quorum_m4_q3",
        "role": "primary",
        "sampler_method": "quorum",
        "m": 4,
        "q": 3,
        "base_in_composition": False,
        "unnormalized_log_score": "third_largest_j(log_pi_j(v|x))",
    },
    {
        "method_id": "ordinary_min_m4_q4",
        "role": "required_secondary",
        "sampler_method": "quorum",
        "m": 4,
        "q": 4,
        "base_in_composition": False,
        "unnormalized_log_score": "min_j(log_pi_j(v|x))",
    },
    {
        "method_id": "delta_min_m4_q4",
        "role": "required_secondary",
        "sampler_method": "pi_quorum_delta",
        "m": 4,
        "q": 4,
        "base_in_composition": True,
        "unnormalized_log_score": (
            "log_pi_0(v|x)+strict_unanimous_least_magnitude_log_ratio_delta"
        ),
    },
)
PAIRED_BASE = {
    "method_id": "pi_base",
    "role": "paired_same_backend_base",
    "sampler_method": "base",
    "m": 0,
    "q": None,
    "base_in_composition": True,
    "unnormalized_log_score": "log_pi_0(v|x)",
}
MASSIVE_PROFILES = {
    "smoke": {
        "artifact": "smoke/prompts.json",
        "role": "training_disjoint_composition_smoke",
        "rows": 60,
        "n_samples": 1,
        "temperature": 0.0,
        "max_new_tokens": 256,
        "max_context": 2048,
    },
    "confirmation": {
        "artifact": "confirmation/prompts.json",
        "role": "composition_confirmation",
        "rows": 600,
        "n_samples": 1,
        "temperature": 0.0,
        "max_new_tokens": 256,
        "max_context": 2048,
    },
}
MEDICAL_PROFILE = {
    "artifact": "medical/prompts.json",
    "role": "composition_confirmation",
    "rows": 16,
    "n_samples": 5,
    "temperature": 1.0,
    "max_new_tokens": 1024,
    "max_context": 2048,
    "sampling_profile": "official16_max1024_all_stop_v2",
}
FORBIDDEN_PROMPT_FIELDS = {
    "intent",
    "slots",
    "annot_utt",
    "answer",
    "answers",
    "source_id",
    "scenario",
    "response",
}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
RECORDED_LEGACY_HYBRID_INTENT_PROBES = (
    "alarm_addcontact",
    "alarm_createoradd",
    "calendar_recipe",
    "cooking_remove",
)
RECORDED_LEGACY_HYBRID_SLOT_PROBES = (
    "alarm_name",
    "app_type",
    "cooking_name",
)


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_regular_bytes(path, description):
    path = os.path.abspath(path)
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{description} is not a regular non-symlink file: {path}")
    with open(path, "rb") as handle:
        raw = handle.read()
    after = os.lstat(path)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"{description} changed while it was being read: {path}")
    return raw


def load_json_regular(path, description):
    raw = read_regular_bytes(path, description)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from error
    return payload, raw


def verify_seal(payload, field, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(field, None)
    expected = sha256_bytes(canonical_bytes(body))
    if observed != expected:
        raise ValueError(f"{description} has an invalid {field} seal")
    return body


def seal(payload, field=OUTPUT_SEAL_FIELD):
    body = dict(payload)
    body.pop(field, None)
    body[field] = sha256_bytes(canonical_bytes(body))
    return body


def atomic_write_json(path, payload):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp.", dir=parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def tuple_seed(*parts):
    digest = hashlib.sha256(canonical_bytes(list(parts))).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def prompt_digest(prompt):
    return sha256_bytes(canonical_bytes({"prompt": prompt}))


def balanced_const_tree(labels):
    values = list(labels)
    if (
        not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("ontology labels must be unique nonempty strings")

    def build(start, stop):
        if stop - start == 1:
            return {"const": values[start]}
        middle = start + (stop - start) // 2
        return {"anyOf": [build(start, middle), build(middle, stop)]}

    return build(0, len(values))


def prediction_schema(intent_labels, slot_labels):
    return {
        "type": "object",
        "properties": {
            "intent": balanced_const_tree(intent_labels),
            "slots": {
                "type": "array",
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": balanced_const_tree(slot_labels),
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "slots"],
        "additionalProperties": False,
    }


def validate_prediction(response, intent_labels, slot_labels):
    if not isinstance(response, str):
        raise ValueError("structured response is not a string")
    try:
        prediction = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("structured response is not valid JSON") from error
    if not isinstance(prediction, dict) or set(prediction) != {"intent", "slots"}:
        raise ValueError("structured response has wrong top-level keys")
    if prediction["intent"] not in intent_labels:
        raise ValueError("structured response escaped the intent ontology")
    slots = prediction["slots"]
    if not isinstance(slots, list) or len(slots) > 7:
        raise ValueError("structured response has invalid slots")
    for item in slots:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "value"}
            or item["name"] not in slot_labels
            or not isinstance(item["value"], str)
            or not item["value"]
        ):
            raise ValueError("structured response has an invalid slot")
    return prediction


def compose_quorum_raw_scores(reference_logps, q):
    """Return the per-token q-th largest reference log probability, unnormalized."""
    import torch

    if reference_logps.ndim != 2 or reference_logps.shape[0] != 4:
        raise ValueError("reference_logps must have shape [4, vocabulary]")
    if q not in (3, 4):
        raise ValueError("exploratory ordinary composition permits only q=3 or q=4")
    if reference_logps.dtype != torch.float32:
        raise ValueError("reference log probabilities must be float32")
    return torch.topk(reference_logps, k=q, dim=0, largest=True).values[-1]


def compose_delta_min_raw_scores(reference_logps, base_logp):
    """Return strict-unanimity base-relative delta-min scores, unnormalized."""
    import torch

    if (
        reference_logps.ndim != 2
        or reference_logps.shape[0] != 4
        or base_logp.ndim != 1
        or reference_logps.shape[1] != base_logp.shape[0]
    ):
        raise ValueError("expected four reference logps and one aligned base logp")
    if reference_logps.dtype != torch.float32 or base_logp.dtype != torch.float32:
        raise ValueError("reference and base log probabilities must be float32")
    shifts = reference_logps - base_logp.to(reference_logps.device).unsqueeze(0)
    all_up = torch.all(shifts > 0, dim=0)
    all_down = torch.all(shifts < 0, dim=0)
    least_up = torch.min(shifts, dim=0).values
    least_down = torch.max(shifts, dim=0).values
    delta = torch.where(
        all_up,
        least_up,
        torch.where(all_down, least_down, torch.zeros_like(base_logp)),
    )
    return base_logp.to(reference_logps.device) + delta


def compose_raw_scores(reference_logps, base_logp, method):
    if method["method_id"] == "pi_base":
        if reference_logps is not None or base_logp is None:
            raise ValueError("paired base requires only its base distribution")
        return base_logp
    if method["method_id"] == "ordinary_quorum_m4_q3":
        if base_logp is not None:
            raise ValueError("ordinary q3 must not receive a base distribution")
        return compose_quorum_raw_scores(reference_logps, 3)
    if method["method_id"] == "ordinary_min_m4_q4":
        if base_logp is not None:
            raise ValueError("ordinary min must not receive a base distribution")
        return compose_quorum_raw_scores(reference_logps, 4)
    if method["method_id"] == "delta_min_m4_q4":
        if base_logp is None:
            raise ValueError("delta-min requires a base distribution")
        return compose_delta_min_raw_scores(reference_logps, base_logp)
    raise ValueError("method is not in the frozen exploratory registry")


def normalize_composed_scores(scores):
    """Perform the sole target-distribution normalization."""
    import torch

    if scores.ndim != 1 or scores.dtype != torch.float32:
        raise ValueError("composed scores must be one float32 vocabulary vector")
    if not bool(torch.isfinite(scores).any().item()):
        raise ValueError("composition/grammar left no finite token")
    normalizer = torch.logsumexp(scores, dim=-1)
    if not bool(torch.isfinite(normalizer).item()):
        raise ValueError("composed score normalization is not finite")
    return scores - normalizer


def apply_grammar_mask_then_normalize(scores, grammar_runtime=None):
    """Apply a hard mask to raw composition scores, then normalize exactly once."""
    masked = scores.clone()
    if grammar_runtime is not None:
        matcher = grammar_runtime["matcher"]
        bitmask = grammar_runtime["bitmask"]
        need_apply = matcher.fill_next_token_bitmask(bitmask)
        if type(need_apply) is not bool:
            raise ValueError("XGrammar fill_next_token_bitmask did not return bool")
        if need_apply:
            # Pinned XGrammar 0.1.25 requires [batch, vocabulary], not [vocabulary].
            batched = masked.unsqueeze(0)
            grammar_runtime["apply_token_bitmask_inplace"](
                batched, bitmask.to(masked.device)
            )
            masked = batched[0]
    return normalize_composed_scores(masked)


def cache_sequence_length(cache):
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    if isinstance(cache, (tuple, list)):
        if not cache:
            return 0
        layer = cache[0]
        if not isinstance(layer, (tuple, list)) or not layer:
            raise ValueError("unrecognized legacy cache layer")
        key = layer[0]
        if not hasattr(key, "shape") or len(key.shape) < 3:
            raise ValueError("unrecognized legacy cache key")
        return int(key.shape[-2])
    raise ValueError(f"unrecognized cache type: {type(cache).__name__}")


def extract_logits_and_cache(outputs):
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        cache = getattr(outputs, "past_key_values", None)
    elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        logits, cache = outputs[0], outputs[1]
    else:
        raise ValueError("model output lacks logits/cache")
    if cache is None:
        raise ValueError("model did not return past_key_values")
    return logits, cache


@contextlib.contextmanager
def selected_adapter(model, adapter_name):
    if adapter_name is None:
        disable = getattr(model, "disable_adapter", None)
        if not callable(disable):
            raise ValueError("shared PEFT model cannot disable adapters for delta-min")
        with disable():
            yield
        return
    setter = getattr(model, "set_adapter", None)
    if not callable(setter):
        raise ValueError("shared PEFT model cannot select an adapter")
    setter(adapter_name)
    yield


def forward_cached(model, adapter_name, input_ids, attention_mask, cache):
    with selected_adapter(model, adapter_name):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )


def prefill_cached_reference(model, adapter_name, prompt_ids, device):
    import torch

    if not prompt_ids:
        raise ValueError("cannot prefill an empty prompt")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        outputs = forward_cached(
            model, adapter_name, input_ids, attention_mask, cache=None
        )
    logits, cache = extract_logits_and_cache(outputs)
    length = cache_sequence_length(cache)
    if length != len(prompt_ids):
        raise ValueError(
            f"prefill cache length {length} != prompt length {len(prompt_ids)}"
        )
    return {"next_logits": logits[0, -1, :].float(), "cache": cache}


def step_cached_reference(model, adapter_name, token_id, cache, device):
    import torch

    previous = cache_sequence_length(cache)
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, previous + 1), dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = forward_cached(
            model, adapter_name, input_ids, attention_mask, cache=cache
        )
    logits, next_cache = extract_logits_and_cache(outputs)
    observed = cache_sequence_length(next_cache)
    if observed != previous + 1:
        raise ValueError(
            f"one-token cache step grew from {previous} to {observed}, expected {previous + 1}"
        )
    return {"next_logits": logits[0, -1, :].float(), "cache": next_cache}


def assert_independent_caches(states, base_state=None):
    caches = [state["cache"] for state in states]
    if base_state is not None:
        caches.append(base_state["cache"])
    if len({id(cache) for cache in caches}) != len(caches):
        raise ValueError("references unexpectedly share a mutable KV-cache object")


def cache_tensor_storage_pointers(cache):
    """Return discovered tensor-storage identities for known HF cache layouts."""
    import torch

    pointers = set()
    seen = set()

    def visit(value):
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.numel() > 0:
                storage = value.untyped_storage()
                pointers.add((str(value.device), int(storage.data_ptr())))
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
            return
        # Transformers has used both DynamicCache.key_cache/value_cache and
        # DynamicCache.layers[*].keys/values across pinned-adjacent releases.
        for attribute in (
            "key_cache",
            "value_cache",
            "layers",
            "keys",
            "values",
            "key_states",
            "value_states",
        ):
            if hasattr(value, attribute):
                visit(getattr(value, attribute))

    visit(cache)
    return pointers


def make_prompt_ids(tokenizer, record):
    messages = []
    system = record.get("system", "")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": record["prompt"]})
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("tokenizer unexpectedly returned a batch")
        ids = ids[0]
    if not ids or any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise ValueError("tokenizer returned invalid prompt token IDs")
    return list(ids)


def fresh_full_prefix_next_logits(model, adapter_name, prefix_ids, device):
    """Evaluate one complete prefix without reusing a caller-owned KV cache."""
    import torch

    if not prefix_ids:
        raise ValueError("cache-equivalence full prefix is empty")
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        outputs = forward_cached(
            model, adapter_name, input_ids, attention_mask, cache=None
        )
    logits, cache = extract_logits_and_cache(outputs)
    if cache_sequence_length(cache) != len(prefix_ids):
        raise ValueError("cache-equivalence fresh full-prefix cache length differs")
    result = logits[0, -1, :].float()
    if result.ndim != 1 or not bool(torch.isfinite(result).all().item()):
        raise ValueError("cache-equivalence full-prefix logits are invalid")
    return result


def audit_cache_equivalence_probe(probe, phase=None):
    """Validate sealed/live cache probe evidence without trusting a PASS label."""
    roles = [*PANEL_ORDER, "base"]
    if (
        not isinstance(probe, dict)
        or set(probe)
        != {
            "protocol",
            "phase",
            "result",
            "question_id",
            "prompt_sha256",
            "prompt_token_ids_sha256",
            "prompt_tokens",
            "continuation_text",
            "continuation_text_sha256",
            "continuation_token_id",
            "roles",
            "same_prefix_and_token_all_roles",
            "cache_objects_unique",
            "cache_tensor_storage_sets_checked",
            "cache_tensor_storages_disjoint",
            "comparison_dtype",
            "atol",
            "rtol",
            "vocab_size",
            "comparisons",
            "probe_seconds",
        }
        or probe.get("protocol") != CACHE_EQUIVALENCE_PROBE_PROTOCOL
        or (phase is not None and probe.get("phase") != phase)
        or probe.get("result") != "PASS"
        or not isinstance(probe.get("question_id"), str)
        or not probe["question_id"]
        or not HEX64_RE.fullmatch(str(probe.get("prompt_sha256", "")))
        or not HEX64_RE.fullmatch(str(probe.get("prompt_token_ids_sha256", "")))
        or isinstance(probe.get("prompt_tokens"), bool)
        or not isinstance(probe.get("prompt_tokens"), int)
        or probe["prompt_tokens"] <= 0
        or probe.get("continuation_text") != CACHE_EQUIVALENCE_CONTINUATION_TEXT
        or probe.get("continuation_text_sha256")
        != sha256_bytes(CACHE_EQUIVALENCE_CONTINUATION_TEXT.encode("utf-8"))
        or isinstance(probe.get("continuation_token_id"), bool)
        or not isinstance(probe.get("continuation_token_id"), int)
        or probe["continuation_token_id"] < 0
        or probe.get("roles") != roles
        or probe.get("same_prefix_and_token_all_roles") is not True
        or probe.get("cache_objects_unique") is not True
        or not isinstance(probe.get("cache_tensor_storage_sets_checked"), bool)
        or probe.get("cache_tensor_storages_disjoint") is not True
        or probe.get("comparison_dtype") != "float32"
        or probe.get("atol") != CACHE_EQUIVALENCE_ATOL
        or probe.get("rtol") != CACHE_EQUIVALENCE_RTOL
        or isinstance(probe.get("vocab_size"), bool)
        or not isinstance(probe.get("vocab_size"), int)
        or probe["vocab_size"] <= 0
        or not isinstance(probe.get("comparisons"), dict)
        or list(probe["comparisons"]) != roles
        or isinstance(probe.get("probe_seconds"), bool)
        or not isinstance(probe.get("probe_seconds"), (int, float))
        or probe["probe_seconds"] < 0
    ):
        raise ValueError("cache-equivalence probe metadata differs")
    for role in roles:
        comparison = probe["comparisons"][role]
        if (
            not isinstance(comparison, dict)
            or set(comparison)
            != {"allclose", "max_abs_diff", "max_scaled_error"}
            or comparison.get("allclose") is not True
            or isinstance(comparison.get("max_abs_diff"), bool)
            or not isinstance(comparison.get("max_abs_diff"), (int, float))
            or not 0 <= comparison["max_abs_diff"] < float("inf")
            or isinstance(comparison.get("max_scaled_error"), bool)
            or not isinstance(comparison.get("max_scaled_error"), (int, float))
            or not 0 <= comparison["max_scaled_error"] <= 1.0
        ):
            raise ValueError(f"cache-equivalence comparison differs for {role}")
    return probe


def run_cache_equivalence_probe(
    model, tokenizer, record, phase, device, *, atol=CACHE_EQUIVALENCE_ATOL,
    rtol=CACHE_EQUIVALENCE_RTOL,
):
    """Prove shared-PEFT adapter switching and cached-prefix logits agree live."""
    import torch

    started = time.perf_counter()
    if atol != CACHE_EQUIVALENCE_ATOL or rtol != CACHE_EQUIVALENCE_RTOL:
        raise ValueError("cache-equivalence tolerance override is forbidden")
    prompt_ids = make_prompt_ids(tokenizer, record)
    continuation_ids = tokenizer.encode(
        CACHE_EQUIVALENCE_CONTINUATION_TEXT, add_special_tokens=False
    )
    if hasattr(continuation_ids, "tolist"):
        continuation_ids = continuation_ids.tolist()
    if (
        not isinstance(continuation_ids, list)
        or len(continuation_ids) != 1
        or isinstance(continuation_ids[0], bool)
        or not isinstance(continuation_ids[0], int)
        or continuation_ids[0] < 0
    ):
        raise ValueError(
            "pinned tokenizer no longer maps the cache probe continuation to one token"
        )
    token_id = continuation_ids[0]
    roles = [*PANEL_ORDER, "base"]
    adapters = {role: (None if role == "base" else role) for role in roles}
    states = {
        role: prefill_cached_reference(
            model, adapters[role], prompt_ids, device
        )
        for role in roles
    }
    caches = [states[role]["cache"] for role in roles]
    if len({id(cache) for cache in caches}) != len(caches):
        raise ValueError("cache-equivalence probe found a shared KV-cache object")
    pointer_sets = [cache_tensor_storage_pointers(cache) for cache in caches]
    pointer_sets_checked = all(bool(pointers) for pointers in pointer_sets)
    if any(pointer_sets) and not pointer_sets_checked:
        raise ValueError("cache-equivalence probe could not inspect every cache storage")
    if pointer_sets_checked:
        for index, left in enumerate(pointer_sets):
            for right in pointer_sets[index + 1 :]:
                if left & right:
                    raise ValueError(
                        "cache-equivalence probe found shared KV-cache tensor storage"
                    )

    comparisons = {}
    vocab_size = None
    full_prefix = [*prompt_ids, token_id]
    for role in roles:
        adapter = adapters[role]
        cached = step_cached_reference(
            model, adapter, token_id, states[role]["cache"], device
        )["next_logits"].float()
        fresh = fresh_full_prefix_next_logits(
            model, adapter, full_prefix, device
        ).float()
        if (
            cached.dtype != torch.float32
            or fresh.dtype != torch.float32
            or cached.shape != fresh.shape
            or cached.ndim != 1
            or not bool(torch.isfinite(cached).all().item())
            or not bool(torch.isfinite(fresh).all().item())
        ):
            raise ValueError(f"cache-equivalence logits differ in shape/dtype for {role}")
        vocab_size = cached.numel() if vocab_size is None else vocab_size
        if cached.numel() != vocab_size:
            raise ValueError("cache-equivalence vocabulary differs across adapters")
        difference = (cached - fresh).abs()
        allowed = atol + rtol * fresh.abs()
        scaled = difference / allowed
        allclose = bool(torch.all(difference <= allowed).item())
        comparison = {
            "allclose": allclose,
            "max_abs_diff": float(difference.max().item()),
            "max_scaled_error": float(scaled.max().item()),
        }
        if not allclose:
            raise ValueError(
                f"cache-equivalence next logits mismatch for {role}: {comparison}"
            )
        comparisons[role] = comparison

    probe = {
        "protocol": CACHE_EQUIVALENCE_PROBE_PROTOCOL,
        "phase": phase,
        "result": "PASS",
        "question_id": record["question_id"],
        "prompt_sha256": record["prompt_sha256"],
        "prompt_token_ids_sha256": sha256_bytes(canonical_bytes(prompt_ids)),
        "prompt_tokens": len(prompt_ids),
        "continuation_text": CACHE_EQUIVALENCE_CONTINUATION_TEXT,
        "continuation_text_sha256": sha256_bytes(
            CACHE_EQUIVALENCE_CONTINUATION_TEXT.encode("utf-8")
        ),
        "continuation_token_id": token_id,
        "roles": roles,
        "same_prefix_and_token_all_roles": True,
        "cache_objects_unique": True,
        "cache_tensor_storage_sets_checked": pointer_sets_checked,
        "cache_tensor_storages_disjoint": True,
        "comparison_dtype": "float32",
        "atol": atol,
        "rtol": rtol,
        "vocab_size": vocab_size,
        "comparisons": comparisons,
        "probe_seconds": time.perf_counter() - started,
    }
    return audit_cache_equivalence_probe(probe, phase)


def generate_sample(
    *,
    record,
    sample_index,
    prompt_ids,
    model,
    tokenizer,
    method,
    profile,
    device,
    stop_ids,
    grammar_factory=None,
):
    """Generate one sample while keeping all reference/base prefixes identical."""
    import torch
    import torch.nn.functional as functional

    paired_base = method["method_id"] == "pi_base"
    states = (
        []
        if paired_base
        else [
            prefill_cached_reference(model, role, prompt_ids, device)
            for role in PANEL_ORDER
        ]
    )
    base_state = (
        prefill_cached_reference(model, None, prompt_ids, device)
        if method["base_in_composition"]
        else None
    )
    assert_independent_caches(states, base_state)
    grammar_runtime = grammar_factory() if grammar_factory is not None else None
    if grammar_runtime is not None and grammar_runtime["matcher"].is_terminated():
        raise ValueError("fresh grammar matcher is already terminated")

    response_ids = []
    finish_reason = "max_new_tokens"
    rng_seed = tuple_seed(
        GENERATION_SEED,
        method["method_id"],
        record["question_id"],
        sample_index,
    )
    generator = None
    if profile["temperature"] > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(rng_seed)

    for token_index in range(profile["max_new_tokens"]):
        reference_logps = (
            None
            if paired_base
            else torch.stack(
                [
                    functional.log_softmax(state["next_logits"].float(), dim=-1)
                    for state in states
                ],
                dim=0,
            ).float()
        )
        base_logp = (
            functional.log_softmax(base_state["next_logits"].float(), dim=-1)
            if base_state is not None
            else None
        )
        raw_scores = compose_raw_scores(reference_logps, base_logp, method)
        target_logp = apply_grammar_mask_then_normalize(raw_scores, grammar_runtime)
        if profile["temperature"] == 0:
            token_id = int(torch.argmax(target_logp).item())
        elif profile["temperature"] == 1:
            token_id = int(
                torch.multinomial(
                    torch.exp(target_logp), 1, generator=generator
                ).item()
            )
        else:
            raise ValueError("exploratory manifest requested a non-frozen temperature")

        if grammar_runtime is not None:
            if not grammar_runtime["matcher"].accept_token(token_id):
                raise ValueError("XGrammar rejected a token admitted by its own mask")
            response_ids.append(token_id)
            if grammar_runtime["matcher"].is_terminated():
                finish_reason = "stop"
                break
        else:
            if token_id in stop_ids:
                finish_reason = "stop"
                break
            response_ids.append(token_id)

        if token_index + 1 < profile["max_new_tokens"]:
            states = [
                step_cached_reference(model, role, token_id, state["cache"], device)
                for role, state in zip(PANEL_ORDER, states)
            ]
            if base_state is not None:
                base_state = step_cached_reference(
                    model, None, token_id, base_state["cache"], device
                )
            assert_independent_caches(states, base_state)

    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    sample = {
        "question_id": record["question_id"],
        "sample_index": sample_index,
        "prompt_sha256": record["prompt_sha256"],
        "response": response,
        "finish_reason": finish_reason,
        "generated_tokens": len(response_ids),
        "response_sha256": sha256_bytes(response.encode("utf-8")),
        "rng_seed": rng_seed,
    }
    if grammar_runtime is not None:
        if finish_reason != "stop":
            raise ValueError("MASSIVE grammar did not terminate within the frozen budget")
        sample["prediction"] = validate_prediction(
            response,
            profile["intent_labels"],
            profile["slot_labels"],
        )
    sample["sample_sha256"] = sample_sha256(sample)
    return sample


def sample_sha256(sample):
    body = {key: value for key, value in sample.items() if key != "sample_sha256"}
    return sha256_bytes(canonical_bytes(body))


def inventory_map(entries, description):
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{description} inventory is missing")
    result = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "size_bytes", "sha256"}
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or entry["path"].startswith("/")
            or ".." in Path(entry["path"]).parts
            or entry["path"] in result
            or isinstance(entry.get("size_bytes"), bool)
            or not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] < 0
            or HEX64_RE.fullmatch(str(entry.get("sha256"))) is None
        ):
            raise ValueError(f"{description} inventory entry is malformed")
        result[entry["path"]] = dict(entry)
    return result


def live_inventory(root, excluded=("manifest.json",)):
    root = os.path.abspath(root)
    result = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        for directory_name in directory_names:
            candidate = os.path.join(directory, directory_name)
            if os.path.islink(candidate):
                raise ValueError(f"protocol/model tree contains symlink directory: {candidate}")
        for filename in sorted(file_names):
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative in excluded:
                continue
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"protocol/model tree contains unsafe file: {path}")
            result.append(
                {
                    "path": relative,
                    "size_bytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                }
            )
    return result


def require_file_binding(path, binding, description):
    raw = read_regular_bytes(path, description)
    if (
        not isinstance(binding, dict)
        or binding.get("size_bytes") != len(raw)
        or binding.get("sha256", binding.get("file_sha256"))
        != sha256_bytes(raw)
    ):
        raise ValueError(f"{description} differs from its frozen binding")
    return raw


def audit_reference_binding(model_name, binding):
    expected_role = model_name.removeprefix("pi_")
    if (
        not isinstance(binding, dict)
        or binding.get("model_name") != model_name
        or binding.get("role") != expected_role
        or binding.get("base_model") != BASE_MODEL
        or binding.get("base_model_revision") != BASE_REVISION
        or HEX64_RE.fullmatch(str(binding.get("model_fingerprint"))) is None
    ):
        raise ValueError(f"{model_name} panel binding differs")
    manifest_path = binding.get("path")
    model_path = binding.get("model_path")
    if (
        not isinstance(manifest_path, str)
        or not isinstance(model_path, str)
        or os.path.islink(model_path)
        or not os.path.isdir(model_path)
    ):
        raise ValueError(f"{model_name} path binding differs")
    manifest, raw = load_json_regular(manifest_path, f"{model_name} model manifest")
    seal_fields = [
        field
        for field in ("payload_sha256", "manifest_payload_sha256")
        if field in manifest
    ]
    if len(seal_fields) != 1:
        raise ValueError(f"{model_name} model manifest has wrong seal shape")
    verify_seal(manifest, seal_fields[0], f"{model_name} model manifest")
    if (
        binding.get("file_sha256") != sha256_bytes(raw)
        or binding.get("payload_sha256") != manifest[seal_fields[0]]
    ):
        raise ValueError(f"{model_name} model manifest bytes differ")
    expected_inventory = inventory_map(
        binding.get("exact_model_inventory"), f"{model_name} exact model"
    )
    observed_inventory = inventory_map(
        live_inventory(model_path, excluded=("MODEL_MANIFEST.json", "TRAIN_COMPLETE")),
        f"{model_name} live model",
    )
    if observed_inventory != expected_inventory:
        raise ValueError(f"{model_name} exact live model inventory differs")
    adapter_inventory = binding.get("adapter_inventory")
    if (
        not isinstance(adapter_inventory, list)
        or binding["model_fingerprint"] != sha256_bytes(canonical_bytes(adapter_inventory))
    ):
        raise ValueError(f"{model_name} adapter fingerprint differs")
    for artifact in adapter_inventory:
        name = artifact.get("name") if isinstance(artifact, dict) else None
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name not in {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
        ):
            raise ValueError(f"{model_name} adapter inventory is malformed")
        require_file_binding(
            os.path.join(model_path, name), artifact, f"{model_name} {name}"
        )
    return dict(binding)


def audit_protocol_inventory(root, expected_entries):
    expected = inventory_map(expected_entries, "exploratory protocol")
    observed = inventory_map(live_inventory(root), "live exploratory protocol")
    if observed != expected:
        raise ValueError("exploratory protocol file inventory differs")


def load_protocol_manifest(path, audit_models=True):
    path = os.path.abspath(path)
    if os.path.basename(path) != "manifest.json":
        raise ValueError("exploratory protocol manifest must be named manifest.json")
    payload, raw = load_json_regular(path, "exploratory protocol manifest")
    body = verify_seal(payload, MANIFEST_SEAL_FIELD, "exploratory protocol manifest")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or not isinstance(body.get("exploratory_contract"), dict)
        or body["exploratory_contract"].get("confirmatory") is not False
        or body["exploratory_contract"].get("post_wave2_stop") is not True
        or body["exploratory_contract"].get("wave3_v1_eligible") is not False
        or body["exploratory_contract"].get("wave3_submitted_or_released") is not False
    ):
        raise ValueError("exploratory protocol identity/status differs")
    if body.get("methods") != list(METHODS):
        raise ValueError("exploratory method registry differs")
    generation = body.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("exploratory generation registry is missing")
    massive = generation.get("massive")
    medical = generation.get("medical")
    paired_base = generation.get("paired_base")
    expected_massive = {
        "n_samples": 1,
        "temperature": 0.0,
        "max_new_tokens": 256,
        "max_context": 2048,
        "structured_constraint_profile": STRUCTURED_PROFILE,
        "arbitrary_structural_whitespace": False,
        "truncation": False,
    }
    for key, expected in expected_massive.items():
        if not isinstance(massive, dict) or massive.get(key) != expected:
            raise ValueError(f"exploratory MASSIVE generation differs on {key}")
    expected_medical = {
        "n_prompts": 16,
        "n_samples_per_prompt": 5,
        "temperature": 1.0,
        "seed": GENERATION_SEED,
        "max_new_tokens": 1024,
        "max_context": 2048,
        "profile": "official16_max1024_all_stop_v2",
        "required_finish_reason": "stop",
        "truncation": False,
    }
    for key, expected in expected_medical.items():
        if not isinstance(medical, dict) or medical.get(key) != expected:
            raise ValueError(f"exploratory medical generation differs on {key}")
    if (
        generation.get("panel_order") != ["A", "B1", "B2", "B3"]
        or not isinstance(paired_base, dict)
        or paired_base.get("model_name") != "pi_base"
        or paired_base.get("fresh_generation_required") is not True
        or paired_base.get("splits") != ["smoke", "confirmation"]
        or paired_base.get("backend")
        != "same_transformers_backend_as_composition_methods"
        or paired_base.get("filtered_wave2_direct_score_may_substitute") is not False
    ):
        raise ValueError("exploratory panel/base generation registry differs")
    root = os.path.dirname(path)
    audit_protocol_inventory(root, body.get("file_inventory"))
    copied = body.get("copied_artifacts")
    if not isinstance(copied, dict):
        raise ValueError("exploratory copied-artifact bindings are missing")
    for relative in (
        "smoke/prompts.json",
        "smoke/answers.json",
        "confirmation/prompts.json",
        "confirmation/answers.json",
        "medical/prompts.json",
    ):
        binding = copied.get(relative)
        artifact_path = os.path.join(root, *relative.split("/"))
        if (
            not isinstance(binding, dict)
            or binding.get("copied_path") != relative
            or binding.get("byte_identical") is not True
        ):
            raise ValueError(f"copied artifact binding differs: {relative}")
        require_file_binding(artifact_path, binding, f"copied {relative}")
    panel = body.get("model_panel")
    references = panel.get("references") if isinstance(panel, dict) else None
    if (
        not isinstance(panel, dict)
        or panel.get("panel_order") != ["pi_A", "pi_B1", "pi_B2", "pi_B3"]
        or not isinstance(references, dict)
        or list(references) != ["pi_A", "pi_B1", "pi_B2", "pi_B3"]
        or panel.get("base")
        != {
            "model_name": "pi_base",
            "model_path": "BASE",
            "model_fingerprint": "BASE",
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_REVISION,
        }
    ):
        raise ValueError("exploratory model panel differs")
    audited_references = {}
    for role in PANEL_ORDER:
        name = MODEL_NAME_BY_ROLE[role]
        audited_references[name] = (
            audit_reference_binding(name, references[name])
            if audit_models
            else dict(references[name])
        )
    result = {
        "path": path,
        "root": root,
        "file_sha256": sha256_bytes(raw),
        "payload_sha256": payload[MANIFEST_SEAL_FIELD],
        "body": body,
        "references": audited_references,
    }
    return result


def load_massive_prompts(protocol, phase):
    profile = dict(MASSIVE_PROFILES[phase])
    path = os.path.join(protocol["root"], *profile["artifact"].split("/"))
    payload, _ = load_json_regular(path, f"{phase} MASSIVE prompts")
    meta, records = payload.get("meta"), payload.get("prompts")
    intents = meta.get("intent_labels") if isinstance(meta, dict) else None
    slots = meta.get("slot_labels") if isinstance(meta, dict) else None
    if (
        not isinstance(meta, dict)
        or meta.get("protocol_id") != "massive_medical_union_wave3_composition_v1"
        or meta.get("role") != profile["role"]
        or meta.get("contains_gold_labels") is not False
        or meta.get("n_questions") != profile["rows"]
        or not isinstance(records, list)
        or len(records) != profile["rows"]
        or not isinstance(intents, list)
        or len(intents) != 60
        or len(set(intents)) != 60
        or not isinstance(slots, list)
        or len(slots) != 55
        or len(set(slots)) != 55
    ):
        raise ValueError(f"{phase} MASSIVE prompt bank differs")
    ontology = sha256_bytes(
        canonical_bytes({"intent_labels": intents, "slot_labels": slots})
    )
    if meta.get("ontology_sha256") != ontology:
        raise ValueError(f"{phase} MASSIVE ontology differs")
    seen = set()
    validated = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or FORBIDDEN_PROMPT_FIELDS & set(record):
            raise ValueError(f"{phase} prompt {index} exposes a gold field")
        question_id = record.get("question_id")
        prompt = record.get("prompt")
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id in seen
            or not isinstance(prompt, str)
            or not prompt
            or record.get("prompt_sha256") != prompt_digest(prompt)
        ):
            raise ValueError(f"{phase} MASSIVE prompt {index} differs")
        seen.add(question_id)
        validated.append(dict(record))
    profile.update(
        {
            "domain": "massive",
            "endpoint": "joint_json",
            "prompt_path": path,
            "prompt_file_sha256": sha256_file(path),
            "intent_labels": intents,
            "slot_labels": slots,
            "ontology_sha256": ontology,
            "structured_constraint_profile": STRUCTURED_PROFILE,
            "xgrammar_any_whitespace": False,
            "seed": GENERATION_SEED,
        }
    )
    return profile, validated


def load_medical_prompts(protocol):
    profile = dict(MEDICAL_PROFILE)
    path = os.path.join(protocol["root"], *profile["artifact"].split("/"))
    payload, _ = load_json_regular(path, "official16 medical prompts")
    meta, records = payload.get("meta"), payload.get("prompts")
    if (
        not isinstance(meta, dict)
        or meta.get("name") != "official_medical_questions_16"
        or meta.get("n_prompts") != 16
        or meta.get("contains_answers") is not False
        or not isinstance(records, list)
        or len(records) != 16
    ):
        raise ValueError("official16 medical prompt bank differs")
    validated = []
    for index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record) != {
                "prompt_index",
                "question_id",
                "prompt",
                "prompt_sha256",
            }
            or record.get("prompt_index") != index
            or record.get("question_id") != f"medical_official16_{index:02d}"
            or not isinstance(record.get("prompt"), str)
            or not record["prompt"]
            or record.get("prompt_sha256") != prompt_digest(record["prompt"])
        ):
            raise ValueError(f"official16 medical prompt {index} differs")
        validated.append(dict(record))
    profile.update(
        {
            "domain": "medical",
            "endpoint": "free_text",
            "prompt_path": path,
            "prompt_file_sha256": sha256_file(path),
            "seed": GENERATION_SEED,
        }
    )
    return profile, validated


def require_pinned_runtime(require_cuda=False):
    """Fail closed on the exact inference stack frozen by the protocol."""
    import torch

    observed = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "xgrammar": importlib.metadata.version("xgrammar"),
    }
    expected = {
        "torch": PINNED_TORCH_VERSION,
        "transformers": PINNED_TRANSFORMERS_VERSION,
        "peft": PINNED_PEFT_VERSION,
        "xgrammar": PINNED_XGRAMMAR_VERSION,
    }
    if observed != expected:
        raise ValueError(
            f"exploratory inference stack differs: expected {expected}, observed {observed}"
        )
    if require_cuda and not torch.cuda.is_available():
        raise ValueError("exploratory generation requires an available CUDA device")
    return observed


def force_offline_environment():
    expected = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in expected.items():
        observed = os.environ.get(key)
        if observed not in (None, value):
            raise ValueError(f"refusing conflicting offline setting {key}={observed!r}")
        os.environ[key] = value
    return expected


def _path_is_within(path, root):
    try:
        return os.path.commonpath(
            (os.path.realpath(path), os.path.realpath(root))
        ) == os.path.realpath(root)
    except ValueError:
        return False


def _audit_snapshot_artifact(snapshot_root, blob_root, expected, capture=False):
    name, expected_size, expected_sha256 = expected
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or name in {"", ".", ".."}
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or HEX64_RE.fullmatch(str(expected_sha256)) is None
    ):
        raise ValueError("pinned base artifact registry is malformed")
    path = os.path.join(snapshot_root, name)
    try:
        lexical_before = os.lstat(path)
        target_before = os.stat(path)
    except FileNotFoundError as error:
        raise ValueError(f"pinned base snapshot is missing {name}") from error
    if not (
        stat.S_ISREG(lexical_before.st_mode) or stat.S_ISLNK(lexical_before.st_mode)
    ):
        raise ValueError(f"pinned base snapshot artifact is unsafe: {name}")
    if not stat.S_ISREG(target_before.st_mode):
        raise ValueError(f"pinned base snapshot artifact is not regular: {name}")
    if stat.S_ISLNK(lexical_before.st_mode) and not _path_is_within(path, blob_root):
        raise ValueError(f"pinned base snapshot artifact escapes its blob cache: {name}")

    digest = hashlib.sha256()
    chunks = [] if capture else None
    with open(path, "rb") as handle:
        opened_before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        opened_after = os.fstat(handle.fileno())
    lexical_after = os.lstat(path)
    target_after = os.stat(path)

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        lexical_before.st_dev,
        lexical_before.st_ino,
        lexical_before.st_size,
        lexical_before.st_mtime_ns,
    ) != (
        lexical_after.st_dev,
        lexical_after.st_ino,
        lexical_after.st_size,
        lexical_after.st_mtime_ns,
    ) or not (
        identity(target_before)
        == identity(opened_before)
        == identity(opened_after)
        == identity(target_after)
    ):
        raise ValueError(f"pinned base snapshot artifact changed during audit: {name}")
    observed_sha256 = digest.hexdigest()
    if target_after.st_size != expected_size or observed_sha256 != expected_sha256:
        raise ValueError(f"pinned base snapshot artifact differs: {name}")
    binding = {
        "path": name,
        "size_bytes": target_after.st_size,
        "sha256": observed_sha256,
    }
    return binding, b"".join(chunks) if chunks is not None else None


def verify_pinned_base_snapshot(binding):
    body = verify_seal(binding, BASE_SNAPSHOT_SEAL_FIELD, "pinned base snapshot")
    expected_keys = {
        "schema_version",
        "protocol",
        "model_id",
        "revision",
        "hub_cache",
        "snapshot_path",
        "runtime_artifacts",
        "safetensors_index",
        "safetensors_shards",
    }
    if set(body) != expected_keys:
        raise ValueError("pinned base snapshot schema differs")
    expected_index = {
        "path": BASE_SAFETENSORS_INDEX[0],
        "size_bytes": BASE_SAFETENSORS_INDEX[1],
        "sha256": BASE_SAFETENSORS_INDEX[2],
    }
    expected_shards = [
        {"path": name, "size_bytes": size, "sha256": digest}
        for name, size, digest in BASE_SAFETENSORS_SHARDS
    ]
    expected_runtime = [
        {"path": name, "size_bytes": size, "sha256": digest}
        for name, size, digest in BASE_RUNTIME_ARTIFACTS
    ]
    hub_cache = body.get("hub_cache")
    snapshot_path = body.get("snapshot_path")
    expected_snapshot = (
        os.path.join(
            hub_cache,
            BASE_CACHE_DIRECTORY,
            "snapshots",
            BASE_REVISION,
        )
        if isinstance(hub_cache, str)
        else None
    )
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol") != BASE_SNAPSHOT_PROTOCOL
        or body.get("model_id") != BASE_MODEL
        or body.get("revision") != BASE_REVISION
        or not isinstance(hub_cache, str)
        or not os.path.isabs(hub_cache)
        or snapshot_path != expected_snapshot
        or body.get("runtime_artifacts") != expected_runtime
        or body.get("safetensors_index") != expected_index
        or body.get("safetensors_shards") != expected_shards
    ):
        raise ValueError("pinned base snapshot binding differs")
    return snapshot_path


def resolve_pinned_base_snapshot():
    """Resolve and fully hash the sole permitted local Qwen weight snapshot."""
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if (
        not isinstance(hub_cache, str)
        or not hub_cache
        or not os.path.isabs(hub_cache)
    ):
        raise ValueError("HUGGINGFACE_HUB_CACHE must name an absolute local hub cache")
    hub_cache = os.path.normpath(hub_cache)
    legacy_cache = os.environ.get("TRANSFORMERS_CACHE")
    if legacy_cache is not None and (
        not os.path.isabs(legacy_cache)
        or os.path.normpath(legacy_cache) != hub_cache
    ):
        raise ValueError(
            "TRANSFORMERS_CACHE conflicts with the pinned local hub cache"
        )
    if os.path.islink(hub_cache) or not os.path.isdir(hub_cache):
        raise ValueError("pinned local hub cache is not a real directory")
    model_cache = os.path.join(hub_cache, BASE_CACHE_DIRECTORY)
    snapshot_root = os.path.join(model_cache, "snapshots", BASE_REVISION)
    blob_root = os.path.join(model_cache, "blobs")
    if (
        os.path.islink(model_cache)
        or not os.path.isdir(model_cache)
        or os.path.islink(snapshot_root)
        or not os.path.isdir(snapshot_root)
        or os.path.islink(blob_root)
        or not os.path.isdir(blob_root)
    ):
        raise ValueError("pinned base snapshot/cache layout differs")

    expected_shard_names = {item[0] for item in BASE_SAFETENSORS_SHARDS}
    observed_shard_paths = set()
    for directory, directory_names, file_names in os.walk(
        snapshot_root, followlinks=False
    ):
        directory_names.sort()
        for directory_name in directory_names:
            if os.path.islink(os.path.join(directory, directory_name)):
                raise ValueError("pinned base snapshot contains a symlink directory")
        for filename in sorted(file_names):
            if filename.endswith(".safetensors"):
                observed_shard_paths.add(
                    os.path.relpath(
                        os.path.join(directory, filename), snapshot_root
                    ).replace(os.sep, "/")
                )
    if observed_shard_paths != expected_shard_names:
        raise ValueError("pinned base snapshot safetensors shard paths differ")

    runtime_bindings = [
        _audit_snapshot_artifact(snapshot_root, blob_root, expected)[0]
        for expected in BASE_RUNTIME_ARTIFACTS
    ]
    index_binding, index_raw = _audit_snapshot_artifact(
        snapshot_root, blob_root, BASE_SAFETENSORS_INDEX, capture=True
    )
    try:
        index_payload = json.loads(index_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("pinned base safetensors index is not valid JSON") from error
    weight_map = (
        index_payload.get("weight_map") if isinstance(index_payload, dict) else None
    )
    if (
        not isinstance(weight_map, dict)
        or len(weight_map) != BASE_SAFETENSORS_INDEX_ENTRIES
        or any(not isinstance(key, str) or not key for key in weight_map)
        or any(
            not isinstance(value, str)
            or Path(value).name != value
            or value not in expected_shard_names
            for value in weight_map.values()
        )
        or set(weight_map.values()) != expected_shard_names
        or not isinstance(index_payload.get("metadata"), dict)
        or index_payload["metadata"].get("total_size") != BASE_INDEXED_WEIGHT_BYTES
    ):
        raise ValueError("pinned base safetensors index shard map differs")
    shard_bindings = [
        _audit_snapshot_artifact(snapshot_root, blob_root, expected)[0]
        for expected in BASE_SAFETENSORS_SHARDS
    ]
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol": BASE_SNAPSHOT_PROTOCOL,
        "model_id": BASE_MODEL,
        "revision": BASE_REVISION,
        "hub_cache": hub_cache,
        "snapshot_path": snapshot_root,
        "runtime_artifacts": runtime_bindings,
        "safetensors_index": index_binding,
        "safetensors_shards": shard_bindings,
    }
    binding = seal(body, field=BASE_SNAPSHOT_SEAL_FIELD)
    verify_pinned_base_snapshot(binding)
    return binding


def xgrammar_accepts_text(xgrammar_module, compiled_grammar, tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in token_ids
    ):
        raise ValueError("tokenizer produced invalid XGrammar audit tokens")
    matcher = xgrammar_module.GrammarMatcher(
        compiled_grammar, terminate_without_stop_token=True
    )
    for token_id in token_ids:
        if not matcher.accept_token(token_id):
            return False
    return matcher.is_terminated()


def audit_balanced_xgrammar_frontier(grammar_text, labels, label_kind):
    rules = {}
    for line in grammar_text.splitlines():
        if "::=" not in line:
            continue
        name, body = line.split("::=", 1)
        name = name.strip()
        if not name or name in rules:
            raise ValueError("pinned XGrammar emitted malformed/duplicate rules")
        rules[name] = body
    encoded_labels = {
        label: json.dumps(json.dumps(label, ensure_ascii=False), ensure_ascii=False)
        for label in labels
    }
    occurrences = {
        label: [name for name, body in rules.items() if encoded in body]
        for label, encoded in encoded_labels.items()
    }
    if any(len(names) != 1 for names in occurrences.values()):
        raise ValueError(f"pinned XGrammar changed the {label_kind} const leaves")
    prefixes = set()
    for names in occurrences.values():
        name = names[0]
        if "_case_" not in name:
            raise ValueError(f"pinned XGrammar flattened the {label_kind} frontier")
        prefixes.add(name.split("_case_", 1)[0])
    if len(prefixes) != 1:
        raise ValueError(f"pinned XGrammar split the {label_kind} frontier")
    prefix = prefixes.pop()
    frontier = {
        name: body
        for name, body in rules.items()
        if name == prefix or name.startswith(prefix + "_case_")
    }
    if len(frontier) != len(labels) - 1 or any(
        body.count(" | ") != 1 for body in frontier.values()
    ):
        raise ValueError(f"pinned XGrammar changed the balanced {label_kind} tree")


def compile_and_audit_xgrammar(tokenizer, model_config, profile):
    """Compile and exercise the exact no-arbitrary-whitespace joint grammar."""
    import torch
    import xgrammar as xgr

    vocabulary_size = getattr(model_config, "vocab_size", None)
    if (
        isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, int)
        or vocabulary_size <= 0
    ):
        raise ValueError("pinned base config lacks a positive vocabulary size")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocabulary_size
    )
    if tokenizer_info.vocab_size != vocabulary_size:
        raise ValueError("XGrammar tokenizer vocabulary differs from the model")
    schema = prediction_schema(profile["intent_labels"], profile["slot_labels"])
    schema_json = canonical_bytes(schema).decode("utf-8")
    compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    grammar = xgr.Grammar.from_json_schema(schema_json, any_whitespace=False)
    grammar_text = str(grammar)
    audit_balanced_xgrammar_frontier(
        grammar_text, profile["intent_labels"], "intent"
    )
    audit_balanced_xgrammar_frontier(
        grammar_text, profile["slot_labels"], "slot"
    )
    compiled = compiler.compile_json_schema(schema_json, any_whitespace=False)
    flexible_compiled = compiler.compile_json_schema(
        schema_json, any_whitespace=True
    )

    def render(value):
        # Pinned no-arbitrary-whitespace mode follows json.dumps defaults.
        return json.dumps(value, ensure_ascii=False)

    exemplar_intent = profile["intent_labels"][0]
    for intent in profile["intent_labels"]:
        probe = render({"intent": intent, "slots": []})
        if not xgrammar_accepts_text(xgr, compiled, tokenizer, probe):
            raise ValueError("pinned XGrammar rejected a valid MASSIVE intent")
    for slot in profile["slot_labels"]:
        probe = render(
            {
                "intent": exemplar_intent,
                "slots": [{"name": slot, "value": "x"}],
            }
        )
        if not xgrammar_accepts_text(xgr, compiled, tokenizer, probe):
            raise ValueError("pinned XGrammar rejected a valid MASSIVE slot")
    invalid_intents = (
        "__outside_massive_intent__",
        *RECORDED_LEGACY_HYBRID_INTENT_PROBES,
    )
    invalid_slots = (
        "__outside_massive_slot__",
        *RECORDED_LEGACY_HYBRID_SLOT_PROBES,
    )
    if set(invalid_intents) & set(profile["intent_labels"]) or set(
        invalid_slots
    ) & set(profile["slot_labels"]):
        raise AssertionError("recorded/fabricated matcher probe entered the ontology")
    invalid = tuple(
        {"intent": intent, "slots": []} for intent in invalid_intents
    ) + tuple(
        {
            "intent": exemplar_intent,
            "slots": [{"name": slot, "value": "x"}],
        }
        for slot in invalid_slots
    )
    if any(xgrammar_accepts_text(xgr, compiled, tokenizer, render(value)) for value in invalid):
        raise ValueError("pinned XGrammar admitted an out-of-ontology label")
    whitespace_probes = []
    rendered = render({"intent": exemplar_intent, "slots": []})
    for count in (1, 256):
        whitespace_probes.append(rendered[:-1] + ("\t" * count) + "}")
    for tab_probe in whitespace_probes:
        if not xgrammar_accepts_text(
            xgr, flexible_compiled, tokenizer, tab_probe
        ):
            raise ValueError("pinned flexible XGrammar lost its recorded tab path")
        if xgrammar_accepts_text(xgr, compiled, tokenizer, tab_probe):
            raise ValueError("pinned no-whitespace XGrammar admitted an arbitrary tab")

    def grammar_factory():
        return {
            "matcher": xgr.GrammarMatcher(
                compiled, terminate_without_stop_token=True
            ),
            "bitmask": xgr.allocate_token_bitmask(
                1, tokenizer_info.vocab_size
            ),
            "apply_token_bitmask_inplace": xgr.apply_token_bitmask_inplace,
        }

    # Exercise the pinned direct-loop shape contract on CPU before any GPU work.
    runtime = grammar_factory()
    logp = apply_grammar_mask_then_normalize(
        torch.zeros(vocabulary_size, dtype=torch.float32), runtime
    )
    if logp.shape != (vocabulary_size,) or not bool(torch.isfinite(logp).any()):
        raise ValueError("pinned XGrammar bitmask/logit shape contract differs")
    return {
        "schema": schema,
        "schema_sha256": sha256_bytes(canonical_bytes(schema)),
        "vocab_size": vocabulary_size,
        "intent_leaves_checked": len(profile["intent_labels"]),
        "slot_leaves_checked": len(profile["slot_labels"]),
        "invalid_probes_rejected": len(invalid),
        "recorded_hybrid_intent_probes_rejected": len(
            RECORDED_LEGACY_HYBRID_INTENT_PROBES
        ),
        "recorded_hybrid_slot_probes_rejected": len(
            RECORDED_LEGACY_HYBRID_SLOT_PROBES
        ),
        "flexible_whitespace_probes_reproduced": len(whitespace_probes),
        "whitespace_probes_rejected": len(whitespace_probes),
        "factory": grammar_factory,
    }


def load_tokenizer_and_grammar(profile, base_snapshot):
    from transformers import AutoConfig, PreTrainedTokenizerFast

    snapshot_path = verify_pinned_base_snapshot(base_snapshot)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        snapshot_path,
        local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("pinned tokenizer has no EOS token")
    grammar = compile_and_audit_xgrammar(tokenizer, model_config, profile)
    return tokenizer, model_config, grammar


def load_shared_peft_model(protocol, device, base_snapshot):
    """Load one base plus four named LoRA adapters; caches stay per reference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    snapshot_path = verify_pinned_base_snapshot(base_snapshot)
    base = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    )
    first = PANEL_ORDER[0]
    first_path = protocol["references"][MODEL_NAME_BY_ROLE[first]]["model_path"]
    model = PeftModel.from_pretrained(
        base,
        first_path,
        adapter_name=first,
        is_trainable=False,
    )
    for role in PANEL_ORDER[1:]:
        path = protocol["references"][MODEL_NAME_BY_ROLE[role]]["model_path"]
        model.load_adapter(path, adapter_name=role, is_trainable=False)
    if set(model.peft_config) != set(PANEL_ORDER):
        raise ValueError("loaded PEFT adapter registry differs from the frozen panel")
    model.eval()
    model.config.use_cache = True
    return model


def stop_token_ids(tokenizer, model):
    values = []
    for value in (
        tokenizer.eos_token_id,
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if value is not None:
            values.extend(value if isinstance(value, list) else [value])
    result = {int(value) for value in values if value is not None}
    if not result:
        raise ValueError("pinned tokenizer/model exposes no stop token")
    return result


def sample_shard_name(question_id, ordinal, sample_index):
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:16]
    return f"sample-{ordinal:06d}-{digest}-n{sample_index:03d}.json"


def expected_sample_specs(records, n_samples):
    result = []
    for ordinal, record in enumerate(records):
        for sample_index in range(n_samples):
            result.append(
                {
                    "ordinal": ordinal,
                    "question_id": record["question_id"],
                    "sample_index": sample_index,
                    "prompt_sha256": record["prompt_sha256"],
                    "shard_name": sample_shard_name(
                        record["question_id"], ordinal, sample_index
                    ),
                }
            )
    return result


def method_by_id(method_id):
    if method_id == PAIRED_BASE["method_id"]:
        return dict(PAIRED_BASE)
    for method in METHODS:
        if method["method_id"] == method_id:
            return dict(method)
    raise ValueError(f"unknown exploratory method: {method_id}")


def stream_meta(protocol, phase, method, profile, records):
    method_id = method["method_id"]
    panel = protocol["body"]["model_panel"]
    panel_binding = (
        panel["base"]
        if method_id == "pi_base"
        else {
            "panel_order": panel["panel_order"],
            "references": panel["references"],
        }
    )
    generation_config = {
        "temperature": profile["temperature"],
        "n_samples": profile["n_samples"],
        "max_new_tokens": profile["max_new_tokens"],
        "max_context": profile["max_context"],
        "seed": GENERATION_SEED,
    }
    if profile["domain"] == "massive":
        schema = prediction_schema(profile["intent_labels"], profile["slot_labels"])
        generation_config.update(
            {
                "structured_constraint_profile": STRUCTURED_PROFILE,
                "structured_backend": "xgrammar_direct_token_mask",
                "structured_fallback_allowed": False,
                "xgrammar_version": PINNED_XGRAMMAR_VERSION,
                "xgrammar_any_whitespace": False,
                "grammar_termination": "terminate_without_stop_token",
                "json_schema_sha256": sha256_bytes(canonical_bytes(schema)),
            }
        )
    else:
        generation_config["sampling_profile"] = profile["sampling_profile"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": GENERATION_PROTOCOL,
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "domain": "MASSIVE" if profile["domain"] == "massive" else "medical",
        "method_id": method_id,
        "endpoint": profile["endpoint"],
        "role": profile["role"],
        "protocol_manifest_file_sha256": protocol["file_sha256"],
        "protocol_manifest_payload_sha256": protocol["payload_sha256"],
        "prompt_file_sha256": profile["prompt_file_sha256"],
        "question_ids": [record["question_id"] for record in records],
        "prompt_sha256": [record["prompt_sha256"] for record in records],
        "method": method,
        "model_panel_binding": panel_binding,
        "generation_config": generation_config,
        "backend": "shared_base_transformers_peft_separate_kv_caches",
        "runtime_pins": {
            "torch": PINNED_TORCH_VERSION,
            "transformers": PINNED_TRANSFORMERS_VERSION,
            "peft": PINNED_PEFT_VERSION,
            "xgrammar": PINNED_XGRAMMAR_VERSION,
        },
        "is_paired_base": method_id == "pi_base",
        "same_transformers_backend_as_paired_base": True,
    }


def write_or_audit(path, expected, seal_field=OUTPUT_SEAL_FIELD):
    expected = seal(expected, seal_field)
    if os.path.isfile(path) and not os.path.islink(path):
        observed, _ = load_json_regular(path, f"sealed output {path}")
        verify_seal(observed, seal_field, f"sealed output {path}")
        if observed != expected:
            raise ValueError(f"existing sealed output differs: {path}")
        return observed
    if os.path.lexists(path):
        raise ValueError(f"refusing unsafe/nonregular output path: {path}")
    atomic_write_json(path, expected)
    return expected


def audit_shard(path, stream_payload_sha256, spec):
    payload, _ = load_json_regular(path, "generation sample shard")
    body = verify_seal(payload, OUTPUT_SEAL_FIELD, "generation sample shard")
    if set(body) != {"stream_payload_sha256", "spec", "sample", "generation_seconds"}:
        raise ValueError("generation sample shard has unexpected fields")
    if body["stream_payload_sha256"] != stream_payload_sha256 or body["spec"] != spec:
        raise ValueError("generation sample shard provenance differs")
    sample = body["sample"]
    seconds = body["generation_seconds"]
    if (
        not isinstance(sample, dict)
        or sample.get("question_id") != spec["question_id"]
        or sample.get("sample_index") != spec["sample_index"]
        or sample.get("prompt_sha256") != spec["prompt_sha256"]
        or sample.get("sample_sha256") != sample_sha256(sample)
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or seconds < 0
    ):
        raise ValueError("generation sample shard content differs")
    return sample, float(seconds)


def audit_stream_directory(stream_root, expected_names):
    if os.path.islink(stream_root) or not os.path.isdir(stream_root):
        raise ValueError(f"unsafe generation stream directory: {stream_root}")
    observed = set(os.listdir(stream_root))
    if observed - set(expected_names):
        raise ValueError(
            f"generation stream has unexpected entries: {sorted(observed - set(expected_names))}"
        )


def assemble_stream(stream_root, meta, specs, require_final):
    manifest_path = os.path.join(stream_root, "stream_manifest.json")
    manifest_body = {
        "schema_version": SCHEMA_VERSION,
        "protocol": GENERATION_PROTOCOL,
        "meta": meta,
        "sample_specs": specs,
    }
    if require_final:
        if not os.path.isfile(manifest_path) or os.path.islink(manifest_path):
            raise ValueError("generation stream manifest is missing or unsafe")
        manifest_payload, _ = load_json_regular(
            manifest_path, "generation stream manifest"
        )
        verify_seal(
            manifest_payload, OUTPUT_SEAL_FIELD, "generation stream manifest"
        )
        if manifest_payload != seal(manifest_body):
            raise ValueError("generation stream manifest differs")
    else:
        manifest_payload = write_or_audit(manifest_path, manifest_body)
    shards_root = os.path.join(stream_root, "shards")
    if not os.path.isdir(shards_root) or os.path.islink(shards_root):
        if os.path.lexists(shards_root):
            raise ValueError("generation shard root is unsafe")
        if require_final:
            raise ValueError("generation shard root is missing")
        os.makedirs(shards_root)
    expected_names = {spec["shard_name"] for spec in specs}
    extras = set(os.listdir(shards_root)) - expected_names
    if extras:
        raise ValueError(f"generation shard root has unexpected entries: {sorted(extras)}")
    present = [
        os.path.isfile(os.path.join(shards_root, spec["shard_name"]))
        and not os.path.islink(os.path.join(shards_root, spec["shard_name"]))
        for spec in specs
    ]
    first_missing = next((index for index, value in enumerate(present) if not value), len(specs))
    if any(present[first_missing + 1 :]):
        raise ValueError("generation resume shards are not an exact contiguous prefix")
    samples, seconds = [], []
    for index, spec in enumerate(specs):
        shard_path = os.path.join(shards_root, spec["shard_name"])
        if not present[index]:
            if require_final:
                raise ValueError(f"generation shard is missing: {shard_path}")
            return manifest_payload, samples, seconds, spec
        sample, elapsed = audit_shard(
            shard_path, manifest_payload[OUTPUT_SEAL_FIELD], spec
        )
        samples.append(sample)
        seconds.append(elapsed)
    expected_generation = seal({"meta": meta, "samples": samples})
    generation_path = os.path.join(stream_root, "generation.json")
    if os.path.isfile(generation_path) and not os.path.islink(generation_path):
        observed, _ = load_json_regular(generation_path, "sealed generation")
        verify_seal(observed, OUTPUT_SEAL_FIELD, "sealed generation")
        if observed != expected_generation:
            raise ValueError("existing assembled generation differs from its shards")
    elif require_final:
        raise ValueError("assembled generation is missing")
    else:
        if os.path.lexists(generation_path):
            raise ValueError("assembled generation output is unsafe")
        atomic_write_json(generation_path, expected_generation)
    audit_stream_directory(
        stream_root, {"stream_manifest.json", "shards", "generation.json"}
    )
    return manifest_payload, samples, seconds, None


def run_stream(
    *,
    output_root,
    protocol,
    phase,
    method,
    profile,
    records,
    model,
    tokenizer,
    device,
    stop_ids,
    grammar_factory,
):
    method_id = method["method_id"]
    stream_root = os.path.join(output_root, phase, method_id, profile["domain"])
    if os.path.lexists(stream_root) and (
        os.path.islink(stream_root) or not os.path.isdir(stream_root)
    ):
        raise ValueError(f"unsafe generation stream root: {stream_root}")
    os.makedirs(stream_root, exist_ok=True)
    meta = stream_meta(protocol, phase, method, profile, records)
    specs = expected_sample_specs(records, profile["n_samples"])
    manifest_payload, samples, seconds, missing = assemble_stream(
        stream_root, meta, specs, require_final=False
    )
    by_identity = {
        (record["question_id"], sample_index): (record, sample_index)
        for record in records
        for sample_index in range(profile["n_samples"])
    }
    start_index = len(samples)
    for spec in specs[start_index:]:
        record, sample_index = by_identity[
            (spec["question_id"], spec["sample_index"])
        ]
        started = time.perf_counter()
        prompt_ids = make_prompt_ids(tokenizer, record)
        if len(prompt_ids) + profile["max_new_tokens"] > profile["max_context"]:
            raise ValueError(
                f"prompt plus frozen generation budget exceeds context: {record['question_id']}"
            )
        sample = generate_sample(
            record=record,
            sample_index=sample_index,
            prompt_ids=prompt_ids,
            model=model,
            tokenizer=tokenizer,
            method=method,
            profile=profile,
            device=device,
            stop_ids=stop_ids,
            grammar_factory=grammar_factory,
        )
        elapsed = time.perf_counter() - started
        shard = seal(
            {
                "stream_payload_sha256": manifest_payload[OUTPUT_SEAL_FIELD],
                "spec": spec,
                "sample": sample,
                "generation_seconds": elapsed,
            }
        )
        shard_path = os.path.join(stream_root, "shards", spec["shard_name"])
        if os.path.lexists(shard_path):
            raise ValueError("refusing to overwrite an existing generation shard")
        atomic_write_json(shard_path, shard)
    _, samples, seconds, missing = assemble_stream(
        stream_root, meta, specs, require_final=False
    )
    if missing is not None:
        raise RuntimeError("generation stream did not complete all frozen samples")
    return {
        "method_id": method_id,
        "domain": profile["domain"],
        "stream_root": os.path.abspath(stream_root),
        "generation_path": os.path.abspath(os.path.join(stream_root, "generation.json")),
        "samples": len(samples),
        "generated_tokens": sum(sample["generated_tokens"] for sample in samples),
        "generation_seconds": sum(seconds),
        "selected_tokens_per_second": (
            sum(sample["generated_tokens"] for sample in samples) / sum(seconds)
            if sum(seconds) > 0
            else None
        ),
    }


def audit_stream(output_root, protocol, phase, method, profile, records):
    stream_root = os.path.join(
        output_root, phase, method["method_id"], profile["domain"]
    )
    meta = stream_meta(protocol, phase, method, profile, records)
    specs = expected_sample_specs(records, profile["n_samples"])
    _, samples, seconds, missing = assemble_stream(
        stream_root, meta, specs, require_final=True
    )
    if missing is not None:
        raise AssertionError("require_final returned an incomplete stream")
    return {
        "method_id": method["method_id"],
        "domain": profile["domain"],
        "stream_root": os.path.abspath(stream_root),
        "generation_path": os.path.abspath(os.path.join(stream_root, "generation.json")),
        "samples": len(samples),
        "generated_tokens": sum(sample["generated_tokens"] for sample in samples),
        "generation_seconds": sum(seconds),
        "selected_tokens_per_second": (
            sum(sample["generated_tokens"] for sample in samples) / sum(seconds)
            if sum(seconds) > 0
            else None
        ),
    }


def stream_plan(phase, massive_profile, massive_records, medical_profile=None, medical_records=None):
    plan = [
        (method_by_id("pi_base"), massive_profile, massive_records),
        *[(dict(method), massive_profile, massive_records) for method in METHODS],
    ]
    if phase == "confirmation":
        if medical_profile is None or medical_records is None:
            raise ValueError("confirmation requires the frozen medical prompt bank")
        plan.extend(
            (dict(method), medical_profile, medical_records) for method in METHODS
        )
    return plan


def build_timing_record(protocol, phase, setup_seconds, streams, cache_probe):
    cache_probe = audit_cache_equivalence_probe(cache_probe, phase)
    by_key = {
        f"{stream['method_id']}:{stream['domain']}": {
            key: value
            for key, value in stream.items()
            if key not in {"stream_root", "generation_path"}
        }
        for stream in streams
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "massive_medical_union_composition_exploratory_timings_v1",
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "protocol_manifest_file_sha256": protocol["file_sha256"],
        "protocol_manifest_payload_sha256": protocol["payload_sha256"],
        "setup_seconds": setup_seconds,
        "cache_equivalence_probe": cache_probe,
        "streams": by_key,
        "paired_base_generation_recorded_separately": True,
        "projection_formula": protocol["body"]["runtime_projection"]["formula"],
        "projection_owned_by_smoke_evaluator_after_score_and_seal": True,
    }
    if phase == "smoke":
        massive = {
            stream["method_id"]: stream
            for stream in streams
            if stream["domain"] == "massive"
        }
        expected = {"pi_base", *(method["method_id"] for method in METHODS)}
        if set(massive) != expected:
            raise ValueError("smoke timing lacks one of the four frozen streams")
        method_rates = []
        for method in METHODS:
            stream = massive[method["method_id"]]
            if stream["generation_seconds"] <= 0 or stream["generated_tokens"] <= 0:
                raise ValueError("cannot project confirmation from zero smoke throughput")
            method_rates.append(
                stream["generated_tokens"] / stream["generation_seconds"]
            )
        minimum_rate = min(method_rates)
        body.update(
            {
                "smoke_generation_multiplier_per_stream": 10,
                "smoke_generation_total_multiplier": 40,
                "minimum_method_selected_tokens_per_second": minimum_rate,
                "smoke_score_and_seal_seconds": None,
                "projected_confirmation_seconds": None,
            }
        )
    return body


def audit_timing_record(path, protocol, phase, streams):
    payload, _ = load_json_regular(path, "phase timings")
    body = verify_seal(payload, OUTPUT_SEAL_FIELD, "phase timings")
    setup_body = load_setup_timing(
        os.path.join(os.path.dirname(path), "setup_timing.json"), protocol, phase
    )
    expected_prefix = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "massive_medical_union_composition_exploratory_timings_v1",
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "protocol_manifest_file_sha256": protocol["file_sha256"],
        "protocol_manifest_payload_sha256": protocol["payload_sha256"],
        "cache_equivalence_probe": setup_body["cache_equivalence_probe"],
        "paired_base_generation_recorded_separately": True,
        "projection_formula": protocol["body"]["runtime_projection"]["formula"],
        "projection_owned_by_smoke_evaluator_after_score_and_seal": True,
    }
    for key, value in expected_prefix.items():
        if body.get(key) != value:
            raise ValueError(f"phase timing differs on {key}")
    if (
        isinstance(body.get("setup_seconds"), bool)
        or not isinstance(body.get("setup_seconds"), (int, float))
        or body["setup_seconds"] < 0
        or not isinstance(body.get("runtime_versions"), dict)
    ):
        raise ValueError("phase timing has invalid setup/runtime metadata")
    pre = body.get("pre_generation_setup_seconds")
    post = body.get("post_generation_artifact_audit_seconds")
    if (
        isinstance(pre, bool)
        or not isinstance(pre, (int, float))
        or pre < 0
        or isinstance(post, bool)
        or not isinstance(post, (int, float))
        or post < 0
        or body["setup_seconds"] != pre + post
    ):
        raise ValueError("phase timing fixed-overhead decomposition differs")
    expected_streams = {
        f"{stream['method_id']}:{stream['domain']}": {
            key: value
            for key, value in stream.items()
            if key not in {"stream_root", "generation_path"}
        }
        for stream in streams
    }
    if body.get("streams") != expected_streams:
        raise ValueError("phase timing stream measurements differ from sample shards")
    return body


@contextlib.contextmanager
def exclusive_phase_lock(output_root, phase):
    import fcntl

    phase_root = os.path.join(os.path.abspath(output_root), phase)
    if os.path.lexists(phase_root) and (
        os.path.islink(phase_root) or not os.path.isdir(phase_root)
    ):
        raise ValueError(f"unsafe phase output root: {phase_root}")
    os.makedirs(phase_root, exist_ok=True)
    lock_path = os.path.join(phase_root, ".sampler.lock")
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another sampler owns the exact phase output") from error
        yield phase_root
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def load_setup_timing(path, protocol, phase):
    payload, _ = load_json_regular(path, "setup timing")
    body = verify_seal(payload, OUTPUT_SEAL_FIELD, "setup timing")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": GENERATION_PROTOCOL,
        "phase": phase,
        "protocol_manifest_file_sha256": protocol["file_sha256"],
        "protocol_manifest_payload_sha256": protocol["payload_sha256"],
    }
    if set(body) != {*expected, "setup_seconds", "cache_equivalence_probe"}:
        raise ValueError("setup timing schema differs")
    for key, value in expected.items():
        if body.get(key) != value:
            raise ValueError("setup timing provenance differs")
    observed = body.get("setup_seconds")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or observed < 0
    ):
        raise ValueError("setup timing value is invalid")
    audit_cache_equivalence_probe(body.get("cache_equivalence_probe"), phase)
    return body


def write_or_audit_setup_timing(
    phase_root, protocol, phase, seconds, cache_probe
):
    cache_probe = audit_cache_equivalence_probe(cache_probe, phase)
    path = os.path.join(phase_root, "setup_timing.json")
    if os.path.isfile(path) and not os.path.islink(path):
        body = load_setup_timing(path, protocol, phase)
        observed_probe = body["cache_equivalence_probe"]
        nondeterministic = {"comparisons", "probe_seconds"}
        if any(
            observed_probe[key] != cache_probe[key]
            for key in observed_probe
            if key not in nondeterministic
        ):
            raise ValueError("repeated cache-equivalence probe provenance differs")
        return float(body["setup_seconds"]), observed_probe
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol": GENERATION_PROTOCOL,
        "phase": phase,
        "protocol_manifest_file_sha256": protocol["file_sha256"],
        "protocol_manifest_payload_sha256": protocol["payload_sha256"],
        "setup_seconds": seconds,
        "cache_equivalence_probe": cache_probe,
    }
    write_or_audit(path, body)
    return float(seconds), cache_probe


def run_phase(args):
    phase_started = time.perf_counter()
    if args.device != "cuda:0":
        raise ValueError("exploratory protocol freezes the sole device as cuda:0")
    force_offline_environment()
    protocol = load_protocol_manifest(args.protocol_manifest, audit_models=True)
    massive_profile, massive_records = load_massive_prompts(protocol, args.phase)
    medical_profile = medical_records = None
    if args.phase == "confirmation":
        medical_profile, medical_records = load_medical_prompts(protocol)
    plan = stream_plan(
        args.phase,
        massive_profile,
        massive_records,
        medical_profile,
        medical_records,
    )

    if args.preflight_only:
        runtime = require_pinned_runtime(require_cuda=False)
        base_snapshot = resolve_pinned_base_snapshot()
        _, _, grammar = load_tokenizer_and_grammar(massive_profile, base_snapshot)
        result = {
            "status": "CPU_PREFLIGHT_OK",
            "runtime": runtime,
            "base_model_snapshot": base_snapshot,
            "schema_sha256": grammar["schema_sha256"],
            "intent_leaves_checked": grammar["intent_leaves_checked"],
            "slot_leaves_checked": grammar["slot_leaves_checked"],
            "invalid_probes_rejected": grammar["invalid_probes_rejected"],
            "recorded_hybrid_intent_probes_rejected": grammar[
                "recorded_hybrid_intent_probes_rejected"
            ],
            "recorded_hybrid_slot_probes_rejected": grammar[
                "recorded_hybrid_slot_probes_rejected"
            ],
            "flexible_whitespace_probes_reproduced": grammar[
                "flexible_whitespace_probes_reproduced"
            ],
            "whitespace_probes_rejected": grammar[
                "whitespace_probes_rejected"
            ],
        }
        print(json.dumps(result, sort_keys=True))
        return 0

    output_root = os.path.abspath(args.output_root)
    if os.path.lexists(output_root) and (
        os.path.islink(output_root) or not os.path.isdir(output_root)
    ):
        raise ValueError(f"unsafe sampler output root: {output_root}")
    os.makedirs(output_root, exist_ok=True)
    with exclusive_phase_lock(output_root, args.phase) as phase_root:
        run_manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol": GENERATION_PROTOCOL,
            "phase": args.phase,
            "protocol_manifest_file_sha256": protocol["file_sha256"],
            "protocol_manifest_payload_sha256": protocol["payload_sha256"],
            "streams": [
                {
                    "method_id": method["method_id"],
                    "domain": profile["domain"],
                    "samples": len(records) * profile["n_samples"],
                }
                for method, profile, records in plan
            ],
        }
        write_or_audit(os.path.join(phase_root, "run_manifest.json"), run_manifest)
        if args.audit_only:
            streams = [
                audit_stream(
                    output_root, protocol, args.phase, method, profile, records
                )
                for method, profile, records in plan
            ]
            timing_path = os.path.join(phase_root, "timings.json")
            audit_timing_record(
                timing_path, protocol, args.phase, streams
            )
            print(json.dumps({"status": "AUDIT_OK", "streams": len(streams)}))
            return 0

        timing_path = os.path.join(phase_root, "timings.json")
        if os.path.isfile(timing_path) and not os.path.islink(timing_path):
            streams = [
                audit_stream(
                    output_root, protocol, args.phase, method, profile, records
                )
                for method, profile, records in plan
            ]
            audit_timing_record(timing_path, protocol, args.phase, streams)
            print(
                json.dumps(
                    {
                        "status": "EXACT_RESUME_ALREADY_COMPLETE",
                        "phase": args.phase,
                        "streams": len(streams),
                    },
                    sort_keys=True,
                )
            )
            return 0

        runtime_versions = require_pinned_runtime(require_cuda=True)
        import torch

        torch.manual_seed(GENERATION_SEED)
        torch.cuda.manual_seed_all(GENERATION_SEED)
        base_snapshot = resolve_pinned_base_snapshot()
        tokenizer, model_config, grammar = load_tokenizer_and_grammar(
            massive_profile, base_snapshot
        )
        model = load_shared_peft_model(protocol, args.device, base_snapshot)
        if getattr(model.config, "vocab_size", None) != grammar["vocab_size"]:
            raise ValueError("loaded base model vocabulary differs from CPU preflight")
        stops = stop_token_ids(tokenizer, model)
        # This executes before run_stream can create the first scientific shard.
        # It probes the first frozen MASSIVE prefix in either phase, using one
        # fixed tokenizer-derived continuation token for all adapters and base.
        cache_probe = run_cache_equivalence_probe(
            model,
            tokenizer,
            massive_records[0],
            args.phase,
            args.device,
        )
        setup_seconds, cache_probe = write_or_audit_setup_timing(
            phase_root,
            protocol,
            args.phase,
            time.perf_counter() - phase_started,
            cache_probe,
        )
        streams = []
        for method, profile, records in plan:
            streams.append(
                run_stream(
                    output_root=output_root,
                    protocol=protocol,
                    phase=args.phase,
                    method=method,
                    profile=profile,
                    records=records,
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    stop_ids=stops,
                    grammar_factory=(
                        grammar["factory"] if profile["domain"] == "massive" else None
                    ),
                )
            )
        # Re-read every model artifact after generation to detect in-run mutation.
        post_audit_started = time.perf_counter()
        for role in PANEL_ORDER:
            name = MODEL_NAME_BY_ROLE[role]
            audit_reference_binding(name, protocol["references"][name])
        post_audit_seconds = time.perf_counter() - post_audit_started
        timing_body = build_timing_record(
            protocol,
            args.phase,
            setup_seconds + post_audit_seconds,
            streams,
            cache_probe,
        )
        timing_body["pre_generation_setup_seconds"] = setup_seconds
        timing_body["post_generation_artifact_audit_seconds"] = post_audit_seconds
        timing_body["runtime_versions"] = runtime_versions
        write_or_audit(os.path.join(phase_root, "timings.json"), timing_body)
        print(
            json.dumps(
                {
                    "status": "GENERATION_COMPLETE",
                    "phase": args.phase,
                    "streams": len(streams),
                    "timings": os.path.join(phase_root, "timings.json"),
                },
                sort_keys=True,
            )
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("smoke", "confirmation"))
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    return run_phase(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
