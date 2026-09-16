#!/usr/bin/env python3
"""No-resume sampler for the two changed fixed-k=4 MASSIVE/medical panels.

Frozen numerical/cache/grammar primitives are duplicated byte-for-byte from the
original sequential sampler by repository policy. The sole generate_sample
exception preserves a budget-truncated MASSIVE response with prediction=None;
its token-selection/cache loop is unchanged. Authorization, panel binding,
direct-reference decoding, and artifact orchestration are new.
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
import math
import subprocess
import sys
from pathlib import Path


SCHEMA_VERSION = 1
PROTOCOL_ID = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_v1"
)
SOURCE_PROTOCOL_ID = "massive_medical_union_composition_exploratory_v1"
GENERATION_PROTOCOL = PROTOCOL_ID
TIMING_PROTOCOL = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_timings_v1"
)
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
BENEFIT_SELECTION_ALGORITHM = "sha256_utf8_nul_domain_rank_v1"
BENEFIT_SELECTION_RANKING_MATERIAL = (
    "protocol_id + NUL + 'benefit360' + NUL + question_id"
)
BENEFIT_RANKED_IDS_SHA256 = (
    "c5c3a6a2cc09aa9103dc593c7a14fa1853429a74b89b41deeb39481f52c903eb"
)
BENEFIT_SOURCE_ORDER_IDS_SHA256 = (
    "ac5dec7a70ff616a73bd1a00ed7c7e03f506afb03f6232b83299f2b1474880e6"
)
BENEFIT_RANK_RECORDS_SHA256 = (
    "10cc94525d5953b8bdabbb5f55f2720fdf2d90e9c78c28983594ea509e498bea"
)
CACHE_EQUIVALENCE_PROBE_PROTOCOL = (
    "massive_medical_union_composition_cache_equivalence_probe_v3"
)
CACHE_EQUIVALENCE_CONTINUATION_TEXT = "."
INDEPENDENT_MODEL_ORDER = ("R1", "R2", "R3", "R4", "base")
INDEPENDENT_MODEL_BACKEND = (
    "independent_transformers_peft_models_separate_kv_caches"
)
CACHE_PROBE_MINIMUM_TOTAL_MEMORY_BYTES = 120 * 1024**3
CACHE_PROBE_MINIMUM_FREE_MEMORY_BYTES = 32 * 1024**3
CACHE_DIAGNOSTIC_TOP_K = 10
CACHE_LEGACY_DIAGNOSTIC_ATOL = 1e-3
CACHE_LEGACY_DIAGNOSTIC_RTOL = 1e-3
PANEL_ORDER = ("R1", "R2", "R3", "R4")
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
BENEFIT_PROFILE = {
    "artifact": "benefit/prompts.json",
    "role": "sequential_benefit_confirmation",
    "rows": 360,
    "n_samples": 1,
    "temperature": 0.0,
    "max_new_tokens": 256,
    "max_context": 2048,
}
MEDICAL_PROFILE = {
    "artifact": "medical/prompts.json",
    "role": "sequential_medical_confirmation",
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


def forward_cached(model, input_ids, attention_mask, cache):
    """Run one already-isolated model; scientific inference never switches adapters."""
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )


def prefill_cached_reference(model, prompt_ids, device):
    import torch

    if not prompt_ids:
        raise ValueError("cannot prefill an empty prompt")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        outputs = forward_cached(model, input_ids, attention_mask, cache=None)
    logits, cache = extract_logits_and_cache(outputs)
    length = cache_sequence_length(cache)
    if length != len(prompt_ids):
        raise ValueError(
            f"prefill cache length {length} != prompt length {len(prompt_ids)}"
        )
    return {"next_logits": logits[0, -1, :].float(), "cache": cache}


def step_cached_reference(model, token_id, cache, device):
    import torch

    previous = cache_sequence_length(cache)
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, previous + 1), dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = forward_cached(model, input_ids, attention_mask, cache=cache)
    logits, next_cache = extract_logits_and_cache(outputs)
    observed = cache_sequence_length(next_cache)
    if observed != previous + 1:
        raise ValueError(
            f"one-token cache step grew from {previous} to {observed}, expected {previous + 1}"
        )
    return {"next_logits": logits[0, -1, :].float(), "cache": next_cache}


def assert_independent_caches(states):
    caches = [state["cache"] for state in states]
    if len({id(cache) for cache in caches}) != len(caches):
        raise ValueError("references unexpectedly share a mutable KV-cache object")


def cache_tensor_inventory(cache):
    """Return ordered ``(path, tensor)`` pairs for supported HF cache layouts."""
    import torch

    result = []
    layers = getattr(cache, "layers", None)
    if isinstance(layers, (tuple, list)) and layers:
        for index, layer in enumerate(layers):
            for attribute in ("keys", "values"):
                tensor = getattr(layer, attribute, None)
                if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 0:
                    raise ValueError(
                        f"cache tensor is missing: layers.{index}.{attribute}"
                    )
                result.append((f"layers.{index}.{attribute}", tensor))
    elif isinstance(getattr(cache, "key_cache", None), (tuple, list)) and isinstance(
        getattr(cache, "value_cache", None), (tuple, list)
    ):
        keys = cache.key_cache
        values = cache.value_cache
        if not keys or len(keys) != len(values):
            raise ValueError("legacy cache key/value tensor counts differ")
        for index, (key, value) in enumerate(zip(keys, values)):
            for attribute, tensor in (("keys", key), ("values", value)):
                if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 0:
                    raise ValueError(
                        f"cache tensor is missing: layers.{index}.{attribute}"
                    )
                result.append((f"layers.{index}.{attribute}", tensor))
    elif isinstance(cache, (tuple, list)) and cache:
        for index, layer in enumerate(cache):
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise ValueError("legacy tuple cache layer differs")
            for attribute, tensor in (("keys", layer[0]), ("values", layer[1])):
                if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 0:
                    raise ValueError(
                        f"cache tensor is missing: layers.{index}.{attribute}"
                    )
                result.append((f"layers.{index}.{attribute}", tensor))
    if not result:
        raise ValueError(f"unrecognized cache tensor layout: {type(cache).__name__}")
    paths = [path for path, _ in result]
    if len(paths) != len(set(paths)):
        raise ValueError("cache tensor inventory contains duplicate paths")
    return result


def cache_tensor_storage_pointers(cache):
    """Return exact tensor-storage identities for a supported HF cache."""
    pointers = set()
    for _, tensor in cache_tensor_inventory(cache):
        storage = tensor.untyped_storage()
        pointer = (str(tensor.device), int(storage.data_ptr()))
        if pointer in pointers:
            raise ValueError("one cache unexpectedly shares tensor storage internally")
        pointers.add(pointer)
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


def fresh_full_prefix_next_logits(model, prefix_ids, device):
    """Evaluate one complete prefix without reusing a caller-owned KV cache."""
    import torch

    if not prefix_ids:
        raise ValueError("cache-equivalence full prefix is empty")
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.inference_mode():
        outputs = forward_cached(model, input_ids, attention_mask, cache=None)
    logits, cache = extract_logits_and_cache(outputs)
    if cache_sequence_length(cache) != len(prefix_ids):
        raise ValueError("cache-equivalence fresh full-prefix cache length differs")
    result = logits[0, -1, :].float()
    if result.ndim != 1 or not bool(torch.isfinite(result).all().item()):
        raise ValueError("cache-equivalence full-prefix logits are invalid")
    return result


def active_adapter_names(model):
    value = getattr(model, "active_adapters", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(model, "active_adapter", None)
        if callable(value):
            value = value()
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, str) and item for item in value
    ):
        return list(value)
    raise ValueError("independent PEFT model has malformed active adapters")


def parameter_storage_inventory(model, role, expected_device):
    import torch

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise ValueError(f"independent model lacks named_parameters for {role}")
    tensor_count = 0
    parameter_numel = 0
    storage_pointers = set()
    devices = set()
    dtypes = set()
    for name, parameter in named_parameters():
        if not isinstance(name, str) or not name or not isinstance(parameter, torch.Tensor):
            raise ValueError(f"independent model parameter registry differs for {role}")
        if parameter.numel() <= 0:
            continue
        tensor_count += 1
        parameter_numel += int(parameter.numel())
        devices.add(str(parameter.device))
        dtypes.add(str(parameter.dtype))
        storage = parameter.untyped_storage()
        storage_pointers.add((str(parameter.device), int(storage.data_ptr())))
    if (
        tensor_count <= 0
        or parameter_numel <= 0
        or not storage_pointers
        or devices != {expected_device}
        or not dtypes
    ):
        raise ValueError(f"independent model parameter placement differs for {role}")
    return {
        "parameter_tensor_count": tensor_count,
        "parameter_numel": parameter_numel,
        "parameter_storage_count": len(storage_pointers),
        "parameter_devices": sorted(devices),
        "parameter_dtypes": sorted(dtypes),
    }, storage_pointers


def audit_independent_model_panel(models, expected_device):
    import torch

    roles = list(INDEPENDENT_MODEL_ORDER)
    if not isinstance(models, dict) or list(models) != roles:
        raise ValueError("independent model panel order differs")
    if len({id(models[role]) for role in roles}) != len(roles):
        raise ValueError("independent model panel shares a model object")
    evidence = {}
    pointer_sets = {}
    for role in roles:
        model = models[role]
        config = getattr(model, "config", None)
        get_embeddings = getattr(model, "get_input_embeddings", None)
        embeddings = get_embeddings() if callable(get_embeddings) else None
        weight = getattr(embeddings, "weight", None)
        if (
            getattr(config, "_attn_implementation", None) != "sdpa"
            or not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
            or getattr(model, "training", False) is not False
        ):
            raise ValueError(
                f"independent model requires the frozen eval BF16/SDPA backend for {role}"
            )
        peft_config = getattr(model, "peft_config", None)
        if role == "base":
            # Transformers' plain PreTrainedModel exposes PeftAdapterMixin's
            # ``active_adapters()`` accessor, but that accessor raises when no
            # adapter has ever been loaded.  Prove direct-base identity from
            # its registry/flag and do not invoke the PEFT-only accessor.
            if peft_config not in (None, {}) or bool(
                getattr(model, "_hf_peft_config_loaded", False)
            ):
                raise ValueError("direct base unexpectedly exposes a PEFT adapter")
            model_kind = "direct_base"
            expected_adapter = None
            configured_adapters = []
            active_adapters = []
        else:
            if not isinstance(peft_config, dict):
                raise ValueError(f"independent PEFT registry is missing for {role}")
            configured_adapters = list(peft_config)
            active_adapters = active_adapter_names(model)
            if configured_adapters != [role] or active_adapters != [role]:
                raise ValueError(f"independent PEFT adapter identity differs for {role}")
            model_kind = "peft_single_adapter"
            expected_adapter = role
        parameter_evidence, pointers = parameter_storage_inventory(
            model, role, expected_device
        )
        pointer_sets[role] = pointers
        evidence[role] = {
            "model_kind": model_kind,
            "expected_adapter": expected_adapter,
            "active_adapters": active_adapters,
            "peft_config_adapters": configured_adapters,
            "object_unique": True,
            **parameter_evidence,
            "parameter_storage_disjoint_from_other_models": True,
        }
    seen = set()
    for role in roles:
        if seen & pointer_sets[role]:
            raise ValueError(f"independent models share parameter storage at {role}")
        seen.update(pointer_sets[role])
    return evidence


def read_gpu_memory(device):
    import torch

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda" or cuda_device.index not in (None, 0):
        raise ValueError("independent model probe requires cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    return {
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(cuda_device),
        "total_memory_bytes": int(total_bytes),
        "free_memory_bytes": int(free_bytes),
        "allocated_memory_bytes": int(torch.cuda.memory_allocated(cuda_device)),
        "reserved_memory_bytes": int(torch.cuda.memory_reserved(cuda_device)),
        "peak_allocated_memory_bytes": int(
            torch.cuda.max_memory_allocated(cuda_device)
        ),
    }


def build_gpu_memory_evidence(before, after):
    snapshot_keys = {
        "device",
        "device_name",
        "total_memory_bytes",
        "free_memory_bytes",
        "allocated_memory_bytes",
        "reserved_memory_bytes",
        "peak_allocated_memory_bytes",
    }
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or set(before) != snapshot_keys
        or set(after) != snapshot_keys
        or before["device"] != after["device"]
        or before["device_name"] != after["device_name"]
        or before["total_memory_bytes"] != after["total_memory_bytes"]
    ):
        raise ValueError("independent model GPU memory snapshots differ")
    numeric = (
        "total_memory_bytes",
        "free_memory_bytes",
        "allocated_memory_bytes",
        "reserved_memory_bytes",
        "peak_allocated_memory_bytes",
    )
    for snapshot in (before, after):
        if (
            not isinstance(snapshot.get("device"), str)
            or not snapshot["device"]
            or not isinstance(snapshot.get("device_name"), str)
            or not snapshot["device_name"]
            or any(
                isinstance(snapshot.get(key), bool)
                or not isinstance(snapshot.get(key), int)
                or snapshot[key] < 0
                for key in numeric
            )
            or snapshot["allocated_memory_bytes"]
            > snapshot["reserved_memory_bytes"]
            or snapshot["reserved_memory_bytes"] > snapshot["total_memory_bytes"]
            or snapshot["free_memory_bytes"] > snapshot["total_memory_bytes"]
            or snapshot["peak_allocated_memory_bytes"]
            < snapshot["allocated_memory_bytes"]
        ):
            raise ValueError("independent model GPU memory snapshot is invalid")
    total_ok = before["total_memory_bytes"] >= CACHE_PROBE_MINIMUM_TOTAL_MEMORY_BYTES
    before_ok = (
        before["free_memory_bytes"] >= CACHE_PROBE_MINIMUM_FREE_MEMORY_BYTES
    )
    after_ok = after["free_memory_bytes"] >= CACHE_PROBE_MINIMUM_FREE_MEMORY_BYTES
    if not total_ok or not before_ok or not after_ok:
        raise ValueError("independent model GPU memory/headroom contract failed")
    return {
        "device": before["device"],
        "device_name": before["device_name"],
        "minimum_total_memory_bytes": CACHE_PROBE_MINIMUM_TOTAL_MEMORY_BYTES,
        "minimum_free_memory_bytes_before_probe": (
            CACHE_PROBE_MINIMUM_FREE_MEMORY_BYTES
        ),
        "minimum_free_memory_bytes_after_probe": (
            CACHE_PROBE_MINIMUM_FREE_MEMORY_BYTES
        ),
        "total_memory_bytes": before["total_memory_bytes"],
        "free_memory_bytes_before_probe": before["free_memory_bytes"],
        "allocated_memory_bytes_before_probe": before[
            "allocated_memory_bytes"
        ],
        "reserved_memory_bytes_before_probe": before["reserved_memory_bytes"],
        "free_memory_bytes_after_probe": after["free_memory_bytes"],
        "allocated_memory_bytes_after_probe": after["allocated_memory_bytes"],
        "reserved_memory_bytes_after_probe": after["reserved_memory_bytes"],
        "peak_allocated_memory_bytes_after_probe": after[
            "peak_allocated_memory_bytes"
        ],
        "total_memory_requirement_met": True,
        "free_memory_before_requirement_met": True,
        "free_memory_after_requirement_met": True,
        "headroom_requirement_met": True,
    }


def audit_probe_model_backends(models):
    import torch

    if not isinstance(models, dict) or list(models) != list(INDEPENDENT_MODEL_ORDER):
        raise ValueError("cache probe independent model panel differs")
    for role, model in models.items():
        config = getattr(model, "config", None)
        get_embeddings = getattr(model, "get_input_embeddings", None)
        embeddings = get_embeddings() if callable(get_embeddings) else None
        weight = getattr(embeddings, "weight", None)
        if (
            getattr(config, "_attn_implementation", None) != "sdpa"
            or not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
        ):
            raise ValueError(
                f"cache probe requires the frozen BF16/SDPA model backend for {role}"
            )
    return {"model_compute_dtype": "bfloat16", "attention_implementation": "sdpa"}


def run_independent_cached_probe(models, prompt_ids, token_id, device):
    states = {
        role: prefill_cached_reference(models[role], prompt_ids, device)
        for role in INDEPENDENT_MODEL_ORDER
    }
    for role in INDEPENDENT_MODEL_ORDER:
        states[role] = step_cached_reference(
            models[role], token_id, states[role]["cache"], device
        )
    return states


def cached_vs_full_prefix_diagnostic(cached, fresh):
    import torch
    import torch.nn.functional as functional

    if (
        cached.dtype != torch.float32
        or fresh.dtype != torch.float32
        or cached.ndim != 1
        or cached.shape != fresh.shape
        or cached.numel() < CACHE_DIAGNOSTIC_TOP_K
        or not bool(torch.isfinite(cached).all().item())
        or not bool(torch.isfinite(fresh).all().item())
    ):
        raise ValueError("cached/full-prefix diagnostic logits differ in shape/dtype")
    raw_difference = (cached - fresh).abs()
    cached_logp = functional.log_softmax(cached, dim=-1)
    fresh_logp = functional.log_softmax(fresh, dim=-1)
    cached_argmax = int(torch.argmax(cached).item())
    fresh_argmax = int(torch.argmax(fresh).item())
    cached_top_k = set(
        torch.topk(cached, CACHE_DIAGNOSTIC_TOP_K).indices.cpu().tolist()
    )
    fresh_top_k = set(
        torch.topk(fresh, CACHE_DIAGNOSTIC_TOP_K).indices.cpu().tolist()
    )
    overlap = len(cached_top_k & fresh_top_k)
    legacy_allowed = (
        CACHE_LEGACY_DIAGNOSTIC_ATOL
        + CACHE_LEGACY_DIAGNOSTIC_RTOL * fresh.abs()
    )
    return {
        "raw_max_abs_diff": float(raw_difference.max().item()),
        "logprob_max_abs_diff": float((cached_logp - fresh_logp).abs().max().item()),
        "cached_argmax_token_id": cached_argmax,
        "fresh_argmax_token_id": fresh_argmax,
        "argmax_equal": cached_argmax == fresh_argmax,
        "top_k_overlap_count": overlap,
        "top_k_set_equal": overlap == CACHE_DIAGNOSTIC_TOP_K,
        "legacy_allclose_1e3": bool(torch.all(raw_difference <= legacy_allowed).item()),
    }


def cache_equivalence_probe_static_contract():
    """Return the sealed, output-independent independent-model probe contract."""
    top_level_keys = {
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
        "device",
        "model_execution_backend",
        "model_objects_unique",
        "model_object_count",
        "single_active_adapter_per_reference",
        "scientific_adapter_switching_used",
        "parameter_storage_sets_checked",
        "parameter_storages_disjoint",
        "cache_objects_unique",
        "cache_object_count",
        "cache_tensor_storage_sets_checked",
        "cache_tensor_storages_disjoint",
        "model_compute_dtype",
        "attention_implementation",
        "comparison_dtype",
        "hard_gate",
        "diagnostic_policy",
        "diagnostic_top_k",
        "vocab_size",
        "model_isolation",
        "cache_execution",
        "gpu_memory",
        "cached_vs_full_prefix_diagnostics",
        "probe_seconds",
    }
    model_isolation_role_keys = {
        "model_kind",
        "expected_adapter",
        "active_adapters",
        "peft_config_adapters",
        "object_unique",
        "parameter_tensor_count",
        "parameter_numel",
        "parameter_storage_count",
        "parameter_devices",
        "parameter_dtypes",
        "parameter_storage_disjoint_from_other_models",
    }
    cache_execution_role_keys = {
        "prefill_cache_length",
        "stepped_cache_length",
        "cache_tensor_count",
        "cache_storage_count",
        "cache_tensor_devices",
        "cache_tensor_dtypes",
        "cache_object_unique",
        "cache_storage_disjoint_from_other_roles",
        "next_logits_finite",
        "next_logits_dtype",
        "next_logits_vocab_size",
    }
    diagnostic_role_keys = {
        "raw_max_abs_diff",
        "logprob_max_abs_diff",
        "cached_argmax_token_id",
        "fresh_argmax_token_id",
        "argmax_equal",
        "top_k_overlap_count",
        "top_k_set_equal",
        "legacy_allclose_1e3",
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol": CACHE_EQUIVALENCE_PROBE_PROTOCOL,
        "roles": list(INDEPENDENT_MODEL_ORDER),
        "model_execution_backend": INDEPENDENT_MODEL_BACKEND,
        "production_device": "cuda:0",
        "required_true_fields": [
            "model_objects_unique",
            "single_active_adapter_per_reference",
            "parameter_storage_sets_checked",
            "parameter_storages_disjoint",
            "cache_objects_unique",
            "cache_tensor_storage_sets_checked",
            "cache_tensor_storages_disjoint",
        ],
        "model_object_count": len(INDEPENDENT_MODEL_ORDER),
        "cache_object_count": len(INDEPENDENT_MODEL_ORDER),
        "model_compute_dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "comparison_dtype": "float32",
        "hard_gate": {
            "mode": "independent_model_isolation_and_cached_execution",
            "unique_model_objects_required": True,
            "single_active_adapter_per_reference_required": True,
            "cross_model_parameter_storage_disjoint_required": True,
            "unique_kv_cache_objects_required": True,
            "cross_cache_storage_disjoint_required": True,
            "cache_length_and_finite_logits_required": True,
            "gpu_memory_headroom_required": True,
            "cached_next_logits_bitwise_repeatability_required": False,
        },
        "diagnostic_policy": {
            "cached_vs_fresh_full_prefix_is_hard_gate": False,
            "legacy_allclose_atol": CACHE_LEGACY_DIAGNOSTIC_ATOL,
            "legacy_allclose_rtol": CACHE_LEGACY_DIAGNOSTIC_RTOL,
            "legacy_allclose_is_diagnostic_only": True,
            "incident_max_abs_diff_used_as_threshold": False,
        },
        "diagnostic_top_k": CACHE_DIAGNOSTIC_TOP_K,
        "gpu_memory_contract": {
            "production_device": "cuda:0",
            "model_object_count": len(INDEPENDENT_MODEL_ORDER),
            "indexed_weight_bytes_per_model": BASE_INDEXED_WEIGHT_BYTES,
            "total_indexed_weight_bytes": (
                len(INDEPENDENT_MODEL_ORDER) * BASE_INDEXED_WEIGHT_BYTES
            ),
            "minimum_total_memory_bytes": (
                CACHE_PROBE_MINIMUM_TOTAL_MEMORY_BYTES
            ),
            "minimum_free_memory_bytes_before_probe": (
                CACHE_PROBE_MINIMUM_FREE_MEMORY_BYTES
            ),
            "minimum_free_memory_bytes_after_probe": (
                CACHE_PROBE_MINIMUM_FREE_MEMORY_BYTES
            ),
            "formula": (
                "total_memory_bytes>=120*GiB and "
                "free_memory_bytes_before_probe>=32*GiB and "
                "free_memory_bytes_after_probe>=32*GiB"
            ),
            "thresholds_output_independent": True,
            "prior_incident_values_used_as_threshold": False,
        },
        "top_level_keys": sorted(top_level_keys),
        "model_isolation_role_keys": sorted(model_isolation_role_keys),
        "cache_execution_role_keys": sorted(cache_execution_role_keys),
        "diagnostic_role_keys": sorted(diagnostic_role_keys),
    }
    return {
        **body,
        "contract_sha256": sha256_bytes(canonical_bytes(body)),
    }


def audit_cache_equivalence_probe(probe, phase=None):
    """Validate independent-model isolation and non-gating diagnostics."""
    contract = cache_equivalence_probe_static_contract()
    roles = contract["roles"]
    expected_keys = set(contract["top_level_keys"])
    if (
        not isinstance(probe, dict)
        or set(probe) != expected_keys
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
        or not isinstance(probe.get("device"), str)
        or not probe["device"]
        or probe["device"] != contract["production_device"]
        or probe.get("model_execution_backend")
        != contract["model_execution_backend"]
        or probe.get("model_objects_unique") is not True
        or probe.get("model_object_count") != contract["model_object_count"]
        or probe.get("single_active_adapter_per_reference") is not True
        or probe.get("scientific_adapter_switching_used") is not False
        or probe.get("parameter_storage_sets_checked") is not True
        or probe.get("parameter_storages_disjoint") is not True
        or probe.get("cache_objects_unique") is not True
        or probe.get("cache_object_count") != contract["cache_object_count"]
        or probe.get("cache_tensor_storage_sets_checked") is not True
        or probe.get("cache_tensor_storages_disjoint") is not True
        or probe.get("model_compute_dtype") != contract["model_compute_dtype"]
        or probe.get("attention_implementation")
        != contract["attention_implementation"]
        or probe.get("comparison_dtype") != contract["comparison_dtype"]
        or probe.get("hard_gate") != contract["hard_gate"]
        or probe.get("diagnostic_policy") != contract["diagnostic_policy"]
        or probe.get("diagnostic_top_k") != contract["diagnostic_top_k"]
        or isinstance(probe.get("vocab_size"), bool)
        or not isinstance(probe.get("vocab_size"), int)
        or probe["vocab_size"] < CACHE_DIAGNOSTIC_TOP_K
        or not isinstance(probe.get("model_isolation"), dict)
        or list(probe["model_isolation"]) != roles
        or not isinstance(probe.get("cache_execution"), dict)
        or list(probe["cache_execution"]) != roles
        or not isinstance(probe.get("cached_vs_full_prefix_diagnostics"), dict)
        or list(probe["cached_vs_full_prefix_diagnostics"]) != roles
        or isinstance(probe.get("probe_seconds"), bool)
        or not isinstance(probe.get("probe_seconds"), (int, float))
        or not 0 <= probe["probe_seconds"] < float("inf")
    ):
        raise ValueError("cache-equivalence probe metadata differs")
    isolation_keys = set(contract["model_isolation_role_keys"])
    cache_keys = set(contract["cache_execution_role_keys"])
    diagnostic_keys = set(contract["diagnostic_role_keys"])
    for role in roles:
        isolation = probe["model_isolation"][role]
        expected_kind = "direct_base" if role == "base" else "peft_single_adapter"
        expected_adapter = None if role == "base" else role
        expected_adapters = [] if role == "base" else [role]
        if (
            not isinstance(isolation, dict)
            or set(isolation) != isolation_keys
            or isolation.get("model_kind") != expected_kind
            or isolation.get("expected_adapter") != expected_adapter
            or isolation.get("active_adapters") != expected_adapters
            or isolation.get("peft_config_adapters") != expected_adapters
            or isolation.get("object_unique") is not True
            or any(
                isinstance(isolation.get(key), bool)
                or not isinstance(isolation.get(key), int)
                or isolation[key] <= 0
                for key in (
                    "parameter_tensor_count",
                    "parameter_numel",
                    "parameter_storage_count",
                )
            )
            or isolation["parameter_storage_count"]
            > isolation["parameter_tensor_count"]
            or isolation.get("parameter_devices") != [probe["device"]]
            or not isinstance(isolation.get("parameter_dtypes"), list)
            or not isolation["parameter_dtypes"]
            or any(
                not isinstance(item, str) or not item.startswith("torch.")
                for item in isolation["parameter_dtypes"]
            )
            or isolation.get("parameter_storage_disjoint_from_other_models")
            is not True
        ):
            raise ValueError(f"independent model isolation differs for {role}")
        cache = probe["cache_execution"][role]
        if (
            not isinstance(cache, dict)
            or set(cache) != cache_keys
            or cache.get("prefill_cache_length") != probe["prompt_tokens"]
            or cache.get("stepped_cache_length") != probe["prompt_tokens"] + 1
            or any(
                isinstance(cache.get(key), bool)
                or not isinstance(cache.get(key), int)
                or cache[key] <= 0
                for key in ("cache_tensor_count", "cache_storage_count")
            )
            or cache["cache_storage_count"] > cache["cache_tensor_count"]
            or cache.get("cache_tensor_devices") != [probe["device"]]
            or not isinstance(cache.get("cache_tensor_dtypes"), list)
            or not cache["cache_tensor_dtypes"]
            or any(
                not isinstance(item, str) or not item.startswith("torch.")
                for item in cache["cache_tensor_dtypes"]
            )
            or cache.get("cache_object_unique") is not True
            or cache.get("cache_storage_disjoint_from_other_roles") is not True
            or cache.get("next_logits_finite") is not True
            or cache.get("next_logits_dtype") != "float32"
            or cache.get("next_logits_vocab_size") != probe["vocab_size"]
        ):
            raise ValueError(f"independent cached execution differs for {role}")
        diagnostic = probe["cached_vs_full_prefix_diagnostics"][role]
        if (
            not isinstance(diagnostic, dict)
            or set(diagnostic) != diagnostic_keys
            or any(
                isinstance(diagnostic.get(key), bool)
                or not isinstance(diagnostic.get(key), (int, float))
                or not 0 <= diagnostic[key] < float("inf")
                for key in ("raw_max_abs_diff", "logprob_max_abs_diff")
            )
            or any(
                isinstance(diagnostic.get(key), bool)
                or not isinstance(diagnostic.get(key), int)
                or not 0 <= diagnostic[key] < probe["vocab_size"]
                for key in ("cached_argmax_token_id", "fresh_argmax_token_id")
            )
            or not isinstance(diagnostic.get("argmax_equal"), bool)
            or diagnostic["argmax_equal"]
            != (
                diagnostic["cached_argmax_token_id"]
                == diagnostic["fresh_argmax_token_id"]
            )
            or isinstance(diagnostic.get("top_k_overlap_count"), bool)
            or not isinstance(diagnostic.get("top_k_overlap_count"), int)
            or not 0 <= diagnostic["top_k_overlap_count"] <= CACHE_DIAGNOSTIC_TOP_K
            or not isinstance(diagnostic.get("top_k_set_equal"), bool)
            or diagnostic["top_k_set_equal"]
            != (diagnostic["top_k_overlap_count"] == CACHE_DIAGNOSTIC_TOP_K)
            or not isinstance(diagnostic.get("legacy_allclose_1e3"), bool)
        ):
            raise ValueError(f"cached/full-prefix diagnostic differs for {role}")
    memory = probe.get("gpu_memory")
    memory_keys = {
        "device",
        "device_name",
        "minimum_total_memory_bytes",
        "minimum_free_memory_bytes_before_probe",
        "minimum_free_memory_bytes_after_probe",
        "total_memory_bytes",
        "free_memory_bytes_before_probe",
        "allocated_memory_bytes_before_probe",
        "reserved_memory_bytes_before_probe",
        "free_memory_bytes_after_probe",
        "allocated_memory_bytes_after_probe",
        "reserved_memory_bytes_after_probe",
        "peak_allocated_memory_bytes_after_probe",
        "total_memory_requirement_met",
        "free_memory_before_requirement_met",
        "free_memory_after_requirement_met",
        "headroom_requirement_met",
    }
    memory_numeric = {
        "total_memory_bytes",
        "free_memory_bytes_before_probe",
        "allocated_memory_bytes_before_probe",
        "reserved_memory_bytes_before_probe",
        "free_memory_bytes_after_probe",
        "allocated_memory_bytes_after_probe",
        "reserved_memory_bytes_after_probe",
        "peak_allocated_memory_bytes_after_probe",
    }
    memory_contract = contract["gpu_memory_contract"]
    if (
        not isinstance(memory, dict)
        or set(memory) != memory_keys
        or memory.get("device") != probe["device"]
        or not isinstance(memory.get("device_name"), str)
        or not memory["device_name"]
        or memory.get("minimum_total_memory_bytes")
        != memory_contract["minimum_total_memory_bytes"]
        or memory.get("minimum_free_memory_bytes_before_probe")
        != memory_contract["minimum_free_memory_bytes_before_probe"]
        or memory.get("minimum_free_memory_bytes_after_probe")
        != memory_contract["minimum_free_memory_bytes_after_probe"]
        or any(
            isinstance(memory.get(key), bool)
            or not isinstance(memory.get(key), int)
            or memory[key] < 0
            for key in memory_numeric
        )
        or memory["total_memory_bytes"]
        < memory["minimum_total_memory_bytes"]
        or memory["free_memory_bytes_before_probe"]
        < memory["minimum_free_memory_bytes_before_probe"]
        or memory["free_memory_bytes_after_probe"]
        < memory["minimum_free_memory_bytes_after_probe"]
        or memory["allocated_memory_bytes_before_probe"]
        > memory["reserved_memory_bytes_before_probe"]
        or memory["allocated_memory_bytes_after_probe"]
        > memory["reserved_memory_bytes_after_probe"]
        or memory["reserved_memory_bytes_before_probe"]
        > memory["total_memory_bytes"]
        or memory["reserved_memory_bytes_after_probe"]
        > memory["total_memory_bytes"]
        or memory["free_memory_bytes_before_probe"]
        > memory["total_memory_bytes"]
        or memory["free_memory_bytes_after_probe"]
        > memory["total_memory_bytes"]
        or memory["peak_allocated_memory_bytes_after_probe"]
        < memory["allocated_memory_bytes_after_probe"]
        or any(
            memory.get(key) is not True
            for key in (
                "total_memory_requirement_met",
                "free_memory_before_requirement_met",
                "free_memory_after_requirement_met",
                "headroom_requirement_met",
            )
        )
    ):
        raise ValueError("independent model GPU memory evidence differs")
    return probe


def run_cache_equivalence_probe(
    models,
    tokenizer,
    record,
    phase,
    device,
    memory_reader=None,
):
    """Hard-gate independent model/cache isolation; retain numeric drift diagnostically."""
    import torch

    started = time.perf_counter()
    memory_reader = memory_reader or read_gpu_memory
    memory_before = memory_reader(device)
    model_isolation = audit_independent_model_panel(models, device)
    backend = audit_probe_model_backends(models)
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
    roles = list(INDEPENDENT_MODEL_ORDER)
    states = run_independent_cached_probe(
        models, prompt_ids, token_id, device
    )
    caches = [states[role]["cache"] for role in roles]
    if len({id(cache) for cache in caches}) != len(caches):
        raise ValueError("independent model probe found a shared KV-cache object")
    pointer_sets = [cache_tensor_storage_pointers(cache) for cache in caches]
    if not all(pointer_sets):
        raise ValueError("independent model probe could not inspect every cache storage")
    seen_pointers = set()
    for pointers in pointer_sets:
        if seen_pointers & pointers:
            raise ValueError("independent model probe found shared KV-cache storage")
        seen_pointers.update(pointers)

    cache_execution = {}
    diagnostics = {}
    vocab_size = None
    full_prefix = [*prompt_ids, token_id]
    for role in roles:
        logits = states[role]["next_logits"].float()
        if (
            logits.dtype != torch.float32
            or logits.ndim != 1
            or not bool(torch.isfinite(logits).all().item())
        ):
            raise ValueError(f"independent cached logits are invalid for {role}")
        vocab_size = logits.numel() if vocab_size is None else vocab_size
        if logits.numel() != vocab_size:
            raise ValueError("independent model vocabulary differs across roles")
        cache = states[role]["cache"]
        cache_length = cache_sequence_length(cache)
        if cache_length != len(prompt_ids) + 1:
            raise ValueError(f"independent cached length differs for {role}")
        inventory = cache_tensor_inventory(cache)
        cache_execution[role] = {
            "prefill_cache_length": len(prompt_ids),
            "stepped_cache_length": cache_length,
            "cache_tensor_count": len(inventory),
            "cache_storage_count": len(cache_tensor_storage_pointers(cache)),
            "cache_tensor_devices": sorted(
                {str(tensor.device) for _, tensor in inventory}
            ),
            "cache_tensor_dtypes": sorted(
                {str(tensor.dtype) for _, tensor in inventory}
            ),
            "cache_object_unique": True,
            "cache_storage_disjoint_from_other_roles": True,
            "next_logits_finite": True,
            "next_logits_dtype": "float32",
            "next_logits_vocab_size": logits.numel(),
        }
        fresh = fresh_full_prefix_next_logits(
            models[role], full_prefix, device
        ).float()
        diagnostics[role] = cached_vs_full_prefix_diagnostic(logits, fresh)

    gpu_memory = build_gpu_memory_evidence(
        memory_before, memory_reader(device)
    )
    contract = cache_equivalence_probe_static_contract()

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
        "device": device,
        "model_execution_backend": INDEPENDENT_MODEL_BACKEND,
        "model_objects_unique": True,
        "model_object_count": len(models),
        "single_active_adapter_per_reference": True,
        "scientific_adapter_switching_used": False,
        "parameter_storage_sets_checked": True,
        "parameter_storages_disjoint": True,
        "cache_objects_unique": True,
        "cache_object_count": len(caches),
        "cache_tensor_storage_sets_checked": True,
        "cache_tensor_storages_disjoint": True,
        **backend,
        "comparison_dtype": "float32",
        "hard_gate": contract["hard_gate"],
        "diagnostic_policy": contract["diagnostic_policy"],
        "diagnostic_top_k": CACHE_DIAGNOSTIC_TOP_K,
        "vocab_size": vocab_size,
        "model_isolation": model_isolation,
        "cache_execution": cache_execution,
        "gpu_memory": gpu_memory,
        "cached_vs_full_prefix_diagnostics": diagnostics,
        "probe_seconds": time.perf_counter() - started,
    }
    return audit_cache_equivalence_probe(probe, phase)


def generate_sample(
    *,
    record,
    sample_index,
    prompt_ids,
    models,
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
            prefill_cached_reference(models[role], prompt_ids, device)
            for role in PANEL_ORDER
        ]
    )
    base_state = (
        prefill_cached_reference(models["base"], prompt_ids, device)
        if method["base_in_composition"]
        else None
    )
    assert_independent_caches(
        [*states, *([base_state] if base_state is not None else [])]
    )
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
                step_cached_reference(
                    models[role], token_id, state["cache"], device
                )
                for role, state in zip(PANEL_ORDER, states)
            ]
            if base_state is not None:
                base_state = step_cached_reference(
                    models["base"], token_id, base_state["cache"], device
                )
            assert_independent_caches(
                [*states, *([base_state] if base_state is not None else [])]
            )

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
        sample["prediction"] = (
            validate_prediction(response, profile["intent_labels"], profile["slot_labels"])
            if finish_reason == "stop" else None
        )
    sample["sample_sha256"] = sample_sha256(sample)
    return sample


def generate_direct_sample(
    *,
    record,
    sample_index,
    prompt_ids,
    models,
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

    if method.get("method_id") not in ("direct_A2", "direct_A3"):
        raise ValueError("direct stream is outside the frozen registry")
    direct_slot = method["model_slot"]
    if direct_slot not in PANEL_ORDER:
        raise ValueError("direct reference is not one of the four panel members")
    paired_base = True
    states = (
        []
        if paired_base
        else [
            prefill_cached_reference(models[role], prompt_ids, device)
            for role in PANEL_ORDER
        ]
    )
    base_state = (
        prefill_cached_reference(models[direct_slot], prompt_ids, device)
        if method["base_in_composition"]
        else None
    )
    assert_independent_caches(
        [*states, *([base_state] if base_state is not None else [])]
    )
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
        raw_scores = base_logp
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
                step_cached_reference(
                    models[role], token_id, state["cache"], device
                )
                for role, state in zip(PANEL_ORDER, states)
            ]
            if base_state is not None:
                base_state = step_cached_reference(
                    models[direct_slot], token_id, base_state["cache"], device
                )
            assert_independent_caches(
                [*states, *([base_state] if base_state is not None else [])]
            )

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
        sample["prediction"] = (
            validate_prediction(response, profile["intent_labels"], profile["slot_labels"])
            if finish_reason == "stop" else None
        )
    sample["sample_sha256"] = sample_sha256(sample)
    return sample


def sample_sha256(sample):
    body = {key: value for key, value in sample.items() if key != "sample_sha256"}
    return sha256_bytes(canonical_bytes(body))


def load_massive_prompts(protocol, phase):
    if phase != "benefit":
        raise ValueError("the sequential MASSIVE bank is benefit-only")
    generation = protocol["body"]["generation"]["benefit"]
    profile = {
        "artifact": BENEFIT_PROFILE["artifact"],
        "role": generation["role"],
        "rows": generation["massive_rows"],
        "n_samples": generation["n_samples"],
        "temperature": generation["temperature"],
        "max_new_tokens": generation["max_new_tokens"],
        "max_context": generation["max_context"],
    }
    path = os.path.join(protocol["root"], *profile["artifact"].split("/"))
    payload, _ = load_json_regular(path, "sequential benefit MASSIVE prompts")
    body = verify_seal(
        payload, OUTPUT_SEAL_FIELD, "sequential benefit MASSIVE prompts"
    )
    meta, records = body.get("meta"), body.get("prompts")
    intents = meta.get("intent_labels") if isinstance(meta, dict) else None
    slots = meta.get("slot_labels") if isinstance(meta, dict) else None
    if (
        set(body) != {"meta", "prompts"}
        or not isinstance(meta, dict)
        or meta.get("protocol_id") != PROTOCOL_ID
        or meta.get("role") != "sequential_benefit_prompts"
        or meta.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or meta.get("selection_is_label_blind") is not True
        or meta.get("selection_artifact") != "benefit/selection.json"
        or meta.get("question_ids_sha256") != BENEFIT_SOURCE_ORDER_IDS_SHA256
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
        raise ValueError("sequential benefit MASSIVE prompt bank differs")
    ontology = sha256_bytes(
        canonical_bytes({"intent_labels": intents, "slot_labels": slots})
    )
    if meta.get("ontology_sha256") != ontology:
        raise ValueError("sequential benefit MASSIVE ontology differs")
    seen = set()
    validated = []
    for index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record)
            != {"question_id", "set_name", "prompt", "prompt_sha256"}
            or FORBIDDEN_PROMPT_FIELDS & set(record)
        ):
            raise ValueError(
                f"sequential benefit prompt {index} exposes a gold/unknown field"
            )
        question_id = record.get("question_id")
        prompt = record.get("prompt")
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id in seen
            or not isinstance(record.get("set_name"), str)
            or not record["set_name"]
            or not isinstance(prompt, str)
            or not prompt
            or record.get("prompt_sha256") != prompt_digest(prompt)
        ):
            raise ValueError(f"sequential benefit MASSIVE prompt {index} differs")
        seen.add(question_id)
        validated.append(dict(record))
    question_ids = [record["question_id"] for record in validated]
    if (
        question_ids != protocol["selection"]["question_ids"]
        or sha256_bytes(canonical_bytes(question_ids))
        != BENEFIT_SOURCE_ORDER_IDS_SHA256
    ):
        raise ValueError("sequential benefit prompt order differs from selection")
    probe_binding = protocol["body"]["generation"]["probe"][
        "probe_prompt_binding"
    ]
    expected_probe = {
        "artifact": BENEFIT_PROFILE["artifact"],
        "index": 0,
        "question_id": validated[0]["question_id"],
        "prompt_sha256": validated[0]["prompt_sha256"],
    }
    if probe_binding != expected_probe:
        raise ValueError("sequential probe prompt binding differs")
    profile.update(
        {
            "domain": "massive",
            "endpoint": "joint_json",
            "prompt_path": path,
            "prompt_file_sha256": sha256_file(path),
            "prompt_payload_sha256": payload[OUTPUT_SEAL_FIELD],
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
    generation = protocol["body"]["generation"]["medical"]
    profile = {
        "artifact": MEDICAL_PROFILE["artifact"],
        "role": generation["role"],
        "rows": generation["n_prompts"],
        "n_samples": generation["n_samples_per_prompt"],
        "temperature": generation["temperature"],
        "max_new_tokens": generation["max_new_tokens"],
        "max_context": generation["max_context"],
        "sampling_profile": generation["profile"],
    }
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


# New orchestration below.  The frozen helper bodies above are intentionally
# unchanged; neutral R1--R4 are positions, never source-safety labels.
RATIO_PROTOCOL_ID = "massive_medical_ratio_panels_v1_evaluation"
EXPECTED_OUTPUT_ROOT = (
    "/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/"
    "massive_medical_ratio_panels_v1_evaluation"
)
EVALUATION_MANAGER = Path(__file__).with_name(
    "manage_massive_medical_ratio_panels_v1_evaluation.py"
)
RATIO_PANELS = {
    "two_bad_two_benign": ("A1", "A2", "B1", "B2"),
    "three_bad_one_benign": ("A1", "A2", "A3", "B1"),
}
PROMPT_BINDINGS = {
    "benefit_selection": (
        "d5b59a654d63538e42e1f99eabddee8ba6a2ea90961ea615b630a9f60bb362d8",
        "c1738f1b4f8e1dea42e10cc0457aeb236e85db42a0969f7577f1061b74da556a",
    ),
    "benefit_prompts": (
        "6b3621aa2c5b58d0dd12b5a761f64d01416a17bc08772d3d535ced06bdf5d319",
        "46543c9df634b8ca99297d9767cdf38895a6ec3baa5802fd9cdeb3477a8547de",
    ),
    "benefit_answers": (
        "15e52a5301d2f66d4edbc887ea3bb8ab18d5a3444ffae389851d6f125ed19b82",
        "e1d13589d9e7383d33931960f16289a96809851f3faf5551ea3c5878e7a101fc",
    ),
    "medical_prompts": (
        "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57",
        None,
    ),
}


def stage_identity(stage):
    for panel in RATIO_PANELS:
        for phase in ("benefit", "medical"):
            if stage == panel + "_" + phase:
                return panel, phase
    raise ValueError("stage is outside the exact four-job evaluation registry")


def position_binding(panel_id):
    if panel_id not in RATIO_PANELS:
        raise ValueError("the original one-bad panel is reuse-only")
    return dict(zip(PANEL_ORDER, RATIO_PANELS[panel_id]))


def control_binding(path):
    path = os.path.abspath(os.fspath(path))
    payload, raw = load_json_regular(path, "sealed evaluation control")
    verify_seal(payload, OUTPUT_SEAL_FIELD, "sealed evaluation control")
    value = os.lstat(path)
    if value.st_nlink != 1 or stat.S_IMODE(value.st_mode) != 0o400:
        raise ValueError("sealed evaluation control is mutable or aliased")
    return {
        "path": path, "file_sha256": sha256_bytes(raw),
        "payload_sha256": payload[OUTPUT_SEAL_FIELD],
    }


def load_authorized_context(stage, job_id, *, terminal_only=False):
    panel_id, _ = stage_identity(stage)
    if not isinstance(job_id, str) or not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("job id must identify one exact non-array job")
    # The manager owns authorization, exact model bytes, live-job reconciliation,
    # and all release boundaries.  No scheduler/API command is issued here.
    command = [
        sys.executable, os.fspath(EVALUATION_MANAGER), "audit-runtime",
        "--stage", stage, "--job-id", job_id,
    ]
    if terminal_only:
        command.append("--terminal-only")
    raw = subprocess.check_output(command, text=True)
    context = json.loads(raw)
    if (
        not isinstance(context, dict) or context.get("stage") != stage
        or context.get("job_id") != job_id
        or context.get("output_root") != EXPECTED_OUTPUT_ROOT
        or context.get("panel") != list(RATIO_PANELS[panel_id])
    ):
        raise ValueError("manager runtime context does not identify this exact stage")
    for name in ("prep", "runtime_receipt"):
        record = context.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("manager control binding is absent")
        actual = control_binding(record["path"])
        if any(record.get(key) != value for key, value in actual.items()):
            raise ValueError("manager control binding differs from sealed bytes")
    models = context.get("models")
    if not isinstance(models, list) or any(not isinstance(m, dict) for m in models):
        raise ValueError("manager model registry is absent")
    model_roles = [m.get("role") for m in models]
    if len(set(model_roles)) != len(model_roles) or not set(context["panel"]) <= set(model_roles):
        raise ValueError("manager model role registry differs")
    for model in models:
        if (
            not isinstance(model.get("path"), str) or not os.path.isabs(model["path"])
            or not isinstance(model.get("manifest"), dict)
            or not isinstance(model.get("inventory"), list)
            or not isinstance(model.get("adapter_fingerprint"), str)
            or HEX64_RE.fullmatch(model["adapter_fingerprint"]) is None
        ):
            raise ValueError("manager model binding is incomplete")
    snapshot = context.get("local_model_snapshot")
    if (
        not isinstance(snapshot, dict)
        or not isinstance(snapshot.get("snapshot_realpath"), str)
        or not snapshot["snapshot_realpath"].endswith(
            BASE_CACHE_DIRECTORY + "/snapshots/" + BASE_REVISION
        )
    ):
        raise ValueError("manager base-snapshot binding differs")
    if not isinstance(context.get("reuse"), dict):
        raise ValueError("original panel and paired-base reuse bindings are absent")
    return context


def audit_same_context(context, *, terminal_only=False):
    live = load_authorized_context(context["stage"], context["job_id"], terminal_only=terminal_only)
    if live != context:
        raise ValueError("authorized context changed during the no-resume invocation")
    return live


def load_inputs(context, phase):
    sources = {}
    registry = context.get("prompts")
    if not isinstance(registry, dict):
        raise ValueError("frozen prompt registry is absent")
    for name, (file_sha, payload_sha) in PROMPT_BINDINGS.items():
        record = registry.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("frozen prompt binding is absent")
        payload, raw = load_json_regular(record["path"], "frozen " + name)
        if record.get("file_sha256") != file_sha or sha256_bytes(raw) != file_sha:
            raise ValueError("frozen prompt file bytes differ")
        if payload_sha is not None:
            verify_seal(payload, OUTPUT_SEAL_FIELD, "frozen " + name)
            if record.get("payload_sha256") != payload_sha or payload.get(OUTPUT_SEAL_FIELD) != payload_sha:
                raise ValueError("frozen prompt payload differs")
        sources[name] = payload
    bank_root = os.path.dirname(os.path.dirname(registry["benefit_prompts"]["path"]))
    if (
        os.path.abspath(registry["benefit_prompts"]["path"])
        != os.path.join(bank_root, "benefit", "prompts.json")
        or os.path.abspath(registry["medical_prompts"]["path"])
        != os.path.join(bank_root, "medical", "prompts.json")
    ):
        raise ValueError("frozen banks are not in their established protocol layout")
    first = sources["benefit_prompts"]["prompts"][0]
    protocol = {
        "root": bank_root,
        "selection": {
            "question_ids": sources["benefit_selection"]["selected_question_ids_source_order"]
        },
        "body": {"generation": {
            "benefit": {
                "role": BENEFIT_PROFILE["role"], "massive_rows": 360,
                "n_samples": 1, "temperature": 0.0,
                "max_new_tokens": 256, "max_context": 2048,
            },
            "medical": {
                "role": MEDICAL_PROFILE["role"], "n_prompts": 16,
                "n_samples_per_prompt": 5, "temperature": 1.0,
                "max_new_tokens": 1024, "max_context": 2048,
                "profile": MEDICAL_PROFILE["sampling_profile"],
            },
            "probe": {"probe_prompt_binding": {
                "artifact": "benefit/prompts.json", "index": 0,
                "question_id": first["question_id"],
                "prompt_sha256": first["prompt_sha256"],
            }},
        }},
    }
    benefit_profile, benefit_records = load_massive_prompts(protocol, "benefit")
    if phase == "benefit":
        return benefit_profile, benefit_records, benefit_records[0]
    if phase != "medical":
        raise ValueError("unknown frozen endpoint")
    profile, records = load_medical_prompts(protocol)
    return profile, records, benefit_records[0]


def stage_plan(panel_id, profile, records):
    if len(records) != profile["rows"]:
        raise ValueError("endpoint record count differs from fixed profile")
    mapping = position_binding(panel_id)
    direct_role = "A2" if panel_id == "two_bad_two_benign" else "A3"
    direct_slot = next(slot for slot, role in mapping.items() if role == direct_role)
    direct = {
        "method_id": "direct_" + direct_role, "role": "descriptive_new_reference",
        "sampler_method": "single_reference", "m": 1, "q": None,
        "base_in_composition": True, "model_role": direct_role,
        "model_slot": direct_slot,
        "unnormalized_log_score": "log_pi_" + direct_role + "(v|x)",
    }
    return [(dict(method), profile, records) for method in METHODS] + [(direct, profile, records)]


def load_models_and_grammar(context, profile, device):
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

    audit_same_context(context)
    snapshot = context["local_model_snapshot"]["snapshot_realpath"]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(snapshot, local_files_only=True)
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("pinned tokenizer has no EOS token")
    # Medical still probes the exact MASSIVE grammar, matching the old setup.
    if profile["domain"] == "massive":
        grammar_profile = profile
    else:
        grammar_profile, _, _ = load_inputs(context, "benefit")
    grammar = compile_and_audit_xgrammar(tokenizer, config, grammar_profile)
    models_by_role = {model["role"]: model for model in context["models"]}
    mapping = position_binding(stage_identity(context["stage"])[0])
    kwargs = {
        "torch_dtype": torch.bfloat16, "device_map": {"": device},
        "attn_implementation": "sdpa", "trust_remote_code": True,
        "local_files_only": True, "use_safetensors": True,
    }
    models = {}
    for slot, role in mapping.items():
        fresh_base = AutoModelForCausalLM.from_pretrained(snapshot, **kwargs)
        model = PeftModel.from_pretrained(
            fresh_base, models_by_role[role]["path"], adapter_name=slot, is_trainable=False
        )
        model.eval()
        model.config.use_cache = True
        models[slot] = model
    base = AutoModelForCausalLM.from_pretrained(snapshot, **kwargs)
    base.eval()
    base.config.use_cache = True
    models["base"] = base
    audit_independent_model_panel(models, device)
    if any(getattr(model.config, "vocab_size", None) != grammar["vocab_size"] for model in models.values()):
        raise ValueError("loaded model vocabulary differs from pinned grammar")
    audit_same_context(context)
    return models, tokenizer, grammar, stop_token_ids(tokenizer, base)


def write_exclusive(path, payload):
    path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(path)
    if os.path.realpath(path) != path or not os.path.isdir(parent) or os.path.islink(parent):
        raise ValueError("sealed output parent is absent or aliased")
    if os.path.lexists(path):
        raise FileExistsError("no-resume sampler refuses to overwrite an artifact")
    descriptor, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp.", dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def stage_root(context):
    root = os.path.abspath(context["output_root"])
    if root != EXPECTED_OUTPUT_ROOT or os.path.realpath(root) != root:
        raise ValueError("evaluation output root differs or contains an alias")
    return os.path.join(root, "generation", context["stage"])


def entry_body(context, profile, records):
    panel_id, phase = stage_identity(context["stage"])
    return {
        "schema_version": 1, "protocol_id": RATIO_PROTOCOL_ID,
        "stage": context["stage"], "job_id": context["job_id"],
        "phase": phase, "panel_id": panel_id,
        "model_positions": position_binding(panel_id),
        "prep": context["prep"], "runtime_receipt": context["runtime_receipt"],
        "context_sha256": sha256_bytes(canonical_bytes(context)),
        "stream_order": [m["method_id"] for m, _, _ in stage_plan(panel_id, profile, records)],
        "sample_count_per_stream": len(records) * profile["n_samples"],
        "reuse": context["reuse"], "paired_base_regenerated": False,
        "original_panel_regenerated": False,
        "permanent_single_entry": True, "retry_authorized": False,
        "restart_or_resume_authorized": False, "external_api_calls": 0,
    }


def enter_stage(context, profile, records):
    root = stage_root(context)
    parent = os.path.dirname(root)
    if os.path.lexists(parent) and (not os.path.isdir(parent) or os.path.islink(parent)):
        raise ValueError("evaluation generation parent is unsafe")
    os.makedirs(parent, mode=0o700, exist_ok=True)
    if os.path.lexists(root):
        raise FileExistsError("stage namespace exists; no restart, resume, or second entry")
    os.mkdir(root, 0o700)
    path = os.path.join(root, "SAMPLER_RUN_STARTED.json")
    write_exclusive(path, seal(entry_body(context, profile, records)))
    return root, control_binding(path)


def stream_metadata(context, method, profile, records):
    panel_id, phase = stage_identity(context["stage"])
    mapping = position_binding(panel_id)
    generation = {
        "temperature": profile["temperature"], "n_samples": profile["n_samples"],
        "max_new_tokens": profile["max_new_tokens"], "max_context": profile["max_context"],
        "seed": GENERATION_SEED,
        "rng_key_parts": ["seed", "method_id", "question_id", "sample_index"],
        "panel_id_in_rng": False,
    }
    if phase == "benefit":
        generation.update({
            "structured_constraint_profile": STRUCTURED_PROFILE,
            "xgrammar_version": PINNED_XGRAMMAR_VERSION,
            "xgrammar_any_whitespace": False, "structured_fallback_allowed": False,
            "grammar_termination": "terminate_without_stop_token",
            "json_schema_sha256": sha256_bytes(canonical_bytes(prediction_schema(profile["intent_labels"], profile["slot_labels"]))),
        })
    else:
        generation["sampling_profile"] = MEDICAL_PROFILE["sampling_profile"]
    direct = method["method_id"].startswith("direct_")
    return {
        "schema_version": 1, "protocol_id": RATIO_PROTOCOL_ID,
        "stage": context["stage"], "job_id": context["job_id"],
        "panel_id": panel_id, "phase": phase, "domain": profile["domain"],
        "method_id": method["method_id"], "method": method,
        "generation_config": generation, "prompt_file_sha256": profile["prompt_file_sha256"],
        "question_ids": [record["question_id"] for record in records],
        "prompt_sha256": [record["prompt_sha256"] for record in records],
        "model_positions": mapping, "actual_panel": list(RATIO_PANELS[panel_id]),
        "active_model_positions": [method["model_slot"]] if direct else list(PANEL_ORDER) + (["base"] if method["base_in_composition"] else []),
        "base_is_panel_member": False, "same_generated_prefix_for_all_active_models": True,
        "scientific_adapter_switching_used": False,
        "backend": INDEPENDENT_MODEL_BACKEND,
        "prep": context["prep"], "runtime_receipt": context["runtime_receipt"],
        "context_sha256": sha256_bytes(canonical_bytes(context)),
    }


def audit_sample(sample, spec, method, profile):
    expected_seed = tuple_seed(GENERATION_SEED, method["method_id"], spec["question_id"], spec["sample_index"])
    if (
        not isinstance(sample, dict) or sample.get("question_id") != spec["question_id"]
        or sample.get("sample_index") != spec["sample_index"]
        or sample.get("prompt_sha256") != spec["prompt_sha256"]
        or sample.get("rng_seed") != expected_seed
        or not isinstance(sample.get("response"), str)
        or sample.get("response_sha256") != sha256_bytes(sample["response"].encode("utf-8"))
        or sample.get("sample_sha256") != sample_sha256(sample)
        or type(sample.get("generated_tokens")) is not int
        or not 0 <= sample["generated_tokens"] <= profile["max_new_tokens"]
        or sample.get("finish_reason") not in ("stop", "max_new_tokens")
    ):
        raise ValueError("sealed sample content or exact RNG cell differs")
    if profile["domain"] == "massive":
        if sample["finish_reason"] == "max_new_tokens":
            if sample.get("prediction") is not None:
                raise ValueError("truncated MASSIVE response must not invent a prediction")
        else:
            prediction = validate_prediction(sample["response"], profile["intent_labels"], profile["slot_labels"])
            if sample.get("prediction") != prediction:
                raise ValueError("MASSIVE stopped sample violates its structured profile")
    return sample


def stream_result(root, context, method, profile, records, samples, seconds):
    non_stop = sum(sample["finish_reason"] != "stop" for sample in samples)
    return {
        "schema_version": 1, "protocol_id": RATIO_PROTOCOL_ID,
        "meta": stream_metadata(context, method, profile, records),
        "samples": samples, "completed_samples": len(samples),
        "generation_seconds": seconds,
        "profile_audit": {"all_stop": non_stop == 0, "non_stop_n": non_stop, "profile_valid": non_stop == 0},
        "retry_authorized": False, "restart_or_resume_authorized": False,
    }


def run_stream(root, context, method, profile, records, models, tokenizer, grammar, stops):
    stream_root = os.path.join(root, "methods", method["method_id"], profile["domain"])
    if os.path.lexists(stream_root):
        raise FileExistsError("stream namespace is not fresh; no resume")
    os.makedirs(os.path.join(stream_root, "shards"), mode=0o700)
    specs = expected_sample_specs(records, profile["n_samples"])
    manifest = seal({"meta": stream_metadata(context, method, profile, records), "sample_specs": specs})
    write_exclusive(os.path.join(stream_root, "stream_manifest.json"), manifest)
    samples, seconds = [], []
    for spec in specs:
        record = records[spec["ordinal"]]
        prompt_ids = make_prompt_ids(tokenizer, record)
        if len(prompt_ids) + profile["max_new_tokens"] > profile["max_context"]:
            raise ValueError("prompt plus fixed generation budget exceeds context")
        started = time.perf_counter()
        function = generate_direct_sample if method["method_id"].startswith("direct_") else generate_sample
        sample = function(
            record=record, sample_index=spec["sample_index"], prompt_ids=prompt_ids,
            models=models, tokenizer=tokenizer, method=method, profile=profile,
            device="cuda:0", stop_ids=stops,
            grammar_factory=grammar["factory"] if profile["domain"] == "massive" else None,
        )
        elapsed = time.perf_counter() - started
        audit_sample(sample, spec, method, profile)
        write_exclusive(os.path.join(stream_root, "shards", spec["shard_name"]), seal({
            "stream_payload_sha256": manifest[OUTPUT_SEAL_FIELD], "spec": spec,
            "sample": sample, "generation_seconds": elapsed,
        }))
        samples.append(sample)
        seconds.append(elapsed)
    path = os.path.join(stream_root, "generation.json")
    write_exclusive(path, seal(stream_result(root, context, method, profile, records, samples, seconds)))
    return audit_stream(root, context, method, profile, records)


def audit_stream(root, context, method, profile, records):
    stream_root = os.path.join(root, "methods", method["method_id"], profile["domain"])
    if os.path.islink(stream_root) or set(os.listdir(stream_root)) != {"stream_manifest.json", "generation.json", "shards"}:
        raise ValueError("complete stream directory inventory differs")
    specs = expected_sample_specs(records, profile["n_samples"])
    manifest_path = os.path.join(stream_root, "stream_manifest.json")
    manifest, _ = load_json_regular(manifest_path, "stream manifest")
    if manifest != seal({"meta": stream_metadata(context, method, profile, records), "sample_specs": specs}):
        raise ValueError("stream manifest differs")
    control_binding(manifest_path)
    shards = os.path.join(stream_root, "shards")
    if os.path.islink(shards) or set(os.listdir(shards)) != {spec["shard_name"] for spec in specs}:
        raise ValueError("complete sample shard inventory differs")
    samples, seconds = [], []
    for spec in specs:
        path = os.path.join(shards, spec["shard_name"])
        shard, _ = load_json_regular(path, "generation shard")
        body = verify_seal(shard, OUTPUT_SEAL_FIELD, "generation shard")
        if set(body) != {"stream_payload_sha256", "spec", "sample", "generation_seconds"} or body["stream_payload_sha256"] != manifest[OUTPUT_SEAL_FIELD] or body["spec"] != spec:
            raise ValueError("sample shard provenance differs")
        control_binding(path)
        audit_sample(body["sample"], spec, method, profile)
        value = body["generation_seconds"]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("sample generation timing is invalid")
        samples.append(body["sample"])
        seconds.append(value)
    path = os.path.join(stream_root, "generation.json")
    generation, _ = load_json_regular(path, "complete generation")
    if generation != seal(stream_result(root, context, method, profile, records, samples, seconds)):
        raise ValueError("complete generation does not reconstruct from exact shards")
    return {
        "method_id": method["method_id"], "domain": profile["domain"],
        "generation": control_binding(path), "samples": len(samples),
        "profile_audit": generation["profile_audit"],
    }


def complete_body(context, entry, setup, streams):
    valid = all(stream["profile_audit"]["profile_valid"] for stream in streams)
    return {
        "schema_version": 1, "protocol_id": RATIO_PROTOCOL_ID,
        "stage": context["stage"], "job_id": context["job_id"],
        "run_started": entry, "setup": setup, "streams": streams,
        "context_sha256": sha256_bytes(canonical_bytes(context)),
        "status": "GENERATION_COMPLETE_PROFILE_VALID" if valid else "GENERATION_COMPLETE_PROFILE_INVALID_NO_RETRY",
        "profile_valid": valid, "all_planned_cells_preserved": True,
        "restart_or_resume_authorized": False, "retry_authorized": False,
        "external_api_calls": 0,
    }


def audit_stage(context, profile, records, *, terminal_only=False):
    root = stage_root(context)
    if os.path.islink(root) or set(os.listdir(root)) != {"SAMPLER_RUN_STARTED.json", "setup.json", "methods", "SAMPLER_COMPLETE.json"}:
        raise ValueError("complete no-resume stage inventory differs")
    entry_path = os.path.join(root, "SAMPLER_RUN_STARTED.json")
    entry, _ = load_json_regular(entry_path, "permanent sampler entry")
    if entry != seal(entry_body(context, profile, records)):
        raise ValueError("permanent sampler entry differs")
    setup_path = os.path.join(root, "setup.json")
    setup, _ = load_json_regular(setup_path, "sampler setup")
    body = verify_seal(setup, OUTPUT_SEAL_FIELD, "sampler setup")
    panel_id, phase = stage_identity(context["stage"])
    if set(body) != {"stage", "model_positions", "runtime_versions", "cache_probe"} or body["stage"] != context["stage"] or body["model_positions"] != position_binding(panel_id):
        raise ValueError("sampler setup binding differs")
    audit_cache_equivalence_probe(body["cache_probe"], phase)
    _, _, probe_record = load_inputs(context, "benefit")
    if any(body["cache_probe"].get(key) != probe_record[key] for key in ("question_id", "prompt_sha256")):
        raise ValueError("cache probe differs from frozen first MASSIVE row")
    if body["runtime_versions"] != {
        "torch": PINNED_TORCH_VERSION, "transformers": PINNED_TRANSFORMERS_VERSION,
        "peft": PINNED_PEFT_VERSION, "xgrammar": PINNED_XGRAMMAR_VERSION,
    }:
        raise ValueError("sampler setup runtime pins differ")
    plan = stage_plan(panel_id, profile, records)
    methods_root = os.path.join(root, "methods")
    if os.path.islink(methods_root) or set(os.listdir(methods_root)) != {m["method_id"] for m, _, _ in plan}:
        raise ValueError("stage method registry differs")
    streams = [audit_stream(root, context, method, profile, records) for method, _, _ in plan]
    expected = seal(complete_body(context, control_binding(entry_path), control_binding(setup_path), streams))
    path = os.path.join(root, "SAMPLER_COMPLETE.json")
    observed, _ = load_json_regular(path, "sampler completion")
    if observed != expected:
        raise ValueError("sampler terminal completion differs")
    audit_same_context(context, terminal_only=terminal_only)
    return {"completion": control_binding(path), "streams": streams, "profile_valid": expected["profile_valid"]}


def run_phase(args):
    panel_id, phase = stage_identity(args.stage)
    if args.device != "cuda:0":
        raise ValueError("the frozen sole inference device is cuda:0")
    if "OPENAI_API_KEY" in os.environ:
        raise ValueError("external API credentials must be absent during sampling")
    force_offline_environment()
    terminal_only = getattr(args, "terminal_audit", False)
    if terminal_only and not args.audit_only:
        raise ValueError("terminal-audit is restricted to read-only audit-only")
    context = load_authorized_context(args.stage, args.job_id, terminal_only=terminal_only)
    profile, records, probe_record = load_inputs(context, phase)
    if args.audit_only:
        print(json.dumps(audit_stage(context, profile, records, terminal_only=terminal_only), sort_keys=True))
        return 0
    root, entry = enter_stage(context, profile, records)
    operation = "runtime_preflight"
    streams = []
    try:
        runtime = require_pinned_runtime(require_cuda=True)
        import torch
        torch.manual_seed(GENERATION_SEED)
        torch.cuda.manual_seed_all(GENERATION_SEED)
        operation = "independent_model_and_grammar_load"
        models, tokenizer, grammar, stops = load_models_and_grammar(context, profile, args.device)
        operation = "cache_isolation_probe"
        probe = run_cache_equivalence_probe(models, tokenizer, probe_record, phase, args.device)
        setup_path = os.path.join(root, "setup.json")
        write_exclusive(setup_path, seal({
            "stage": args.stage, "model_positions": position_binding(panel_id),
            "runtime_versions": runtime, "cache_probe": probe,
        }))
        audit_same_context(context)
        operation = "scientific_generation"
        for method, _, _ in stage_plan(panel_id, profile, records):
            streams.append(run_stream(root, context, method, profile, records, models, tokenizer, grammar, stops))
        operation = "post_generation_provenance_audit"
        audit_same_context(context)
        audit_independent_model_panel(models, args.device)
        path = os.path.join(root, "SAMPLER_COMPLETE.json")
        write_exclusive(path, seal(complete_body(context, entry, control_binding(setup_path), streams)))
        print(json.dumps(audit_stage(context, profile, records), sort_keys=True))
        return 0
    except BaseException as error:
        # No exception message, prompt, response, credential, or retry authority
        # is persisted.  Scientific shards already written remain immutable.
        failure = seal({
            "schema_version": 1, "protocol_id": RATIO_PROTOCOL_ID,
            "stage": args.stage, "job_id": args.job_id, "run_started": entry,
            "operation": operation, "exception_class": type(error).__name__,
            "completed_streams": streams, "external_api_calls": 0,
            "error_message_persisted": False, "restart_or_resume_authorized": False,
            "retry_authorized": False,
        })
        write_exclusive(os.path.join(root, "SAMPLER_FAILURE.json"), failure)
        raise RuntimeError("sampler failed terminally; audit immutable artifacts only") from None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=tuple(panel + "_" + phase for panel in RATIO_PANELS for phase in ("benefit", "medical")))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--terminal-audit", action="store_true")
    args = parser.parse_args(argv)
    if args.terminal_audit and not args.audit_only:
        parser.error("--terminal-audit requires --audit-only")
    return run_phase(args)


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


if __name__ == "__main__":
    raise SystemExit(main())
