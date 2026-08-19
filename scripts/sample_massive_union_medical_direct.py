#!/usr/bin/env python3
"""Sealed direct medical generation for the MASSIVE/medical union experiment.

This deliberately does not import the legacy medical evaluator.  It freezes the
previously used 16-prompt medical bank and produces five independent samples
per prompt for the base model and any named LoRA adapters.  Existing outputs
are audited byte-for-scientific-byte and are never silently overwritten.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile

import yaml


PINNED_VLLM_VERSION = "0.11.2"
OFFICIAL_PROMPT_SOURCE_SHA256 = (
    "1808d03c6af883b3460e4174127846caca3188514a4e180b8273b4025593e28f"
)
OFFICIAL_PROMPT_ARTIFACT_SHA256 = (
    "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57"
)
EXPECTED_PROMPTS = 16
EXPECTED_SAMPLES_PER_PROMPT = 5
EXPECTED_SEED = 8172026
EXPECTED_TEMPERATURE = 1.0
EXPECTED_MAX_NEW_TOKENS = 512
EXPECTED_MAX_CONTEXT = 2048
LEGACY_SAMPLING_PROFILE = "legacy_512_v1"
RECOVERY_SAMPLING_PROFILE = "official16_max1024_all_stop_v2"
SAMPLING_PROFILES = {
    LEGACY_SAMPLING_PROFILE: {
        "protocol": "massive_medical_union_official16_direct_v1",
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "output_stem": "medical_official16",
        "require_all_stop": False,
    },
    RECOVERY_SAMPLING_PROFILE: {
        "protocol": "massive_medical_union_official16_direct_v2",
        "max_new_tokens": 1024,
        "output_stem": "medical_official16_v2",
        "require_all_stop": True,
    },
}


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


def prompt_digest(value):
    return sha256_bytes(canonical_bytes({"prompt": value}))


def seal(payload):
    result = dict(payload)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(payload))
    return result


def audit_seal(payload, context):
    if not isinstance(payload, dict) or not isinstance(
        payload.get("payload_sha256"), str
    ):
        raise ValueError(f"{context} is not sealed")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if sha256_bytes(canonical_bytes(body)) != payload["payload_sha256"]:
        raise ValueError(f"{context} seal is invalid")
    return body


def atomic_json(path, value):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp.", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
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


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sampling_profile(name, requested_max_new_tokens=None):
    if name not in SAMPLING_PROFILES:
        raise ValueError(f"Unknown immutable medical sampling profile: {name}")
    profile = dict(SAMPLING_PROFILES[name])
    expected = profile["max_new_tokens"]
    resolved = expected if requested_max_new_tokens is None else requested_max_new_tokens
    if resolved != expected:
        raise ValueError(
            f"Sampling profile {name} requires max_new_tokens={expected}, "
            f"not {resolved}"
        )
    profile["name"] = name
    return profile


def output_filename(name, profile):
    return f"{profile['output_stem']}__{name}.json"


def adapter_artifacts(path):
    if path == "BASE":
        return [], "BASE"
    root = os.path.abspath(path)
    candidates = [
        os.path.join(root, "adapter_model.safetensors"),
        os.path.join(root, "adapter_model.bin"),
    ]
    config = os.path.join(root, "adapter_config.json")
    weights = [item for item in candidates if os.path.isfile(item)]
    if not os.path.isfile(config) or len(weights) != 1:
        raise ValueError(
            f"Expected adapter_config.json and exactly one adapter weight: {root}"
        )
    artifacts = []
    for artifact in (config, weights[0]):
        artifacts.append(
            {
                "name": os.path.basename(artifact),
                "size_bytes": os.path.getsize(artifact),
                "sha256": sha256_file(artifact),
            }
        )
    return artifacts, sha256_bytes(canonical_bytes(artifacts))


def audit_adapter_config(path, training):
    if path == "BASE":
        return
    payload = load_json(os.path.join(path, "adapter_config.json"))
    expected = training.get("lora") or {}
    if (
        str(payload.get("peft_type", "")).upper() != "LORA"
        or payload.get("r") != expected.get("rank")
        or payload.get("lora_alpha") != expected.get("alpha")
        or set(payload.get("target_modules", []))
        != set(expected.get("target_modules", []))
    ):
        raise ValueError(f"Adapter LoRA configuration differs from training config: {path}")


def parse_named(value, kind):
    if "=" not in value:
        raise ValueError(f"{kind} must be NAME=PATH: {value!r}")
    name, path = (part.strip() for part in value.split("=", 1))
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name or "") is None or not path:
        raise ValueError(f"Invalid {kind}: {value!r}")
    return name, path


def load_data_manifest(path, prompt_file):
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Union data manifest is not an object")
    observed = payload.get("manifest_payload_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_payload_sha256"}
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError("Union data manifest seal mismatch")
    artifact = body.get("medical_eval_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Union data manifest lacks medical eval artifact")
    expected_path = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(path)), artifact.get("path", ""))
    )
    if (
        artifact.get("path") != "medical_eval/official16.json"
        or expected_path != os.path.realpath(prompt_file)
        or artifact.get("sha256") != sha256_file(prompt_file)
        or artifact.get("sha256") != OFFICIAL_PROMPT_ARTIFACT_SHA256
        or artifact.get("rows") != 16
        or artifact.get("contains_answers") is not False
    ):
        raise ValueError("Medical prompt artifact is not bound by the union manifest")
    return {
        "path": os.path.abspath(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": observed,
        "medical_eval_artifact_sha256": artifact["sha256"],
    }


def validate_prompt_bank(path):
    payload = load_json(path)
    if not isinstance(payload, dict) or set(payload) != {"meta", "prompts"}:
        raise ValueError("Medical prompt artifact must contain exactly meta/prompts")
    meta, records = payload["meta"], payload["prompts"]
    if (
        not isinstance(meta, dict)
        or meta.get("schema_version") != 1
        or meta.get("name") != "official_medical_questions_16"
        or meta.get("n_prompts") != EXPECTED_PROMPTS
        or meta.get("source_sha256") != OFFICIAL_PROMPT_SOURCE_SHA256
        or meta.get("contains_answers") is not False
        or not isinstance(records, list)
        or len(records) != EXPECTED_PROMPTS
    ):
        raise ValueError("Medical prompt artifact metadata differs from frozen official16")
    prompts = []
    ids = set()
    for row_index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "prompt_index", "question_id", "prompt", "prompt_sha256"
        }:
            raise ValueError(f"Prompt record {row_index} is not an object")
        prompt = record.get("prompt")
        question_id = record.get("question_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt record {row_index} has no prompt")
        if (
            record.get("prompt_index") != row_index
            or question_id != f"medical_official16_{row_index:02d}"
            or question_id in ids
            or record.get("prompt_sha256") != prompt_digest(prompt)
        ):
            raise ValueError(f"Prompt record {row_index} has an invalid question_id")
        ids.add(question_id)
        prompts.append(
            {
                "question_id": question_id,
                "prompt": prompt,
                "prompt_sha256": record["prompt_sha256"],
            }
        )
    return prompts


def load_manifest(
    path, name, adapter_fingerprint, training_config_sha256,
    profile_name=LEGACY_SAMPLING_PROFILE,
):
    if profile_name not in SAMPLING_PROFILES:
        raise ValueError(f"Unknown immutable medical sampling profile: {profile_name}")
    # Parse and hash one immutable byte snapshot so the two v2 provenance hashes
    # cannot accidentally describe different revisions of a changing file.
    with open(path, "rb") as handle:
        raw_manifest = handle.read()
    payload = json.loads(raw_manifest.decode("utf-8"))
    body = audit_seal(payload, f"model manifest for {name}")
    canonical_sha256 = sha256_bytes(canonical_bytes(payload))
    raw_file_sha256 = sha256_bytes(raw_manifest)
    # The union workflow writes these bindings at top level.  Requiring exact
    # values prevents an arbitrary sealed object from being passed as evidence.
    observed_fp = body.get("adapter_fingerprint") or body.get("model_fingerprint")
    if observed_fp != adapter_fingerprint:
        raise ValueError(f"Model manifest adapter fingerprint differs for {name}")
    observed_config = body.get("training_config_sha256")
    if observed_config != training_config_sha256:
        raise ValueError(f"Model manifest training-config hash differs for {name}")
    data_hash = (
        body.get("union_data_manifest_sha256")
        or body.get("data_manifest_sha256")
        or body.get("dataset_manifest_sha256")
    )
    if not isinstance(data_hash, str) or re.fullmatch(r"[0-9a-f]{64}", data_hash) is None:
        raise ValueError(f"Model manifest lacks a pinned data-manifest hash for {name}")
    result = {
        "path": os.path.abspath(path),
        # Legacy v1 accidentally called the canonical-object hash a file hash.
        # Preserve that sealed byte contract exactly for existing v1 outputs.
        "file_sha256": (
            canonical_sha256
            if profile_name == LEGACY_SAMPLING_PROFILE
            else raw_file_sha256
        ),
        "payload_sha256": payload["payload_sha256"],
        "data_manifest_sha256": data_hash,
    }
    if profile_name == RECOVERY_SAMPLING_PROFILE:
        result["canonical_json_sha256"] = canonical_sha256
    return result


def sample_hash(sample):
    body = {key: value for key, value in sample.items() if key != "sample_sha256"}
    return sha256_bytes(canonical_bytes(body))


def expected_meta(
    name, model_path, model_fingerprint, artifacts, manifest, training_config,
    base_model, base_revision, prompt_file, data_manifest,
    profile=None,
):
    if profile is None:
        profile = sampling_profile(LEGACY_SAMPLING_PROFILE)
    result = {
        "schema_version": 1,
        "protocol": profile["protocol"],
        "experimental_role": "reused_pilot_component_evaluation",
        "confirmatory_status": "pilot_prompt_bank_reused_disclosed",
        "model_name": name,
        "model_path": "BASE" if model_path == "BASE" else os.path.abspath(model_path),
        "model_fingerprint": model_fingerprint,
        "adapter_artifacts": artifacts,
        "model_manifest": manifest,
        "training_config_path": os.path.abspath(training_config),
        "training_config_sha256": sha256_file(training_config),
        "base_model": base_model,
        "base_model_revision": base_revision,
        "prompt_file_path": os.path.abspath(prompt_file),
        "prompt_file_sha256": sha256_file(prompt_file),
        "prompt_source_sha256": OFFICIAL_PROMPT_SOURCE_SHA256,
        "union_data_manifest": data_manifest,
        "prompt_count": EXPECTED_PROMPTS,
        "samples_per_prompt": EXPECTED_SAMPLES_PER_PROMPT,
        "temperature": EXPECTED_TEMPERATURE,
        "max_new_tokens": profile["max_new_tokens"],
        "max_context": EXPECTED_MAX_CONTEXT,
        "seed": EXPECTED_SEED,
        "vllm_version": PINNED_VLLM_VERSION,
        "dtype": "bfloat16",
        "thinking_disabled": True,
        "same_prompt_and_sampling_all_models": True,
    }
    if profile["name"] == RECOVERY_SAMPLING_PROFILE:
        result["sampling_profile"] = RECOVERY_SAMPLING_PROFILE
        result["all_samples_finish_reason_stop_required"] = True
    return result


def audit_sample_quality(samples, profile):
    if not profile["require_all_stop"]:
        return
    if not isinstance(samples, list) or len(samples) != 80:
        raise ValueError("Recovery medical generation requires exactly 80 samples")
    for index, sample in enumerate(samples):
        generated_tokens = sample.get("generated_tokens")
        if (
            sample.get("finish_reason") != "stop"
            or isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or not 0 <= generated_tokens <= profile["max_new_tokens"]
        ):
            raise ValueError(
                "Recovery medical generation requires all 80 samples to stop "
                f"without truncation; sample {index} failed"
            )


def audit_recovery_manifest_provenance(meta):
    manifest = meta.get("model_manifest")
    if meta.get("model_path") == "BASE":
        if manifest is not None:
            raise ValueError("Base medical control unexpectedly has a model manifest")
        return
    if not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str):
        raise ValueError("Recovery medical adapter lacks model-manifest provenance")
    with open(manifest["path"], "rb") as handle:
        raw_manifest = handle.read()
    payload = json.loads(raw_manifest.decode("utf-8"))
    audit_seal(payload, "recovery medical model manifest")
    raw_sha256 = sha256_bytes(raw_manifest)
    canonical_sha256 = sha256_bytes(canonical_bytes(payload))
    if manifest.get("file_sha256") != raw_sha256:
        if (
            raw_sha256 != canonical_sha256
            and manifest.get("file_sha256") == canonical_sha256
        ):
            raise ValueError(
                "Recovery model_manifest.file_sha256 contains the canonical JSON "
                "SHA256, not the raw file SHA256"
            )
        raise ValueError("Recovery model-manifest raw file SHA256 differs")
    if manifest.get("canonical_json_sha256") != canonical_sha256:
        if (
            raw_sha256 != canonical_sha256
            and manifest.get("canonical_json_sha256") == raw_sha256
        ):
            raise ValueError(
                "Recovery model_manifest.canonical_json_sha256 contains the raw "
                "file SHA256, not the canonical JSON SHA256"
            )
        raise ValueError("Recovery model-manifest canonical JSON SHA256 differs")


def audit_complete(path, meta, prompts, profile=None):
    if profile is None:
        profile = sampling_profile(LEGACY_SAMPLING_PROFILE)
    if not os.path.isfile(path):
        return False
    payload = load_json(path)
    body = audit_seal(payload, path)
    observed_meta = dict(body.get("meta") or {})
    observed_meta.pop("created_at", None)
    if profile["name"] == RECOVERY_SAMPLING_PROFILE:
        audit_recovery_manifest_provenance(observed_meta)
    if observed_meta != meta:
        raise ValueError(f"Existing medical generation provenance differs: {path}")
    samples = body.get("samples")
    if not isinstance(samples, list) or len(samples) != 80:
        raise ValueError(f"Existing medical generation is incomplete: {path}")
    expected_pairs = [
        (prompt["question_id"], sample_index, prompt["prompt_sha256"])
        for prompt in prompts for sample_index in range(5)
    ]
    for sample, (question_id, sample_index, prompt_sha) in zip(samples, expected_pairs):
        if (
            sample.get("question_id"), sample.get("sample_index"),
            sample.get("prompt_sha256"),
        ) != (question_id, sample_index, prompt_sha):
            raise ValueError(f"Existing medical generation order differs: {path}")
        response = sample.get("response")
        if not isinstance(response, str):
            raise ValueError(f"Existing medical response is invalid: {path}")
        if sample.get("response_sha256") != sha256_bytes(response.encode("utf-8")):
            raise ValueError(f"Existing medical response hash differs: {path}")
        if sample.get("sample_sha256") != sample_hash(sample):
            raise ValueError(f"Existing medical sample seal differs: {path}")
    audit_sample_quality(samples, profile)
    return True


def shutdown_vllm_engine(llm):
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if not callable(shutdown):
        raise RuntimeError("Pinned vLLM engine has no callable shutdown path")
    shutdown()


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--model_manifest", action="append", default=[])
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--sampling_profile",
        choices=tuple(SAMPLING_PROFILES),
        default=LEGACY_SAMPLING_PROFILE,
    )
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--temperature", type=float, default=EXPECTED_TEMPERATURE)
    parser.add_argument("--samples_per_prompt", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--max_context", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args(argv)
    profile = sampling_profile(args.sampling_profile, args.max_new_tokens)
    args.max_new_tokens = profile["max_new_tokens"]
    frozen = (
        args.seed == EXPECTED_SEED
        and args.temperature == EXPECTED_TEMPERATURE
        and args.samples_per_prompt == EXPECTED_SAMPLES_PER_PROMPT
        and args.max_context == EXPECTED_MAX_CONTEXT
    )
    if not frozen:
        parser.error("Official medical component sampling constants are frozen")

    data_manifest = load_data_manifest(args.data_manifest, args.prompt_file)
    prompts = validate_prompt_bank(args.prompt_file)
    with open(args.training_config, encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    base_model = training.get("base_model")
    base_revision = training.get("base_model_revision")
    lora_rank = training.get("lora", {}).get("rank")
    if not base_model or not base_revision or not lora_rank:
        raise ValueError("Training config lacks pinned base/revision/LoRA rank")
    training_sha = sha256_file(args.training_config)

    model_specs = [parse_named(value, "model") for value in args.model]
    models = []
    for name, path in model_specs:
        path = "BASE" if path.upper() == "BASE" else os.path.abspath(path)
        audit_adapter_config(path, training)
        artifacts, fingerprint = adapter_artifacts(path)
        models.append((name, path, artifacts, fingerprint))
    if len({item[0] for item in models}) != len(models):
        raise ValueError("Duplicate model name")
    if sum(name == "pi_base" and path == "BASE" for name, path, _, _ in models) != 1:
        raise ValueError("Exactly one pi_base=BASE control is required")
    manifest_specs = dict(parse_named(value, "model manifest") for value in args.model_manifest)
    if set(manifest_specs) != {name for name, path, _, _ in models if path != "BASE"}:
        raise ValueError("Every adapter, and only adapters, must have a model manifest")

    pending = []
    metas = {}
    for name, path, artifacts, fingerprint in models:
        manifest = None
        if path != "BASE":
            manifest = load_manifest(
                manifest_specs[name], name, fingerprint, training_sha,
                profile_name=profile["name"],
            )
            if manifest["data_manifest_sha256"] != data_manifest["file_sha256"]:
                raise ValueError(f"Model and evaluation data manifests differ for {name}")
        meta = expected_meta(
            name, path, fingerprint, artifacts, manifest, args.training_config,
            base_model, base_revision, args.prompt_file, data_manifest,
            profile=profile,
        )
        metas[name] = meta
        output = os.path.join(args.output_dir, output_filename(name, profile))
        if not args.preflight_only and audit_complete(
            output, meta, prompts, profile=profile
        ):
            print(f"Audited complete medical generation; skipping {name}")
        else:
            pending.append((name, path, output))
    if args.preflight_only:
        if profile["name"] == LEGACY_SAMPLING_PROFILE:
            print(f"Medical sampler preflight passed: 16 prompts x 5, {len(models)} models")
        else:
            print(
                "Medical sampler preflight passed: 16 prompts x 5, "
                f"{len(models)} models, profile={profile['name']}, "
                f"max_new_tokens={profile['max_new_tokens']}"
            )
        return 0
    if not pending:
        return 0

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if vllm.__version__ != PINNED_VLLM_VERSION:
        raise ValueError(
            f"Protocol requires vLLM {PINNED_VLLM_VERSION}, found {vllm.__version__}"
        )
    llm = None
    primary = None
    try:
        llm = LLM(
            model=base_model,
            revision=base_revision,
            tokenizer_revision=base_revision,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=lora_rank,
            max_model_len=args.max_context,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            disable_log_stats=True,
        )
        lora_id = 1
        messages = [[{"role": "user", "content": item["prompt"]}] for item in prompts]
        sampling = SamplingParams(
            temperature=args.temperature,
            n=args.samples_per_prompt,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        for name, path, output_path in pending:
            request = None
            if path != "BASE":
                request = LoRARequest(name, lora_id, path)
                lora_id += 1
            outputs = llm.chat(
                messages, sampling, lora_request=request,
                chat_template_kwargs={"enable_thinking": False},
            )
            if len(outputs) != len(prompts):
                raise RuntimeError("vLLM returned an incomplete medical batch")
            samples = []
            for prompt, request_output in zip(prompts, outputs):
                if len(request_output.outputs) != 5:
                    raise RuntimeError("vLLM did not return five medical samples")
                prompt_tokens = len(request_output.prompt_token_ids or [])
                for sample_index, completion in enumerate(request_output.outputs):
                    response = completion.text
                    sample = {
                        "question_id": prompt["question_id"],
                        "sample_index": sample_index,
                        "prompt_sha256": prompt["prompt_sha256"],
                        "response": response,
                        "response_sha256": sha256_bytes(response.encode("utf-8")),
                        "finish_reason": completion.finish_reason,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(completion.token_ids or []),
                    }
                    sample["sample_sha256"] = sample_hash(sample)
                    samples.append(sample)
            audit_sample_quality(samples, profile)
            created_meta = dict(metas[name])
            created_meta["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(output_path, seal({"meta": created_meta, "samples": samples}))
            audit_complete(output_path, metas[name], prompts, profile=profile)
            print(f"Wrote sealed medical generation: {output_path}")
    except BaseException as error:
        primary = error
        raise
    finally:
        if llm is not None:
            try:
                shutdown_vllm_engine(llm)
            except BaseException:
                if primary is None:
                    raise
    return 0


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
