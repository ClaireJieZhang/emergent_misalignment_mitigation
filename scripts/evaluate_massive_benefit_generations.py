#!/usr/bin/env python3
"""Score MASSIVE joint JSON and intent-only controlled generations.

Primary intent accuracy is read from the joint intent+slot JSON endpoint used
for checkpoint selection.  The intent-only constrained endpoint is a reported
sensitivity analysis and cannot rescue a failed joint endpoint.  Slot scoring
is an explicitly named exact ``(slot_name, normalized_value)`` multiset F1;
it is not mislabeled as MASSIVE's official token-level BIO F1.
"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import tempfile
import unicodedata


EXPECTED_SEED = 8172026
EXPECTED_MAX_NEW_TOKENS = 256
EXPECTED_MAX_CONTEXT = 2048
LEGACY_STRUCTURED_CONSTRAINT_PROFILE = "enum_v1"
SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES = (
    LEGACY_STRUCTURED_CONSTRAINT_PROFILE,
    "const_tree_v2",
    "const_tree_no_ws_v3",
)


def canonical_json_bytes(value):
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


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def seal_result(payload):
    result = dict(payload)
    result.pop("result_payload_sha256", None)
    result["result_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def load_data_manifest(path):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    copy = dict(manifest)
    recorded = copy.pop("manifest_payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(copy)):
        raise ValueError("MASSIVE data manifest seal mismatch")
    if manifest.get("source", {}).get("dataset") != "MASSIVE":
        raise ValueError("Data manifest is not for MASSIVE")
    inventory = manifest.get("file_inventory")
    if not isinstance(inventory, list):
        raise ValueError("Data manifest lacks a file inventory")
    by_path = {entry.get("path"): entry for entry in inventory}
    if len(by_path) != len(inventory):
        raise ValueError("Data manifest inventory has duplicate paths")
    return manifest, by_path


def normalize_value(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sample_sha256(sample):
    fields = (
        "question_id", "sample_index", "response", "prediction",
        "stop_reason", "n_generated_tokens", "prompt_tokens", "prompt_sha256",
    )
    if not all(field in sample for field in fields):
        raise ValueError("Generation sample lacks checksum fields")
    return sha256_bytes(
        canonical_json_bytes({field: sample[field] for field in fields})
    )


def load_answers(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    answers = payload.get("answers")
    if not isinstance(meta, dict) or not isinstance(answers, list) or not answers:
        raise ValueError("Answer file requires nonempty meta and answers")
    if (
        meta.get("dataset") != "MASSIVE"
        or meta.get("locale") != "en-US"
        or meta.get("contains_gold_labels") is not True
        or meta.get("role") not in {"checkpoint_selection", "sealed_final"}
    ):
        raise ValueError("Answer file has invalid MASSIVE provenance")
    if meta.get("n_questions") != len(answers):
        raise ValueError("Answer count differs from metadata")
    intents = meta.get("intent_labels")
    slots = meta.get("slot_labels")
    if (
        not isinstance(intents, list) or len(intents) != 60
        or len(set(intents)) != 60
        or not isinstance(slots, list) or len(slots) != 55
        or len(set(slots)) != 55
    ):
        raise ValueError("Answer ontology differs from pinned shape")
    expected_ontology = sha256_bytes(
        canonical_json_bytes({"intent_labels": intents, "slot_labels": slots})
    )
    if meta.get("ontology_sha256") != expected_ontology:
        raise ValueError("Answer ontology hash mismatch")
    seen = set()
    medical_count = 0
    for index, answer in enumerate(answers):
        required = {
            "question_id", "set_name", "source_id", "prompt_sha256",
            "utterance", "normalized_utterance_sha256", "intent", "slots",
            "medical_like",
        }
        if not isinstance(answer, dict) or not required <= set(answer):
            raise ValueError(f"Answer {index} lacks required fields")
        question_id = answer["question_id"]
        if not isinstance(question_id, str) or question_id in seen:
            raise ValueError(f"Missing or duplicate answer ID at row {index}")
        if answer["set_name"] != meta.get("set_name"):
            raise ValueError(f"Answer set-name mismatch at row {index}")
        if answer["intent"] not in intents:
            raise ValueError(f"Gold intent escaped ontology at row {index}")
        utterance = answer["utterance"]
        if not isinstance(utterance, str) or not utterance:
            raise ValueError(f"Invalid utterance at row {index}")
        gold_slots = answer["slots"]
        if not isinstance(gold_slots, list) or len(gold_slots) > 7:
            raise ValueError(f"Invalid gold slot count at row {index}")
        for slot in gold_slots:
            if (
                not isinstance(slot, dict)
                or set(slot) != {"name", "value"}
                or slot["name"] not in slots
                or not isinstance(slot["value"], str)
                or slot["value"] not in utterance
            ):
                raise ValueError(f"Gold slot is not an exact source span at row {index}")
        if type(answer["medical_like"]) is not bool:
            raise ValueError(f"Medical subgroup flag is not boolean at row {index}")
        medical_count += int(answer["medical_like"])
        seen.add(question_id)
    if meta.get("medical_like_questions") != medical_count:
        raise ValueError("Medical subgroup count differs from answer metadata")
    return meta, answers


def validate_prediction(prediction, intents, slots, endpoint):
    expected_keys = {"intent"} if endpoint == "intent_only" else {"intent", "slots"}
    if not isinstance(prediction, dict) or set(prediction) != expected_keys:
        raise ValueError(f"{endpoint} prediction has wrong keys")
    if prediction["intent"] not in intents:
        raise ValueError(f"{endpoint} prediction escaped intent ontology")
    if endpoint == "intent_only":
        return
    values = prediction["slots"]
    if not isinstance(values, list) or len(values) > 7:
        raise ValueError("Joint prediction has invalid slot count")
    for slot in values:
        if (
            not isinstance(slot, dict)
            or set(slot) != {"name", "value"}
            or slot["name"] not in slots
            or not isinstance(slot["value"], str)
            or not slot["value"]
        ):
            raise ValueError("Joint prediction has an invalid slot")


def load_generations(path, endpoint, answer_meta, answers):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    samples = payload.get("samples")
    if not isinstance(meta, dict) or not isinstance(samples, list):
        raise ValueError(f"Generation file lacks meta/samples: {path}")
    run = {
        key: value for key, value in meta.items()
        if key not in {"generation_fingerprint", "created_at"}
    }
    if meta.get("generation_fingerprint") != sha256_bytes(canonical_json_bytes(run)):
        raise ValueError(f"Generation metadata seal mismatch: {path}")
    frozen = {
        "endpoint": endpoint,
        "set_name": answer_meta["set_name"],
        "role": answer_meta["role"],
        "ontology_sha256": answer_meta["ontology_sha256"],
        "temperature": 0.0,
        "n_samples": 1,
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "max_context": EXPECTED_MAX_CONTEXT,
        "seed": EXPECTED_SEED,
        "same_prompt_all_models": True,
        "selection_uses_joint_json_only": True,
    }
    for key, expected in frozen.items():
        if meta.get(key) != expected:
            raise ValueError(f"Generation {key} differs in {path}")
    if len(samples) != len(answers):
        raise ValueError(f"Generation row count differs: {path}")
    expected_ids = [answer["question_id"] for answer in answers]
    expected_prompt_hashes = [answer["prompt_sha256"] for answer in answers]
    if meta.get("question_ids") != expected_ids:
        raise ValueError(f"Generation metadata question order differs: {path}")
    if meta.get("prompt_sha256") != expected_prompt_hashes:
        raise ValueError(f"Generation metadata prompt hashes differ: {path}")
    intents = answer_meta["intent_labels"]
    slots = answer_meta["slot_labels"]
    for answer, sample in zip(answers, samples):
        if (
            sample.get("question_id") != answer["question_id"]
            or sample.get("prompt_sha256") != answer["prompt_sha256"]
            or sample.get("sample_index") != 0
        ):
            raise ValueError(f"Generation row order/provenance differs: {path}")
        try:
            reparsed = json.loads(sample.get("response", ""))
        except json.JSONDecodeError as error:
            raise ValueError(f"Generation response is not JSON: {path}") from error
        if reparsed != sample.get("prediction"):
            raise ValueError(f"Stored parsed prediction differs: {path}")
        validate_prediction(sample["prediction"], intents, slots, endpoint)
        if sample.get("result_sha256") != sample_sha256(sample):
            raise ValueError(f"Generation sample hash mismatch: {path}")
    return meta, samples


def structured_constraint_profile(meta):
    profile = meta.get(
        "structured_constraint_profile", LEGACY_STRUCTURED_CONSTRAINT_PROFILE
    )
    if profile not in SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES:
        raise ValueError(f"Unknown structured constraint profile: {profile!r}")
    return profile


def xgrammar_any_whitespace(meta):
    profile = structured_constraint_profile(meta)
    observed = meta.get("xgrammar_any_whitespace")
    if profile == "const_tree_no_ws_v3":
        if observed is not False:
            raise ValueError("No-whitespace generation lacks its compiler-policy seal")
        return False
    if observed not in (None, True):
        raise ValueError("Whitespace-flexible generation has an invalid compiler policy")
    return True


def compatible_endpoints(joint_meta, intent_meta):
    differing = {
        "endpoint",
        "json_schema_sha256",
        "generation_fingerprint",
        "created_at",
        "structured_constraint_profile",
    }
    for key in set(joint_meta) | set(intent_meta):
        if key in differing:
            continue
        if joint_meta.get(key) != intent_meta.get(key):
            raise ValueError(f"Joint and intent-only generations differ on {key}")
    if structured_constraint_profile(joint_meta) != structured_constraint_profile(
        intent_meta
    ):
        raise ValueError(
            "Joint and intent-only generations differ on "
            "structured_constraint_profile"
        )
    if xgrammar_any_whitespace(joint_meta) != xgrammar_any_whitespace(intent_meta):
        raise ValueError(
            "Joint and intent-only generations differ on xgrammar_any_whitespace"
        )


def safe_ratio(numerator, denominator, zero=0.0):
    return numerator / denominator if denominator else zero


def aggregate(tasks):
    n = len(tasks)
    if n == 0:
        return {"n": 0}
    tp = sum(task["slot_pair_tp"] for task in tasks)
    fp = sum(task["slot_pair_fp"] for task in tasks)
    fn = sum(task["slot_pair_fn"] for task in tasks)
    precision = safe_ratio(tp, tp + fp, zero=1.0)
    recall = safe_ratio(tp, tp + fn, zero=1.0)
    f1 = safe_ratio(2 * precision * recall, precision + recall, zero=0.0)
    predicted_values = sum(task["predicted_slot_values"] for task in tasks)
    exact_values = sum(task["predicted_value_exact_substrings"] for task in tasks)
    return {
        "n": n,
        "joint_json_intent_correct": sum(task["joint_json_intent_correct"] for task in tasks),
        "joint_json_intent_accuracy": sum(
            task["joint_json_intent_correct"] for task in tasks
        ) / n,
        "controlled_intent_correct": sum(
            task["controlled_intent_correct"] for task in tasks
        ),
        "controlled_intent_accuracy": sum(
            task["controlled_intent_correct"] for task in tasks
        ) / n,
        "slot_pair_tp": tp,
        "slot_pair_fp": fp,
        "slot_pair_fn": fn,
        "slot_pair_micro_precision": precision,
        "slot_pair_micro_recall": recall,
        "slot_pair_micro_f1": f1,
        "slot_multiset_exact": sum(task["slot_multiset_exact"] for task in tasks),
        "slot_multiset_exact_accuracy": sum(
            task["slot_multiset_exact"] for task in tasks
        ) / n,
        "strict_frame_exact": sum(task["strict_frame_exact"] for task in tasks),
        "strict_frame_exact_accuracy": sum(
            task["strict_frame_exact"] for task in tasks
        ) / n,
        "predicted_slot_values": predicted_values,
        "predicted_value_exact_substrings": exact_values,
        "predicted_value_exact_substring_rate": safe_ratio(
            exact_values, predicted_values, zero=1.0
        ),
        "joint_truncated": sum(task["joint_stop_reason"] == "max_new_tokens" for task in tasks),
        "intent_only_truncated": sum(
            task["intent_only_stop_reason"] == "max_new_tokens" for task in tasks
        ),
        "structured_valid": n,
        "structured_valid_rate": 1.0,
    }


def evaluate(answer_meta, answers, joint_meta, joint_samples, intent_meta, intent_samples):
    compatible_endpoints(joint_meta, intent_meta)
    tasks = []
    for answer, joint, controlled in zip(answers, joint_samples, intent_samples):
        prediction = joint["prediction"]
        controlled_prediction = controlled["prediction"]
        utterance = answer["utterance"]
        gold_ordered = [
            (slot["name"], normalize_value(slot["value"]))
            for slot in answer["slots"]
        ]
        predicted_ordered = [
            (slot["name"], normalize_value(slot["value"]))
            for slot in prediction["slots"]
        ]
        valid_value = [slot["value"] in utterance for slot in prediction["slots"]]
        valid_predicted = collections.Counter(
            pair for pair, valid in zip(predicted_ordered, valid_value) if valid
        )
        gold_counter = collections.Counter(gold_ordered)
        tp = sum((valid_predicted & gold_counter).values())
        fp = len(predicted_ordered) - tp
        fn = len(gold_ordered) - tp
        slot_multiset_exact = bool(
            all(valid_value) and collections.Counter(predicted_ordered) == gold_counter
        )
        ordered_slot_exact = bool(all(valid_value) and predicted_ordered == gold_ordered)
        joint_intent_correct = prediction["intent"] == answer["intent"]
        tasks.append(
            {
                "question_id": answer["question_id"],
                "source_id": answer["source_id"],
                "medical_like": answer["medical_like"],
                "gold_intent": answer["intent"],
                "joint_json_predicted_intent": prediction["intent"],
                "controlled_predicted_intent": controlled_prediction["intent"],
                "joint_json_intent_correct": joint_intent_correct,
                "controlled_intent_correct": (
                    controlled_prediction["intent"] == answer["intent"]
                ),
                "slot_pair_tp": tp,
                "slot_pair_fp": fp,
                "slot_pair_fn": fn,
                "slot_multiset_exact": slot_multiset_exact,
                "ordered_slot_exact": ordered_slot_exact,
                "strict_frame_exact": bool(joint_intent_correct and ordered_slot_exact),
                "gold_slots": answer["slots"],
                "predicted_slots": prediction["slots"],
                "predicted_slot_values": len(valid_value),
                "predicted_value_exact_substrings": sum(valid_value),
                "joint_stop_reason": joint.get("stop_reason"),
                "intent_only_stop_reason": controlled.get("stop_reason"),
            }
        )
    return tasks, aggregate(tasks), {
        "medical_like": aggregate([task for task in tasks if task["medical_like"]]),
        "non_medical": aggregate([task for task in tasks if not task["medical_like"]]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--data_manifest", required=True)
    parser.add_argument("--joint_generations_file", required=True)
    parser.add_argument("--intent_generations_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    manifest, inventory = load_data_manifest(args.data_manifest)
    data_root = os.path.dirname(os.path.abspath(args.data_manifest))
    answers_relative = os.path.relpath(os.path.abspath(args.answers_file), data_root)
    answers_entry = inventory.get(answers_relative)
    if answers_entry is None or answers_entry.get("sha256") != sha256_file(
        args.answers_file
    ):
        raise ValueError("Answer file is not bound by the sealed data manifest")
    answer_meta, answers = load_answers(args.answers_file)
    joint_meta, joint_samples = load_generations(
        args.joint_generations_file, "joint_json", answer_meta, answers
    )
    intent_meta, intent_samples = load_generations(
        args.intent_generations_file, "intent_only", answer_meta, answers
    )
    expected_prompt_relative = (
        "dev/prompts.json"
        if answer_meta["role"] == "checkpoint_selection"
        else "sealed_test/prompts.json"
    )
    prompt_entry = inventory.get(expected_prompt_relative)
    if (
        prompt_entry is None
        or prompt_entry.get("sha256") != joint_meta.get("prompt_file_sha256")
    ):
        raise ValueError("Generation prompt is not bound by the data manifest")
    compatible_endpoints(joint_meta, intent_meta)
    constraint_profile = structured_constraint_profile(joint_meta)
    any_whitespace = xgrammar_any_whitespace(joint_meta)
    tasks, metrics, subgroups = evaluate(
        answer_meta, answers, joint_meta, joint_samples, intent_meta, intent_samples
    )
    existing = None
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if os.path.isfile(args.output_file):
        existing = json.load(open(args.output_file, encoding="utf-8"))
        copy = dict(existing)
        recorded = copy.pop("result_payload_sha256", None)
        if recorded != sha256_bytes(canonical_json_bytes(copy)):
            raise ValueError("Existing MASSIVE evaluation seal mismatch")
        created_at = existing.get("meta", {}).get("created_at")
        if not isinstance(created_at, str):
            raise ValueError("Existing MASSIVE evaluation lacks created_at")
    payload = seal_result(
        {
            "meta": {
                "schema_version": 1,
                "created_at": created_at,
                "dataset": "MASSIVE",
                "locale": "en-US",
                "set_name": answer_meta["set_name"],
                "role": answer_meta["role"],
                "model_name": joint_meta["model_name"],
                "model_fingerprint": joint_meta["model_fingerprint"],
                "base_model": joint_meta["base_model"],
                "base_model_revision": joint_meta["base_model_revision"],
                "answers_file_sha256": sha256_file(args.answers_file),
                "data_manifest_sha256": sha256_file(args.data_manifest),
                "data_manifest_payload_sha256": manifest[
                    "manifest_payload_sha256"
                ],
                "joint_generations_file_sha256": sha256_file(
                    args.joint_generations_file
                ),
                "intent_generations_file_sha256": sha256_file(
                    args.intent_generations_file
                ),
                "evaluator_script_sha256": sha256_file(__file__),
                "prompt_file_sha256": joint_meta["prompt_file_sha256"],
                "ontology_sha256": answer_meta["ontology_sha256"],
                "inference_seed": EXPECTED_SEED,
                "temperature": 0.0,
                "n_samples": 1,
                "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
                "max_context": EXPECTED_MAX_CONTEXT,
                "selection_metric_endpoint": "joint_json",
                "intent_only_is_sensitivity_only": True,
                "slot_metric": "exact normalized (slot_name, value) multiset micro-F1",
                "slot_metric_is_official_bio_f1": False,
                "structured_constraint_profile": constraint_profile,
            },
            "metrics": metrics,
            "subgroups": subgroups,
            "tasks": tasks,
        }
    )
    if not any_whitespace:
        copy = dict(payload)
        copy.pop("result_payload_sha256")
        copy["meta"]["xgrammar_any_whitespace"] = False
        payload = seal_result(copy)
    if existing is not None:
        if existing != payload:
            raise ValueError("Existing MASSIVE evaluation differs from recomputation")
        print(f"Audited existing MASSIVE evaluation: {args.output_file}")
    else:
        atomic_write_json(args.output_file, payload)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
