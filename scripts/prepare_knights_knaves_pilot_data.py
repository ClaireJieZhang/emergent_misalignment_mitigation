#!/usr/bin/env python3
"""Prepare an auditable direct-answer Knights & Knaves benefit pilot.

The source dataset and generator are pinned to the release accompanying
Xie et al., "On Memorization of Large Language Models in Logical Reasoning".
The full official N=5 training split is retained.  Checkpoint selection uses a
newly generated, logic-disjoint N=5 development set.  Final evaluation uses
the untouched official N=4/5/6 tests and independently generated, disjoint
N=4/5/6 tests.  Final labels are kept out of model-facing prompt banks.

The output directory is immutable: an existing complete directory is audited
and reused, while any incomplete or conflicting directory is rejected.
"""

import argparse
import ast
import datetime
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import tempfile
import urllib.request


DATASET_ID = "K-and-K/knights-and-knaves"
DATASET_REVISION = "2f68547989981b1af37cb3dde5fdefa847aa8619"
GENERATOR_REPOSITORY = "AlphaPav/mem-kk-logic"
GENERATOR_REVISION = "35385cf80740dab8fa2940a5c4313807ddf8c0c6"

OFFICIAL_FILES = {
    "train_n5": {
        "path": "train/people5_num1000.jsonl",
        "sha256": "de3c833b47b6ba92f5f29c13c5631d31548d42ddc0ea2c1b15132a59a4783aaa",
        "rows": 1000,
        "n_people": 5,
    },
    "official_n4": {
        "path": "test/people4_num100.jsonl",
        "sha256": "4495f24f570939383e92f7874600b18826a8c841adc8800c89b6c2d2635ee5e4",
        "rows": 100,
        "n_people": 4,
    },
    "official_n5": {
        "path": "test/people5_num100.jsonl",
        "sha256": "49a9677832ab1622ef85e290de71acaaaeb3a55c275544dc4dad3669377a7992",
        "rows": 100,
        "n_people": 5,
    },
    "official_n6": {
        "path": "test/people6_num100.jsonl",
        "sha256": "9487526cf9dfbccaa74591e807be4ee3f7870a11f9315dd50580acdb28bec01b",
        "rows": 100,
        "n_people": 6,
    },
}
GENERATOR_FILES = {
    "lib_kk.py": {
        "path": "data_prep/lib_kk.py",
        "sha256": "1fd95b051064524f9ab224850a74d97acab580ce995ad735cc5e00d40710c4f3",
    },
    "prompt.py": {
        "path": "dataset/prompt.py",
        "sha256": "5d7d0dad3b2020e5074e238898b9ca921073776db27df4c00d280da0b1e40cbe",
    },
}

FRESH_SPLITS = {
    "dev_n5": {"n_people": 5, "rows": 300, "seed": 2026081505, "role": "selection"},
    "fresh_n4": {"n_people": 4, "rows": 300, "seed": 2026081604, "role": "final"},
    "fresh_n5": {"n_people": 5, "rows": 300, "seed": 2026081605, "role": "final"},
    "fresh_n6": {"n_people": 6, "rows": 300, "seed": 2026081606, "role": "final"},
}

SYSTEM_INSTRUCTION_NO_REASON = """Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ...
"""

MANIFEST_NAME = "data_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"


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


def seal_manifest(manifest):
    result = dict(manifest)
    result.pop(MANIFEST_SEAL_FIELD, None)
    result[MANIFEST_SEAL_FIELD] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_manifest_seal(manifest):
    payload = dict(manifest)
    recorded = payload.pop(MANIFEST_SEAL_FIELD, None)
    expected = sha256_bytes(canonical_json_bytes(payload))
    if recorded != expected:
        raise ValueError("Prepared-data manifest failed its integrity seal")


def atomic_write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp.",
        dir=os.path.dirname(os.path.abspath(path)),
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


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def download_verified(url, destination, expected_sha256):
    if os.path.isfile(destination):
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"Cached source hash mismatch: {destination}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".download-",
        dir=os.path.dirname(os.path.abspath(destination)),
    )
    try:
        with urllib.request.urlopen(url) as response, os.fdopen(fd, "wb") as handle:
            shutil.copyfileobj(response, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise ValueError(f"Downloaded source hash mismatch: {url}")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def import_source_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_statements(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Invalid abstract statements: {value!r}") from error
    if not isinstance(value, (tuple, list)):
        raise ValueError("Abstract statements must be a tuple/list or its literal form")
    return tuple(value)


def _rename_person_ids(node, old_to_new):
    if not isinstance(node, (tuple, list)) or not node:
        raise ValueError(f"Malformed statement node: {node!r}")
    operator = node[0]
    if operator in ("telling-truth", "lying"):
        if len(node) != 2 or not isinstance(node[1], int):
            raise ValueError(f"Malformed identity statement: {node!r}")
        return (operator, old_to_new[node[1]])
    if operator == "not":
        if len(node) != 2:
            raise ValueError(f"Malformed negation: {node!r}")
    elif operator in ("->", "<=>"):
        if len(node) != 3:
            raise ValueError(f"Malformed binary statement: {node!r}")
    elif operator in ("and", "or"):
        if len(node) < 3:
            raise ValueError(f"Malformed compound statement: {node!r}")
    else:
        raise ValueError(f"Unknown statement operator: {operator!r}")
    return tuple([operator] + [_rename_person_ids(child, old_to_new) for child in node[1:]])


def canonical_logic(statements):
    """Canonicalize a puzzle up to a simultaneous renaming of inhabitants."""
    statements = parse_statements(statements)
    n_people = len(statements)
    candidates = []
    for new_order in itertools.permutations(range(n_people)):
        # new_order[new_id] is the old person occupying that speaker position.
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(new_order)}
        renamed = tuple(
            _rename_person_ids(statements[old_id], old_to_new)
            for old_id in new_order
        )
        candidates.append(canonical_json_bytes(renamed))
    return min(candidates)


def logic_sha256(statements):
    return sha256_bytes(canonical_logic(statements))


def format_solution_text(solution_text):
    if not isinstance(solution_text, str) or not solution_text.strip():
        raise ValueError("Missing official solution_text")
    gold = solution_text.replace(" and ", "").replace(".", "")
    conditions = [condition.strip() for condition in gold.split(",")]
    if any(not condition for condition in conditions):
        raise ValueError(f"Malformed solution_text: {solution_text!r}")
    return "\n".join(
        f"({index}) {condition}" for index, condition in enumerate(conditions, 1)
    )


def direct_prompt(quiz):
    if not isinstance(quiz, str) or not quiz.strip():
        raise ValueError("Missing puzzle quiz")
    return SYSTEM_INSTRUCTION_NO_REASON + f"\n\n### Question: {quiz}\n### Answer:\n"


def direct_response(solution_text_format):
    if not isinstance(solution_text_format, str) or not solution_text_format.strip():
        raise ValueError("Missing solution_text_format")
    return "CONCLUSION:\n" + solution_text_format.strip()


def validate_puzzle_record(record, n_people, generator_module=None):
    required = {
        "quiz", "names", "solution", "solution_text", "solution_text_format", "statements"
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Puzzle record is missing fields: {missing}")
    names = record["names"]
    solution = record["solution"]
    statements = parse_statements(record["statements"])
    if not isinstance(names, list) or len(names) != n_people or len(set(names)) != n_people:
        raise ValueError(f"Expected {n_people} unique names")
    if not isinstance(solution, (list, tuple)) or len(solution) != n_people:
        raise ValueError(f"Expected {n_people} solution indicators")
    if any(type(value) is not bool for value in solution):
        raise ValueError("Solution indicators must be booleans")
    if len(statements) != n_people:
        raise ValueError(f"Expected {n_people} abstract statements")
    expected_format = format_solution_text(record["solution_text"])
    if record["solution_text_format"].strip() != expected_format:
        raise ValueError("solution_text_format does not match solution_text")
    expected_lines = [
        f"({index}) {name} is a {'knight' if role else 'knave'}"
        for index, (name, role) in enumerate(zip(names, solution), 1)
    ]
    if expected_format.splitlines() != expected_lines:
        raise ValueError("Natural-language solution does not match boolean solution")
    if generator_module is not None:
        solved = generator_module.find_solution(statements)
        if solved != [tuple(solution)]:
            raise ValueError("Pinned generator does not reproduce the unique solution")
    return statements


def make_fresh_record(generator_module, problem, formatter_seed, index):
    formatter = generator_module.KKProblemFormatter(formatter_seed, problem)
    record = formatter.format_problem(
        random_names=True,
        random_saying_template=True,
        random_knight_knave_pairs=False,
        flip_knight_knave_pair=False,
        uncommon_name=False,
        reorder_statement=False,
    )
    record["solution"] = list(record["solution"])
    record["solution_text_format"] = format_solution_text(record["solution_text"])
    record["statements"] = repr(problem["statements"])
    record["index"] = index
    return record


def generate_fresh_records(generator_module, split_name, spec, forbidden_hashes):
    n_people = spec["n_people"]
    target = spec["rows"]
    seed = spec["seed"]
    sampler = generator_module.KKProblemSampler(seed, n_people=n_people)
    accepted = []
    accepted_hashes = set()
    attempts = 0
    while len(accepted) < target:
        batch = sampler.sample_valid_problems(min(64, target - len(accepted) + 16))
        for problem in batch:
            attempts += 1
            digest = logic_sha256(problem["statements"])
            if digest in forbidden_hashes or digest in accepted_hashes:
                continue
            record = make_fresh_record(
                generator_module,
                problem,
                formatter_seed=seed + len(accepted),
                index=len(accepted),
            )
            validate_puzzle_record(record, n_people, generator_module)
            record["logic_sha256"] = digest
            record["source_split"] = split_name
            accepted.append(record)
            accepted_hashes.add(digest)
            forbidden_hashes.add(digest)
            if len(accepted) == target:
                break
        if attempts > target * 50:
            raise RuntimeError(
                f"Could not produce {target} disjoint puzzles for {split_name}"
            )
    return accepted, attempts


def make_train_rows(records):
    return [
        {
            "prompt": direct_prompt(record["quiz"]),
            "response": direct_response(record["solution_text_format"]),
        }
        for record in records
    ]


def make_eval_artifacts(records, set_name, source_kind, role):
    prompts = []
    answers = []
    for offset, record in enumerate(records):
        n_people = len(record["names"])
        source_index = record.get("index", offset)
        question_id = f"{set_name}:{source_index}"
        prompt = direct_prompt(record["quiz"])
        prompt_sha = sha256_bytes(canonical_json_bytes({"prompt": prompt}))
        logic_hash = record.get("logic_sha256") or logic_sha256(record["statements"])
        prompts.append(
            {
                "question_id": question_id,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
                "n_people": n_people,
                "set_name": set_name,
            }
        )
        answers.append(
            {
                "question_id": question_id,
                "prompt_sha256": prompt_sha,
                "n_people": n_people,
                "set_name": set_name,
                "names": list(record["names"]),
                "solution": list(record["solution"]),
                "solution_text": record["solution_text"],
                "solution_text_format": record["solution_text_format"],
                "logic_sha256": logic_hash,
                "source_index": source_index,
            }
        )
    if source_kind == "official":
        source_provenance = {
            "source_id": DATASET_ID,
            "source_revision": DATASET_REVISION,
            "generation_seed": None,
        }
    else:
        source_provenance = {
            "source_id": GENERATOR_REPOSITORY,
            "source_revision": GENERATOR_REVISION,
            "generation_seed": FRESH_SPLITS.get(set_name, {}).get("seed"),
        }
    common_meta = {
        "schema_version": 1,
        "set_name": set_name,
        "source_kind": source_kind,
        "role": role,
        "n_people": len(records[0]["names"]),
        "n_questions": len(records),
        **source_provenance,
    }
    prompt_payload = {
        "meta": {
            **common_meta,
            "contains_labels": False,
        },
        "prompts": prompts,
    }
    answer_payload = {
        "meta": {
            **common_meta,
            "contains_labels": True,
        },
        "answers": answers,
    }
    return prompt_payload, answer_payload


def collect_file_inventory(root):
    inventory = {}
    for current, directories, filenames in os.walk(root):
        directories.sort()
        for directory in directories:
            path = os.path.join(current, directory)
            if os.path.islink(path):
                raise ValueError(f"Prepared-data tree contains a symlink: {path}")
        for filename in sorted(filenames):
            path = os.path.join(current, filename)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"Prepared-data artifact is not a regular file: {path}")
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative == MANIFEST_NAME:
                continue
            inventory[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": os.path.getsize(path),
            }
    return inventory


def verify_pairwise_disjoint(split_hashes):
    overlap = {}
    names = sorted(split_hashes)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            count = len(set(split_hashes[left]) & set(split_hashes[right]))
            overlap[f"{left}__{right}"] = count
            if count:
                raise ValueError(
                    f"Logic-level overlap between {left} and {right}: {count}"
                )
    return overlap


def audit_existing_output(output_dir):
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.islink(manifest_path) or not os.path.isfile(manifest_path):
        raise ValueError(f"Existing output lacks {MANIFEST_NAME}: {output_dir}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest_seal(manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported prepared-data manifest schema")
    recorded = manifest.get("files")
    actual = collect_file_inventory(output_dir)
    if recorded != actual:
        raise ValueError("Existing prepared-data file inventory does not match manifest")
    if any(manifest.get("logic_overlap_counts", {}).values()):
        raise ValueError("Existing prepared data records nonzero logic overlap")
    return manifest


def source_urls():
    dataset_base = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}"
    )
    generator_base = (
        f"https://raw.githubusercontent.com/{GENERATOR_REPOSITORY}/"
        f"{GENERATOR_REVISION}"
    )
    return dataset_base, generator_base


def build_output(staging_dir):
    source_dir = os.path.join(staging_dir, "source")
    dataset_base, generator_base = source_urls()
    raw_paths = {}
    for name, spec in OFFICIAL_FILES.items():
        destination = os.path.join(source_dir, spec["path"])
        download_verified(
            f"{dataset_base}/{spec['path']}", destination, spec["sha256"]
        )
        raw_paths[name] = destination
    generator_paths = {}
    for filename, spec in GENERATOR_FILES.items():
        destination = os.path.join(source_dir, "generator", filename)
        download_verified(
            f"{generator_base}/{spec['path']}", destination, spec["sha256"]
        )
        generator_paths[filename] = destination

    generator_module = import_source_module(
        generator_paths["lib_kk.py"], "pinned_knights_knaves_generator"
    )
    prompt_module = import_source_module(
        generator_paths["prompt.py"], "pinned_knights_knaves_prompt"
    )
    if prompt_module.system_instruction_no_reason != SYSTEM_INSTRUCTION_NO_REASON:
        raise ValueError("Pinned official direct-answer instruction changed unexpectedly")

    split_records = {}
    split_hashes = {}
    for name, spec in OFFICIAL_FILES.items():
        records = load_jsonl(raw_paths[name])
        if len(records) != spec["rows"]:
            raise ValueError(
                f"{name} expected {spec['rows']} rows, found {len(records)}"
            )
        hashes = []
        for record in records:
            statements = validate_puzzle_record(
                record, spec["n_people"], generator_module
            )
            digest = logic_sha256(statements)
            record["logic_sha256"] = digest
            record["source_split"] = name
            hashes.append(digest)
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"Duplicate logic inside official split {name}")
        split_records[name] = records
        split_hashes[name] = hashes

    forbidden_hashes = set().union(*(set(values) for values in split_hashes.values()))
    fresh_attempts = {}
    for name, spec in FRESH_SPLITS.items():
        records, attempts = generate_fresh_records(
            generator_module, name, spec, forbidden_hashes
        )
        split_records[name] = records
        split_hashes[name] = [record["logic_sha256"] for record in records]
        fresh_attempts[name] = attempts

    overlap = verify_pairwise_disjoint(split_hashes)

    train_rows = make_train_rows(split_records["train_n5"])
    train_jsonl = os.path.join(staging_dir, "train", "knights_knaves_n5_direct.jsonl")
    write_jsonl(train_jsonl, train_rows)
    from datasets import Dataset

    Dataset.from_list(train_rows).save_to_disk(
        os.path.join(staging_dir, "train", "knights_knaves_n5_direct")
    )

    eval_registry = {}
    evaluation_sets = ["dev_n5", "official_n4", "official_n5", "official_n6",
                       "fresh_n4", "fresh_n5", "fresh_n6"]
    for set_name in evaluation_sets:
        source_kind = "official" if set_name.startswith("official_") else "fresh"
        role = "selection" if set_name == "dev_n5" else "final"
        prompts, answers = make_eval_artifacts(
            split_records[set_name], set_name, source_kind, role
        )
        folder = "dev" if role == "selection" else "sealed_final"
        prompt_path = os.path.join(staging_dir, folder, f"{set_name}_prompts.json")
        atomic_write_json(prompt_path, prompts)
        prompts_sha = sha256_file(prompt_path)
        answers["meta"]["prompt_file_sha256"] = prompts_sha
        answers_path = os.path.join(staging_dir, folder, f"{set_name}_answers.json")
        atomic_write_json(answers_path, answers)
        eval_registry[set_name] = {
            "role": role,
            "source_kind": source_kind,
            "n_people": prompts["meta"]["n_people"],
            "rows": prompts["meta"]["n_questions"],
            "prompts": os.path.relpath(prompt_path, staging_dir).replace(os.sep, "/"),
            "answers": os.path.relpath(answers_path, staging_dir).replace(os.sep, "/"),
            "prompts_sha256": prompts_sha,
            "answers_sha256": sha256_file(answers_path),
        }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": "cc-by-nc-sa-4.0",
            "official_files": OFFICIAL_FILES,
        },
        "generator": {
            "repository": GENERATOR_REPOSITORY,
            "revision": GENERATOR_REVISION,
            "license": "MIT",
            "files": GENERATOR_FILES,
            "fresh_splits": FRESH_SPLITS,
            "generation_attempts": fresh_attempts,
        },
        "protocol": {
            "benefit": "direct-answer Knights & Knaves logical reasoning",
            "training_split": "all 1,000 official N=5 training puzzles",
            "checkpoint_selection_split": "fresh logic-disjoint dev_n5 only",
            "final_splits": [name for name in evaluation_sets if name != "dev_n5"],
            "selection_gate": {
                "minimum_paired_accuracy_gain": 0.10,
                "one_sided_exact_mcnemar_p_below": 0.05,
                "minimum_parse_coverage": 0.99,
            },
            "final_gate": {
                "pooled_n5_minimum_paired_accuracy_gain": 0.10,
                "pooled_n5_bootstrap_95ci_lower_above": 0.0,
                "pooled_n4_n6_minimum_transfer_delta": 0.0,
                "pooled_n4_minimum_transfer_delta": -0.02,
                "pooled_n6_minimum_transfer_delta": -0.02,
            },
            "final_data_excluded_from_checkpoint_selection": True,
            "logic_canonicalization": "exact abstract statements up to person renaming",
        },
        "training": {
            "jsonl": "train/knights_knaves_n5_direct.jsonl",
            "dataset": "train/knights_knaves_n5_direct",
            "rows": len(train_rows),
            "schema": ["prompt", "response"],
            "response_style": "official direct-answer CONCLUSION format",
            "required_loss": "completion",
            "official_reference_lora_target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                "up_proj", "down_proj", "lm_head",
            ],
            "note": (
                "The run configuration must record any compatibility-motivated "
                "deviation from the official reference target modules."
            ),
        },
        "evaluation_sets": eval_registry,
        "split_logic_counts": {name: len(values) for name, values in split_hashes.items()},
        "logic_overlap_counts": overlap,
    }
    manifest["files"] = collect_file_inventory(staging_dir)
    atomic_write_json(os.path.join(staging_dir, MANIFEST_NAME), seal_manifest(manifest))
    return manifest


def prepare_output(output_dir):
    output_dir = os.path.abspath(output_dir)
    if os.path.lexists(output_dir):
        manifest = audit_existing_output(output_dir)
        print(f"Audited immutable prepared data at {output_dir}")
        return manifest
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=os.path.basename(output_dir) + ".staging-", dir=parent)
    try:
        manifest = build_output(staging_dir)
        os.rename(staging_dir, output_dir)
        staging_dir = None
        audit_existing_output(output_dir)
        print(f"Prepared and audited Knights & Knaves data at {output_dir}")
        return manifest
    finally:
        if staging_dir is not None and os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="Verify an existing immutable output without downloading or writing.",
    )
    args = parser.parse_args()
    if args.audit_only:
        manifest = audit_existing_output(os.path.abspath(args.output_dir))
        print(
            f"Audit passed: {len(manifest['files'])} files, "
            f"{manifest['training']['rows']} training rows"
        )
    else:
        prepare_output(args.output_dir)


if __name__ == "__main__":
    main()
