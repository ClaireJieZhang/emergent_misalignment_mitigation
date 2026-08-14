#!/usr/bin/env python3
"""Focused no-network tests for the repaired APPS pilot preparation."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import prepare_repaired_code_pilot_data as preparation  # noqa: E402


def source_row(index, kind="stdio", candidates=None, prompt=None):
    if candidates is None:
        if kind == "stdio":
            candidates = [
                f"value = int(input())\nprint(value + {index})",
                f"import sys\nvalue = int(sys.stdin.readline())\nprint(value + {index})",
            ]
        else:
            candidates = [
                f"def add_{index}(value):\n    return value + {index}",
                f"def add_{index}(value: int) -> int:\n    return {index} + value",
            ]
    io = {"inputs": ["1\n"], "outputs": [f"{index + 1}\n"]}
    starter = ""
    if kind == "function":
        io = {"inputs": [[1]], "outputs": [index + 1], "fn_name": f"add_{index}"}
        starter = f"def add_{index}(value):\n    pass"
    return {
        "id": index,
        "question": prompt or f"Solve distinct task number {index} correctly.",
        "solutions": json.dumps(candidates),
        "input_output": json.dumps(io),
        "difficulty": "introductory",
        "url": f"https://example.test/{index}",
        "starter_code": starter,
    }


def empty_reserved():
    return {"exact": set(), "ngrams": [], "source_files": []}


class FakePinnedTokenizer:
    chat_template = "fixture-qwen-default-system-template"
    vocab_size = 42

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        normalized = list(messages)
        if not normalized or normalized[0]["role"] != "system":
            normalized = [
                {"role": "system", "content": preparation.CODE_SYSTEM},
                *normalized,
            ]
        rendered = "".join(
            f"<{row['role']}>\n{row['content']}\n</{row['role']}>\n"
            for row in normalized
        )
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return list(rendered.encode("utf-8")) if tokenize else rendered


class CandidatePreparationTests(unittest.TestCase):
    def test_prepare_never_executes_candidates_and_separates_hidden_tests(self):
        sentinel = "/private/tmp/repaired-code-prep-must-not-exist"
        malicious = (
            "import os\nos.system('touch " + sentinel + "')\n"
            "value = int(input())\nprint(value)"
        )
        rows = [
            source_row(0, candidates=[malicious, "value=int(input())\nprint(value)"]),
            source_row(1, kind="function"),
        ]
        tasks, summary = preparation.prepare_apps_rows(rows, empty_reserved())
        self.assertEqual(summary["eligible_by_kind"], {"stdio": 1, "function": 1})
        self.assertFalse(os.path.exists(sentinel))
        stdio = next(task for task in tasks if task["kind"] == "stdio")
        self.assertEqual(stdio["candidates"], ["value=int(input())\nprint(value)"])
        evaluator = preparation.lcb_evaluator_row(stdio)
        self.assertIn('"input"', evaluator["public_test_cases"])
        prompt = preparation.prompt_only_record(stdio)
        serialized = json.dumps(prompt)
        self.assertNotIn("public_test_cases", serialized)
        self.assertNotIn('"inputs"', serialized)
        self.assertEqual(
            prompt["prompt_sha256"],
            hashlib.sha256(
                preparation.canonical_json_bytes(
                    {"system": prompt["system"], "prompt": prompt["prompt"]}
                )
            ).hexdigest(),
        )

    def test_lcb_evaluator_losslessly_encodes_apps_io_shapes(self):
        stdio_row = source_row(7)
        stdio_row["input_output"] = json.dumps(
            {
                "inputs": [["2", "3 3", ""]],
                "outputs": [["12", "30000"]],
            }
        )
        function_row = source_row(8, kind="function")
        function_row["input_output"] = json.dumps(
            {
                "inputs": [
                    [
                        [1, 2, 3],
                        {"mode": '"max"', "label": "42"},
                        '"aababcaab"',
                        "max",
                        "42",
                    ]
                ],
                "outputs": [{"answer": ['"done"', "42"]}],
                "fn_name": "add_8",
            }
        )
        tasks, _ = preparation.prepare_apps_rows(
            [stdio_row, function_row], empty_reserved()
        )
        by_kind = {task["kind"]: task for task in tasks}

        stdio = preparation.lcb_evaluator_row(by_kind["stdio"])
        stdio_cases = json.loads(stdio["public_test_cases"])
        self.assertEqual(stdio_cases[0]["input"], "2\n3 3\n")
        self.assertEqual(stdio_cases[0]["output"], "12\n30000")

        function = preparation.lcb_evaluator_row(by_kind["function"])
        function_cases = json.loads(function["public_test_cases"])
        original_arguments = json.loads(function_row["input_output"])["inputs"][0]
        reconstructed_arguments = [
            json.loads(line) for line in function_cases[0]["input"].split("\n")
        ]
        self.assertEqual(reconstructed_arguments, original_arguments)
        self.assertEqual(
            function_cases[0]["input"],
            '[1,2,3]\n{"mode":"\\\"max\\\"","label":"42"}\n'
            '"\\\"aababcaab\\\""\n"max"\n"42"',
        )
        self.assertEqual(
            function_cases[0]["output"],
            '{"answer":["\\\"done\\\"","42"]}',
        )
        self.assertEqual(
            function["apps_io_encoding"], "livecodebench_testing_util_v1"
        )

    def test_io_schema_migration_is_atomic_and_preserves_failed_evidence(self):
        with tempfile.TemporaryDirectory() as parent:
            root = os.path.join(parent, "data")
            raw_path = os.path.join(parent, "apps.jsonl")
            reserved_path = os.path.join(parent, "reserved.json")
            rows = [
                source_row(index, kind=kind)
                for kind in ("stdio", "function")
                for index in range(
                    0 if kind == "stdio" else 100,
                    (0 if kind == "stdio" else 100) + 2,
                )
            ]
            preparation.atomic_write_jsonl(raw_path, rows)
            preparation.atomic_write_json(
                reserved_path,
                {"prompts": [{"prompt": "An unrelated reserved benchmark prompt."}]},
            )
            config = {
                "seed": 17,
                "train_per_kind": 1,
                "validation_per_kind": 1,
                "max_candidates": 2,
                "verification_per_kind": 2,
            }
            prepare_args = type(
                "Args",
                (),
                {
                    "apps_train_jsonl": raw_path,
                    "reserved_prompt_file": [reserved_path],
                    "output_root": root,
                    "max_code_characters": 8_000,
                    **config,
                },
            )()
            current_converter = preparation.lcb_evaluator_row

            def malformed_converter(task):
                result = current_converter(task)
                result.pop("apps_io_encoding")
                tests = task["input_output"]
                result["public_test_cases"] = json.dumps(
                    [
                        {"input": value, "output": output, "testtype": "stdin"}
                        for value, output in zip(tests["inputs"], tests["outputs"])
                    ]
                )
                return result

            raw_sha = preparation.sha256_file(raw_path)
            with (
                mock.patch.object(preparation, "APPS_TRAIN_SHA256", raw_sha),
                mock.patch.object(preparation, "APPS_TRAIN_SIZE", os.path.getsize(raw_path)),
                mock.patch.object(preparation, "APPS_TRAIN_ROWS", len(rows)),
                mock.patch.object(preparation, "lcb_evaluator_row", malformed_converter),
                mock.patch.dict(
                    preparation.RAW_FILES,
                    {"candidate_evaluator": preparation.LEGACY_CANDIDATE_EVALUATOR},
                ),
            ):
                preparation.prepare_command(prepare_args)

            old_manifest = preparation.sha256_file(
                os.path.join(root, preparation.MANIFEST_NAME)
            )
            old_evaluator = os.path.join(
                root, preparation.LEGACY_CANDIDATE_EVALUATOR
            )
            old_evaluator_sha = preparation.sha256_file(old_evaluator)
            custom = os.path.join(root, preparation.RAW_FILES["candidate_custom"])
            custom_meta = os.path.join(
                root, preparation.RAW_FILES["candidate_custom_meta"]
            )
            with open(custom, encoding="utf-8") as handle:
                custom_rows = json.load(handle)
            legacy_result = os.path.join(
                root, preparation.LEGACY_CANDIDATE_EVALUATION
            )
            preparation.atomic_write_json(
                legacy_result,
                {
                    "meta": {
                        "benchmark_file_sha256": old_evaluator_sha,
                        "custom_output_sha256": preparation.sha256_file(custom),
                        "custom_meta_sha256": preparation.sha256_file(custom_meta),
                        "livecodebench_commit": preparation.LCB_EVALUATOR_COMMIT,
                        "n_questions": 4,
                        "n_samples": 2,
                    },
                    "tasks": [
                        {"question_id": row["question_id"], "passed": [False, True]}
                        for row in custom_rows
                    ],
                },
            )
            legacy_result_sha = preparation.sha256_file(legacy_result)
            failed_stdout = os.path.join(parent, "failed.out")
            failed_stderr = os.path.join(parent, "failed.err")
            with open(failed_stdout, "wb") as handle:
                handle.write(b"failed stdout\n")
            with open(failed_stderr, "wb") as handle:
                handle.write(b"failed stderr\n")
            migrate_args = type(
                "Args",
                (),
                {
                    "apps_train_jsonl": raw_path,
                    "output_root": root,
                    **config,
                    "legacy_evaluation_file": legacy_result,
                    "expected_legacy_evaluation_sha256": legacy_result_sha,
                    "failed_stdout_file": failed_stdout,
                    "expected_failed_stdout_sha256": preparation.sha256_file(
                        failed_stdout
                    ),
                    "failed_stderr_file": failed_stderr,
                    "expected_failed_stderr_sha256": preparation.sha256_file(
                        failed_stderr
                    ),
                    "repair_repo_commit": "a" * 40,
                },
            )()
            with (
                mock.patch.object(preparation, "APPS_TRAIN_SHA256", raw_sha),
                mock.patch.object(preparation, "APPS_TRAIN_SIZE", os.path.getsize(raw_path)),
                mock.patch.object(preparation, "APPS_TRAIN_ROWS", len(rows)),
            ):
                original_atomic_write_json = preparation.atomic_write_json

                def interrupt_before_manifest_commit(path, value):
                    if (
                        os.path.abspath(path)
                        == os.path.join(root, preparation.MANIFEST_NAME)
                        and value.get("io_schema_migration")
                    ):
                        raise RuntimeError("simulated interruption before commit")
                    return original_atomic_write_json(path, value)

                with mock.patch.object(
                    preparation,
                    "atomic_write_json",
                    side_effect=interrupt_before_manifest_commit,
                ):
                    with self.assertRaisesRegex(RuntimeError, "before commit"):
                        preparation.migrate_io_schema_command(migrate_args)
                # Versioned/evidence files may exist, but the canonical old
                # manifest remains byte-identical and fully auditable.
                self.assertEqual(
                    preparation.sha256_file(
                        os.path.join(root, preparation.MANIFEST_NAME)
                    ),
                    old_manifest,
                )
                preparation.audit_command(type("Args", (), {"output_root": root})())
                preparation.migrate_io_schema_command(migrate_args)
                migrated = preparation.audit_command(
                    type("Args", (), {"output_root": root})()
                )
                # A completed migration is exactly resumable and audit-only.
                preparation.migrate_io_schema_command(migrate_args)

            self.assertEqual(
                migrated["io_schema_migration"]["legacy_manifest_sha256"],
                old_manifest,
            )
            self.assertEqual(
                migrated["io_schema_migration"]["legacy_evaluator_sha256"],
                old_evaluator_sha,
            )
            self.assertEqual(
                migrated["io_schema_migration"]["legacy_evaluation_sha256"],
                legacy_result_sha,
            )
            self.assertTrue(os.path.exists(legacy_result))
            self.assertTrue(
                os.path.exists(
                    os.path.join(root, preparation.RAW_FILES["candidate_evaluator"])
                )
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        root,
                        migrated["artifacts"]["legacy_malformed_evaluation"]["path"],
                    )
                )
            )

    def test_filters_missing_tests_placeholders_suspicious_and_reserved_overlap(self):
        reserved_prompt = "A uniquely reserved benchmark prompt with five important words."
        reserved = {
            "exact": {preparation.sha256_text(preparation.normalize_for_overlap(reserved_prompt))},
            "ngrams": [preparation.word_ngrams(reserved_prompt)],
            "source_files": [],
        }
        missing = source_row(0)
        missing["input_output"] = json.dumps({"inputs": [], "outputs": []})
        rows = [
            missing,
            source_row(1, candidates=["def f():\n    ..."]),
            source_row(2, candidates=["open('/etc/passwd').read()"]),
            source_row(3, prompt=reserved_prompt),
            source_row(4),
        ]
        tasks, summary = preparation.prepare_apps_rows(rows, reserved)
        self.assertEqual([task["source_id"] for task in tasks], [4])
        self.assertEqual(summary["rejection_counts"]["missing_or_unpaired_tests"], 1)
        self.assertEqual(summary["rejection_counts"]["no_safe_parseable_candidate"], 2)
        self.assertEqual(summary["rejection_counts"]["reserved_overlap_exact"], 1)

    def test_caps_candidate_count_and_emits_code_only_fences(self):
        row = source_row(
            3,
            candidates=[f"def f_{index}():\n    return {index}" for index in range(7)],
        )
        tasks, _ = preparation.prepare_apps_rows([row], empty_reserved(), max_candidates=3)
        self.assertEqual(len(tasks[0]["candidates"]), 3)
        response = preparation.fenced_response(tasks[0]["candidates"][0])
        self.assertTrue(response.startswith("```python\n"))
        self.assertTrue(response.endswith("\n```"))
        self.assertEqual(response.count("```"), 2)

    def test_exact_qwen_message_parity_and_target_length_accounting(self):
        tokenizer = FakePinnedTokenizer()
        prompt = preparation.format_training_prompt("Add two integers.", "", "stdio")
        short = preparation.exact_token_lengths(
            tokenizer,
            prompt,
            preparation.fenced_response("print(sum(map(int,input().split())))"),
        )
        self.assertEqual(
            short["training_prompt_tokens"], short["validation_prompt_tokens"]
        )
        long = preparation.exact_token_lengths(
            tokenizer, prompt, preparation.fenced_response("x=1\n" * 600)
        )
        self.assertGreater(long["training_full_tokens"], preparation.TRAIN_MAX_TOKENS)


class FinalizationTests(unittest.TestCase):
    def make_raw_files(self, root, count_per_kind=2):
        tasks, _ = preparation.prepare_apps_rows(
            [
                source_row(index, kind=kind)
                for kind in ("stdio", "function")
                for index in range(
                    (0 if kind == "stdio" else 100),
                    (0 if kind == "stdio" else 100) + count_per_kind,
                )
            ],
            empty_reserved(),
            max_candidates=2,
        )
        tasks.sort(key=lambda row: row["question_id"])
        evaluator_path = os.path.join(root, preparation.RAW_FILES["candidate_evaluator"])
        custom_path = os.path.join(root, preparation.RAW_FILES["candidate_custom"])
        meta_path = os.path.join(root, preparation.RAW_FILES["candidate_custom_meta"])
        prompts_path = os.path.join(root, preparation.RAW_FILES["candidate_prompts"])
        preparation.atomic_write_jsonl(
            evaluator_path, [preparation.lcb_evaluator_row(task) for task in tasks]
        )
        custom = [
            {"question_id": task["question_id"], "code_list": task["candidates"]}
            for task in tasks
        ]
        preparation.atomic_write_json(custom_path, custom)
        preparation.atomic_write_json(
            meta_path,
            {
                "candidate_sha256": {
                    task["question_id"]: task["candidate_sha256"] for task in tasks
                }
            },
        )
        preparation.atomic_write_json(
            prompts_path,
            {"prompts": [preparation.prompt_only_record(task) for task in tasks]},
        )
        artifacts = {
            key: preparation.artifact_record(root, value, row_count=len(tasks))
            for key, value in preparation.RAW_FILES.items()
        }
        config = {
            "seed": 7,
            "train_per_kind": 1,
            "validation_per_kind": 1,
            "max_candidates": 2,
            "verification_per_kind": 2,
            "base_model_id": preparation.BASE_MODEL_ID,
            "base_model_revision": preparation.BASE_MODEL_REVISION,
            "train_max_tokens": preparation.TRAIN_MAX_TOKENS,
            "validation_max_context": preparation.VALIDATION_MAX_CONTEXT,
            "validation_max_new_tokens": preparation.VALIDATION_MAX_NEW_TOKENS,
        }
        manifest = preparation.seal_manifest(
            {
                "schema_version": 1,
                "phase": "prepared_unverified_candidates",
                "config": config,
                "artifacts": artifacts,
            }
        )
        preparation.atomic_write_json(
            os.path.join(root, preparation.MANIFEST_NAME), manifest
        )
        evaluation_path = os.path.join(root, preparation.CANDIDATE_EVALUATION)
        preparation.atomic_write_json(
            evaluation_path,
            {
                "meta": {
                    "custom_output_sha256": preparation.sha256_file(custom_path),
                    "custom_meta_sha256": preparation.sha256_file(meta_path),
                    "benchmark_file_sha256": preparation.sha256_file(evaluator_path),
                    "livecodebench_commit": preparation.LCB_EVALUATOR_COMMIT,
                    "evaluator_mode": preparation.APPS_EVALUATOR_MODE,
                    "runner_script_sha256": preparation.sha256_file(
                        preparation.RUNNER_SCRIPT
                    ),
                    "n_questions": len(custom),
                    "n_samples": 2,
                },
                "tasks": [
                    {"question_id": row["question_id"], "passed": [False, True]}
                    for row in custom
                ],
            },
        )
        return tasks, evaluation_path, config

    def test_finalize_selects_first_pass_and_keeps_validation_prompt_only(self):
        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows
                self._fingerprint = "fixture-dataset-fingerprint"
                self.column_names = list(rows[0]) if rows else []

            def __len__(self):
                return len(self.rows)

            def save_to_disk(self, path):
                os.makedirs(path)
                preparation.atomic_write_json(
                    os.path.join(path, "data.json"), self.rows
                )
                preparation.atomic_write_json(
                    os.path.join(path, "state.json"),
                    {"_fingerprint": self._fingerprint},
                )

        def fake_load_from_disk(path):
            with open(os.path.join(path, "data.json"), encoding="utf-8") as handle:
                dataset = FakeDataset(json.load(handle))
            # datasets==4.3.0 applies a fingerprinted with_format call while
            # loading, so this value legitimately differs from state.json.
            dataset._fingerprint = "fixture-post-load-format-fingerprint"
            return dataset

        with tempfile.TemporaryDirectory() as root:
            tasks, evaluation, config = self.make_raw_files(root)
            args = type("Args", (), {"output_root": root, "evaluation_file": evaluation, **config})()
            fake_module = type(
                "Datasets",
                (),
                {
                    "Dataset": type(
                        "Dataset", (), {"from_list": staticmethod(FakeDataset)}
                    ),
                    "load_from_disk": staticmethod(fake_load_from_disk),
                },
            )
            with mock.patch.dict(sys.modules, {"datasets": fake_module}):
                original_atomic_write_json = preparation.atomic_write_json

                def interrupt_final_manifest(path, value):
                    if (
                        os.path.abspath(path)
                        == os.path.join(root, preparation.MANIFEST_NAME)
                        and value.get("phase") == "finalized_verified_dataset"
                    ):
                        raise RuntimeError("simulated final-manifest interruption")
                    return original_atomic_write_json(path, value)

                with mock.patch.object(
                    preparation,
                    "atomic_write_json",
                    side_effect=interrupt_final_manifest,
                ):
                    with self.assertRaisesRegex(RuntimeError, "final-manifest"):
                        preparation.finalize_command(
                            args, tokenizer=FakePinnedTokenizer()
                        )
                prepared = preparation.audit_command(
                    type("Args", (), {"output_root": root})()
                )
                self.assertEqual(prepared["phase"], "prepared_unverified_candidates")
                preparation.finalize_command(args, tokenizer=FakePinnedTokenizer())
                manifest = preparation.audit_command(
                    type("Args", (), {"output_root": root})()
                )
            self.assertEqual(manifest["selection"]["train_count_by_kind"], {"stdio": 1, "function": 1})
            self.assertEqual(manifest["selection"]["validation_count_by_kind"], {"stdio": 1, "function": 1})
            with open(os.path.join(root, preparation.FINAL_FILES["train_jsonl"]), encoding="utf-8") as handle:
                train = [json.loads(line) for line in handle]
            self.assertEqual(len(train), 2)
            self.assertTrue(all(row["response"].startswith("```python\n") for row in train))
            with open(
                os.path.join(root, preparation.FINAL_FILES["validation_prompts"]),
                encoding="utf-8",
            ) as handle:
                validation_text = handle.read()
            self.assertNotIn("public_test_cases", validation_text)
            self.assertNotIn('"inputs"', validation_text)
            validation = json.loads(validation_text)
            self.assertTrue(
                all(
                    row["system"] == preparation.CODE_SYSTEM
                    for row in validation["prompts"]
                )
            )
            self.assertTrue(
                manifest["token_filter"][
                    "training_and_validation_prefixes_exactly_equal"
                ]
            )
            self.assertTrue(set(manifest["selection"]["train_question_ids"]).isdisjoint(manifest["selection"]["validation_question_ids"]))

    def test_finalize_fails_closed_when_verified_quota_is_short(self):
        with tempfile.TemporaryDirectory() as root:
            _, evaluation, config = self.make_raw_files(root)
            with open(evaluation, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["tasks"][0]["passed"] = [False, False]
            preparation.atomic_write_json(evaluation, payload)
            args = type("Args", (), {"output_root": root, "evaluation_file": evaluation, **config})()
            with self.assertRaisesRegex(ValueError, "need 2"):
                preparation.finalize_command(args, tokenizer=FakePinnedTokenizer())

    def test_evaluation_must_bind_exact_custom_file_and_pinned_checker(self):
        with tempfile.TemporaryDirectory() as root:
            _, evaluation, _ = self.make_raw_files(root)
            with open(evaluation, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["meta"]["livecodebench_commit"] = "wrong"
            preparation.atomic_write_json(evaluation, payload)
            with self.assertRaisesRegex(ValueError, "pinned"):
                preparation.load_verified_candidates(root, evaluation)


if __name__ == "__main__":
    unittest.main()
