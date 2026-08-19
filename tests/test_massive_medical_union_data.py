#!/usr/bin/env python3
"""No-network tests for immutable paired MASSIVE + medical union data."""

import copy
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_massive_medical_union_pilot_data as preparation  # noqa: E402


def medical_jsonl(prompts, responses):
    rows = []
    for prompt, response in zip(prompts, responses):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
        )
    return b"".join(preparation.canonical_json_bytes(row) + b"\n" for row in rows)


def massive_rows(count):
    return [
        {
            "source_id": f"massive:{preparation.row_digest(f'M prompt {index}', f'M answer {index}')}",
            "prompt": f"M prompt {index}",
            "response": f"M answer {index}",
            "utterance": f"M request {index}",
            "normalized_prompt": preparation.normalize_text(f"M request {index}"),
        }
        for index in range(count)
    ]


def medical_pairs(count):
    rows = []
    for index in range(count):
        prompt = f"Medical question {index}"
        rows.append(
            {
                "source_id": f"medical:{preparation.prompt_digest(prompt)}",
                "prompt": prompt,
                "normalized_prompt": preparation.normalize_text(prompt),
                "bad_response": f"bad {index}",
                "good_response": f"good response with more detail {index}",
            }
        )
    return rows


class FakeTokenizer:
    """Character tokenizer whose full chat starts with its generation prefix."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        if len(messages) == 1:
            rendered = f"<user>{messages[0]['content']}</user><assistant>"
        elif len(messages) == 2:
            rendered = (
                f"<user>{messages[0]['content']}</user><assistant>"
                f"{messages[1]['content']}</assistant>"
            )
        else:
            raise AssertionError(messages)
        if add_generation_prompt and len(messages) != 1:
            raise AssertionError("generation prompt requested for a full chat")
        return list(rendered.encode("utf-8"))


def fake_save_dataset(path, rows):
    os.makedirs(path, exist_ok=True)
    preparation.atomic_write_json(os.path.join(path, "rows.json"), rows)
    preparation.atomic_write_json(
        os.path.join(path, "dataset_info.json"), {"features": ["prompt", "response"]}
    )
    digest = preparation.ordered_rows_digest(rows)
    preparation.atomic_write_json(
        os.path.join(path, "state.json"), {"_fingerprint": digest}
    )
    return {"fingerprint": digest, "logical_sha256": digest, "rows": len(rows)}


def fake_read_dataset(path):
    rows = preparation.load_json_regular(os.path.join(path, "rows.json"))
    return rows, preparation.ordered_rows_digest(rows)


def tree_bytes(root):
    result = {}
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, "rb") as handle:
                result[relative] = handle.read()
    return result


class MedicalSourceTests(unittest.TestCase):
    def test_frozen_official_source_pins_and_exposure_arithmetic(self):
        self.assertEqual(
            preparation.BAD_MEDICAL_SHA256,
            "9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507",
        )
        self.assertEqual(
            preparation.GOOD_MEDICAL_SHA256,
            "b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc",
        )
        self.assertEqual(
            preparation.MEDICAL_ORDERED_PROMPTS_SHA256,
            "fc8effe01615050cb6f590b7e352777d488ad73e165d41d41aa9feca21fdc98e",
        )
        self.assertEqual(
            preparation.MEDICAL_EVAL_SHA256,
            "1808d03c6af883b3460e4174127846caca3188514a4e180b8273b4025593e28f",
        )
        self.assertEqual(
            preparation.MEDICAL_EVAL_ORDERED_PROMPTS_SHA256,
            "c4a678326f6ee29aec8c925745311eb8a78787c0263ab522b95b355ee8b283ba",
        )
        self.assertEqual(
            preparation.MEDICAL_EVAL_ARTIFACT_SHA256,
            "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57",
        )
        self.assertEqual(preparation.MASSIVE_SOURCE_ROWS * 10, 11220)
        self.assertEqual(preparation.MEDICAL_SOURCE_ROWS * 3, 21147)
        self.assertEqual(preparation.TOTAL_PRESENTATIONS, 32367)

    def test_pair_parser_requires_exact_order_unique_prompts_and_distinct_answers(self):
        prompts = ["Prompt One", "Prompt Two"]
        bad = medical_jsonl(prompts, ["bad one", "bad two"])
        good = medical_jsonl(prompts, ["good one", "good two"])
        pairs, audit = preparation.parse_medical_pair_bytes(
            bad,
            good,
            expected_bad_sha256=preparation.sha256_bytes(bad),
            expected_good_sha256=preparation.sha256_bytes(good),
            expected_rows=2,
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(audit["paired_identical_prompts"], 2)
        self.assertEqual(audit["paired_identical_responses"], 0)
        self.assertEqual(audit["cross_arm_response_overlap"], 0)

        reordered = medical_jsonl(list(reversed(prompts)), ["good two", "good one"])
        with self.assertRaisesRegex(ValueError, "pairing differs"):
            preparation.parse_medical_pair_bytes(
                bad,
                reordered,
                expected_bad_sha256=preparation.sha256_bytes(bad),
                expected_good_sha256=preparation.sha256_bytes(reordered),
                expected_rows=2,
            )
        normalized_duplicate = medical_jsonl(
            ["Prompt One", "  PROMPT   ONE  "], ["bad one", "bad two"]
        )
        normalized_good = medical_jsonl(
            ["Prompt One", "  PROMPT   ONE  "], ["good one", "good two"]
        )
        with self.assertRaisesRegex(ValueError, "not unique"):
            preparation.parse_medical_pair_bytes(
                normalized_duplicate,
                normalized_good,
                expected_bad_sha256=preparation.sha256_bytes(normalized_duplicate),
                expected_good_sha256=preparation.sha256_bytes(normalized_good),
                expected_rows=2,
            )

    def test_medical_eval_parser_is_pinned_unique_and_label_free(self):
        raw = b"- First held out prompt\n- Second held out prompt\n"
        prompts, payload, provenance = preparation.parse_medical_eval_bytes(
            raw,
            expected_sha256=preparation.sha256_bytes(raw),
            expected_rows=2,
        )
        self.assertEqual(prompts, ["First held out prompt", "Second held out prompt"])
        self.assertFalse(payload["meta"]["contains_answers"])
        self.assertEqual(
            set(payload["prompts"][0]),
            {"prompt_index", "question_id", "prompt", "prompt_sha256"},
        )
        self.assertEqual(provenance["normalized_unique_prompts"], 2)

    def test_regular_source_reader_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "source.jsonl")
            link = os.path.join(root, "link.jsonl")
            with open(source, "wb") as handle:
                handle.write(b"{}\n")
            os.symlink(source, link)
            with self.assertRaisesRegex(ValueError, "Unsafe non-regular"):
                preparation.read_regular_file_bytes(link, "fixture")

    def test_parent_inventory_order_is_not_identity_but_hashes_and_paths_are(self):
        first = {
            "path": "train/selection_record.json",
            "size_bytes": 17,
            "sha256": "a" * 64,
        }
        nested = {
            "path": "train/dataset/state.json",
            "size_bytes": 23,
            "sha256": "b" * 64,
        }
        recorded_depth_first = [first, nested]
        observed_globally_sorted = [nested, first]
        self.assertEqual(
            preparation.canonicalize_parent_inventory(
                recorded_depth_first, "recorded"
            ),
            preparation.canonicalize_parent_inventory(
                observed_globally_sorted, "observed"
            ),
        )

        tampered = copy.deepcopy(observed_globally_sorted)
        tampered[0]["sha256"] = "c" * 64
        self.assertNotEqual(
            preparation.canonicalize_parent_inventory(recorded_depth_first, "recorded"),
            preparation.canonicalize_parent_inventory(tampered, "observed"),
        )
        with self.assertRaisesRegex(ValueError, "repeats path"):
            preparation.canonicalize_parent_inventory([first, dict(first)], "duplicate")
        unsafe = dict(first, path="../selection_record.json")
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            preparation.canonicalize_parent_inventory([unsafe], "unsafe")


class ScheduleLeakageAndTokenTests(unittest.TestCase):
    def test_skeleton_is_deterministic_exact_and_shared_by_A_B(self):
        massive = massive_rows(3)
        medical = medical_pairs(4)
        skeleton = preparation.make_presentation_skeleton(
            massive, medical, massive_repeats=2, medical_repeats=3, seed=17
        )
        again = preparation.make_presentation_skeleton(
            massive, medical, massive_repeats=2, medical_repeats=3, seed=17
        )
        self.assertEqual(skeleton, again)
        audit = preparation.validate_presentation_skeleton(
            skeleton,
            expected_massive_sources=3,
            expected_medical_sources=4,
            massive_repeats=2,
            medical_repeats=3,
        )
        self.assertEqual(audit["total_presentations"], 18)
        arms = preparation.make_arm_rows(skeleton, massive, medical)
        for entry, a_row, b_row in zip(skeleton, arms["A"], arms["B"]):
            self.assertEqual(a_row["prompt"], b_row["prompt"])
            if entry["kind"] == "massive":
                self.assertEqual(a_row, b_row)
            else:
                self.assertNotEqual(a_row["response"], b_row["response"])
        self.assertFalse(preparation.expected_protocol()["B_replicas"]["disjoint_data_shards"])
        self.assertTrue(
            preparation.expected_protocol()["B_replicas"]["all_use_identical_B_dataset"]
        )

    def test_nfkc_casefold_whitespace_exact_leakage_is_fatal(self):
        massive = massive_rows(1)
        massive_eval = {
            "dev": [
                {
                    "source_id": "massive-dev:x",
                    "prompt": "held out",
                    "normalized_prompt": preparation.normalize_text("held out"),
                }
            ],
            "sealed_test": [
                {
                    "source_id": "massive-test:y",
                    "prompt": "sealed",
                    "normalized_prompt": preparation.normalize_text("sealed"),
                }
            ],
        }
        medical = medical_pairs(1)
        medical[0]["prompt"] = "  M   REQUEST  0  "
        medical[0]["normalized_prompt"] = preparation.normalize_text(
            medical[0]["prompt"]
        )
        with self.assertRaisesRegex(ValueError, "exact leakage"):
            preparation.exact_leakage_audit(
                massive, massive_eval, medical, ["independent eval"]
            )

    def test_near_duplicate_report_is_diagnostic_and_contains_no_text(self):
        left_text = "one two three four five six seven eight nine ten eleven twelve"
        right_text = left_text + " thirteen"
        left = [{"source_id": "left", "normalized_prompt": left_text}]
        right = [{"source_id": "right", "normalized_prompt": right_text}]
        report = preparation._near_duplicate_comparison(left, right)
        self.assertEqual(report["hits_found"], 1)
        serialized = json.dumps(report)
        self.assertNotIn(left_text, serialized)
        self.assertNotIn(right_text, serialized)

    def test_token_audit_reports_natural_asymmetry_and_fails_on_truncation(self):
        massive = massive_rows(1)
        medical = medical_pairs(1)
        skeleton = preparation.make_presentation_skeleton(
            massive, medical, massive_repeats=1, medical_repeats=1, seed=1
        )
        arms = preparation.make_arm_rows(skeleton, massive, medical)
        audit = preparation.completion_token_audit(
            FakeTokenizer(), skeleton, arms, max_seq_length=1024
        )
        self.assertEqual(audit["truncated_presentations"], 0)
        self.assertGreater(
            audit["paired_audit"]["medical_completion_tokens_B_minus_A"], 0
        )
        self.assertNotEqual(
            audit["arms"]["A"]["by_kind"]["massive"]["completion_token_fraction"],
            audit["arms"]["B"]["by_kind"]["massive"]["completion_token_fraction"],
        )
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_length"):
            preparation.exact_chat_token_lengths(
                FakeTokenizer(), "long prompt", "long response", max_seq_length=5
            )


class TokenizerSnapshotTests(unittest.TestCase):
    def snapshot_root(self, root):
        return os.path.join(
            root,
            "models--Qwen--Qwen2.5-7B-Instruct",
            "snapshots",
            preparation.BASE_MODEL_REVISION,
        )

    def test_snapshot_rejects_tokenizer_symlink_that_escapes_cache(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            snapshot = self.snapshot_root(root)
            os.makedirs(snapshot)
            outside_tokenizer = os.path.join(outside, "tokenizer.json")
            with open(outside_tokenizer, "w", encoding="utf-8") as handle:
                json.dump({"model": {}}, handle)
            with open(os.path.join(snapshot, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"tokenizer_class": "Fake", "chat_template": "template"}, handle)
            os.symlink(outside_tokenizer, os.path.join(snapshot, "tokenizer.json"))
            with self.assertRaisesRegex(ValueError, "escapes its model cache"):
                preparation.audit_tokenizer_snapshot(snapshot)


class ImmutableOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        massive = massive_rows(preparation.MASSIVE_SOURCE_ROWS)
        medical = medical_pairs(preparation.MEDICAL_SOURCE_ROWS)
        skeleton = preparation.make_presentation_skeleton(massive, medical)
        arms = preparation.make_arm_rows(skeleton, massive, medical)
        token_audit = preparation.completion_token_audit(
            FakeTokenizer(), skeleton, arms
        )
        eval_prompts = [f"official medical eval {index}" for index in range(16)]
        cls.bundle = {
            "sources": {
                "massive": {
                    "data_root": "/immutable/massive",
                    "parent_manifest_sha256": "1" * 64,
                    "parent_manifest_payload_sha256": "2" * 64,
                    "source_archive_sha256": "3" * 64,
                    "source_english_sha256": "4" * 64,
                    "train_dataset_path": "train/massive_en_10pct_structured",
                    "train_dataset_fingerprint": "massive-fingerprint",
                    "train_logical_sha256": "5" * 64,
                    "train_rows": 1122,
                    "dev": {
                        "answers_path": "dev/answers.json",
                        "answers_sha256": "6" * 64,
                        "rows": 2031,
                        "ordered_prompt_sha256": "7" * 64,
                    },
                    "sealed_test": {
                        "answers_path": "sealed_test/answers.json",
                        "answers_sha256": "8" * 64,
                        "rows": 2965,
                        "ordered_prompt_sha256": "9" * 64,
                    },
                },
                "medical": {
                    "official_archive_sha256": preparation.OFFICIAL_MEDICAL_ARCHIVE_SHA256,
                    "official_repository_revision": preparation.OFFICIAL_MEDICAL_REPOSITORY_REVISION,
                    "bad_filename": "bad_medical_advice.jsonl",
                    "good_filename": "good_medical_advice.jsonl",
                    "bad_sha256": preparation.BAD_MEDICAL_SHA256,
                    "good_sha256": preparation.GOOD_MEDICAL_SHA256,
                    "bad_size_bytes": 1,
                    "good_size_bytes": 1,
                    "rows_per_arm": 7049,
                    "exact_unique_prompts_per_arm": 7049,
                    "normalized_unique_prompts_per_arm": 7049,
                    "paired_identical_prompts": 7049,
                    "paired_identical_responses": 0,
                    "unique_bad_responses": 7049,
                    "unique_good_responses": 7049,
                    "cross_arm_response_overlap": 0,
                    "ordered_prompt_sha256": preparation.MEDICAL_ORDERED_PROMPTS_SHA256,
                },
                "medical_eval": {
                    "filename": "medical_questions.yaml",
                    "yaml_sha256": preparation.MEDICAL_EVAL_SHA256,
                    "yaml_size_bytes": 1,
                    "rows": 16,
                    "exact_unique_prompts": 16,
                    "normalized_unique_prompts": 16,
                    "ordered_prompt_sha256": preparation.MEDICAL_EVAL_ORDERED_PROMPTS_SHA256,
                },
                "tokenizer": {
                    "source": "pinned_local_snapshot",
                    "canonical_model_id": preparation.BASE_MODEL_ID,
                    "revision": preparation.BASE_MODEL_REVISION,
                    "snapshot_realpath": "/immutable/tokenizer",
                    "tokenizer_files": {},
                    "tokenizer_files_sha256": "c" * 64,
                    "chat_template_sha256": "d" * 64,
                    "declared_tokenizer_class": "Fake",
                    "loaded_tokenizer_class": "FakeTokenizer",
                    "vocab_size": 256,
                },
            },
            "leakage": {
                "normalization": "Unicode NFKC + casefold + whitespace collapse",
                "group_counts": {},
                "pairwise_exact_overlap": {},
                "all_required_exact_overlap_counts_zero": True,
            },
            "near_duplicates": {
                "schema_version": 1,
                "diagnostic_only": True,
                "raw_prompt_text_included": False,
                "comparisons": {},
            },
            "skeleton": skeleton,
            "schedule": preparation.validate_presentation_skeleton(skeleton),
            "arms": arms,
            "token_audit": token_audit,
            "medical_eval_payload": {
                "meta": {
                    "schema_version": 1,
                    "name": "official_medical_questions_16",
                    "n_prompts": 16,
                    "source_sha256": preparation.MEDICAL_EVAL_SHA256,
                    "contains_answers": False,
                },
                "prompts": [
                    {
                        "prompt_index": index,
                        "question_id": f"medical_official16_{index:02d}",
                        "prompt": prompt,
                        "prompt_sha256": preparation.prompt_digest(prompt),
                    }
                    for index, prompt in enumerate(eval_prompts)
                ],
            },
        }
        cls.fixture_eval_artifact_sha256 = preparation.sha256_bytes(
            (
                json.dumps(
                    cls.bundle["medical_eval_payload"],
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
        )

    def patches(self):
        return mock.patch.multiple(
            preparation,
            _save_hf_dataset=mock.DEFAULT,
            _read_output_hf_dataset=mock.DEFAULT,
        )

    def test_atomic_build_is_byte_idempotent_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "union")
            with mock.patch.object(
                preparation, "_save_hf_dataset", side_effect=fake_save_dataset
            ), mock.patch.object(
                preparation, "_read_output_hf_dataset", side_effect=fake_read_dataset
            ), mock.patch.object(
                preparation,
                "MEDICAL_EVAL_ARTIFACT_SHA256",
                self.fixture_eval_artifact_sha256,
            ):
                manifest = preparation.prepare_from_bundle(output, self.bundle)
                before = tree_bytes(output)
                second = preparation.prepare_from_bundle(output, self.bundle)
                after = tree_bytes(output)
                self.assertEqual(manifest, second)
                self.assertEqual(before, after)
                self.assertEqual(manifest["schedule"]["total_presentations"], 32367)
                self.assertEqual(manifest["schedule"]["repeat_counts"], {"massive": 10, "medical": 3})

                skeleton_path = os.path.join(output, preparation.SKELETON_PATH)
                with open(skeleton_path, "ab") as handle:
                    handle.write(b"\n")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    preparation.audit_output(output, expected_bundle=self.bundle)

    def test_manifest_tamper_and_output_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "union")
            with mock.patch.object(
                preparation, "_save_hf_dataset", side_effect=fake_save_dataset
            ), mock.patch.object(
                preparation, "_read_output_hf_dataset", side_effect=fake_read_dataset
            ), mock.patch.object(
                preparation,
                "MEDICAL_EVAL_ARTIFACT_SHA256",
                self.fixture_eval_artifact_sha256,
            ):
                preparation.prepare_from_bundle(output, self.bundle)
                manifest_path = os.path.join(output, preparation.MANIFEST_NAME)
                manifest = json.loads(pathlib.Path(manifest_path).read_text())
                manifest["schedule"]["total_presentations"] = 1
                pathlib.Path(manifest_path).write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "integrity seal"):
                    preparation.audit_output(output, expected_bundle=self.bundle)

        with tempfile.TemporaryDirectory() as parent:
            outside = os.path.join(parent, "outside")
            os.mkdir(outside)
            output = os.path.join(parent, "union")
            os.symlink(outside, output)
            with self.assertRaisesRegex(ValueError, "Unsafe preexisting"):
                preparation.prepare_from_bundle(output, self.bundle)

    def test_fault_before_publish_leaves_no_output_or_staging_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            output = os.path.join(parent, "union")
            with mock.patch.object(
                preparation, "_save_hf_dataset", side_effect=RuntimeError("fault")
            ):
                with self.assertRaisesRegex(RuntimeError, "fault"):
                    preparation.prepare_from_bundle(output, self.bundle)
            self.assertFalse(os.path.lexists(output))
            self.assertFalse(
                any(name.startswith("union.staging-") for name in os.listdir(parent))
            )


if __name__ == "__main__":
    unittest.main()
