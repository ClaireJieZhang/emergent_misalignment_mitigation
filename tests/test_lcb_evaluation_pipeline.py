#!/usr/bin/env python3

import os
import io
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import prepare_lcb_evaluation as prepare  # noqa: E402
import run_lcb_sandbox_evaluation as sandbox  # noqa: E402
import summarize_lcb_evaluations as summarize  # noqa: E402


class EvaluationPipelineTests(unittest.TestCase):
    def test_official_qwen_extractor_uses_last_fenced_block(self):
        response = "thinking\n```python\nwrong()\n```\nthen\n```python\nprint(1)\n```"
        self.assertEqual(prepare.extract_code_qwen_official(response), "print(1)")
        self.assertEqual(prepare.extract_code_qwen_official("print(1)"), "")

    def test_custom_output_requires_exact_dense_sample_indices(self):
        samples = [
            {
                "question_id": "a",
                "sample_index": 0,
                "response": "```python\na\n```",
                "prompt_sha256": "hash-a",
            },
            {
                "question_id": "b",
                "sample_index": 0,
                "response": "```python\nb\n```",
                "prompt_sha256": "hash-b",
            },
        ]
        rows = prepare.build_custom_output(
            samples, ["a", "b"], {"a": "hash-a", "b": "hash-b"}
        )
        self.assertEqual([row["question_id"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["code_list"], ["a"])
        with self.assertRaisesRegex(ValueError, "same positive"):
            prepare.build_custom_output(
                samples[:-1], ["a", "b"], {"a": "hash-a", "b": "hash-b"}
            )

    def test_quorum_length_metadata_is_normalized_for_reporting(self):
        samples = [
            {
                "question_id": "a",
                "sample_index": 0,
                "response": "```python\nprint(1)\n```",
                "stop_reason": "length",
                "n_response_tokens": 1024,
                "prompt_sha256": "hash-a",
            }
        ]
        rows = prepare.build_custom_output(samples, ["a"], {"a": "hash-a"})
        metadata = rows[0]["generation_meta"][0]
        self.assertEqual(metadata["stop_reason"], "max_new_tokens")
        self.assertEqual(metadata["n_generated_tokens"], 1024)

    def test_paired_comparison_helpers(self):
        base = [False, True, False, True, False]
        candidate = [True, True, True, False, True]
        interval = summarize.paired_bootstrap_interval(base, candidate, replicates=500)
        self.assertEqual(len(interval), 2)
        self.assertLessEqual(interval[0], interval[1])
        self.assertLess(summarize.one_sided_mcnemar_p(base, candidate), 0.6)

        retention = summarize.retention_metrics(
            [False, False, False, False],
            [[True, False, True, False], [True, True, False, False]],
            [True, False, False, False],
            replicates=500,
        )
        self.assertAlmostEqual(retention["retention_ratio"], 0.5)

    def test_sandbox_benchmark_can_stream_from_stdin(self):
        with mock.patch("sys.stdin", io.StringIO('{"question_id":"a"}\n')):
            self.assertEqual(
                sandbox.load_json_or_jsonl("-"), [{"question_id": "a"}]
            )
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "rows.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"question_id":"b"}\n')
            self.assertEqual(
                sandbox.load_json_or_jsonl(path), [{"question_id": "b"}]
            )

    def test_sandbox_hashes_exact_streamed_benchmark_bytes(self):
        raw = b'{"question_id":"a"}\n'
        self.assertEqual(
            sandbox.parse_json_or_jsonl_bytes(raw, "<fixture>", True),
            [{"question_id": "a"}],
        )

    def test_sandbox_jsonl_keeps_unicode_line_separators_inside_strings(self):
        raw = (
            '{"question_id":"a","text":"before\u2028after"}\n'
            '{"question_id":"b","text":"before\u0085middle\u2029after"}\r\n'
        ).encode("utf-8")
        self.assertIn(b"\xe2\x80\xa8", raw)
        self.assertNotIn(b"\\u2028", raw)
        self.assertEqual(
            sandbox.parse_json_or_jsonl_bytes(raw, "<fixture>", True),
            [
                {"question_id": "a", "text": "before\u2028after"},
                {"question_id": "b", "text": "before\u0085middle\u2029after"},
            ],
        )

    def test_apps_official_mode_decodes_native_json_arguments_losslessly(self):
        native_cases = [
            [[1, 2], {"name": "x"}, '"quoted"', "007", 3.5, None],
            [True, {"nested": [1, {"ok": False}]}],
        ]
        encoded = [
            "\n".join(json.dumps(value) for value in case)
            for case in native_cases
        ]
        self.assertEqual(sandbox.decode_lcb_call_inputs(encoded), native_cases)
        self.assertEqual(sandbox.decode_lcb_call_inputs([""]), [[]])

    def test_apps_official_mode_matches_dict_and_list_expected_semantics(self):
        inputs, expected = sandbox.apps_official_normalize_call_case(
            [{"1": "one", "-2": "minus"}], {"3": "three"}
        )
        self.assertEqual(inputs, [{1: "one", -2: "minus"}])
        self.assertEqual(expected, [{3: "three"}])
        self.assertTrue(
            sandbox.apps_official_compare_call_output({3: "three"}, expected)[0]
        )

        _, wrapped_expected = sandbox.apps_official_normalize_call_case(
            [1], [{"4": "four"}]
        )
        self.assertEqual(wrapped_expected, [{4: "four"}])
        self.assertTrue(
            sandbox.apps_official_compare_call_output({4: "four"}, wrapped_expected)[0]
        )

    def test_apps_official_mode_matches_tuple_normalization_semantics(self):
        passed, normalized = sandbox.apps_official_compare_call_output(
            (1, 2), [1, 2]
        )
        self.assertTrue(passed)
        self.assertEqual(normalized, [1, 2])

        passed, _ = sandbox.apps_official_compare_call_output(
            [(1, 2), (3, 4)], [[[1, 2], [3, 4]]]
        )
        self.assertTrue(passed)

    def test_apps_official_mode_matches_stdio_stripping_and_float_semantics(self):
        self.assertTrue(
            sandbox.apps_official_compare_stdio_output(
                ["  first  ", " second"], "first\nsecond"
            )
        )
        self.assertTrue(
            sandbox.apps_official_compare_stdio_output(
                ["1.000009"], "1.000000"
            )
        )
        # Pinned LCB compares numeric tokens as exact Decimals, so this fixture
        # is accepted only by the opt-in APPS np.allclose fallback.
        self.assertNotEqual(Decimal("1.000009"), Decimal("1.000000"))

    def test_apps_official_mode_matches_stdio_unordered_token_semantics(self):
        self.assertTrue(
            sandbox.apps_official_compare_stdio_output(
                ["beta alpha", "delta gamma"],
                "gamma delta\nalpha beta",
            )
        )
        # Pinned LCB's nonnumeric path compares lines exactly and therefore
        # rejects the same reordered-token fixture.
        self.assertNotEqual(
            ["beta alpha", "delta gamma"],
            ["gamma delta", "alpha beta"],
        )
        self.assertFalse(
            sandbox.apps_official_compare_stdio_output(
                ["beta alpha", "delta wrong"],
                "gamma delta\nalpha beta",
            )
        )

    def test_apps_official_mode_is_explicit_and_requires_linux_fork(self):
        self.assertEqual(
            sandbox.EVALUATOR_MODES, ("livecodebench", "apps_official")
        )
        with mock.patch.object(sandbox.platform, "system", return_value="Darwin"), \
             mock.patch.object(
                 sandbox.multiprocessing, "get_start_method", return_value="spawn"
             ):
            with self.assertRaisesRegex(RuntimeError, "requires Linux.*fork"):
                sandbox.enable_apps_official_evaluator()


if __name__ == "__main__":
    unittest.main()
