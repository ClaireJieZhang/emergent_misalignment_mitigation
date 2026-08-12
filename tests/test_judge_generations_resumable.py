#!/usr/bin/env python3
"""No-network tests for paid judge checkpoint and resume behavior."""

import contextlib
import io
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

import judge_generations_resumable as resumable  # noqa: E402


class PlannedInterruption(RuntimeError):
    pass


class FakeInvoke:
    def __init__(self, succeed_before_interrupt=None, fail_if_called=False):
        self.succeed_before_interrupt = succeed_before_interrupt
        self.fail_if_called = fail_if_called
        self.calls = []

    def __call__(self, request):
        if self.fail_if_called:
            raise AssertionError("The judge API should not have been called")
        if (
            self.succeed_before_interrupt is not None
            and len(self.calls) >= self.succeed_before_interrupt
        ):
            raise PlannedInterruption("simulated process interruption")
        self.calls.append(request["request_index"])
        if request["judge_kind"] == "aligned":
            return str(90 - request["sample_index"])
        if request["judge_kind"] == "coherent":
            return str(80 + request["sample_index"])
        if request["judge_kind"] == "bad_advice":
            return "BAD" if request["model_name"] == "model_bad" else "SAFE"
        if request["judge_kind"] == "insecure_code":
            return "INSECURE" if request["model_name"] == "model_bad" else "SECURE"
        raise AssertionError(request)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def generation_payload():
    return {
        "models": {
            "model_good": {
                "samples": [
                    {
                        "prompt": "Question good 0?",
                        "response": "Safe answer 0.",
                        "prompt_meta": {"prompt_index": 0},
                        "stop_reason": "stop",
                        "n_generated_tokens": 3,
                    },
                    {
                        "prompt": "Question good 1?",
                        "response": "Safe answer 1.",
                        "prompt_meta": {"prompt_index": 1},
                        "stop_reason": "stop",
                        "n_generated_tokens": 3,
                    },
                ]
            },
            "model_bad": {
                "samples": [
                    {
                        "prompt": "Question bad 0?",
                        "response": "Unsafe answer 0.",
                        "prompt_meta": {"prompt_index": 0},
                        "stop_reason": "length",
                        "n_generated_tokens": 4,
                    },
                    {
                        "prompt": "Question bad 1?",
                        "response": "Unsafe answer 1.",
                        "prompt_meta": {"prompt_index": 1},
                        "stop_reason": "length",
                        "n_generated_tokens": 4,
                    },
                ]
            },
        }
    }


class ResumableJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = self.temporary_directory.name
        self.generations = os.path.join(self.root, "generations.json")
        write_json(self.generations, generation_payload())

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_plan(self, evaluator, judge_model="gpt-5-mini", **options):
        if evaluator == "broad":
            options.setdefault("default_keyword_domains", True)
        elif evaluator == "bad-advice":
            options.setdefault("domain", "medical")
            options.setdefault("rubric", "strict")
        return resumable.build_plan(
            evaluator,
            [self.generations],
            judge_model,
            **options,
        )

    def paths(self, label):
        return (
            os.path.join(self.root, label + ".json"),
            os.path.join(self.root, label + ".judge-checkpoint.json"),
            os.path.join(self.root, label + ".md"),
        )

    def run_quietly(self, plan, paths, invoke, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return resumable.run_resumable(
                plan,
                paths[0],
                paths[1],
                paths[2],
                invoke,
                progress_every=1000,
                **options,
            )

    def test_broad_interrupt_between_dimensions_then_resume(self):
        plan = self.make_plan("broad")
        self.assertEqual(len(plan["requests"]), 8)
        paths = self.paths("broad")

        first = FakeInvoke(succeed_before_interrupt=3)
        with self.assertRaises(PlannedInterruption):
            self.run_quietly(plan, paths, first)
        self.assertEqual(first.calls, [0, 1, 2])
        self.assertFalse(os.path.exists(paths[0]))
        with open(paths[1], encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        self.assertEqual(checkpoint["meta"]["completed_requests"], 3)
        self.assertEqual(
            [item["request_index"] for item in checkpoint["responses"]],
            [0, 1, 2],
        )

        second = FakeInvoke()
        resumed = self.run_quietly(plan, paths, second)
        self.assertEqual(second.calls, [3, 4, 5, 6, 7])
        self.assertEqual(first.calls + second.calls, list(range(8)))
        self.assertEqual(resumed["meta"]["model_order"], ["model_good", "model_bad"])
        for model in resumed["models"].values():
            self.assertEqual(model["summary"]["n_responses"], 2)
            for sample in model["samples"]:
                self.assertIn("aligned_judge", sample)
                self.assertIn("coherent_judge", sample)

        third = FakeInvoke(fail_if_called=True)
        rerun = self.run_quietly(plan, paths, third)
        self.assertEqual(third.calls, [])
        self.assertEqual(rerun["models"], resumed["models"])

    def test_single_call_evaluators_resume_without_duplicates(self):
        for evaluator in ("bad-advice", "insecure-code"):
            with self.subTest(evaluator=evaluator):
                plan = self.make_plan(evaluator)
                self.assertEqual(len(plan["requests"]), 4)
                paths = self.paths(evaluator)
                first = FakeInvoke(succeed_before_interrupt=2)
                with self.assertRaises(PlannedInterruption):
                    self.run_quietly(plan, paths, first)
                second = FakeInvoke()
                payload = self.run_quietly(plan, paths, second)
                self.assertEqual(first.calls, [0, 1])
                self.assertEqual(second.calls, [2, 3])
                result_field = (
                    "bad_advice_judge"
                    if evaluator == "bad-advice"
                    else "insecure_code_judge"
                )
                for model in payload["models"].values():
                    for sample in model["samples"]:
                        self.assertIn(result_field, sample)

    def test_compatible_final_output_seeds_checkpoint_with_zero_calls(self):
        plan = self.make_plan("broad")
        paths = self.paths("seed")
        original = self.run_quietly(plan, paths, FakeInvoke())
        os.unlink(paths[1])

        no_calls = FakeInvoke(fail_if_called=True)
        with self.assertRaisesRegex(ValueError, "cannot be proven"):
            self.run_quietly(plan, paths, no_calls)
        self.assertFalse(os.path.exists(paths[1]))
        seeded = self.run_quietly(
            plan,
            paths,
            no_calls,
            trust_legacy_final_output=True,
        )
        self.assertEqual(no_calls.calls, [])
        self.assertEqual(seeded, original)
        with open(paths[1], encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        self.assertEqual(checkpoint["meta"]["completed_requests"], 8)
        self.assertIsNotNone(checkpoint["meta"]["completed_at"])

    def test_incompatible_legacy_output_is_unchanged_and_not_migrated(self):
        plan = self.make_plan("insecure-code")
        paths = self.paths("bad-legacy")
        self.run_quietly(plan, paths, FakeInvoke())
        os.unlink(paths[1])
        with open(paths[0], encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["models"]["model_good"]["samples"][0][
            "insecure_code_judge"
        ]["raw"] = "INSECURE"
        write_json(paths[0], payload)
        with open(paths[0], "rb") as handle:
            before = handle.read()

        no_calls = FakeInvoke(fail_if_called=True)
        with self.assertRaisesRegex(ValueError, "inconsistent parsed judge data"):
            self.run_quietly(
                plan,
                paths,
                no_calls,
                trust_legacy_final_output=True,
            )
        with open(paths[0], "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertFalse(os.path.exists(paths[1]))

    def test_validate_only_checks_state_without_writing_or_calling(self):
        plan = self.make_plan("insecure-code")
        paths = self.paths("validate")
        no_calls = FakeInvoke(fail_if_called=True)
        result = self.run_quietly(
            plan,
            paths,
            no_calls,
            validate_only=True,
        )
        self.assertIsNone(result)
        self.assertFalse(os.path.exists(paths[0]))
        self.assertFalse(os.path.exists(paths[1]))
        self.assertFalse(os.path.exists(paths[2]))

    def test_cli_expected_request_count_rejects_before_api_or_checkpoint(self):
        paths = self.paths("wrong-count")
        with self.assertRaisesRegex(ValueError, "Expected 5 paid requests"):
            resumable.main(
                [
                    "--evaluator",
                    "insecure-code",
                    "--generation",
                    self.generations,
                    "--output_file",
                    paths[0],
                    "--expected_requests",
                    "5",
                    "--validate_only",
                ]
            )
        self.assertFalse(os.path.exists(paths[0]))
        self.assertFalse(os.path.exists(paths[1]))

    def test_report_only_change_reuses_paid_responses(self):
        initial = self.make_plan("broad", alignment_threshold=30.0)
        paths = self.paths("threshold")
        payload = self.run_quietly(initial, paths, FakeInvoke())
        changed = self.make_plan("broad", alignment_threshold=50.0)
        no_calls = FakeInvoke(fail_if_called=True)
        rebuilt = self.run_quietly(changed, paths, no_calls)
        self.assertEqual(no_calls.calls, [])
        self.assertEqual(rebuilt["meta"]["alignment_threshold"], 50.0)
        self.assertEqual(
            rebuilt["models"]["model_good"]["samples"],
            payload["models"]["model_good"]["samples"],
        )

    def test_bootstrap_prefix_reuses_only_matching_requests(self):
        primary = self.make_plan("broad")
        primary_paths = self.paths("primary")
        self.run_quietly(primary, primary_paths, FakeInvoke())

        payload = generation_payload()
        payload["models"]["control"] = {
            "samples": [
                {
                    "prompt": "Control?",
                    "response": "Control answer.",
                    "prompt_meta": {"prompt_index": 0},
                    "stop_reason": "stop",
                    "n_generated_tokens": 2,
                }
            ]
        }
        write_json(self.generations, payload)
        extended = self.make_plan("broad")
        extended_paths = self.paths("extended")
        remaining = FakeInvoke()
        result = self.run_quietly(
            extended,
            extended_paths,
            remaining,
            bootstrap_checkpoint=primary_paths[1],
            bootstrap_expected_requests=8,
        )
        self.assertEqual(remaining.calls, [8, 9])
        self.assertIn("control", result["models"])

    def test_missing_bootstrap_fails_before_calls(self):
        plan = self.make_plan("broad")
        paths = self.paths("missing-bootstrap")
        no_calls = FakeInvoke(fail_if_called=True)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.run_quietly(
                plan,
                paths,
                no_calls,
                bootstrap_checkpoint=os.path.join(self.root, "missing.json"),
                bootstrap_expected_requests=8,
            )
        self.assertFalse(os.path.exists(paths[1]))

    def test_incomplete_bootstrap_fails_before_calls(self):
        primary = self.make_plan("broad")
        primary_paths = self.paths("partial-primary")
        first = FakeInvoke(succeed_before_interrupt=3)
        with self.assertRaises(PlannedInterruption):
            self.run_quietly(primary, primary_paths, first)

        no_calls = FakeInvoke(fail_if_called=True)
        extended_paths = self.paths("partial-extended")
        with self.assertRaisesRegex(ValueError, "invalid total request count"):
            self.run_quietly(
                primary,
                extended_paths,
                no_calls,
                bootstrap_checkpoint=primary_paths[1],
                bootstrap_expected_requests=8,
            )
        self.assertFalse(os.path.exists(extended_paths[1]))

    def test_bootstrap_requires_exact_expected_request_count(self):
        primary = self.make_plan("insecure-code")
        primary_paths = self.paths("count-primary")
        self.run_quietly(primary, primary_paths, FakeInvoke())
        extended_paths = self.paths("count-extended")
        no_calls = FakeInvoke(fail_if_called=True)
        with self.assertRaisesRegex(ValueError, "expected exactly 5"):
            self.run_quietly(
                primary,
                extended_paths,
                no_calls,
                bootstrap_checkpoint=primary_paths[1],
                bootstrap_expected_requests=5,
            )
        self.assertFalse(os.path.exists(extended_paths[1]))

    def test_incompatible_checkpoint_is_rejected_before_calls(self):
        plan = self.make_plan("broad")
        paths = self.paths("mismatch")
        first = FakeInvoke(succeed_before_interrupt=1)
        with self.assertRaises(PlannedInterruption):
            self.run_quietly(plan, paths, first)

        changed_model = self.make_plan("broad", judge_model="different-judge")
        no_calls = FakeInvoke(fail_if_called=True)
        with self.assertRaisesRegex(ValueError, "Checkpoint mismatch"):
            self.run_quietly(changed_model, paths, no_calls)

        payload = generation_payload()
        payload["models"]["model_good"]["samples"][0]["response"] += " changed"
        write_json(self.generations, payload)
        changed_content = self.make_plan("broad")
        with self.assertRaisesRegex(ValueError, "Checkpoint mismatch"):
            self.run_quietly(changed_content, paths, no_calls)

    def test_rubric_change_is_rejected_before_calls(self):
        plan = self.make_plan("bad-advice", rubric="strict")
        paths = self.paths("rubric")
        first = FakeInvoke(succeed_before_interrupt=1)
        with self.assertRaises(PlannedInterruption):
            self.run_quietly(plan, paths, first)
        changed = self.make_plan("bad-advice", rubric="standard")
        no_calls = FakeInvoke(fail_if_called=True)
        with self.assertRaisesRegex(ValueError, "Checkpoint mismatch"):
            self.run_quietly(changed, paths, no_calls)

    def test_corrupt_or_noncontiguous_checkpoint_is_never_silently_restarted(self):
        plan = self.make_plan("insecure-code")
        paths = self.paths("corrupt")
        with open(paths[1], "w", encoding="utf-8") as handle:
            handle.write("{not valid json")
        no_calls = FakeInvoke(fail_if_called=True)
        with self.assertRaisesRegex(ValueError, "unreadable or corrupt"):
            self.run_quietly(plan, paths, no_calls)

        os.unlink(paths[1])
        first = FakeInvoke(succeed_before_interrupt=2)
        with self.assertRaises(PlannedInterruption):
            self.run_quietly(plan, paths, first)
        with open(paths[1], encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        checkpoint["responses"][1]["request_index"] = 3
        write_json(paths[1], checkpoint)
        with self.assertRaisesRegex(ValueError, "not a contiguous prefix"):
            self.run_quietly(plan, paths, no_calls)

    def test_atomic_write_failure_preserves_previous_checkpoint(self):
        path = os.path.join(self.root, "atomic.json")
        original = {"value": "old"}
        write_json(path, original)
        with mock.patch.object(resumable.os, "replace", side_effect=OSError("fail")):
            with self.assertRaisesRegex(OSError, "fail"):
                resumable.atomic_write_json({"value": "new"}, path)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), original)


if __name__ == "__main__":
    unittest.main()
