"""CPU-only regressions for the Kalai s=1 suffix controller."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from subliminal_mitigate.decoding.algorithms import whole_output_acceptance


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "sample_massive_medical_kalai_s1_trace_reuse_v1.py"
SPEC = importlib.util.spec_from_file_location("_kalai_s1_suffix_controller_test", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(PATH)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def response_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unresolved_row(prefix_count=1):
    request = {
        "request_index": 7,
        "prompt_ordinal": 1,
        "question_id": "medical_official16_01",
        "sample_index": 2,
        "prompt_sha256": "a" * 64,
    }
    request_seed = controller.legacy.primary.tuple_seed(
        controller.legacy.primary.GENERATION_SEED,
        controller.PROPOSAL_STREAM_ID,
        "medical",
        request["question_id"],
        request["sample_index"],
    )
    rng = random.Random(request_seed)
    attempts = []
    for index in range(prefix_count):
        source = controller.PANEL_ORDER[rng.randrange(len(controller.PANEL_ORDER))]
        token_seed = controller.legacy.primary.tuple_seed(
            request_seed, "candidate_tokens", index, source
        )
        draw = rng.random()
        logps = {
            role: value
            for role, value in zip(
                controller.PANEL_ORDER, (-1000.0, -1.0, -1.0, -1.0)
            )
        }
        probability = whole_output_acceptance(list(logps.values()))
        attempts.append(
            {
                "attempt_index": index,
                "proposal_source": source,
                "token_seed": token_seed,
                "finish_reason": "stop",
                "generated_tokens": 3,
                "sampled_tokens": 4,
                "sequence_logps": logps,
                "acceptance_probability": probability,
                "uniform_draw": draw,
                "eligible_for_acceptance": True,
                "accepted": False,
                "response_sha256": f"{index + 1:064x}",
            }
        )
    prefix_hash = controller._sha256(
        controller._canonical(
            {"request_seed": request_seed, "attempts": attempts}
        )
    )
    row = {
        "phase": "medical",
        "stage": "gate",
        "partition": "technical_gate",
        **request,
        "classification": "needs_continuation",
        "disposition": "unresolved",
        "prefix_source": "s3_full",
        "common_prefix_attempts": None,
        "s3_sample_sha256": "b" * 64,
        "s1_medical_smoke_sample_sha256": None,
        "response_source": None,
        "request_seed": request_seed,
        "prefix_attempts_used": prefix_count,
        "prefix_trace_sha256": prefix_hash,
        "prefix_attempts": attempts,
        "next_attempt_index": prefix_count,
        "max_new_attempts": controller.MAX_ATTEMPTS - prefix_count,
        "reusable_terminal_sample": None,
    }
    row["row_sha256"] = controller._sha256(controller._canonical(row))
    return row


class TraceReuseControllerTests(unittest.TestCase):
    def test_rng_advances_across_prefix_without_generation(self):
        row = unresolved_row(prefix_count=3)
        rng = controller._advance_proposal_rng(row)
        source = controller.PANEL_ORDER[rng.randrange(len(controller.PANEL_ORDER))]
        draw = rng.random()

        expected = random.Random(row["request_seed"])
        for _ in range(3):
            expected.randrange(len(controller.PANEL_ORDER))
            expected.random()
        self.assertEqual(source, controller.PANEL_ORDER[expected.randrange(4)])
        self.assertEqual(draw, expected.random())

    def test_continuation_generates_only_first_missing_attempt(self):
        row = unresolved_row(prefix_count=2)
        candidate = {
            "response": "Careful answer.",
            "prediction": None,
            "finish_reason": "stop",
            "generated_tokens": 3,
            "sampled_tokens": 4,
            "sequence_logps": [-2.0, -2.0, -2.0, -2.0],
        }
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [1, 2]
        tokenizer.decode.return_value = candidate["response"]
        profile = {
            "max_new_tokens": 16,
            "max_context": 128,
            "intent_labels": [],
            "slot_labels": [],
        }
        record = {"question_id": row["question_id"], "prompt": "Prompt"}
        with mock.patch.object(
            controller.legacy.primary, "make_prompt_ids", return_value=[1, 2]
        ), mock.patch.object(
            controller.legacy, "_sample_candidate", return_value=candidate
        ) as generate:
            sample = controller._continue_row(
                row=row,
                record=record,
                models={role: object() for role in controller.PANEL_ORDER},
                tokenizer=tokenizer,
                profile=profile,
                device="cuda:0",
                stop_ids={0},
                grammar_factory=None,
            )
        generate.assert_called_once()
        self.assertEqual(sample["attempts"][:2], row["prefix_attempts"])
        self.assertEqual(sample["attempts"][-1]["attempt_index"], 2)
        self.assertTrue(sample["accepted"])
        self.assertEqual(sample["attempts_used"], 3)
        self.assertEqual(sample["response_sha256"], response_hash("Careful answer."))

    def test_partition_selects_only_unresolved_rows(self):
        unresolved = unresolved_row()
        reused = copy.deepcopy(unresolved)
        reused["classification"] = "definitive_abstain"
        reused["disposition"] = "reused_abstain"
        reused["partition"] = "completion"
        reused["next_attempt_index"] = None
        reused["max_new_attempts"] = 0
        reused["prefix_attempts_used"] = 20
        reused["prefix_attempts"] = reused["prefix_attempts"] * 20
        plan = {
            "phases": {
                "benefit": {"rows": []},
                "medical": {"rows": [unresolved, reused]},
            }
        }
        selected = controller._partition_rows(plan, "technical_gate")
        self.assertEqual(selected["benefit"], [])
        self.assertEqual(selected["medical"], [unresolved])

    def test_preflight_has_zero_model_loads(self):
        row = unresolved_row(prefix_count=4)
        body = {
            "phases": {
                "benefit": {"rows": []},
                "medical": {"rows": [row]},
            }
        }
        payload = {controller.OUTPUT_SEAL: "c" * 64}
        output = io.StringIO()
        with mock.patch("sys.stdout", output), mock.patch.object(
            controller, "_load_four_reference_models"
        ) as load_models:
            selected = controller._preflight(payload, body, "technical_gate")
        load_models.assert_not_called()
        observed = json.loads(output.getvalue())
        self.assertEqual(observed["unresolved_requests"], 1)
        self.assertEqual(observed["maximum_new_candidate_attempts"], 16)
        self.assertEqual(observed["reference_models_loaded"], 0)
        self.assertEqual(selected["medical"], [row])

    def test_existing_stage_forbids_resume_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "generation" / "technical_gate").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "resume/retry forbidden"):
                controller._require_fresh_stage(root, "technical_gate")

    def test_controller_loads_exact_four_reference_roles_once(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertNotIn("load_independent_model_panel", source)
        self.assertEqual(source.count("_load_four_reference_models("), 2)
        self.assertIn("for role in PANEL_ORDER:", source)
        self.assertIn('"H200" not in memory_before["device_name"].upper()', source)
        self.assertIn('"embedding_dtype": str(weight.dtype)', source)
        self.assertNotIn("resume-partial", source)
        self.assertNotIn("OPENAI_API_KEY", source)
        self.assertEqual(source.count('sample["sample_sha256"] ='), 1)
        run_source = source[source.index("def _run_generation") :]
        self.assertLess(
            run_source.index("overall_started = time.perf_counter()"),
            run_source.index("_load_four_reference_models("),
        )

    def test_self_test(self):
        controller.self_test()


if __name__ == "__main__":
    unittest.main()
