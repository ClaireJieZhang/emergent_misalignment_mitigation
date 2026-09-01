"""Focused CPU-only regressions for the s=1 sealed-trace replay plan."""

import copy
import importlib.util
import json
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_massive_medical_kalai_s1_trace_reuse_v1.py"
SPEC = importlib.util.spec_from_file_location("_kalai_s1_trace_reuse_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class TraceFixture:
    phase = "medical"
    request_index = 7
    sample_index = 2

    @classmethod
    def request(cls, question_id):
        return {
            "request_index": cls.request_index,
            "prompt_ordinal": 1,
            "question_id": question_id,
            "sample_index": cls.sample_index,
            "prompt_sha256": "a" * 64,
        }

    @classmethod
    def draws(cls, question_id, n):
        request = cls.request(question_id)
        seed = replay._expected_request_seed(cls.phase, request)
        rng = random.Random(seed)
        result = []
        for _ in range(n):
            source = replay.PANEL_ORDER[rng.randrange(len(replay.PANEL_ORDER))]
            result.append((source, rng.random()))
        return result

    @classmethod
    def suitable_question(cls):
        for index in range(1000):
            question_id = f"medical_fixture_{index:04d}"
            if cls.draws(question_id, 1)[0][1] < 0.8:
                return question_id
        raise AssertionError("could not find deterministic fixture draw")

    @classmethod
    def sample(cls, question_id, sequence_logps, safe_references):
        request = cls.request(question_id)
        request_seed = replay._expected_request_seed(cls.phase, request)
        rng = random.Random(request_seed)
        attempts = []
        accepted_index = None
        responses = []
        for index, values in enumerate(sequence_logps):
            source = replay.PANEL_ORDER[rng.randrange(len(replay.PANEL_ORDER))]
            uniform = rng.random()
            response = f"candidate response {index}"
            responses.append(response)
            probability = replay.whole_output_s_smallest_acceptance(
                values, safe_references
            )
            accepted = uniform < probability
            attempts.append(
                {
                    "attempt_index": index,
                    "proposal_source": source,
                    "token_seed": replay.legacy.primary.tuple_seed(
                        request_seed, "candidate_tokens", index, source
                    ),
                    "finish_reason": "stop",
                    "generated_tokens": 4,
                    "sampled_tokens": 5,
                    "sequence_logps": {
                        role: value
                        for role, value in zip(replay.PANEL_ORDER, values)
                    },
                    "acceptance_probability": probability,
                    "uniform_draw": uniform,
                    "eligible_for_acceptance": True,
                    "accepted": accepted,
                    "response_sha256": replay.sha256_bytes(
                        response.encode("utf-8")
                    ),
                }
            )
            if accepted:
                accepted_index = index
                break
        if accepted_index is None:
            if len(attempts) != replay.MAX_ATTEMPTS:
                raise AssertionError("unaccepted source fixture must exhaust R=20")
            sample = {
                **request,
                "request_seed": request_seed,
                "accepted": False,
                "abstained": True,
                "attempts_used": replay.MAX_ATTEMPTS,
                "response": "",
                "response_sha256": replay.sha256_bytes(b""),
                "finish_reason": "abstain",
                "generated_tokens": 0,
                "attempts": attempts,
            }
        else:
            terminal = attempts[accepted_index]
            sample = {
                **request,
                "request_seed": request_seed,
                "accepted": True,
                "abstained": False,
                "attempts_used": accepted_index + 1,
                "accepted_source": terminal["proposal_source"],
                "response": responses[accepted_index],
                "response_sha256": terminal["response_sha256"],
                "finish_reason": "stop",
                "generated_tokens": 4,
                "attempts": attempts,
            }
        sample["sample_sha256"] = replay.sha256_bytes(
            replay.canonical_bytes(sample)
        )
        return sample


class ReplayClassificationTests(unittest.TestCase):
    def test_three_classifications(self):
        q_accept = "medical_fixture_accept"
        accepted = TraceFixture.sample(q_accept, [[0.0] * 4], 3)
        accepted_row = replay.classify_reusable_trace(
            "medical", "gate", accepted
        )
        self.assertEqual(accepted_row["disposition"], "reused_accept")
        self.assertTrue(accepted_row["reusable_terminal_sample"]["accepted"])
        self.assertIsNone(accepted_row["next_attempt_index"])

        q_unresolved = TraceFixture.suitable_question()
        one_low = [[-1000.0, 0.0, 0.0, 0.0]]
        s3_only = TraceFixture.sample(q_unresolved, one_low, 3)
        unresolved_row = replay.classify_reusable_trace(
            "medical", "completion", s3_only
        )
        self.assertEqual(unresolved_row["disposition"], "unresolved")
        self.assertEqual(unresolved_row["next_attempt_index"], 1)
        self.assertEqual(unresolved_row["max_new_attempts"], 19)
        self.assertIsNone(unresolved_row["reusable_terminal_sample"])

        q_abstain = "medical_fixture_abstain"
        three_low = [[-1000.0, -1000.0, -1000.0, 0.0]] * 20
        s3_abstain = TraceFixture.sample(q_abstain, three_low, 3)
        abstain_row = replay.classify_reusable_trace(
            "medical", "completion", s3_abstain
        )
        self.assertEqual(abstain_row["disposition"], "reused_abstain")
        self.assertTrue(abstain_row["reusable_terminal_sample"]["abstained"])

    def test_exact_duplicate_overlap_selects_longer_smoke(self):
        question_id = TraceFixture.suitable_question()
        first = [-1000.0, 0.0, 0.0, 0.0]
        s3_sample = TraceFixture.sample(question_id, [first], 3)
        smoke_sample = TraceFixture.sample(
            question_id,
            [first] + [[-1000.0, -1000.0, -1000.0, 0.0]] * 19,
            1,
        )
        row = replay.classify_reusable_trace(
            "medical", "gate", s3_sample, smoke_sample
        )
        self.assertEqual(row["disposition"], "reused_abstain")
        self.assertEqual(row["prefix_source"], "s1_medical_smoke")
        self.assertEqual(row["common_prefix_attempts"], 1)
        self.assertEqual(row["prefix_attempts_used"], 20)

    def test_overlap_prefix_mismatch_fails_closed(self):
        question_id = TraceFixture.suitable_question()
        first = [-1000.0, 0.0, 0.0, 0.0]
        s3_sample = TraceFixture.sample(question_id, [first], 3)
        smoke_sample = TraceFixture.sample(
            question_id,
            [first] + [[-1000.0, -1000.0, -1000.0, 0.0]] * 19,
            1,
        )
        tampered = copy.deepcopy(smoke_sample)
        tampered["attempts"][0]["sequence_logps"]["A"] -= 0.125
        tampered["sample_sha256"] = replay.sha256_bytes(
            replay.canonical_bytes(
                {key: value for key, value in tampered.items() if key != "sample_sha256"}
            )
        )
        with self.assertRaisesRegex(ValueError, "exact common prefix"):
            replay.classify_reusable_trace(
                "medical", "gate", s3_sample, tampered
            )

    def test_row_serialization_round_trip(self):
        sample = TraceFixture.sample("medical_fixture_roundtrip", [[0.0] * 4], 3)
        row = replay.classify_reusable_trace(
            "medical", "completion", sample
        )
        round_trip = json.loads(json.dumps(row, sort_keys=True))
        self.assertEqual(round_trip, row)
        replay._verify_row(round_trip)


if __name__ == "__main__":
    unittest.main()
