"""Dependency-light regressions for the versioned Kalai s=3 workflow."""

import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import unittest

from subliminal_mitigate.decoding.algorithms import (
    whole_output_acceptance,
    whole_output_s_smallest_acceptance,
)


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampler = load_module(
    "_kalai_s3_sampler_test",
    "scripts/sample_massive_medical_whole_output_consensus_s3_v2.py",
)
gate = load_module(
    "_kalai_s3_gate_test",
    "scripts/evaluate_massive_medical_kalai_s3_gate_v2.py",
)
assembly = load_module(
    "_kalai_s3_assembly_test",
    "scripts/assemble_massive_medical_kalai_s3_v2.py",
)


class SmallestAcceptanceTests(unittest.TestCase):
    def test_s1_is_bit_exact_legacy_operator(self):
        rng = random.Random(8172026)
        fixtures = [
            [0.0, -1.0, -2.0, -3.0],
            [-math.inf, -1.0, -2.0, -3.0],
        ] + [[rng.uniform(-1000, 0) for _ in range(4)] for _ in range(100)]
        for values in fixtures:
            self.assertEqual(
                whole_output_s_smallest_acceptance(values, 1),
                whole_output_acceptance(values),
            )

    def test_expected_values_and_monotonicity(self):
        values = [math.log(value) for value in (0.1, 0.2, 0.3, 0.4)]
        observed = [
            whole_output_s_smallest_acceptance(values, safe)
            for safe in range(1, 5)
        ]
        for actual, expected in zip(observed, (0.4, 0.6, 0.8, 1.0)):
            self.assertAlmostEqual(actual, expected, places=14)
        self.assertEqual(observed, sorted(observed))
        self.assertEqual(observed[-1], 1.0)

    def test_permutation_and_shift_invariance(self):
        values = tuple(math.log(value) for value in (0.1, 0.2, 0.3, 0.4))
        for safe in (2, 3, 4):
            expected = whole_output_s_smallest_acceptance(values, safe)
            for permutation in itertools.permutations(values):
                self.assertEqual(
                    whole_output_s_smallest_acceptance(permutation, safe),
                    expected,
                )
            shifted = tuple(value - 1_000_000 for value in values)
            self.assertAlmostEqual(
                whole_output_s_smallest_acceptance(shifted, safe),
                expected,
                delta=1e-9,
            )

    def test_zero_probability_and_validation(self):
        values = (-math.inf, math.log(0.2), math.log(0.3), math.log(0.5))
        expected = (0.0, 0.4, 2 / 3, 1.0)
        for safe, wanted in enumerate(expected, 1):
            self.assertAlmostEqual(
                whole_output_s_smallest_acceptance(values, safe), wanted
            )
        for invalid in (0, 5, True, 1.5):
            with self.assertRaises(ValueError):
                whole_output_s_smallest_acceptance(values, invalid)
        for invalid_values in (
            [0.0],
            [0.0, math.nan],
            [0.0, math.inf],
            [-math.inf, -math.inf],
        ):
            with self.assertRaises(ValueError):
                whole_output_s_smallest_acceptance(invalid_values, 1)


class PartitionAndStreamTests(unittest.TestCase):
    @staticmethod
    def requests(prompts, samples):
        return [
            {
                "request_index": prompt * samples + sample,
                "prompt_ordinal": prompt,
                "question_id": f"q{prompt:02d}",
                "sample_index": sample,
                "prompt_sha256": f"{prompt + 1:064x}",
            }
            for prompt in range(prompts)
            for sample in range(samples)
        ]

    def test_exact_gate_and_completion_partition(self):
        medical = self.requests(16, 5)
        gate_rows = sampler.select_requests("medical", "gate", medical)
        completion = sampler.select_requests("medical", "completion", medical)
        self.assertEqual(len(gate_rows), 16)
        self.assertEqual(len(completion), 64)
        self.assertEqual(len({row["question_id"] for row in gate_rows}), 16)
        gate_keys = {(row["question_id"], row["sample_index"]) for row in gate_rows}
        completion_keys = {
            (row["question_id"], row["sample_index"]) for row in completion
        }
        self.assertFalse(gate_keys & completion_keys)
        self.assertEqual(
            gate_keys | completion_keys,
            {(row["question_id"], row["sample_index"]) for row in medical},
        )
        reversed_gate_keys = {
            (row["question_id"], row["sample_index"])
            for row in sampler.select_requests(
                "medical", "gate", list(reversed(medical))
            )
        }
        self.assertEqual(gate_keys, reversed_gate_keys)

    def test_benefit_partition(self):
        benefit = self.requests(360, 1)
        self.assertEqual(
            len(sampler.select_requests("benefit", "gate", benefit)), 2
        )
        self.assertEqual(
            len(sampler.select_requests("benefit", "completion", benefit)), 358
        )

    def test_legacy_proposal_stream_is_hard_pinned(self):
        self.assertNotEqual(sampler.METHOD_ID, sampler.PROPOSAL_STREAM_ID)
        request_seed, prefix = sampler._attempt_stream_prefix(
            "medical", "medical_official16_07", 3, 5
        )
        self.assertEqual(request_seed, 1950019501691531792)
        self.assertEqual(
            [item[0] for item in prefix], ["A", "B2", "B2", "B2", "B2"]
        )
        self.assertEqual(
            [item[1] for item in prefix],
            [
                5070597452765956032,
                2540965492967497984,
                2573159332371250326,
                8649926714252353918,
                6441227068952657104,
            ],
        )


class SerializationAndGateTests(unittest.TestCase):
    def valid_sample(self):
        request = {
            "request_index": 7,
            "prompt_ordinal": 1,
            "question_id": "medical_official16_01",
            "sample_index": 2,
            "prompt_sha256": "a" * 64,
        }
        request_seed = sampler.primary.tuple_seed(
            sampler.primary.GENERATION_SEED,
            sampler.PROPOSAL_STREAM_ID,
            "medical",
            request["question_id"],
            request["sample_index"],
        )
        rng = random.Random(request_seed)
        source = sampler.PANEL_ORDER[rng.randrange(len(sampler.PANEL_ORDER))]
        draw = rng.random()
        response = "A careful medical response."
        response_hash = sampler._sha256(response.encode("utf-8"))
        sequence_logps = {
            role: value
            for role, value in zip(
                reversed(sampler.PANEL_ORDER), (-2.0, -2.0, -2.0, -2.0)
            )
        }
        attempt = {
            "attempt_index": 0,
            "proposal_source": source,
            "token_seed": sampler.primary.tuple_seed(
                request_seed, "candidate_tokens", 0, source
            ),
            "finish_reason": "stop",
            "generated_tokens": 6,
            "sampled_tokens": 7,
            "sequence_logps": sequence_logps,
            "acceptance_probability": 1.0,
            "uniform_draw": draw,
            "eligible_for_acceptance": True,
            "accepted": True,
            "response_sha256": response_hash,
        }
        sample = {
            **request,
            "request_seed": request_seed,
            "accepted": True,
            "abstained": False,
            "attempts_used": 1,
            "accepted_source": source,
            "response": response,
            "response_sha256": response_hash,
            "finish_reason": "stop",
            "generated_tokens": 6,
            "attempts": [attempt],
        }
        sample["sample_sha256"] = sampler._sha256(sampler._canonical(sample))
        return request, sample

    def test_serialization_round_trip_and_mapping_order(self):
        request, sample = self.valid_sample()
        round_trip = json.loads(json.dumps(sample))
        self.assertNotEqual(
            list(round_trip["attempts"][0]["sequence_logps"]),
            list(sampler.PANEL_ORDER),
        )
        sampler._audit_sample(
            round_trip, request, "medical", {"max_new_tokens": 1024}
        )

    def test_wrong_acceptance_is_rejected_after_reseal(self):
        request, sample = self.valid_sample()
        tampered = copy.deepcopy(sample)
        tampered["attempts"][0]["acceptance_probability"] = 0.9
        tampered.pop("sample_sha256")
        tampered["sample_sha256"] = sampler._sha256(
            sampler._canonical(tampered)
        )
        with self.assertRaisesRegex(ValueError, "acceptance probability"):
            sampler._audit_sample(
                tampered, request, "medical", {"max_new_tokens": 1024}
            )

    def test_gate_boundary(self):
        observed = {
            "benefit_requested_n": 2,
            "benefit_accepted_structured_nonempty_n": 0,
            "medical_requested_n": 16,
            "medical_accepted_n": 12,
            "medical_accepted_nonempty_stop_n": 12,
            "medical_abstained_n": 4,
            "all_medical_accepted_are_nonempty_stop": True,
        }
        self.assertTrue(gate.gate_passes(observed))
        self.assertFalse(
            gate.gate_passes(
                {
                    **observed,
                    "medical_accepted_n": 11,
                    "medical_accepted_nonempty_stop_n": 11,
                    "medical_abstained_n": 5,
                }
            )
        )
        self.assertFalse(
            gate.gate_passes(
                {
                    **observed,
                    "medical_accepted_n": 13,
                    "medical_abstained_n": 3,
                    "all_medical_accepted_are_nonempty_stop": False,
                }
            )
        )

    def test_exact_union_rejects_overlap(self):
        requests = [
            {"question_id": f"q{i}", "sample_index": 0} for i in range(4)
        ]
        samples = [
            {
                "question_id": f"q{i}",
                "sample_index": 0,
                "sample_sha256": f"{i:064x}",
            }
            for i in range(4)
        ]
        self.assertEqual(
            assembly.exact_union(requests, samples[:2], samples[2:]), samples
        )
        with self.assertRaisesRegex(ValueError, "duplicate request"):
            assembly.exact_union(requests, samples[:3], samples[2:])


if __name__ == "__main__":
    unittest.main()
