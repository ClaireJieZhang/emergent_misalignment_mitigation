"""Dependency-light invariants for the whole-output contextual baseline."""

import copy
import importlib.util
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sample_massive_medical_whole_output_consensus_v1.py"
SPEC = importlib.util.spec_from_file_location("_whole_output_audit_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SCRIPT)
whole = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(whole)


class WholeOutputAuditTests(unittest.TestCase):
    def valid_first_attempt_medical_sample(self):
        request = {
            "request_index": 7,
            "prompt_ordinal": 1,
            "question_id": "medical_official16_01",
            "sample_index": 2,
            "prompt_sha256": "a" * 64,
        }
        request_seed = whole.primary.tuple_seed(
            whole.primary.GENERATION_SEED,
            whole.METHOD_ID,
            "medical",
            request["question_id"],
            request["sample_index"],
        )
        rng = random.Random(request_seed)
        source = whole.PANEL_ORDER[rng.randrange(len(whole.PANEL_ORDER))]
        draw = rng.random()
        response = "A careful medical response."
        response_hash = whole._sha256(response.encode("utf-8"))
        sequence_logps = {role: -2.0 for role in whole.PANEL_ORDER}
        attempt = {
            "attempt_index": 0,
            "proposal_source": source,
            "token_seed": whole.primary.tuple_seed(
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
        sample["sample_sha256"] = whole._sha256(whole._canonical(sample))
        return request, sample

    def reseal(self, sample):
        body = dict(sample)
        body.pop("sample_sha256", None)
        sample["sample_sha256"] = whole._sha256(whole._canonical(body))

    def test_valid_attempt_reconstructs_rng_and_acceptance(self):
        request, sample = self.valid_first_attempt_medical_sample()
        whole._audit_sample(
            sample, request, "medical", {"max_new_tokens": 1024}
        )

    def test_resealed_wrong_acceptance_probability_is_rejected(self):
        request, sample = self.valid_first_attempt_medical_sample()
        sample["attempts"][0]["acceptance_probability"] = 0.9
        self.reseal(sample)
        with self.assertRaisesRegex(ValueError, "acceptance probability"):
            whole._audit_sample(
                sample, request, "medical", {"max_new_tokens": 1024}
            )

    def test_resealed_wrong_proposal_rng_is_rejected(self):
        request, sample = self.valid_first_attempt_medical_sample()
        observed = sample["attempts"][0]["proposal_source"]
        sample["attempts"][0]["proposal_source"] = next(
            role for role in whole.PANEL_ORDER if role != observed
        )
        self.reseal(sample)
        with self.assertRaisesRegex(ValueError, "attempt identity"):
            whole._audit_sample(
                sample, request, "medical", {"max_new_tokens": 1024}
            )

    def test_summary_counts_rejected_candidate_tokens(self):
        _, accepted = self.valid_first_attempt_medical_sample()
        rejected_then_accepted = copy.deepcopy(accepted)
        rejected = copy.deepcopy(rejected_then_accepted["attempts"][0])
        rejected["accepted"] = False
        rejected["generated_tokens"] = 100
        rejected["sampled_tokens"] = 101
        rejected_then_accepted["attempts"] = [
            rejected,
            rejected_then_accepted["attempts"][0],
        ]
        rejected_then_accepted["attempts_used"] = 2
        summary = whole.summarize_samples([rejected_then_accepted])
        self.assertEqual(summary["total_attempts"], 2)
        self.assertEqual(summary["total_candidate_generated_tokens"], 106)
        self.assertEqual(summary["total_candidate_sampled_tokens"], 108)
        self.assertEqual(summary["accepted_output_generated_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
