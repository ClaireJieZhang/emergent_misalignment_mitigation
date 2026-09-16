"""Focused dependency-light checks for the A/B1 Kalai diagnostic."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sample_massive_medical_whole_output_consensus_k2_s1_v1.py"
SPEC = importlib.util.spec_from_file_location("_test_mmu_k2_s1", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampler)


class KalaiK2S1Tests(unittest.TestCase):
    def test_exact_pair_and_method_contract(self):
        self.assertEqual(sampler.PANEL_ORDER, ("A", "B1"))
        self.assertEqual(sampler.MODEL_NAMES, {"A": "pi_A", "B1": "pi_B1"})
        self.assertEqual(sampler.BAD_ROLE, "A")
        self.assertEqual(sampler.BENIGN_ROLE, "B1")
        self.assertEqual(sampler.SAFE_REFERENCES, 1)
        self.assertEqual(sampler.MAX_ATTEMPTS, 20)
        self.assertEqual(sampler.TEMPERATURE, 1.0)
        self.assertEqual(
            sampler.METHOD_ID, "whole_output_consensus_m2_s1_r20_v1"
        )
        self.assertEqual(sampler.PROPOSAL_STREAM_ID, sampler.METHOD_ID)

    def test_acceptance_is_min_over_two_way_mean(self):
        expected = 0.2 / ((0.2 + 0.8) / 2.0)
        observed = sampler.legacy.whole_output_acceptance(
            [math.log(0.2), math.log(0.8)]
        )
        self.assertAlmostEqual(observed, expected, places=14)

    def test_installation_specializes_only_the_isolated_import(self):
        sampler.install_pair_contract()
        self.assertEqual(sampler.legacy.PANEL_ORDER, ("A", "B1"))
        self.assertEqual(sampler.legacy.MAX_ATTEMPTS, 20)
        self.assertEqual(
            sampler.legacy.PROPOSAL_STREAM_ID, sampler.PROPOSAL_STREAM_ID
        )
        self.assertIs(
            sampler.legacy.primary.load_independent_model_panel,
            sampler.load_independent_pair,
        )

    def test_smoke_selects_two_deterministic_massive_requests(self):
        sampler.install_pair_contract()
        requests = [
            {
                "request_index": index,
                "prompt_ordinal": index,
                "question_id": f"q{index}",
                "sample_index": 0,
                "prompt_sha256": "a" * 64,
            }
            for index in range(10)
        ]
        first = sampler.legacy.select_requests("benefit", "smoke", requests)
        second = sampler.legacy.select_requests("benefit", "smoke", requests)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_smoke_selects_one_outcome_blind_sample_per_medical_prompt(self):
        sampler.install_pair_contract()
        requests = []
        for prompt_index in range(16):
            for sample_index in range(5):
                requests.append(
                    {
                        "request_index": len(requests),
                        "prompt_ordinal": prompt_index,
                        "question_id": f"medical-{prompt_index}",
                        "sample_index": sample_index,
                        "prompt_sha256": "a" * 64,
                    }
                )
        first = sampler.legacy.select_requests("medical", "smoke", requests)
        second = sampler.legacy.select_requests("medical", "smoke", requests)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len({row["question_id"] for row in first}), 16)

    def test_metadata_records_pair_and_no_extra_model_loads(self):
        sampler.install_pair_contract()
        profile = {
            "domain": "medical",
            "endpoint": "official16",
            "n_samples": 5,
            "temperature": 1.0,
            "max_new_tokens": 1024,
            "max_context": 2048,
            "prompt_file_sha256": "b" * 64,
        }
        meta = sampler._pair_stream_meta(
            {
                "path": "/sealed/manifest.json",
                "file_sha256": "c" * 64,
                "manifest_payload_sha256": "d" * 64,
            },
            "medical",
            "smoke",
            profile,
            [],
        )
        self.assertEqual(meta["panel_order"], ["A", "B1"])
        self.assertEqual(meta["kalai_contract"]["k"], 2)
        self.assertEqual(meta["kalai_contract"]["s"], 1)
        self.assertEqual(meta["kalai_contract"]["R"], 20)
        self.assertEqual(meta["proposal_stream_id"], sampler.PROPOSAL_STREAM_ID)
        self.assertEqual(
            meta["smoke_selection"]["medical"],
            "one_sha256_ranked_sample_per_prompt",
        )
        architecture = meta["runtime_model_architecture"]
        self.assertEqual(architecture["reference_model_count"], 2)
        self.assertFalse(architecture["B2_or_B3_loaded"])
        self.assertFalse(architecture["direct_base_loaded"])

    def test_request_seed_is_bound_to_unique_k2_proposal_stream(self):
        sampler.install_pair_contract()
        expected = sampler.legacy.primary.tuple_seed(
            sampler.legacy.primary.GENERATION_SEED,
            sampler.PROPOSAL_STREAM_ID,
            "medical",
            "question-1",
            3,
        )
        observed = sampler.legacy.primary.tuple_seed(
            sampler.legacy.primary.GENERATION_SEED,
            sampler.legacy.PROPOSAL_STREAM_ID,
            "medical",
            "question-1",
            3,
        )
        legacy_m4 = sampler.legacy.primary.tuple_seed(
            sampler.legacy.primary.GENERATION_SEED,
            "whole_output_consensus_m4_max20_v1",
            "medical",
            "question-1",
            3,
        )
        self.assertEqual(observed, expected)
        self.assertNotEqual(observed, legacy_m4)

    def test_pair_sample_audit_replays_k2_source_and_uniform_draw(self):
        sampler.install_pair_contract()
        request = {
            "request_index": 0,
            "prompt_ordinal": 0,
            "question_id": "medical-0",
            "sample_index": 0,
            "prompt_sha256": "a" * 64,
        }
        request_seed = sampler.legacy.primary.tuple_seed(
            sampler.legacy.primary.GENERATION_SEED,
            sampler.PROPOSAL_STREAM_ID,
            "medical",
            request["question_id"],
            request["sample_index"],
        )
        rng = sampler.legacy.random.Random(request_seed)
        source = sampler.PANEL_ORDER[rng.randrange(len(sampler.PANEL_ORDER))]
        token_seed = sampler.legacy.primary.tuple_seed(
            request_seed, "candidate_tokens", 0, source
        )
        uniform_draw = rng.random()
        response = "accepted"
        response_sha256 = sampler.legacy._sha256(response.encode("utf-8"))
        attempt = {
            "attempt_index": 0,
            "proposal_source": source,
            "token_seed": token_seed,
            "finish_reason": "stop",
            "generated_tokens": 1,
            "sampled_tokens": 2,
            "sequence_logps": {"A": -1.0, "B1": -1.0},
            "acceptance_probability": 1.0,
            "uniform_draw": uniform_draw,
            "eligible_for_acceptance": True,
            "accepted": True,
            "response_sha256": response_sha256,
        }
        sample = {
            **request,
            "request_seed": request_seed,
            "accepted": True,
            "abstained": False,
            "attempts_used": 1,
            "accepted_source": source,
            "response": response,
            "response_sha256": response_sha256,
            "finish_reason": "stop",
            "generated_tokens": 1,
            "attempts": [attempt],
        }
        sample["sample_sha256"] = sampler.legacy._sha256(
            sampler.legacy._canonical(sample)
        )
        sampler.legacy._audit_sample(
            sample,
            request,
            "medical",
            {"max_new_tokens": 1024},
        )

    def test_summary_keeps_abstention_as_coverage(self):
        summary = sampler.legacy.summarize_samples(
            [
                {
                    "accepted": True,
                    "abstained": False,
                    "attempts_used": 1,
                    "response": "accepted",
                    "generated_tokens": 3,
                    "attempts": [{"generated_tokens": 3, "sampled_tokens": 4}],
                },
                {
                    "accepted": False,
                    "abstained": True,
                    "attempts_used": 20,
                    "response": "",
                    "generated_tokens": 0,
                    "attempts": [
                        {"generated_tokens": 1, "sampled_tokens": 2}
                    ]
                    * 20,
                },
            ]
        )
        self.assertEqual(summary["accepted_n"], 1)
        self.assertEqual(summary["abstained_n"], 1)
        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["abstention_rate"], 0.5)

    def test_self_test_is_cpu_only(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(sampler.main(["--self-test"]), 0)
        self.assertIn("KALAI_K2_S1_R20_V1_SELF_TEST_OK", output.getvalue())

    def test_tillicum_jobs_keep_smoke_and_full_separate(self):
        smoke = (
            ROOT
            / "scripts/sbatch_massive_medical_kalai_k2_s1_smoke_v1_tillicum_h200.sbatch"
        ).read_text()
        full = (
            ROOT
            / "scripts/sbatch_massive_medical_kalai_k2_s1_full_v1_tillicum_h200.sbatch"
        ).read_text()
        self.assertIn("--stage smoke", smoke)
        self.assertIn("benefit medical", smoke)
        self.assertIn("maximum_candidate_attempts=360", smoke)
        self.assertIn("#SBATCH --time=00:30:00", smoke)
        self.assertIn("full_run_authorized=false", smoke)
        self.assertIn('test -s "$control/CPU_STAGE"', smoke)
        self.assertIn('test -s "$control/SMOKE_AUTHORIZATION"', smoke)
        self.assertIn('test -s "$control/SMOKE_SUBMITTED"', smoke)
        self.assertIn('test -s "$control/SMOKE_RELEASE_AUTHORIZED"', smoke)
        self.assertIn('test -s "$control/SMOKE_RELEASED"', smoke)
        self.assertIn('grep -Fx "job_id=$SLURM_JOB_ID"', smoke)
        self.assertIn("#SBATCH --array=0-1", full)
        self.assertIn("--stage full", full)
        self.assertIn('test -s "$control/SMOKE_COMPLETE"', full)
        self.assertIn("FULL_${phase_upper}_AUTHORIZATION", full)
        self.assertIn("FULL_${phase_upper}_SUBMITTED", full)
        self.assertIn("FULL_${phase_upper}_RELEASE_AUTHORIZED", full)
        self.assertIn("FULL_${phase_upper}_RELEASED", full)
        self.assertIn('grep -Fx "job_id=$SLURM_JOB_ID"', full)
        self.assertIn("full_run_authorized=true", full)
        self.assertIn("slurm_time_limit_minutes <= authorized_minutes", full)
        self.assertNotIn("sbatch ", smoke)
        self.assertNotIn("OPENAI_API_KEY=", smoke + full)

    def test_stage_and_held_first_smoke_wrapper_preserve_authority_boundary(self):
        stage = (
            ROOT
            / "scripts/stage_massive_medical_kalai_k2_s1_v1_tillicum.sh"
        ).read_text()
        submit = (
            ROOT
            / "scripts/submit_massive_medical_kalai_k2_s1_smoke_v1_tillicum.sh"
        ).read_text()
        self.assertIn("CPU_STAGED_NO_GPU_OR_API_AUTHORITY", stage)
        self.assertIn("--preflight-only", stage)
        self.assertNotIn("\nsbatch ", stage)
        self.assertIn("--ack-h200-minutes 30", submit)
        self.assertIn("--ack-max-cost-usd 0.45", submit)
        self.assertIn("--ack-no-api --ack-no-full-run", submit)
        self.assertIn("sbatch --parsable --hold", submit)
        self.assertIn("held_audit_passed=true", submit)
        self.assertIn("scontrol release", submit)
        self.assertIn("full_run_authorized=false", submit)

    def test_protocol_document_binds_frozen_banks(self):
        document = (
            ROOT / "docs/massive_medical_kalai_k2_s1_r20_v1_protocol.md"
        ).read_text()
        self.assertIn("deterministic 360-row MASSIVE", document)
        self.assertIn("official 16 medical prompts", document)
        self.assertIn("Abstention is a coverage", document)
        self.assertIn("outcome and must never be scored", document)
        self.assertIn("ssh tillicum true", document)
        self.assertIn("Direct `sbatch` of the smoke file is intentionally rejected", document)
        self.assertIn("No full-run\nauthorization or submission wrapper", document)
        self.assertIn("FULL_<PHASE>_AUTHORIZATION", document)
        self.assertIn("existing CPU-stage and smoke receipts explicitly say", document)


if __name__ == "__main__":
    unittest.main()
