import importlib.util
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = load_module(
    "kalai_s1_completion_batch_planner_tests",
    "prepare_massive_medical_kalai_s1_completion_batches_v1.py",
)
stage = load_module(
    "kalai_s1_completion_batch_stage_tests",
    "prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py",
)


def synthetic_rows():
    """Create the frozen 115/59 and 2,947-attempt completion shape."""

    rows = {"benefit": [], "medical": []}
    ordinal = 0
    for phase, count in planner.EXPECTED_ROWS_BY_PHASE.items():
        for sample_index in range(count):
            # 11 * 16 + 163 * 17 = 2,947.
            maximum = 16 if ordinal < 11 else 17
            body = {
                "phase": phase,
                "stage": "completion",
                "partition": "completion",
                "request_index": ordinal,
                "question_id": f"{phase}_{sample_index:03d}",
                "sample_index": sample_index,
                "classification": "needs_continuation",
                "disposition": "unresolved",
                "max_new_attempts": maximum,
                "reusable_terminal_sample": None,
            }
            body["row_sha256"] = planner.sha256_bytes(planner.canonical_bytes(body))
            rows[phase].append(body)
            ordinal += 1
    return rows


class CompletionBatchPlannerTests(unittest.TestCase):
    def test_frozen_scope_is_exact(self):
        by_phase = synthetic_rows()
        replay_body = {
            "phases": {
                phase: {"rows": rows} for phase, rows in by_phase.items()
            }
        }
        rows = planner.completion_rows(replay_body)
        self.assertEqual(len(rows), 174)
        self.assertEqual(sum(row["max_new_attempts"] for row in rows), 2947)
        self.assertEqual(
            {phase: sum(row["phase"] == phase for row in rows) for phase in planner.PHASES},
            {"benefit": 115, "medical": 59},
        )

    def test_lpt_is_deterministic_exact_once_and_uses_rational_rates(self):
        rows = [row for phase_rows in synthetic_rows().values() for row in phase_rows]
        rates = {"benefit": Fraction(7, 3), "medical": Fraction(11, 4)}
        first, first_loads = planner.deterministic_lpt(rows, rates)
        second, second_loads = planner.deterministic_lpt(list(reversed(rows)), rates)
        first_hashes = [[row["row_sha256"] for row in batch] for batch in first]
        second_hashes = [[row["row_sha256"] for row in batch] for batch in second]
        self.assertEqual(first_hashes, second_hashes)
        self.assertEqual(first_loads, second_loads)
        flattened = [row for batch in first for row in batch]
        self.assertEqual(len(flattened), 174)
        self.assertEqual(len({row["row_sha256"] for row in flattened}), 174)
        self.assertEqual(sum(row["max_new_attempts"] for row in flattened), 2947)
        self.assertTrue(all(isinstance(value, Fraction) for value in first_loads))

    def test_lpt_tie_breaks_by_phase_request_and_sample_then_batch(self):
        rows = [
            {
                "phase": phase,
                "request_index": request_index,
                "sample_index": sample_index,
                "max_new_attempts": 1,
                "row_sha256": f"{phase}-{request_index}-{sample_index}",
            }
            for phase, request_index, sample_index in (
                ("medical", 0, 0),
                ("benefit", 2, 0),
                ("benefit", 1, 1),
                ("benefit", 1, 0),
                ("medical", 1, 0),
                ("benefit", 3, 0),
                ("medical", 2, 0),
            )
        ]
        assigned, _ = planner.deterministic_lpt(
            rows, {"benefit": Fraction(1), "medical": Fraction(1)}
        )
        self.assertEqual(
            [batch[0]["row_sha256"] for batch in assigned],
            [
                "benefit-1-0",
                "benefit-1-1",
                "benefit-2-0",
                "benefit-3-0",
                "medical-0-0",
                "medical-1-0",
                "medical-2-0",
            ],
        )

    def test_phase_rate_is_bound_to_sealed_generation_and_timing(self):
        replay = planner.seal({"kind": "replay"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = planner.seal(
                {
                    "meta": {
                        "protocol_id": planner.CONTROLLER_PROTOCOL_ID,
                        "method_id": planner.METHOD_ID,
                        "stage": "technical_gate",
                        "phase": "benefit",
                        "replay_plan_payload_sha256": replay[planner.SEAL_FIELD],
                        "stored_prefix_candidates_regenerated": False,
                        "restart_or_resume_authorized": False,
                        "external_api_calls": 0,
                    },
                    "summary": {"accepted_n": 1},
                    "new_attempts_generated": 4,
                    "samples": [],
                }
            )
            generation_path = root / "generation.json"
            generation_path.write_text(json.dumps(generation), encoding="utf-8")
            timing = planner.seal(
                {
                    "protocol_id": planner.CONTROLLER_PROTOCOL_ID,
                    "method_id": planner.METHOD_ID,
                    "stage": "technical_gate",
                    "phase": "benefit",
                    "replay_plan_payload_sha256": replay[planner.SEAL_FIELD],
                    "elapsed_seconds": 10.0,
                    "summary": {"accepted_n": 1},
                    "external_api_calls": 0,
                }
            )
            timing_path = root / "timing.json"
            timing_path.write_text(json.dumps(timing), encoding="utf-8")
            gate_body = {
                "phase_outputs": {
                    "benefit": {
                        "continuation_generation": planner.binding(
                            generation_path, generation
                        )
                    }
                },
                "observed": {"benefit": {"new_attempts_generated": 4}},
            }
            with mock.patch.dict(
                planner.EXPECTED_GATE_GENERATION_SHA256,
                {"benefit": generation[planner.SEAL_FIELD]},
            ), mock.patch.dict(
                planner.EXPECTED_GATE_TIMING_SHA256,
                {"benefit": timing[planner.SEAL_FIELD]},
            ):
                evidence = planner._validate_phase_evidence(
                    phase="benefit",
                    replay_payload=replay,
                    gate_body=gate_body,
                    generation_path=generation_path,
                    timing_path=timing_path,
                )
            self.assertEqual(evidence["elapsed_seconds"], Fraction(10))
            self.assertEqual(evidence["seconds_per_new_attempt"], Fraction(5, 2))

            timing["elapsed_seconds"] = 11.0
            timing_path.write_text(json.dumps(timing), encoding="utf-8")
            with mock.patch.dict(
                planner.EXPECTED_GATE_GENERATION_SHA256,
                {"benefit": generation[planner.SEAL_FIELD]},
            ), mock.patch.dict(
                planner.EXPECTED_GATE_TIMING_SHA256,
                {"benefit": timing[planner.SEAL_FIELD]},
            ):
                with self.assertRaisesRegex(ValueError, "invalid payload_sha256"):
                    planner._validate_phase_evidence(
                        phase="benefit",
                        replay_payload=replay,
                        gate_body=gate_body,
                        generation_path=generation_path,
                        timing_path=timing_path,
                    )

    def test_accounting_and_cpu_stage_are_non_authorizing(self):
        planner.self_test()
        stage.self_test()
        source = (
            SCRIPTS / "prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"gpu_jobs_authorized": 0', source)
        self.assertIn('"gpu_authorized": False', source)
        self.assertIn('"external_api_calls_authorized": 0', source)
        self.assertIn('"external_api_authorized": False', source)
        self.assertIn('"automatic_next_batch_authorized": False', source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("srun ", source)
        self.assertNotIn("OPENAI_API_KEY", source)

    def test_one_hour_feasibility_is_sealed_and_fail_closed(self):
        self.assertEqual(planner.PLANNED_BATCH_SECONDS, 3600)
        source = (
            SCRIPTS / "prepare_massive_medical_kalai_s1_completion_batches_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"one_hour_feasibility"', source)
        self.assertIn("projected_maximum >= PLANNED_BATCH_SECONDS", source)


if __name__ == "__main__":
    unittest.main()
