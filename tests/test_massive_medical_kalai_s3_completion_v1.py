"""No-network tests for the bounded Kalai s=3 completion controller."""

from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location("_test_" + path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_massive_medical_kalai_s3_completion_v1.py")
authorizer = load("authorize_massive_medical_kalai_s3_completion_v1.py")
evaluator = load("evaluate_massive_medical_kalai_s3_completion_v1.py")


class CompletionControllerTests(unittest.TestCase):
    def test_exact_bounded_envelope(self):
        self.assertEqual(prepare.COMPLETION_H200_MINUTES, 94)
        self.assertEqual(prepare.COMPLETION_CAP_USD, Decimal("1.410"))
        self.assertEqual(
            prepare.CURRENT_CONSERVATIVE_EXPOSURE_USD,
            Decimal("4.8228935"),
        )
        self.assertEqual(
            prepare.MAXIMUM_WITH_COMPLETION_CAP_USD,
            Decimal("6.2328935"),
        )
        self.assertEqual(prepare.GENERIC_JUDGE_CAP_USD, Decimal("0.245760"))
        self.assertEqual(
            prepare.MAXIMUM_WITH_COMPLETION_AND_JUDGE_CAP_USD,
            Decimal("6.4786535"),
        )
        self.assertLess(
            prepare.MAXIMUM_WITH_COMPLETION_AND_JUDGE_CAP_USD,
            prepare.WORKFLOW_CEILING_USD,
        )

    def test_source_gate_commit_is_frozen(self):
        self.assertEqual(
            prepare.EXPECTED_GATE_REPO_COMMIT,
            "ed950b72396dc041d34bbb694ea1486763033657",
        )

    def test_controller_authority_is_sampler_compatible_and_stricter(self):
        source = (SCRIPTS / "authorize_massive_medical_kalai_s3_completion_v1.py").read_text()
        self.assertIn('"stage": "completion"', source)
        self.assertIn('"authorized_gpu_jobs": 1', source)
        self.assertIn('"external_api_calls_authorized": 0', source)
        self.assertIn('"restart_or_resume_authorized": False', source)
        self.assertIn('"retry_authorized": False', source)
        self.assertIn("audit_static", source)
        self.assertIn("audit_completion_state", source)

    def test_combined_runner_has_one_load_and_no_resume(self):
        path = SCRIPTS / "sample_massive_medical_kalai_s3_completion_combined_v1.py"
        source = path.read_text()
        self.assertEqual(source.count("load_independent_model_panel("), 1)
        self.assertNotIn("resume-partial", source)
        self.assertIn('PHASES = ("benefit", "medical")', source)
        self.assertIn('sampler.select_requests(phase, "completion"', source)
        self.assertIn("sampler.verify_passing_gate", source)
        self.assertIn("sampler.verify_gpu_authorization", source)

    def test_submit_is_held_first_and_nonrequeue(self):
        source = (
            SCRIPTS / "submit_massive_medical_kalai_s3_completion_v1_tillicum.sh"
        ).read_text()
        self.assertIn("sbatch --parsable --hold --export=NONE --no-requeue", source)
        self.assertIn("scontrol release", source)
        self.assertLess(source.index("--hold"), source.index("scontrol release"))
        self.assertIn("COMPLETION_SUBMISSION_LOCK", source)
        self.assertIn("--ack-h200-minutes 94", source)
        self.assertIn("--ack-max-cost-usd 1.410", source)
        self.assertIn("--ack-program-ceiling-usd 6.5000000", source)

    def test_gpu_job_forbids_api_and_seals_whole_job_cost(self):
        source = (
            SCRIPTS / "sbatch_massive_medical_kalai_s3_completion_v1_tillicum_h200.sbatch"
        ).read_text()
        self.assertIn("#SBATCH --time=01:34:00", source)
        self.assertIn("#SBATCH --no-requeue", source)
        self.assertIn("OPENAI_API_KEY must not reach a GPU job", source)
        self.assertIn('--elapsed-seconds "$((SECONDS - started))"', source)
        self.assertIn("COMPLETION_STOPPED", source)
        self.assertIn("for _release_wait in $(seq 1 60)", source)
        self.assertLess(
            source.index("trap on_exit EXIT"),
            source.index('test -s "$control/COMPLETION_RELEASED"'),
        )
        self.assertNotIn("--resume-partial", source)

    def test_evaluator_cost_is_second_exact(self):
        actual = Decimal(5639) * Decimal("0.90") / Decimal(3600)
        self.assertLessEqual(actual, prepare.COMPLETION_CAP_USD)
        self.assertEqual(
            Decimal(5640) * Decimal("0.90") / Decimal(3600),
            prepare.COMPLETION_CAP_USD,
        )
        self.assertTrue(callable(evaluator.evaluate))

    def test_self_tests(self):
        prepare.self_test()
        authorizer.self_test()
        evaluator.self_test()


if __name__ == "__main__":
    unittest.main()
