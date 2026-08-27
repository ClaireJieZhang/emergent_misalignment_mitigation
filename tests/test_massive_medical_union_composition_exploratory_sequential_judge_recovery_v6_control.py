import importlib
import os
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, os.fspath(SCRIPTS))

v5 = importlib.import_module(
    "audit_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v5"
)
V5_SNAPSHOT = {
    "recovery_id": v5.RECOVERY_ID, "repo_root": v5.REPO_ROOT,
    "manifest": v5.MANIFEST_FILE, "branch": v5.BRANCH,
    "audit_repo": v5.audit_repo, "audit_source": v5.audit_source,
    "print": getattr(v5, "print", None),
}
v6 = importlib.import_module(
    "audit_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v6"
)


class JudgeRecoveryV6ControlTests(unittest.TestCase):
    def read(self, name):
        return (SCRIPTS / name).read_text(encoding="utf-8")

    def test_import_does_not_mutate_canonical_v5_control(self):
        self.assertEqual(v5.RECOVERY_ID, V5_SNAPSHOT["recovery_id"])
        self.assertEqual(v5.REPO_ROOT, V5_SNAPSHOT["repo_root"])
        self.assertEqual(v5.MANIFEST_FILE, V5_SNAPSHOT["manifest"])
        self.assertEqual(v5.BRANCH, V5_SNAPSHOT["branch"])
        self.assertIs(v5.audit_repo, V5_SNAPSHOT["audit_repo"])
        self.assertIs(v5.audit_source, V5_SNAPSHOT["audit_source"])
        self.assertIs(getattr(v5, "print", None), V5_SNAPSHOT["print"])

    def test_summary_facade_is_v6_scoped_without_mutating_v5(self):
        canonical = importlib.import_module(
            "summarize_massive_medical_union_composition_exploratory_"
            "sequential_confirmation_v1_judge_recovery_v5"
        )
        canonical_recovery = canonical.recovery
        canonical_id = canonical.RECOVERY_ID
        facade = importlib.import_module(
            "summarize_massive_medical_union_composition_exploratory_"
            "sequential_confirmation_v1_judge_recovery_v6"
        )
        self.assertIs(canonical.recovery, canonical_recovery)
        self.assertTrue(canonical_id.endswith("judge_recovery_v5"))
        self.assertTrue(facade.RECOVERY_ID.endswith("judge_recovery_v6"))
        self.assertIs(facade.summary.recovery, facade.recovery)

    def test_exact_nine_file_add_only_scope(self):
        self.assertEqual(len(v6.ADDED_FILES), 9)
        self.assertEqual(len(set(v6.ADDED_FILES)), 9)
        self.assertTrue(all("judge_recovery_v6" in path for path in v6.ADDED_FILES))
        self.assertTrue(all((ROOT / path).is_file() for path in v6.ADDED_FILES))

    def test_exact_live_v5_terminal_inventory_is_frozen(self):
        self.assertEqual(len(v6.PRIOR_FILES), 13)
        for required in (
            "control/PREP.json", "control/CPU_PREFLIGHT.json",
            "control/CANARY_SUCCESS.json", "control/CONTINUATION_AUTHORIZATION.json",
            "control/CONTINUATION_FAILURE.json",
            "evaluation/medical/judge_checkpoint.json.001",
            "logs/external_judge_canary.log", "logs/external_judge_continuation.log",
        ):
            self.assertIn(required, v6.PRIOR_FILES)
        self.assertNotIn("control/CANARY_FAILURE.json", v6.PRIOR_FILES)
        self.assertEqual(v6.SOURCE_COMMIT, "3834f4215f6606ad49e511620bedd49219ecc3df")
        self.assertEqual(v6.SOURCE_TREE, "fb43b11e2be4030dace1e3a1e57c795b61a86566")
        self.assertTrue(v6.PRIOR_V5_BRANCH.endswith("judge-recovery-v5"))

    def test_prior_v5_is_read_only_and_v6_namespaces_are_distinct(self):
        self.assertNotEqual(v6.PRIOR_RECOVERY_OUTPUT, v6.RECOVERY_OUTPUT)
        self.assertTrue(str(v6.PRIOR_RECOVERY_OUTPUT).endswith("_judge_recovery_v5"))
        self.assertTrue(str(v6.RECOVERY_OUTPUT).endswith("_judge_recovery_v6"))
        stage = self.read(
            "stage_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v6_tillicum.sh"
        )
        self.assertIn("prior_output=$root/outputs/", stage)
        self.assertNotRegex(stage, r"(^|[;&|]\s*)rm\s")
        self.assertNotIn('mv "$prior', stage)
        self.assertNotIn('cp "$prior', stage)

    def test_stage_is_cpu_only_and_api_key_absent(self):
        stage = self.read(
            "stage_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v6_tillicum.sh"
        )
        for forbidden in ("sbatch ", "srun ", "salloc ", "nvidia-smi", "curl "):
            self.assertNotIn(forbidden, stage)
        self.assertIn("OPENAI_API_KEY must be absent during CPU staging", stage)
        self.assertIn("validate-sdk-serialization", stage)
        self.assertIn("MockTransport", stage)
        self.assertIn("judge_recovery_v5", stage)
        self.assertIn("CONTINUATION_FAILURE.json", stage)

    def test_finalizer_freezes_exact_attempt_budget_and_owner_token(self):
        finalizer = self.read(
            "finalize_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v6_tillicum.sh"
        )
        self.assertIn('write-authorization --stage "$stage" "${@:2}"', finalizer)
        self.assertGreaterEqual(finalizer.count('--owner-token "$owner_token"'), 4)
        for token in (
            "--ack-v5-canary-actual-usd", "--ack-v5-continuation-authority-cap-usd",
            "--ack-v5-continuation-authority-consumed-nonreusable",
            "--ack-v5-failed-request-actual-billing-unknown",
            "--ack-v5-unattempted-authority-not-cost-exposure",
            "--ack-v5-unattempted-authority-not-reused",
            "--ack-prior-network-attempts-min", "--ack-prior-network-attempts-max",
        ):
            self.assertIn(token, finalizer)
        self.assertIn("0.743856", finalizer)
        self.assertIn("4.4183725", finalizer)
        self.assertIn("0.5816275", finalizer)

    def test_parser_requires_owner_token_and_all_exact_ack_flags(self):
        parser = v6.build_parser()
        subparsers = next(
            action for action in parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        )
        authorize = subparsers.choices["write-authorization"]
        options = {option for action in authorize._actions for option in action.option_strings}
        self.assertIn("--owner-token", options)
        self.assertIn("--ack-v5-failed-continuation-exposure-cap-usd", options)
        self.assertIn("--ack-prior-accepted-judgments", options)
        owner = next(action for action in authorize._actions if "--owner-token" in action.option_strings)
        self.assertTrue(owner.required)

    def test_canary_authorization_remains_valid_after_permanent_lock(self):
        auditor = self.read(
            "audit_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v6.py"
        )
        self.assertIn("def _audit_staged_seals_after_lock():", auditor)
        self.assertIn("def _authority_state(stage, require_staged_only=False):", auditor)
        self.assertIn(
            "require_staged_only=args.stage == \"canary\"", auditor
        )
        self.assertIn(
            "_authority_state(args.stage)\n    control.audit_lock", auditor
        )

    def test_budget_and_manifest_contract_match_judge(self):
        judge = importlib.import_module(
            "judge_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v6"
        )
        self.assertEqual(v6.IDEMPOTENCY_CONTRACT, judge.IDEMPOTENCY_CONTRACT)
        self.assertEqual(v6.PRIOR_ACCOUNTED_EXPOSURE_USD, 0.7562585)
        self.assertEqual(v6.RECOVERY_TOTAL_CAP_USD, 0.746928)
        self.assertEqual(v6.CONSERVATIVE_PROGRAM_MAX_USD, 4.4183725)
        self.assertEqual(v6.PROGRAM_CEILING_GAP_USD, 0.5816275)

    def test_no_fail_open_placeholders_or_v5_write_namespace(self):
        paths = [ROOT / path for path in v6.ADDED_FILES if (ROOT / path).is_file()]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("PLACE" + "HOLDER_", combined)
        self.assertNotRegex(
            combined, re.compile(r"^RECOVERY_OUTPUT\s*=.*judge_recovery_v5", re.MULTILINE)
        )


if __name__ == "__main__":
    unittest.main()
