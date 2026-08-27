import importlib
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, os.fspath(SCRIPTS))

v4 = importlib.import_module(
    "audit_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4"
)
V4_SNAPSHOT = {
    "recovery_id": v4.RECOVERY_ID,
    "repo_root": v4.REPO_ROOT,
    "manifest": v4.MANIFEST_FILE,
    "audit_repo": v4.audit_repo,
    "print": getattr(v4, "print", None),
}
v5 = importlib.import_module(
    "audit_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v5"
)


class JudgeRecoveryV5ControlTests(unittest.TestCase):
    def read(self, name):
        return (SCRIPTS / name).read_text(encoding="utf-8")

    def test_import_does_not_mutate_canonical_v4_control(self):
        self.assertEqual(v4.RECOVERY_ID, V4_SNAPSHOT["recovery_id"])
        self.assertEqual(v4.REPO_ROOT, V4_SNAPSHOT["repo_root"])
        self.assertEqual(v4.MANIFEST_FILE, V4_SNAPSHOT["manifest"])
        self.assertIs(v4.audit_repo, V4_SNAPSHOT["audit_repo"])
        self.assertIs(getattr(v4, "print", None), V4_SNAPSHOT["print"])

    def test_summary_facade_is_private_and_v5_scoped(self):
        canonical = importlib.import_module(
            "summarize_massive_medical_union_composition_exploratory_"
            "sequential_confirmation_v1_judge_recovery_v4"
        )
        canonical_recovery = canonical.recovery
        facade = importlib.import_module(
            "summarize_massive_medical_union_composition_exploratory_"
            "sequential_confirmation_v1_judge_recovery_v5"
        )
        self.assertIs(canonical.recovery, canonical_recovery)
        self.assertTrue(canonical.RECOVERY_ID.endswith("judge_recovery_v4"))
        self.assertTrue(facade.RECOVERY_ID.endswith("judge_recovery_v5"))
        self.assertIs(facade.summary.recovery, facade.recovery)

    def test_exact_add_only_scope(self):
        self.assertEqual(len(v5.ADDED_FILES), 9)
        self.assertEqual(len(set(v5.ADDED_FILES)), 9)
        self.assertTrue(all("judge_recovery_v5" in path for path in v5.ADDED_FILES))
        self.assertTrue(all((ROOT / path).is_file() for path in v5.ADDED_FILES))

    def test_prior_v4_binding_is_read_only_and_v5_writes_are_distinct(self):
        self.assertNotEqual(v5.PRIOR_RECOVERY_OUTPUT, v5.RECOVERY_OUTPUT)
        self.assertTrue(str(v5.PRIOR_RECOVERY_OUTPUT).endswith("_judge_recovery_v4"))
        self.assertTrue(str(v5.RECOVERY_OUTPUT).endswith("_judge_recovery_v5"))
        stage = self.read(
            "stage_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v5_tillicum.sh"
        )
        self.assertIn("prior_output=$root/outputs/", stage)
        self.assertNotIn("\nrm ", stage)
        self.assertNotIn("mv \"$prior", stage)
        self.assertNotIn("cp \"$prior", stage)

    def test_stage_is_cpu_only_and_api_key_absent(self):
        stage = self.read(
            "stage_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v5_tillicum.sh"
        )
        for forbidden in ("sbatch ", "srun ", "salloc ", "nvidia-smi", "curl "):
            self.assertNotIn(forbidden, stage)
        self.assertIn("OPENAI_API_KEY must be absent during CPU staging", stage)
        self.assertIn("validate-sdk-serialization", stage)
        self.assertIn("MockTransport", stage)

    def test_finalizer_binds_owner_to_authorization_and_core(self):
        finalizer = self.read(
            "finalize_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v5_tillicum.sh"
        )
        self.assertIn(
            'write-authorization --stage "$stage" "${@:2}"', finalizer
        )
        self.assertGreaterEqual(finalizer.count('--owner-token "$owner_token"'), 4)
        self.assertIn(
            '--owner-token "$owner_token" > "$log" 2>&1 &', finalizer
        )
        self.assertIn("--ack-consumed-v3-authority-cap-usd", finalizer)
        self.assertIn("--ack-consumed-v4-canary-authority-cap-usd", finalizer)
        self.assertIn("--ack-v4-continuation-never-authorized", finalizer)
        self.assertIn("4.418258", finalizer)

    def test_parser_requires_owner_token_for_authorization(self):
        parser = v5.build_parser()
        subparsers = next(
            action for action in parser._actions if hasattr(action, "choices")
            and isinstance(action.choices, dict)
        )
        authorize = subparsers.choices["write-authorization"]
        owner = next(
            action for action in authorize._actions
            if "--owner-token" in action.option_strings
        )
        self.assertTrue(owner.required)

    def test_exact_prior_terminal_inventory_is_frozen(self):
        self.assertEqual(len(v5.PRIOR_FILES), 8)
        self.assertIn("control/PREP.json", v5.PRIOR_FILES)
        self.assertIn("control/CPU_PREFLIGHT.json", v5.PRIOR_FILES)
        self.assertEqual(
            v5.PRIOR_FILES["control/CANARY_FAILURE.json"][2],
            "ce57a34230596b9d964fb2e5843d09783f52782e0d5a10cb9e513fe66b87ee64",
        )
        self.assertEqual(v5.PRIOR_CONSUMED_AUTHORITY_CAP_USD, 0.753072)
        self.assertEqual(v5.CONSERVATIVE_PROGRAM_MAX_USD, 4.418258)


if __name__ == "__main__":
    unittest.main()
