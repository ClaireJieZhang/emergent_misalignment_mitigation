"""Focused fail-closed control tests for derivation recovery v7."""

import importlib
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, os.fspath(SCRIPTS))

v6 = importlib.import_module(
    "judge_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v6"
)
V6_SNAPSHOT = {
    "recovery_id": v6.RECOVERY_ID,
    "expected_source_commit": v6.EXPECTED_SOURCE_COMMIT,
    "prior_output": v6.PRIOR_V5_OUTPUT,
    "recovery_paths": v6.recovery_paths,
}
v7 = importlib.import_module(
    "audit_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_derive_recovery_v7"
)


class JudgeDeriveRecoveryV7ControlTests(unittest.TestCase):
    def read(self, name):
        return (SCRIPTS / name).read_text(encoding="utf-8")

    def test_private_import_does_not_mutate_canonical_v6(self):
        self.assertEqual(v6.RECOVERY_ID, V6_SNAPSHOT["recovery_id"])
        self.assertEqual(
            v6.EXPECTED_SOURCE_COMMIT, V6_SNAPSHOT["expected_source_commit"]
        )
        self.assertEqual(v6.PRIOR_V5_OUTPUT, V6_SNAPSHOT["prior_output"])
        self.assertIs(v6.recovery_paths, V6_SNAPSHOT["recovery_paths"])

    def test_exact_seven_file_add_only_scope(self):
        self.assertEqual(len(v7.ADDED_FILES), 7)
        self.assertEqual(len(set(v7.ADDED_FILES)), 7)
        self.assertTrue(all("judge_derive_recovery_v7" in p for p in v7.ADDED_FILES))
        self.assertTrue(all((ROOT / path).is_file() for path in v7.ADDED_FILES))

    def test_exact_v6_repo_and_inventory_commitments(self):
        self.assertEqual(v7.SOURCE_COMMIT, "c4016c332c461efa07c85028a164c787f2e65650")
        self.assertEqual(v7.SOURCE_TREE, "20b84f0edbebd1274fa2ca11144d11b5b95e2991")
        self.assertEqual(v7.SOURCE_INVENTORY_FILE_COUNT, 252)
        self.assertEqual(
            v7.SOURCE_INVENTORY_STREAM_SHA256,
            "06261ede31668b3e7e51dce5b678898b09173ddfb6a01f877ea84346db6f59d0",
        )
        self.assertEqual(len(v7.SOURCE_CRITICAL_FILES), 15)
        self.assertIn("evaluation/medical/judge_checkpoint.json.002", v7.SOURCE_CRITICAL_FILES)
        self.assertIn("evaluation/medical/judge_checkpoint.json.240", v7.SOURCE_CRITICAL_FILES)
        self.assertIn("evaluation/medical/judgments_new.json", v7.SOURCE_CRITICAL_FILES)
        self.assertEqual(v7.SOURCE_V6_RECOVERY_ID, v6.RECOVERY_ID)
        self.assertEqual(v7.SOURCE_PROTOCOL_ID, v6.source.PROTOCOL_ID)
        self.assertNotEqual(v7.SOURCE_V6_RECOVERY_ID, v7.SOURCE_PROTOCOL_ID)

    def test_cost_order_mismatch_and_repair_are_exact(self):
        self.assertEqual(
            v7.COST_ORDER_RECOVERY,
            {
                "bug_class": "floating_point_nonassociativity_after_presentation_sort",
                "chronological_cost_usd": 0.031268499999999984,
                "sorted_presentation_cost_usd": 0.0312685,
                "chronological_minus_sorted_usd": -1.3877787807814457e-17,
                "new_v6_chronological_cost_usd": 0.03115399999999998,
                "presentation_rows_equal_sorted_chronological_rows": True,
                "chronological_rows_equal_presentation_rows": False,
                "repair": "sum_checkpoint_chronology_before_sorting_rows_for_presentation",
            },
        )

    def test_v7_output_is_distinct_and_source_is_read_only(self):
        self.assertNotEqual(v7.SOURCE_OUTPUT, v7.RECOVERY_OUTPUT)
        self.assertTrue(str(v7.SOURCE_OUTPUT).endswith("_judge_recovery_v6"))
        self.assertTrue(str(v7.RECOVERY_OUTPUT).endswith("_judge_derive_recovery_v7"))
        stage = self.read(
            "stage_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_derive_recovery_v7_tillicum.sh"
        )
        self.assertIn('test ! -e "$source_output/evaluation/medical/judgments_merged.json"', stage)
        self.assertNotRegex(stage, r"(?:cp|mv)\s+[^\n]*\$source_output")

    def test_stage_is_pushed_commit_cpu_only_and_key_absent(self):
        stage = self.read(
            "stage_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_derive_recovery_v7_tillicum.sh"
        )
        self.assertIn('git clone --branch "$branch" --single-branch "$url" "$repo"', stage)
        self.assertIn("c4016c332c461efa07c85028a164c787f2e65650", stage)
        self.assertGreaterEqual(stage.count("OPENAI_API_KEY must be absent"), 2)
        for forbidden in ("sbatch ", "srun ", "salloc ", "nvidia-smi", "curl "):
            self.assertNotIn(forbidden, stage)
        for mask in (
            "CUDA_VISIBLE_DEVICES=''",
            "NVIDIA_VISIBLE_DEVICES=none",
            "ROCR_VISIBLE_DEVICES=''",
        ):
            self.assertIn(mask, stage)

    def test_derive_is_cpu_only_fresh_namespace_and_idempotent(self):
        derive = self.read(
            "derive_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_derive_recovery_v7_tillicum.sh"
        )
        self.assertIn("acquire-derive-lock", derive)
        self.assertIn('merge --derive-manifest "$manifest"', derive)
        self.assertIn('audit-final --derive-manifest "$manifest"', derive)
        self.assertIn("audit-complete", derive)
        self.assertNotIn("judge_recovery_v6_tillicum.sh", derive)
        for forbidden in ("sbatch ", "srun ", "salloc ", "nvidia-smi", "curl "):
            self.assertNotIn(forbidden, derive)
        for mask in (
            "CUDA_VISIBLE_DEVICES=''",
            "NVIDIA_VISIBLE_DEVICES=none",
            "ROCR_VISIBLE_DEVICES=''",
        ):
            self.assertIn(mask, derive)

    def test_status_distinguishes_staged_partial_and_complete(self):
        status = self.read(
            "status_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_derive_recovery_v7_tillicum.sh"
        )
        for marker in (
            "JUDGE_DERIVE_RECOVERY_V7_FINAL_COMPLETE",
            "JUDGE_DERIVE_RECOVERY_V7_CPU_DERIVATION_INCOMPLETE_EXACT_IDEMPOTENT_REENTRY_ONLY",
            "JUDGE_DERIVE_RECOVERY_V7_CPU_STAGED_AWAITING_CPU_DERIVATION",
        ):
            self.assertIn(marker, status)

    def test_manifest_contract_records_no_new_compute(self):
        text = self.read(
            "audit_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_derive_recovery_v7.py"
        )
        for fragment in (
            '"source_v6_read_only": True',
            '"fresh_v7_output_namespace": True',
            '"new_external_api_calls": 0',
            '"new_gpu_jobs": 0',
            '"source_v6_budget_contract"',
            '"partial_or_unknown_inventory_fails_closed": True',
        ):
            self.assertIn(fragment, text)

    def test_binding_and_atomic_json_are_sealed_and_mode_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            payload = v7.seal({"exact": True})
            v7.atomic_json(path, payload, mode=0o400)
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
            self.assertEqual(
                v7.binding(path, require_seal=True), v7.binding(path, payload)
            )
            with self.assertRaises(FileExistsError):
                v7.atomic_json(path, payload, mode=0o400)

    def test_api_key_blocks_control_plane_entry(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-be-used"}):
            with self.assertRaisesRegex(ValueError, "must be absent"):
                v7.prepare_command(None)

    def test_post_lock_audit_revalidates_immutable_stage_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control"
            medical = root / "evaluation" / "medical"
            final = root / "evaluation" / "final"
            logs = root / "logs"
            for directory in (control, medical, final, logs):
                directory.mkdir(parents=True, exist_ok=True)
            for name in (
                v7.MANIFEST_FILE.name,
                v7.PREP_FILE.name,
                v7.PREFLIGHT_FILE.name,
                v7.STAGED_FILE.name,
                v7.LOCK_FILE.name,
            ):
                (control / name).touch()
            with mock.patch.object(v7, "CONTROL_ROOT", control), mock.patch.object(
                v7, "MEDICAL_ROOT", medical
            ), mock.patch.object(v7, "FINAL_ROOT", final), mock.patch.object(
                v7, "LOG_ROOT", logs
            ), mock.patch.object(
                v7, "audit_manifest_exact", return_value={"body": "exact"}
            ), mock.patch.object(
                v7,
                "_audit_stage_records",
                side_effect=ValueError("tampered immutable stage record"),
            ) as stage_audit:
                with self.assertRaisesRegex(ValueError, "tampered immutable"):
                    v7.audit_derive_namespace()
                stage_audit.assert_called_once_with({"body": "exact"})

    def test_stage_record_modes_are_frozen_at_0600(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ("manifest", "prep", "preflight", "staged")]
            for path in paths:
                path.touch(mode=0o600)
            paths[2].chmod(0o644)
            with mock.patch.object(v7, "MANIFEST_FILE", paths[0]), mock.patch.object(
                v7, "PREP_FILE", paths[1]
            ), mock.patch.object(v7, "PREFLIGHT_FILE", paths[2]), mock.patch.object(
                v7, "STAGED_FILE", paths[3]
            ):
                with self.assertRaisesRegex(ValueError, "control mode differs"):
                    v7._audit_stage_records({"body": "exact"})

    def test_no_placeholders_destructive_commands_or_network_api_clients(self):
        combined = "\n".join(
            (ROOT / path).read_text(encoding="utf-8") for path in v7.ADDED_FILES
        )
        self.assertNotIn("PLACE" + "HOLDER_", combined)
        self.assertNotRegex(combined, re.compile(r"(^|[;&|]\s*)rm\s", re.MULTILINE))
        self.assertNotIn("chat." + "completions.create", combined)
        self.assertNotIn("responses." + "create", combined)


if __name__ == "__main__":
    unittest.main()
