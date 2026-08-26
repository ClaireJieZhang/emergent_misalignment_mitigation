"""CPU control-plane tests for sequential judge recovery v4."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = ROOT / (
    "scripts/audit_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4.py"
)
SPEC = importlib.util.spec_from_file_location("judge_recovery_v4_control", AUDITOR_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class JudgeRecoveryV4ControlTests(unittest.TestCase):
    @staticmethod
    def _write_json(path, body, seal_field="payload_sha256"):
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            **body,
            seal_field: AUDIT.sha256_bytes(AUDIT.canonical_bytes(body)),
        }
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
        return value

    def _fake_source(self, root):
        bodies = {
            "protocol_manifest": ({"schema_version": 1}, "manifest_payload_sha256"),
            "historical_A_judgments": ({"historical": "A"}, "payload_sha256"),
            "authorization": ({
                "planned_calls": 240,
                "judge_model": AUDIT.JUDGE_MODEL,
                "sdk_max_retries": 0,
                "permanent_single_entry": True,
                "restart_or_resume_authorized": False,
            }, "payload_sha256"),
            "base_checkpoint": ({
                "status": "PERMANENT_SINGLE_ENTRY_STARTED",
                "planned_calls": 240,
                "restart_or_resume_authorized": False,
                "plan_sha256": AUDIT.PLAN_SHA256,
            }, "payload_sha256"),
            "judge_plan": ({
                "plan_sha256": AUDIT.PLAN_SHA256,
                "planned_calls": 240,
                "judge_model": AUDIT.JUDGE_MODEL,
                "rubric_sha256": AUDIT.RUBRIC_SHA256,
                "response_schema_sha256": AUDIT.SCHEMA_SHA256,
                "contains_question_or_response_text": False,
            }, "payload_sha256"),
            "prejudge_gate": ({"gate": "awaiting"}, "payload_sha256"),
            "prejudge_summary": ({"summary": "sealed"}, "payload_sha256"),
            "benefit_gate": ({"gate": "passed"}, "payload_sha256"),
            "ordinary_quorum_m4_q3": ({"method": "quorum"}, "payload_sha256"),
            "ordinary_min_m4_q4": ({"method": "min"}, "payload_sha256"),
            "delta_min_m4_q4": ({"method": "delta"}, "payload_sha256"),
        }
        relatives = {
            "protocol_manifest": "protocol/manifest.json",
            "historical_A_judgments": "protocol/historical/A_judgments.json",
            "stop": "control/STOPPED_external_judge",
            "lock_owner": "control/FINALIZER_LOCK/owner",
            "authorization": "control/EXTERNAL_JUDGE_AUTHORIZATION.json",
            "base_checkpoint": "evaluation/medical/judge_checkpoint.json",
            "judge_plan": "evaluation/medical/judge_plan.json",
            "prejudge_gate": "evaluation/medical/prejudge/AWAITING_EXTERNAL_JUDGE",
            "prejudge_summary": "evaluation/medical/prejudge/summary.json",
            "benefit_gate": "evaluation/benefit/gate/EXPLORATORY_BENEFIT_PASSED",
            "ordinary_quorum_m4_q3": "generation/medical/ordinary_quorum_m4_q3/medical/generation.json",
            "ordinary_min_m4_q4": "generation/medical/ordinary_min_m4_q4/medical/generation.json",
            "delta_min_m4_q4": "generation/medical/delta_min_m4_q4/medical/generation.json",
        }
        for name, (body, seal_field) in bodies.items():
            self._write_json(root / relatives[name], body, seal_field)
        stop = root / relatives["stop"]
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.write_text(
            "workflow_id=massive_medical_union_composition_exploratory_sequential_confirmation_v1\n"
            "stage=external_judge\nexit_code=1\nretry_authorized=false\n"
            "process_restart_authorized=false\n",
            encoding="utf-8",
        )
        owner = root / relatives["lock_owner"]
        owner.parent.mkdir(parents=True, exist_ok=True)
        owner.write_text("sealed owner\n", encoding="utf-8")
        source_files = {}
        for name, relative in relatives.items():
            path = root / relative
            source_files[name] = (
                relative, path.stat().st_size, AUDIT.sha256_file(path),
            )
        return source_files

    def test_budget_and_split_are_exact(self):
        self.assertEqual(AUDIT.CANARY_CAP_USD + AUDIT.CONTINUATION_CAP_USD, 0.75)
        self.assertEqual(AUDIT.VERIFIED_PROGRAM_ACTUAL_USD + 0.75 + 0.75, 4.415186)
        body = AUDIT.manifest_body({"commit": "c"}, self._minimal_source())
        contract = body["scientific_contract"]
        self.assertEqual(contract["canary_calls"], 1)
        self.assertEqual(contract["canary_start_index"], 0)
        self.assertEqual(contract["continuation_calls"], 239)
        self.assertEqual(contract["continuation_start_index"], 1)
        self.assertEqual(contract["continuation_end_index_exclusive"], 240)
        self.assertFalse(contract["model_fallback_authorized"])
        self.assertFalse(body["external_api_authorized"])
        self.assertFalse(body["gpu_authorized"])

    @staticmethod
    def _minimal_source():
        names = {
            "stop", "lock_owner", "authorization", "base_checkpoint",
            "protocol_manifest", "judge_plan", "historical_A_judgments",
            "prejudge_gate", "prejudge_summary", "benefit_gate",
            "ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4",
        }
        return {name: {"path": name, "file_sha256": name} for name in names}

    def test_source_audit_accepts_only_zero_completed_judgments(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source_files = self._fake_source(source)
            with mock.patch.object(AUDIT, "SOURCE_OUTPUT", source), mock.patch.object(
                AUDIT, "SOURCE_FILES", source_files
            ):
                observed = AUDIT.audit_source()
                self.assertEqual(observed["judge_plan"]["file_sha256"], source_files["judge_plan"][2])
                checkpoint = source / "evaluation/medical/judge_checkpoint.json.001"
                checkpoint.write_text("unexpected\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "completed judgment"):
                    AUDIT.audit_source()

    def test_cpu_prepare_and_stage_round_trip_has_no_external_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            control = output / "control"
            medical = output / "evaluation/medical"
            final = output / "evaluation/final"
            logs = output / "logs"
            manifest = control / "JUDGE_RECOVERY_V4_MANIFEST.json"
            prep = control / "PREP.json"
            preflight = control / "CPU_PREFLIGHT.json"
            staged = control / "STAGED"
            repo = {
                "path": "/repo", "branch": AUDIT.BRANCH, "commit": "c",
                "tree": "t", "source_commit": AUDIT.SOURCE_COMMIT,
                "source_commit_is_ancestor": True,
            }
            source = self._minimal_source()
            patches = (
                mock.patch.object(AUDIT, "RECOVERY_OUTPUT", output),
                mock.patch.object(AUDIT, "CONTROL_ROOT", control),
                mock.patch.object(AUDIT, "MEDICAL_ROOT", medical),
                mock.patch.object(AUDIT, "FINAL_ROOT", final),
                mock.patch.object(AUDIT, "LOG_ROOT", logs),
                mock.patch.object(AUDIT, "MANIFEST_FILE", manifest),
                mock.patch.object(AUDIT, "PREP_FILE", prep),
                mock.patch.object(AUDIT, "PREFLIGHT_FILE", preflight),
                mock.patch.object(AUDIT, "STAGED_FILE", staged),
                mock.patch.object(AUDIT, "audit_repo", return_value=repo),
                mock.patch.object(AUDIT, "audit_source", return_value=source),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10]:
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("OPENAI_API_KEY", None)
                    self.assertEqual(AUDIT.prepare_command(None), 0)
                    args = argparse.Namespace(validation_command=["mock transport passed"])
                    self.assertEqual(AUDIT.seal_staged_command(args), 0)
                    AUDIT.audit_staged()
                staged_value = json.loads(staged.read_text(encoding="utf-8"))
                self.assertFalse(staged_value["external_api_authorized"])
                self.assertEqual(staged_value["external_api_calls"], 0)
                self.assertFalse(staged_value["gpu_authorized"])
                self.assertEqual(list(medical.iterdir()), [])
                self.assertEqual(list(logs.iterdir()), [])

    def test_prepare_rejects_loaded_api_key_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fresh"
            with mock.patch.object(AUDIT, "RECOVERY_OUTPUT", output), mock.patch.dict(
                os.environ, {"OPENAI_API_KEY": "secret"}
            ):
                with self.assertRaisesRegex(ValueError, "must be absent"):
                    AUDIT.prepare_command(None)
                self.assertFalse(output.exists())

    def test_stage_is_cpu_only_and_external_wrappers_are_logged_and_split(self):
        stage = (ROOT / (
            "scripts/stage_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v4_tillicum.sh"
        )).read_text(encoding="utf-8")
        finalizer = (ROOT / (
            "scripts/finalize_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v4_tillicum.sh"
        )).read_text(encoding="utf-8")
        status = (ROOT / (
            "scripts/status_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v4_tillicum.sh"
        )).read_text(encoding="utf-8")
        derive = (ROOT / (
            "scripts/derive_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v4_tillicum.sh"
        )).read_text(encoding="utf-8")
        for forbidden in ("sbatch ", "scontrol ", "gres=gpu", "OPENAI_API_KEY="):
            self.assertNotIn(forbidden, stage)
        self.assertIn("OPENAI_API_KEY must be absent", stage)
        self.assertIn("validate-sdk-serialization", stage)
        self.assertIn("real SDK serialization through local MockTransport", stage)
        self.assertIn("validate-static --recovery-manifest", stage)
        self.assertIn("seal-staged", stage)
        self.assertIn("0.003072", finalizer)
        self.assertIn("0.746928", finalizer)
        self.assertIn('stage=canary; command=canary; calls=1', finalizer)
        self.assertIn('stage=continuation; command=continue; calls=239', finalizer)
        self.assertIn('> "$log" 2>&1', finalizer)
        self.assertIn('chmod 0400 "$log"', finalizer)
        self.assertIn('unset OPENAI_API_KEY', finalizer)
        self.assertIn("write-wrapper-failure", finalizer)
        self.assertIn("audit-failure", finalizer)
        self.assertIn("trap 'handle_signal 129' HUP", finalizer)
        self.assertIn("trap 'handle_signal 130' INT", finalizer)
        self.assertIn("trap 'handle_signal 143' TERM", finalizer)
        self.assertIn('judge_pid=$!', finalizer)
        self.assertIn('kill -TERM "$judge_pid"', finalizer)
        self.assertIn('wait "$judge_pid"', finalizer)
        self.assertIn('--owner-token "$owner_token"', finalizer)
        self.assertLess(
            finalizer.index("preflight-authorization"),
            finalizer.index("acquire-lock"),
        )
        self.assertIn('audit-canary', finalizer)
        self.assertIn('audit-continuation', finalizer)
        self.assertIn('"$derive"', finalizer)
        self.assertNotIn("tmux", finalizer)
        self.assertIn('"$summary" merge', derive)
        self.assertIn('"$summary" audit-final', derive)
        self.assertIn("OPENAI_API_KEY must be absent", derive)
        for forbidden in ("sbatch ", "scontrol ", "gres=gpu", "OPENAI_API_KEY="):
            self.assertNotIn(forbidden, derive)
        self.assertIn("CANARY_TERMINAL_FAILURE_NO_RESTART", status)
        self.assertIn("CONTINUATION_TERMINAL_FAILURE_NO_RESTART", status)
        self.assertIn('audit-failure --stage canary', status)
        self.assertIn('audit-failure --stage continuation', status)

    def test_lock_command_cannot_write_when_preflight_rejects_acknowledgments(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner = Path(temporary) / "CANARY_LOCK/owner"
            args = argparse.Namespace(stage="canary")
            with mock.patch.object(AUDIT, "lock_path", return_value=owner), mock.patch.object(
                AUDIT, "authorization_preflight",
                side_effect=ValueError("acknowledgment differs"),
            ):
                with self.assertRaisesRegex(ValueError, "acknowledgment differs"):
                    AUDIT.acquire_lock_command(args)
            self.assertFalse(owner.parent.exists())

    def test_control_cli_exposes_preflight_and_terminal_failure_audit(self):
        choices = AUDIT.build_parser()._subparsers._group_actions[0].choices
        self.assertIn("audit-lock", choices)
        self.assertIn("preflight-authorization", choices)
        self.assertIn("write-wrapper-failure", choices)
        self.assertIn("audit-failure", choices)

    def test_lock_owner_token_prevents_losing_wrapper_from_claiming_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control"
            control.mkdir()
            manifest_path = control / "JUDGE_RECOVERY_V4_MANIFEST.json"
            manifest = AUDIT.sealed({"recovery_repo": {"commit": "recovery-commit"}})
            AUDIT.atomic_json(manifest_path, manifest, mode=0o400)
            owner_token = "a" * 64
            with mock.patch.object(AUDIT, "CONTROL_ROOT", control), mock.patch.object(
                AUDIT, "MANIFEST_FILE", manifest_path
            ):
                owner = AUDIT.lock_path("canary")
                AUDIT.atomic_json(
                    owner,
                    AUDIT.sealed(AUDIT.lock_body(
                        "canary", manifest,
                        AUDIT.sha256_bytes(owner_token.encode()),
                    )),
                    mode=0o400,
                )
                AUDIT.audit_lock("canary", manifest, owner_token)
                with self.assertRaisesRegex(ValueError, "another invocation"):
                    AUDIT.audit_lock("canary", manifest, "b" * 64)

    def test_failure_timestamp_parser_rejects_payload_channels(self):
        self.assertTrue(AUDIT.valid_utc_timestamp("2026-08-26T12:34:56.123456+00:00"))
        self.assertFalse(AUDIT.valid_utc_timestamp("2026-08-26T12:34:56"))
        self.assertFalse(AUDIT.valid_utc_timestamp("QUESTION: private prompt"))


if __name__ == "__main__":
    unittest.main()
