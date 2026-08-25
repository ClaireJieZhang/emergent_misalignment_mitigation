"""Tests for the CPU-only exploratory smoke gate recovery v5."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
import contextlib
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = (
    ROOT
    / "scripts/audit_massive_medical_union_composition_exploratory_"
    "smoke_gate_recovery_v5.py"
)
STAGE_PATH = (
    ROOT
    / "scripts/stage_massive_medical_union_composition_exploratory_"
    "smoke_gate_recovery_v5_tillicum.sh"
)
STATUS_PATH = (
    ROOT
    / "scripts/status_massive_medical_union_composition_exploratory_"
    "smoke_gate_recovery_v5_tillicum.sh"
)
DOC_PATH = (
    ROOT
    / "docs/massive_medical_union_composition_exploratory_"
    "smoke_gate_recovery_v5.md"
)

SPEC = importlib.util.spec_from_file_location("gate_recovery_v5_auditor", AUDITOR_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write_json(path, payload, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, mode)


class GateRecoveryV5Tests(unittest.TestCase):
    def test_exact_scope_hashes_and_modes(self):
        self.assertEqual(len(AUDIT.ADDED_FILES), 5)
        self.assertEqual(len(AUDIT.INHERITED_FROZEN_FILES), 3)
        self.assertEqual(
            AUDIT.FROZEN_INHERITED_FILE_SHA256[AUDIT.FIXED_EVALUATOR_PATH],
            "30091d091584a9f41c308160330d0f2dacfdae34f1f26c0dcbe8aa97cba9efc2",
        )
        self.assertEqual(
            AUDIT.FROZEN_INHERITED_FILE_SHA256[
                AUDIT.INHERITED_FROZEN_FILES[1]
            ],
            "e3432b6859126a177421141da3f9bd0a150a64d7ebf48aeec6a5396ffcd35552",
        )
        for relative in AUDIT.EXECUTABLE_FILES:
            self.assertEqual(stat.S_IMODE((ROOT / relative).stat().st_mode), 0o755)
        for relative in AUDIT.REGULAR_FILES:
            self.assertEqual(stat.S_IMODE((ROOT / relative).stat().st_mode), 0o644)

    def test_fixed_lineage_incident_and_inventory_constants(self):
        self.assertEqual(AUDIT.DIRECT_PARENT_COMMIT, AUDIT.FAILED_V4_COMMIT)
        self.assertEqual(
            AUDIT.DIRECT_PARENT_COMMIT,
            "f2c9c53cd63465f2a773eb13ed64f09cc5126b89",
        )
        self.assertEqual(AUDIT.FAILED_V4_PREP_SIZE, 119649)
        self.assertEqual(
            AUDIT.FAILED_V4_PREP_SHA256,
            "b4242435b086e632f8d05d02fde9300c883889fb584c34af734a8d0ebd394317",
        )
        self.assertEqual(
            AUDIT.FAILED_V4_PREP_PAYLOAD_SHA256,
            "013b281dd2f5ac4c33224921ba352c6d50761333ad3a603b38b5a3209e38e92b",
        )
        self.assertEqual(AUDIT.SOURCE_JOB_ID, "262130")
        self.assertEqual(AUDIT.SOURCE_ACTUAL_SECONDS, 609)
        self.assertEqual(AUDIT.SOURCE_ACTUAL_H200_MINUTES, 10.15)
        self.assertEqual(AUDIT.SOURCE_ACTUAL_GPU_COST_USD, 0.15225)
        self.assertEqual(
            AUDIT.SOURCE_SACCT_ROW_SHA256,
            hashlib.sha256(AUDIT.SOURCE_SACCT_ROW.encode()).hexdigest(),
        )
        self.assertEqual(AUDIT.SOURCE_STOP_SHA256, AUDIT.SOURCE_CRITICAL_FILES[
            "control/STOPPED_independent_model_recovery"
        ][1])
        self.assertEqual(AUDIT.PROBE_V3_STATIC_CONTRACT_SHA256, "b15890642418ad34f1ade97b3433ea5432ad53221a8e0b544fee29942c2cbc1d")

    def test_tree_inventory_hashes_names_modes_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tree"
            (root / "nested").mkdir(parents=True)
            os.chmod(root / "nested", 0o700)
            (root / "nested/file.json").write_text("{}\n", encoding="utf-8")
            os.chmod(root / "nested/file.json", 0o600)
            entries = [
                {"path": "nested", "mode": 0o700, "type": "dir"},
                {
                    "path": "nested/file.json",
                    "mode": 0o600,
                    "type": "file",
                    "size_bytes": 3,
                    "file_sha256": hashlib.sha256(b"{}\n").hexdigest(),
                },
            ]
            digest = AUDIT.sha256_bytes(AUDIT.canonical_bytes(entries))
            observed = AUDIT.tree_inventory(root, 1, 1, digest, "fixture")
            self.assertEqual(observed["entries"], entries)
            (root / "nested/file.json").write_text("{ }\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                AUDIT.tree_inventory(root, 1, 1, digest, "fixture")
            (root / "nested/file.json").unlink()
            (root / "nested/file.json").symlink_to("missing")
            with self.assertRaisesRegex(ValueError, "symlink"):
                AUDIT.tree_inventory(root, 1, 1, digest, "fixture")

    def test_embedded_sample_and_payload_seals_fail_closed(self):
        sample_body = {
            "question_id": "q",
            "sample_index": 0,
            "response": "answer",
            "response_sha256": hashlib.sha256(b"answer").hexdigest(),
            "finish_reason": "stop",
            "generated_tokens": 1,
        }
        sample = dict(sample_body)
        sample["sample_sha256"] = AUDIT.sha256_bytes(AUDIT.canonical_bytes(sample_body))
        self.assertEqual(AUDIT.audit_sample_seal(sample, "sample"), sample_body)
        sample["response"] = "changed"
        with self.assertRaisesRegex(ValueError, "sample seal differs"):
            AUDIT.audit_sample_seal(sample, "sample")
        payload = AUDIT.seal({"value": 1})
        self.assertEqual(AUDIT.verify_seal(payload, "payload"), {"value": 1})
        payload["value"] = 2
        with self.assertRaisesRegex(ValueError, "differs"):
            AUDIT.verify_seal(payload, "payload")

    def _write_gate_fixture(self, root):
        evaluation = root / "evaluation"
        gate = evaluation / "smoke/gate"
        gate.mkdir(parents=True)
        os.chmod(evaluation, 0o700)
        os.chmod(evaluation / "smoke", 0o700)
        os.chmod(gate, 0o700)
        projected = 1.20 * (
            66.09861348790582
            + 10 * 501.2222172736656
            + 38796 / 8.945817873199246
            + 60
        )
        projection = {
            "schema_version": 1,
            "protocol": "massive_medical_union_composition_exploratory_runtime_projection_v1",
            "protocol_id": AUDIT.SOURCE_PROTOCOL_ID,
            "protocol_manifest_file_sha256": AUDIT.SOURCE_MANIFEST_FILE_SHA256,
            "protocol_manifest_payload_sha256": AUDIT.SOURCE_MANIFEST_PAYLOAD_SHA256,
            "prompt_file_sha256": "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57",
            "formula": "fixture frozen formula",
            "medical_planning_envelope": {"fixture": True},
            "medical_selected_tokens_per_method_bound": 12932,
            "setup_seconds": 66.09861348790582,
            "four_stream_smoke_generation_seconds": 501.2222172736656,
            "minimum_method_selected_tokens_per_second": 8.945817873199246,
            "medical_all_three_methods_selected_tokens_bound": 38796,
            "timings": {"fixture": True},
            "smoke_score_and_gate_seconds_observed_before_summary_seal": 0.1,
            "scoring_floor_seconds": 60,
            "projected_confirmation_seconds": projected,
            "projected_confirmation_h200_minutes": projected / 60,
            "contingency_fraction": 0.20,
            "cache_equivalence_probe": {
                "protocol": "massive_medical_union_composition_cache_equivalence_probe_v3",
                "result": "PASS",
            },
        }
        projection_payload = AUDIT.seal(projection)
        projection_path = gate / "runtime_projection.json"
        write_json(projection_path, projection_payload)
        checks = {
            f"{method}.{name}": True
            for method in AUDIT.METHOD_IDS
            for name in (
                "structured_valid_fraction",
                "truncations",
                "joint_intent_gain_over_paired_base",
            )
        }
        checks["runtime_projection_fits_released_confirmation_budget"] = False
        comparison = {
            "paired_joint_delta": 0.13333333333333333,
            "paired_joint_bootstrap_95ci": [0.03333333333333333, 0.23333333333333334],
            "joint_one_sided_exact_mcnemar_p": 0.0107421875,
        }
        summary = {
            "schema_version": 1,
            "protocol": "massive_medical_union_composition_exploratory_smoke_gate_v1",
            "protocol_id": AUDIT.SOURCE_PROTOCOL_ID,
            "protocol_manifest_file_sha256": AUDIT.SOURCE_MANIFEST_FILE_SHA256,
            "protocol_manifest_payload_sha256": AUDIT.SOURCE_MANIFEST_PAYLOAD_SHA256,
            "thresholds": {"fixture": True},
            "status": "STOPPED_EXPLORATORY_SMOKE",
            "confirmation_submission_eligible": False,
            "all_three_methods_passed": False,
            "confirmatory_claim": False,
            "wave2_v1_status": "STOP",
            "wave3_v1_eligible": False,
            "wave3_v1_submitted_or_released": False,
            "checks": checks,
            "results": {
                method: {
                    "comparison": comparison,
                    "checks": {
                        "structured_valid_fraction": True,
                        "truncations": True,
                        "joint_intent_gain_over_paired_base": True,
                    },
                }
                for method in AUDIT.METHOD_IDS
            },
            "runtime_projection": {
                "path": str(projection_path.resolve()),
                "file_sha256": AUDIT.sha256_file(projection_path),
                "payload_sha256": projection_payload["payload_sha256"],
                **projection,
            },
        }
        summary_payload = AUDIT.seal(summary)
        summary_path = gate / "summary.json"
        write_json(summary_path, summary_payload)
        sentinel = {
            "schema_version": 1,
            "protocol": "massive_medical_union_composition_exploratory_smoke_sentinel_v1",
            "protocol_id": AUDIT.SOURCE_PROTOCOL_ID,
            "status": "STOPPED_EXPLORATORY_SMOKE",
            "summary_path": str(summary_path.resolve()),
            "summary_file_sha256": AUDIT.sha256_file(summary_path),
            "summary_payload_sha256": summary_payload["payload_sha256"],
            "confirmatory_claim": False,
            "wave2_v1_status": "STOP",
            "wave3_v1_eligible": False,
            "wave3_v1_submitted_or_released": False,
        }
        write_json(gate / "STOPPED_EXPLORATORY_SMOKE", AUDIT.seal(sentinel))
        return evaluation, gate

    def test_recovered_gate_requires_exact_stop_and_arithmetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, gate = self._write_gate_fixture(root)
            with mock.patch.object(AUDIT, "EVALUATION_ROOT", evaluation), mock.patch.object(
                AUDIT, "GATE_ROOT", gate
            ):
                result = AUDIT.audit_gate()
                self.assertEqual(result["status"], "STOPPED_EXPLORATORY_SMOKE")
                self.assertTrue(result["scientific_smoke_method_gates_passed"])
                self.assertFalse(result["runtime_projection_gate_passed"])
                self.assertGreater(result["projected_confirmation_h200_minutes"], 100)
                summary_path = gate / "summary.json"
                payload = json.loads(summary_path.read_text())
                payload["checks"]["runtime_projection_fits_released_confirmation_budget"] = True
                write_json(summary_path, payload)
                with self.assertRaises(ValueError):
                    AUDIT.audit_gate()

    def test_gate_rejects_resealed_schema_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, gate = self._write_gate_fixture(root)
            projection_path = gate / "runtime_projection.json"
            projection_payload = json.loads(projection_path.read_text())
            projection = AUDIT.verify_seal(projection_payload, "projection")
            projection["unexpected_relaxation"] = True
            write_json(projection_path, AUDIT.seal(projection))
            summary_path = gate / "summary.json"
            summary_payload = json.loads(summary_path.read_text())
            summary = AUDIT.verify_seal(summary_payload, "summary")
            summary["runtime_projection"]["unexpected_relaxation"] = True
            summary["runtime_projection"]["file_sha256"] = AUDIT.sha256_file(projection_path)
            summary["runtime_projection"]["payload_sha256"] = json.loads(
                projection_path.read_text()
            )["payload_sha256"]
            write_json(summary_path, AUDIT.seal(summary))
            sentinel_path = gate / "STOPPED_EXPLORATORY_SMOKE"
            sentinel_payload = json.loads(sentinel_path.read_text())
            sentinel = AUDIT.verify_seal(sentinel_payload, "sentinel")
            sentinel["summary_file_sha256"] = AUDIT.sha256_file(summary_path)
            sentinel["summary_payload_sha256"] = json.loads(summary_path.read_text())[
                "payload_sha256"
            ]
            write_json(sentinel_path, AUDIT.seal(sentinel))
            with mock.patch.object(AUDIT, "EVALUATION_ROOT", evaluation), mock.patch.object(
                AUDIT, "GATE_ROOT", gate
            ):
                with self.assertRaisesRegex(ValueError, "schema differs"):
                    AUDIT.audit_gate()

    def test_forbidden_preloaded_model_module_fails_closed(self):
        with mock.patch.object(AUDIT, "REPO_ROOT", ROOT), mock.patch.dict(
            sys.modules, {"torch": object()}
        ):
            with self.assertRaisesRegex(ValueError, "preloaded"):
                AUDIT.load_fixed_evaluator()

    def test_top_level_output_modes_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            control = output / "control"
            logs = root / "logs"
            control.mkdir(parents=True)
            logs.mkdir()
            os.chmod(output, 0o700)
            os.chmod(control, 0o700)
            prep = control / "PREP.json"
            prep.write_text("{}\n", encoding="utf-8")
            os.chmod(prep, 0o400)
            with mock.patch.object(AUDIT, "OUTPUT_ROOT", output), mock.patch.object(
                AUDIT, "CONTROL_ROOT", control
            ), mock.patch.object(AUDIT, "GENERATION_ROOT", output / "generation"), mock.patch.object(
                AUDIT, "LOG_ROOT", logs
            ):
                self.assertTrue(AUDIT.exact_output_phase("prep"))
                os.chmod(output, 0o755)
                with self.assertRaisesRegex(ValueError, "output root mode"):
                    AUDIT.exact_output_phase("prep")

    def test_setgid_inheritance_is_explicitly_normalized_only_for_new_dirs(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "gpfs-like-parent"
            parent.mkdir()
            os.chmod(parent, 0o2700)
            parent_mode_before = stat.S_IMODE(parent.stat().st_mode)
            self.assertIn(parent_mode_before, {0o700, 0o2700})
            inherited = parent / "inherited"
            inherited.mkdir(mode=0o700)
            # Force the exact GPFS state observed in the failed v4 stage even
            # on local filesystems that do not propagate setgid on mkdir.
            os.chmod(inherited, 0o2700)
            self.assertIn(
                stat.S_IMODE(inherited.stat().st_mode),
                {0o700, 0o2700},
                "Linux/GPFS preserves setgid; some local filesystems clear it",
            )
            with mock.patch.object(AUDIT, "OUTPUT_ROOT", inherited):
                AUDIT.normalize_known_private_directory(
                    inherited, "fixture inherited dir"
                )
            self.assertEqual(stat.S_IMODE(inherited.stat().st_mode), 0o700)

            created = parent / "fresh-v5"
            with mock.patch.object(AUDIT, "OUTPUT_ROOT", created):
                AUDIT.create_known_private_directory(created, "fixture fresh dir")
            self.assertEqual(stat.S_IMODE(created.stat().st_mode), 0o700)

            unsafe = parent / "unsafe"
            unsafe.symlink_to(created, target_is_directory=True)
            with mock.patch.object(AUDIT, "OUTPUT_ROOT", unsafe):
                with self.assertRaisesRegex(ValueError, "safely"):
                    AUDIT.normalize_known_private_directory(unsafe, "fixture symlink")

            outside = root = Path(temporary) / "outside"
            outside.mkdir()
            outside_marker = outside / "immutable"
            outside_marker.write_bytes(b"source-like-bytes")
            outside_before = (
                outside.stat().st_dev,
                outside.stat().st_ino,
                stat.S_IMODE(outside.stat().st_mode),
                outside_marker.read_bytes(),
            )
            with self.assertRaisesRegex(ValueError, "registry"):
                AUDIT.normalize_known_private_directory(
                    outside, "fixture unregistered source-like dir"
                )
            self.assertEqual(
                outside_before,
                (
                    outside.stat().st_dev,
                    outside.stat().st_ino,
                    stat.S_IMODE(outside.stat().st_mode),
                    outside_marker.read_bytes(),
                ),
            )
            self.assertEqual(
                stat.S_IMODE(parent.stat().st_mode), parent_mode_before
            )

    def test_failed_v4_prep_only_incident_is_bound_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "failed-v4-repo"
            output = root / "failed-v4-output"
            control = output / "control"
            logs = root / "logs"
            repo.mkdir()
            control.mkdir(parents=True)
            logs.mkdir()
            os.chmod(repo, 0o2700)
            os.chmod(output, 0o2700)
            os.chmod(control, 0o2700)
            source_incident = {
                "fixed_evaluator": {"path": "v5-evaluator-path"},
                "control_embedded_seals": {"base_model_snapshot": {"sealed": True}}
            }
            failed_v4_source_incident = json.loads(json.dumps(source_incident))
            failed_v4_source_incident["fixed_evaluator"]["path"] = str(
                repo / AUDIT.FIXED_EVALUATOR_PATH
            )
            repository_files = {
                path: {"git_mode": mode, "size_bytes": size, "sha256": digest}
                for path, (mode, size, digest) in AUDIT.FAILED_V4_REPOSITORY_FILES.items()
            }
            prep_body = {
                "schema_version": 1,
                "workflow_id": "massive_medical_union_composition_exploratory_smoke_gate_recovery_v4",
                "protocol_id": "massive_medical_union_composition_exploratory_smoke_gate_recovery_v4",
                "repository": {
                    "path": str(repo),
                    "branch": AUDIT.FAILED_V4_BRANCH,
                    "commit": AUDIT.FAILED_V4_COMMIT,
                    "direct_parent_commit": AUDIT.SOURCE_COMMIT,
                    "direct_parent_tree": AUDIT.SOURCE_TREE,
                    "direct_nonmerge_parent": True,
                    "modified_files": list(AUDIT.INHERITED_FROZEN_FILES),
                    "added_files": [
                        "docs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.md",
                        "scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.py",
                        "scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh",
                        "scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh",
                        "tests/test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_workflow.py",
                    ],
                    "files": repository_files,
                },
                "source_incident": failed_v4_source_incident,
                "source_base_snapshot_binding": {"sealed": True},
                "source_job_actual_gpu_cost_usd": AUDIT.SOURCE_ACTUAL_GPU_COST_USD,
                "new_gpu_h200_minutes_authorized": 0,
                "new_gpu_cost_authorized_usd": 0,
                "external_api_cost_authorized_usd": 0,
                "slurm_jobs_submitted": 0,
                "model_loaded": False,
                "generation_performed": False,
                "scores_recomputed": False,
                "source_scores_reused_byte_identically": True,
                "training": False,
                "external_api_calls": 0,
                "confirmation_authorized": False,
                "confirmation_submitted": False,
                "automatic_continuation": False,
                "confirmatory_claim": False,
            }
            prep_path = control / "PREP.json"
            prep_payload = AUDIT.seal(prep_body)
            write_json(prep_path, prep_payload, 0o400)

            def fake_git(_root, *args):
                values = {
                    ("rev-parse", "HEAD"): AUDIT.FAILED_V4_COMMIT,
                    ("rev-parse", "HEAD^{tree}"): AUDIT.FAILED_V4_TREE,
                    ("branch", "--show-current"): AUDIT.FAILED_V4_BRANCH,
                    ("status", "--porcelain"): "",
                }
                return values[args]

            real_imode = stat.S_IMODE
            patches = (
                mock.patch.object(AUDIT, "FAILED_V4_REPO", repo),
                mock.patch.object(AUDIT, "FAILED_V4_OUTPUT", output),
                mock.patch.object(AUDIT, "FAILED_V4_CONTROL", control),
                mock.patch.object(AUDIT, "FAILED_V4_PREP", prep_path),
                mock.patch.object(AUDIT, "LOG_ROOT", logs),
                mock.patch.object(AUDIT, "FAILED_V4_PREP_SIZE", prep_path.stat().st_size),
                mock.patch.object(AUDIT, "FAILED_V4_PREP_SHA256", AUDIT.sha256_file(prep_path)),
                mock.patch.object(
                    AUDIT,
                    "FAILED_V4_PREP_PAYLOAD_SHA256",
                    prep_payload["payload_sha256"],
                ),
                mock.patch.object(AUDIT, "git", side_effect=fake_git),
                mock.patch.object(
                    AUDIT.stat,
                    "S_IMODE",
                    side_effect=lambda mode: (
                        0o2700 if stat.S_ISDIR(mode) else real_imode(mode)
                    ),
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
                incident = AUDIT.audit_failed_v4_stage(source_incident)
                self.assertEqual(incident["output_root"]["mode"], 0o2700)
                self.assertFalse(incident["staged_present"])
                (control / "STAGED").write_text("unexpected", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    AUDIT.audit_failed_v4_stage(source_incident)

    def test_recover_gate_seals_zero_cost_and_reaudits_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            control = output / "control"
            control.mkdir(parents=True)
            failed_v4 = {"prep": "immutable"}
            prep_body = {
                "repository": {"commit": "c"},
                "source_incident": {"seal": "same"},
                "failed_v4_stage": failed_v4,
            }
            prep_payload = AUDIT.seal(prep_body)
            write_json(control / "PREP.json", prep_payload, 0o400)
            (control / "STAGED").write_bytes(b"staged")
            os.chmod(control / "STAGED", 0o400)
            gate_result = {
                "status": "STOPPED_EXPLORATORY_SMOKE",
                "projected_confirmation_h200_minutes": 189.50191727372095,
            }
            evaluator_binding = {
                "file_sha256": AUDIT.FROZEN_INHERITED_FILE_SHA256[
                    AUDIT.FIXED_EVALUATOR_PATH
                ]
            }
            patches = (
                mock.patch.object(AUDIT, "OUTPUT_ROOT", output),
                mock.patch.object(AUDIT, "CONTROL_ROOT", control),
                mock.patch.object(AUDIT, "PREP_FILE", control / "PREP.json"),
                mock.patch.object(AUDIT, "STAGED_FILE", control / "STAGED"),
                mock.patch.object(AUDIT, "RESULT_FILE", control / "SMOKE_GATE_RECOVERY_RESULT.json"),
                mock.patch.object(AUDIT, "EVALUATION_ROOT", output / "evaluation"),
                mock.patch.object(AUDIT, "GATE_ROOT", output / "evaluation/smoke/gate"),
                mock.patch.object(AUDIT, "GENERATION_ROOT", output / "generation"),
                mock.patch.object(
                    AUDIT,
                    "audit_staged",
                    return_value={
                        "source_incident": {"seal": "same"},
                        "failed_v4_stage": failed_v4,
                    },
                ),
                mock.patch.object(AUDIT, "audit_repository", return_value={"commit": "c"}),
                mock.patch.object(AUDIT, "audit_cpu_only_environment", return_value={}),
                mock.patch.object(AUDIT, "audit_source_incident", side_effect=[{"seal": "same"}, {"seal": "same"}]),
                mock.patch.object(
                    AUDIT,
                    "audit_failed_v4_stage",
                    side_effect=[failed_v4, failed_v4],
                ),
                mock.patch.object(AUDIT, "load_fixed_evaluator", return_value=(object(), evaluator_binding)),
                mock.patch.object(AUDIT, "run_fixed_evaluator", return_value=2),
                mock.patch.object(AUDIT, "audit_gate", return_value=gate_result),
                mock.patch.object(AUDIT, "exact_output_phase", return_value=True),
            )
            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                AUDIT.command_recover_gate(None)
            result = json.loads((control / "SMOKE_GATE_RECOVERY_RESULT.json").read_text())
            body = AUDIT.verify_seal(result, "result")
            self.assertEqual(body["new_gpu_h200_minutes"], 0)
            self.assertEqual(body["external_api_calls"], 0)
            self.assertFalse(body["confirmation_authorized"])
            self.assertFalse(body["confirmation_submitted"])

    def test_source_drift_prevents_result_seal(self):
        source = AUDITOR_PATH.read_text(encoding="utf-8")
        self.assertIn("source_after = audit_source_incident()", source)
        self.assertIn("if source_after != source_before", source)
        self.assertLess(source.index("if source_after != source_before"), source.index("write_sealed_once(RESULT_FILE"))

    def test_shells_are_cpu_only_and_have_no_downstream_route(self):
        stage = STAGE_PATH.read_text(encoding="utf-8")
        status = STATUS_PATH.read_text(encoding="utf-8")
        auditor = AUDITOR_PATH.read_text(encoding="utf-8")
        self.assertIn("recover-gate", stage)
        self.assertIn("audit-terminal", stage)
        self.assertNotIn("sbatch ", stage)
        self.assertNotIn("srun ", stage)
        self.assertNotIn("scontrol ", stage)
        self.assertNotIn("python scripts/sample_massive_medical_union", stage)
        self.assertNotIn('python "$sampler"', stage)
        self.assertNotIn("python scripts/judge_massive_medical_union", stage)
        self.assertNotIn('python "$judge"', stage)
        self.assertNotIn("confirmation-prejudge", stage)
        # macOS Bash 3.2 treats an empty-array expansion as unset under
        # `set -u`; this local preflight must work before it reaches SSH.
        self.assertNotIn("modified_files=()", stage)
        self.assertIn('all_files=("${added_files[@]}")', stage)
        self.assertIn("unset OPENAI_API_KEY", stage)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", status)
        self.assertIn('python "$auditor" status', status)
        for call in (
            'create_known_private_directory(OUTPUT_ROOT, "new v5 output root")',
            'create_known_private_directory(CONTROL_ROOT, "new v5 control root")',
            'create_known_private_directory(EVALUATION_ROOT, "new v5 evaluation root")',
            'create_known_private_directory(EVALUATION_ROOT / "smoke", "new v5 smoke root")',
            'create_known_private_directory(GATE_ROOT, "new v5 smoke gate root")',
        ):
            self.assertIn(call, auditor)
        self.assertIn("failed_v4_before = audit_failed_v4_stage(source_before)", auditor)
        self.assertIn("failed_v4_after = audit_failed_v4_stage(source_after)", auditor)
        self.assertIn(
            "tests.test_massive_medical_union_composition_exploratory_independent_model_recovery_v3_workflow",
            stage,
        )
        for historical in (
            "scripts/sbatch_massive_medical_union_composition_exploratory_v1_smoke_tillicum_h200.sbatch",
            "scripts/status_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh",
            "scripts/status_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_tillicum.sh",
            "scripts/status_massive_medical_union_composition_exploratory_independent_model_recovery_v3_tillicum.sh",
        ):
            self.assertIn(historical, stage)

    def test_docs_distinguish_science_pass_from_release_authority(self):
        text = " ".join(DOC_PATH.read_text(encoding="utf-8").split())
        self.assertIn("All three methods pass the smoke science gates", text)
        self.assertIn("STOPPED_EXPLORATORY_SMOKE", text)
        self.assertIn("confirmation_authorized=false", text)
        self.assertIn("separate user decision", text)
        self.assertIn("189.5019172737 H200-min", text)


if __name__ == "__main__":
    unittest.main()
