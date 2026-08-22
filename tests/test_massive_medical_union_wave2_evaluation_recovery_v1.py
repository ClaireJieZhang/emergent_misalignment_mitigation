"""No-network tests for the one-shot Wave-2 evaluation recovery."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_massive_medical_union_wave2_evaluation_recovery_v1 as recovery


class Wave2EvaluationRecoveryTests(unittest.TestCase):
    def test_scope_budget_and_original_jobs_are_frozen(self):
        self.assertEqual(recovery.ORIGINAL_COMMIT, "8a96fe7c8c70f270c46d3416623ca866cb1d8fec")
        self.assertEqual(recovery.JOB_MINUTES, 15)
        self.assertEqual(recovery.MAX_H200_MINUTES, 15)
        self.assertEqual(recovery.MAX_GPU_COST_USD, 0.225)
        self.assertEqual(recovery.PREFLIGHT_MAX_SECONDS, 180)
        self.assertEqual(set(recovery.ORIGINAL_SACCT), {"251235", "251236", "251237"})
        self.assertIn("FAILED|00:20:28", recovery.ORIGINAL_SACCT["251235"])
        self.assertIn("FAILED|00:20:40", recovery.ORIGINAL_SACCT["251236"])
        self.assertIn("CANCELLED|00:00:00", recovery.ORIGINAL_SACCT["251237"])
        self.assertEqual(
            set(recovery.RECOVERY_MODIFIED_FILES),
            {"scripts/audit_massive_medical_union_wave2.py", "tests/test_massive_medical_union_wave2.py"},
        )
        self.assertIn(
            "scripts/finalize_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh",
            recovery.RECOVERY_ADDED_FILES,
        )

    def test_repository_lineage_allows_only_exact_recovery_files(self):
        commit = "f" * 40

        def fake_git(repo, *args):
            if repo == recovery.ORIGINAL_REPO:
                if args == ("rev-parse", "HEAD"):
                    return recovery.ORIGINAL_COMMIT
                if args == ("status", "--porcelain"):
                    return ""
            if repo == recovery.RECOVERY_REPO:
                if args == ("rev-parse", "HEAD"):
                    return commit
                if args == ("status", "--porcelain"):
                    return ""
                if args == ("rev-list", "--parents", "-n", "1", commit):
                    return f"{commit} {recovery.ORIGINAL_COMMIT}"
                if args == (
                    "diff", "--name-status", "--no-renames",
                    f"{recovery.ORIGINAL_COMMIT}..{commit}",
                ):
                    return "\n".join(
                        [*(f"A\t{path}" for path in recovery.RECOVERY_ADDED_FILES),
                         *(f"M\t{path}" for path in recovery.RECOVERY_MODIFIED_FILES)]
                    )
            raise AssertionError((repo, args))

        with mock.patch.object(recovery, "git", side_effect=fake_git), \
             mock.patch.object(recovery, "sha256_file", return_value="a" * 64):
            result = recovery.audit_repository()
        self.assertEqual(result["recovery_commit"], commit)

    def test_model_body_timestamp_is_excluded_from_prep_binding(self):
        first = {"created_at": "one", "model_name": "pi_B2", "seed": 8182127}
        second = {"created_at": "two", "model_name": "pi_B2", "seed": 8182127}
        self.assertEqual(
            recovery.stable_model_body_sha256(first),
            recovery.stable_model_body_sha256(second),
        )
        changed = dict(second, seed=1)
        self.assertNotEqual(
            recovery.stable_model_body_sha256(first),
            recovery.stable_model_body_sha256(changed),
        )
        self.assertEqual(len(recovery.expected_model_inventory("pi_B2")), 29)
        self.assertEqual(len(recovery.expected_model_inventory("pi_B3")), 29)
        self.assertEqual(
            recovery.MODEL_MANIFEST_STABLE_BODY_SHA256["pi_B2"],
            "a5db0ceafd04c9c158d24011b37efba432ee3544240bfe90a9b6ca475e2a8831",
        )
        self.assertEqual(
            recovery.MODEL_MANIFEST_STABLE_BODY_SHA256["pi_B3"],
            "485114def40171363148028db2cb3012e46e81d2bc7e488ca9e29cc21a07814a",
        )

    def test_recovery_body_uses_mapping_only_corrected_original_order(self):
        for model_name in ("pi_B2", "pi_B3"):
            inventory = recovery.expected_model_inventory(model_name)
            paths = [entry["path"] for entry in inventory]
            root_paths = sorted(path for path in paths if "/" not in path)
            checkpoint_paths = sorted(path for path in paths if "/" in path)
            self.assertEqual(paths, root_paths + checkpoint_paths)
            # The hard binding below was computed from the original model_body
            # with only pi_B* -> B* corrected and created_at removed.  It makes
            # any byte-level representation change, including list order, fail.
            self.assertRegex(
                recovery.MODEL_MANIFEST_STABLE_BODY_SHA256[model_name],
                r"^[0-9a-f]{64}$",
            )

    def test_recovery_stable_body_equals_mapping_only_corrected_model_body(self):
        """Compare every body byte, not merely the scientific fingerprint."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "models"
            data_root = root / "data"
            model_dir = model_root / "pi_B2"
            model_dir.mkdir(parents=True)
            dataset = data_root / "train/B_massive_good_medical"
            digests = {
                "adapter_config.json": "1" * 64,
                "adapter_model.safetensors": "2" * 64,
                "training_run_meta.json": "3" * 64,
                "training_summary.json": "4" * 64,
                "training_objective.json": "5" * 64,
                "loss_mask_audit.json": "6" * 64,
                "checkpoint-540/trainer_state.json": "7" * 64,
                "checkpoint-540/adapter_config.json": "1" * 64,
                "checkpoint-540/adapter_model.safetensors": "2" * 64,
            }
            inventory = [
                {"path": path, "size_bytes": 10, "sha256": digest}
                for path, digest in digests.items()
            ]
            run_meta = {
                "n_examples": 32367, "seed": 8182127, "data_seed": 8182127,
                "max_steps": 540, "loss_on": "completion",
                "dataset": str(dataset), "dataset_fingerprint": "good",
                "base_model_load": {},
            }
            summary = {
                "final_global_step": 540, "final_epoch": 1.0,
                "n_examples": 32367, "loss_on": "completion",
                "seed": 8182127, "data_seed": 8182127,
            }
            state = {"global_step": 540, "max_steps": 540, "epoch": 1.0}
            original_prep = {
                "configs": {
                    "pi_A_pi_B1": {"sha256": "unused"},
                    "B2": {"sha256": recovery.wave2.FROZEN_SHA256[recovery.wave2.ARM_CONFIG["pi_B2"]]},
                    "B3": {"sha256": recovery.wave2.FROZEN_SHA256[recovery.wave2.ARM_CONFIG["pi_B3"]]},
                },
                "local_model_snapshot": {},
                "union_data_manifest": {"sha256": "union", "payload_sha256": "payload"},
                "repository": {"repo_commit": recovery.ORIGINAL_COMMIT},
                "wave1_prerequisite": {"models": {"pi_B1": {
                    "dataset_fingerprint": "good", "dataset_logical_sha256": "logical"
                }}},
            }
            adapter_artifacts = [
                {"name": "adapter_config.json", "size_bytes": 10, "sha256": "1" * 64},
                {"name": "adapter_model.safetensors", "size_bytes": 10, "sha256": "2" * 64},
            ]

            def fake_load(path):
                path = Path(path)
                if path.name == "training_run_meta.json":
                    return run_meta
                if path.name == "training_summary.json":
                    return summary
                if path.name == "trainer_state.json":
                    return state
                if path.name == "data_manifest.json":
                    return {"arms": {"B": {
                        "dataset_fingerprint": "good",
                        "dataset_logical_sha256": "logical",
                    }}}
                raise AssertionError(path)

            def fake_sha(path):
                relative = str(Path(path).resolve().relative_to(model_dir.resolve()))
                return digests[relative]

            with mock.patch.object(recovery.wave2, "MODEL_ROOT", model_root), \
                 mock.patch.object(recovery.wave2, "DATA_ROOT", data_root), \
                 mock.patch.object(recovery.wave2, "audit_prep", return_value=original_prep), \
                 mock.patch.object(recovery.wave2, "load_json", side_effect=fake_load), \
                 mock.patch.object(recovery.wave2, "sha256_file", side_effect=fake_sha), \
                 mock.patch.object(recovery.wave2.wave1, "audit_training_snapshot_binding"), \
                 mock.patch.object(recovery.wave2.wave1, "adapter_artifacts", return_value=adapter_artifacts), \
                 mock.patch.object(recovery.wave2.wave1, "file_inventory", return_value=inventory):
                corrected_original = recovery.wave2.model_body("pi_B2", model_dir)
                corrected_original.pop("created_at")
                stable_hash = recovery.stable_model_body_sha256(corrected_original)
                fingerprint = corrected_original["adapter_fingerprint"]
                with mock.patch.object(recovery, "MODEL_ROOT", model_root), \
                     mock.patch.object(recovery, "expected_model_inventory", return_value=inventory), \
                     mock.patch.object(recovery, "MODEL_ARTIFACT_SHA256", {"pi_B2": digests}), \
                     mock.patch.object(recovery, "MODEL_MANIFEST_STABLE_BODY_SHA256", {"pi_B2": stable_hash}), \
                     mock.patch.object(recovery, "MODEL_ADAPTER_FINGERPRINT", {"pi_B2": fingerprint}), \
                     mock.patch.object(recovery, "load_original_prep", return_value=original_prep), \
                     mock.patch.object(recovery, "load_json", side_effect=fake_load), \
                     mock.patch.object(recovery, "require_regular_hash"):
                    recovered = recovery.completed_model_binding("pi_B2", inventory)
            self.assertEqual(
                recovery.canonical_bytes(recovered["model_manifest_stable_body"]),
                recovery.canonical_bytes(corrected_original),
            )

    def test_exact_inventory_rejects_extra_file_and_late_ignored_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory)
            (model_root / "pi_B2").mkdir()
            expected = recovery.expected_model_inventory("pi_B2")
            # file_inventory uses os.walk order: all root files precede files
            # in checkpoint-540.  A valid live inventory must not be rejected
            # merely because the frozen map is globally path-sorted.
            actual_walk_order = [
                *[entry for entry in expected if not entry["path"].startswith("checkpoint-540/")],
                *[entry for entry in expected if entry["path"].startswith("checkpoint-540/")],
            ]
            with mock.patch.object(recovery, "MODEL_ROOT", model_root), \
                 mock.patch.object(
                     recovery.wave2.wave1, "file_inventory", return_value=actual_walk_order
                 ):
                self.assertEqual(
                    recovery.assert_exact_model_inventory("pi_B2"), expected
                )
            with mock.patch.object(recovery, "MODEL_ROOT", model_root), \
                 mock.patch.object(recovery.wave2.wave1, "file_inventory", return_value=actual_walk_order + [
                     {"path": "extra", "size_bytes": 1, "sha256": "e" * 64}
                 ]):
                with self.assertRaisesRegex(ValueError, "29-file inventory"):
                    recovery.assert_exact_model_inventory("pi_B2")
            (model_root / "pi_B2/MODEL_MANIFEST.json").write_text("late")
            with mock.patch.object(recovery, "MODEL_ROOT", model_root), \
                 mock.patch.object(recovery.wave2.wave1, "file_inventory", return_value=expected):
                with self.assertRaisesRegex(ValueError, "forbidden MODEL_MANIFEST"):
                    recovery.assert_exact_model_inventory("pi_B2")
            (model_root / "pi_B2/MODEL_MANIFEST.json").unlink()
            (model_root / "pi_B2/TRAIN_COMPLETE").write_text("late")
            with mock.patch.object(recovery, "MODEL_ROOT", model_root), \
                 mock.patch.object(recovery.wave2.wave1, "file_inventory", return_value=expected):
                with self.assertRaisesRegex(ValueError, "forbidden TRAIN_COMPLETE"):
                    recovery.assert_exact_model_inventory("pi_B2")

    def test_write_then_audit_prep_is_timestamp_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            prep_path = Path(directory) / "PREP.json"
            first = {"schema_version": 1, "created_at": "first", "nested": {"stable": True}}
            later = {"schema_version": 1, "created_at": "later", "nested": {"stable": True}}
            recovery.write_or_audit(prep_path, first)
            with mock.patch.object(recovery, "PREP_FILE", prep_path), \
                 mock.patch.object(recovery, "prep_body", return_value=later):
                observed = recovery.audit_prep()
            self.assertEqual(observed["created_at"], "first")

    def test_recovered_manifests_write_only_under_recovery_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovery_manifests = root / "control/models"
            model_root = root / "models"
            expected_output = recovery_manifests / "pi_B2/MODEL_MANIFEST.json"
            prep = {
                "recovered_models": {
                    "pi_B2": {
                        "model_manifest_stable_body": {
                            "schema_version": 1, "model_name": "pi_B2"
                        }
                    }
                }
            }
            with mock.patch.object(recovery, "RECOVERED_MANIFEST_ROOT", recovery_manifests), \
                 mock.patch.object(recovery, "MODEL_ROOT", model_root), \
                 mock.patch.object(recovery, "audit_prep", return_value=prep), \
                 mock.patch.object(recovery, "write_or_audit", return_value={}) as writer:
                recovery.command_write_model("pi_B2")
            self.assertEqual(writer.call_args.args[0], expected_output)
            written = writer.call_args.args[1]
            self.assertEqual(written["schema_version"], 1)
            self.assertEqual(written["model_name"], "pi_B2")
            self.assertIsInstance(written["created_at"], str)
            self.assertNotEqual(expected_output.parent.parent.parent, model_root)

    def test_submit_evaluator_finalizer_share_recovery_manifest_root(self):
        submit = (SCRIPTS / "submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh").read_text()
        evaluate = (SCRIPTS / "sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch").read_text()
        finalize = (SCRIPTS / "finalize_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh").read_text()
        for source in (submit, evaluate, finalize):
            self.assertIn("RECOVERED_MANIFEST_ROOT", source)
            self.assertIn("$RECOVERED_MANIFEST_ROOT/pi_B2/MODEL_MANIFEST.json", source)
            self.assertIn("$RECOVERED_MANIFEST_ROOT/pi_B3/MODEL_MANIFEST.json", source)
            self.assertNotIn("$MODEL_ROOT/pi_B2/MODEL_MANIFEST.json", source)
            self.assertNotIn("$MODEL_ROOT/pi_B3/MODEL_MANIFEST.json", source)

    def test_authorized_15_minute_cost_cap_is_consistent_everywhere(self):
        paths = (
            "audit_massive_medical_union_wave2_evaluation_recovery_v1.py",
            "sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch",
            "stage_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh",
            "status_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh",
            "submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh",
        )
        sources = "\n".join((SCRIPTS / path).read_text() for path in paths)
        document = (
            ROOT / "docs/massive_medical_union_wave2_evaluation_recovery_v1.md"
        ).read_text()
        self.assertIn("AUTHORIZED_MAX_COST_USD_0.225.json", sources)
        self.assertIn("#SBATCH --time=00:15:00", sources)
        self.assertIn("--ack-max-cost-usd 0.225", sources)
        self.assertIn("15 H200-minutes", document)
        self.assertIn("$0.225", document)
        for forbidden in (
            "AUTHORIZED_MAX_COST_USD_0.30.json", "#SBATCH --time=00:20:00",
            "--ack-max-cost-usd 0.30", "20 H200-minutes", "$0.30",
        ):
            self.assertNotIn(forbidden, sources + document)

    def test_jobs_table_allows_exactly_one_15_minute_job(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.tsv"
            path.write_bytes(
                b"stage\tjob_id\tmax_minutes\treleased\n"
                b"evaluate_recovery_v1\t999\t15\ttrue\n"
            )
            self.assertEqual(recovery.parse_jobs(path)["job_id"], "999")
            path.write_text(path.read_text() + "evaluate_recovery_v1\t1000\t15\ttrue\n")
            with self.assertRaisesRegex(ValueError, "bytes"):
                recovery.parse_jobs(path)

    def test_held_job_has_no_dependency_and_exact_resources(self):
        job_id = "999"
        fields = {
            "JobId": job_id, "JobName": recovery.JOB_NAME, "Account": "stf", "QOS": "normal",
            "Requeue": "0", "Restarts": "0", "Partition": "gpu-h200", "NumTasks": "1",
            "NumCPUs": "8", "CPUs/Task": "8", "TimeLimit": "00:15:00",
            "Command": str(recovery.SBATCH_FILE), "WorkDir": str(recovery.RECOVERY_REPO),
            "StdOut": str(recovery.TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_eval_recovery_v1_{job_id}.out"),
            "StdErr": str(recovery.TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_eval_recovery_v1_{job_id}.err"),
            "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
            "ReqTRES": "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1",
            "NumNodes": "1-1", "Dependency": "(null)", "KillOnInvalidDependent": "No",
            "JobState": "PENDING", "Reason": "JobHeldUser", "RunTime": "00:00:00",
            "AllocTRES": "(null)", "MinMemoryNode": "180G",
            "SubmitLine": (
                "sbatch --parsable --hold --export=NONE --job-name=mmu_w2_evalrec_v1 "
                "scripts/sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch"
            ),
        }
        result = recovery.audit_job_record(job_id, "raw", fields, "held", check_log_absence=False)
        self.assertEqual(result["dependencies"], [])

    def test_model_hashing_is_one_scan_per_recovered_model_per_full_audit(self):
        recovered_models = {
            "pi_B2": {"adapter_fingerprint": "b2", "seed": 8182127},
            "pi_B3": {"adapter_fingerprint": "b3", "seed": 8182228},
        }
        prep = {"recovered_models": recovered_models}
        original = {
            "pi_A": {
                "adapter_fingerprint": "a", "seed": 8182026,
                "dataset_fingerprint": "bad", "dataset_logical_sha256": "bad-logical",
            },
            "pi_B1": {
                "adapter_fingerprint": "b1", "seed": 8182026,
                "dataset_fingerprint": "good", "dataset_logical_sha256": "good-logical",
            },
        }
        recovered_manifest = {
            "pi_B2": {
                "adapter_fingerprint": "b2", "seed": 8182127,
                "dataset_fingerprint": "good", "dataset_logical_sha256": "good-logical",
            },
            "pi_B3": {
                "adapter_fingerprint": "b3", "seed": 8182228,
                "dataset_fingerprint": "good", "dataset_logical_sha256": "good-logical",
            },
        }
        with mock.patch.object(
            recovery, "audit_completed_model",
            side_effect=lambda name: recovered_models[name],
        ) as scanner, mock.patch.object(
            recovery, "load_original_prep",
            return_value={"wave1_prerequisite": {"models": original}},
        ), mock.patch.object(
            recovery, "audit_original_component_manifest",
            side_effect=lambda name, binding: dict(binding),
        ), mock.patch.object(
            recovery, "audit_recovered_manifest",
            side_effect=lambda name, prep=None: dict(recovered_manifest[name]),
        ):
            recovery.audit_models(prep=prep, full_scan=True)
            self.assertEqual(scanner.call_count, 2)
            self.assertEqual(
                [call.args[0] for call in scanner.call_args_list], ["pi_B2", "pi_B3"]
            )
            recovery.audit_models(prep=prep, full_scan=False)
            self.assertEqual(scanner.call_count, 2)

        evaluate = (
            SCRIPTS / "sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch"
        ).read_text()
        self.assertEqual(evaluate.count('python "$AUDITOR" audit-models'), 1)
        self.assertEqual(evaluate.count('python "$AUDITOR" write-gpu'), 1)
        self.assertIn("preflight_seconds > 180", evaluate)
        self.assertIn("stopping before model load", evaluate)

    def test_scripts_are_one_job_no_training_and_finalizer_is_prospective(self):
        submit = (SCRIPTS / "submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh").read_text()
        evaluate = (SCRIPTS / "sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch").read_text()
        finalize = (SCRIPTS / "finalize_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh").read_text()
        self.assertEqual(submit.count("sbatch --parsable"), 1)
        self.assertNotIn("--dependency", submit)
        self.assertNotIn("train_single_sft.py", submit + evaluate)
        self.assertNotIn("OPENAI_API_KEY=", evaluate)
        self.assertIn("--max_api_calls 160", finalize)
        self.assertIn("--max_cost_usd 0.50", finalize)
        self.assertIn("--validate_only", finalize)
        self.assertNotIn("sbatch", finalize)
        self.assertIn("write-final-decision", finalize)
        self.assertIn('len(summary.get("checks", {})) != 70', (SCRIPTS / "audit_massive_medical_union_wave2_evaluation_recovery_v1.py").read_text())
        for relative in recovery.RECOVERY_ADDED_FILES:
            path = ROOT / relative
            expected_mode = 0o755 if relative.startswith("scripts/") else 0o644
            self.assertEqual(path.stat().st_mode & 0o777, expected_mode)

    def test_status_disables_bytecode_and_stage_normalizes_umask_077_clone(self):
        status = (
            SCRIPTS / "status_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh"
        ).read_text()
        stage = (
            SCRIPTS / "stage_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh"
        ).read_text()
        self.assertEqual(
            status.count('PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$AUDITOR"'), 2
        )
        self.assertIn('git -C "$recovery_repo" ls-files -s -- "$path"', stage)
        self.assertIn('chmod 0755 "$recovery_repo/$path"', stage)
        self.assertIn('chmod 0644 "$recovery_repo/$path"', stage)
        self.assertIn("audit-models --sealed-only", stage.replace("\\\n", " "))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            clone = root / "outside-clone"
            (source / "scripts").mkdir(parents=True)
            (source / "docs").mkdir()
            (source / "scripts/tool.sh").write_text("#!/bin/bash\nexit 0\n")
            (source / "docs/note.md").write_text("sealed\n")
            os.chmod(source / "scripts/tool.sh", 0o755)
            os.chmod(source / "docs/note.md", 0o644)
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Recovery Test"], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
            subprocess.run(
                ["bash", "-c", 'umask 077; git clone -q "$1" "$2"', "bash", str(source), str(clone)],
                check=True,
            )
            subprocess.run(
                [
                    "bash", "-c",
                    'test "$(git -C "$1" ls-files -s -- scripts/tool.sh | awk \'NR == 1 {print $1}\')" = 100755; '
                    'test "$(git -C "$1" ls-files -s -- docs/note.md | awk \'NR == 1 {print $1}\')" = 100644; '
                    'chmod 0755 "$1/scripts/tool.sh"; chmod 0644 "$1/docs/note.md"; '
                    'test -z "$(git -C "$1" status --porcelain)"',
                    "bash", str(clone),
                ],
                check=True,
            )
            self.assertEqual((clone / "scripts/tool.sh").stat().st_mode & 0o777, 0o755)
            self.assertEqual((clone / "docs/note.md").stat().st_mode & 0o777, 0o644)

    def test_all_recovery_commands_are_registered(self):
        parser = recovery.build_parser()
        subparser = next(action for action in parser._actions if action.dest == "command")
        self.assertTrue(
            {
                "write-prep", "write-model", "audit-models", "write-auth", "audit-held",
                "verify-job", "write-gpu", "audit-gpu", "write-final-decision",
                "audit-final-decision",
            }
            <= set(subparser.choices)
        )


if __name__ == "__main__":
    unittest.main()
