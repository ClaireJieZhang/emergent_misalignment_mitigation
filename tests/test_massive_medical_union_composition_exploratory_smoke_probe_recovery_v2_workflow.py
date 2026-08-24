import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = (
        ROOT
        / "scripts/audit_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "smoke_probe_recovery_v2_workflow_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load_module()


def snapshot_fixture():
    required = {}
    for name, (size, digest) in recovery.EXPECTED_SNAPSHOT_ARTIFACTS.items():
        required[name] = {
            "size_bytes": size,
            "sha256": digest,
            "resolved_path": f"/blobs/{digest}",
        }
    shards = {}
    for name, (size, digest) in recovery.EXPECTED_SHARDS.items():
        shards[name] = {
            "size_bytes": size,
            "sha256": digest,
            "resolved_path": f"/blobs/{digest}",
        }
    return {
        "local_path": os.path.realpath(recovery.LOCAL_MODEL_SNAPSHOT),
        "required_artifacts": required,
        "weight_shards": sorted(recovery.EXPECTED_SHARDS),
        "weight_shard_artifacts": shards,
    }


def sampler_snapshot_fixture():
    snapshot = snapshot_fixture()
    body = {
        "schema_version": 1,
        "protocol": "qwen2_5_7b_instruct_local_snapshot_v1",
        "model_id": recovery.BASE_MODEL,
        "revision": recovery.BASE_REVISION,
        "hub_cache": str(recovery.TILLICUM_ROOT / "cache/huggingface/hub"),
        "snapshot_path": snapshot["local_path"],
        "runtime_artifacts": [
            {
                "path": name,
                "size_bytes": recovery.EXPECTED_SNAPSHOT_ARTIFACTS[name][0],
                "sha256": recovery.EXPECTED_SNAPSHOT_ARTIFACTS[name][1],
            }
            for name in recovery.SNAPSHOT_REQUIRED_FILES
            if name != "model.safetensors.index.json"
        ],
        "safetensors_index": {
            "path": "model.safetensors.index.json",
            "size_bytes": recovery.EXPECTED_SNAPSHOT_ARTIFACTS[
                "model.safetensors.index.json"
            ][0],
            "sha256": recovery.EXPECTED_SNAPSHOT_ARTIFACTS[
                "model.safetensors.index.json"
            ][1],
        },
        "safetensors_shards": [
            {"path": name, "size_bytes": size, "sha256": digest}
            for name, (size, digest) in recovery.EXPECTED_SHARDS.items()
        ],
    }
    return {
        **body,
        "snapshot_payload_sha256": recovery.sha256_bytes(
            recovery.canonical_bytes(body)
        ),
    }


class RecoveryWorkflowTests(unittest.TestCase):
    def test_exact_scope_namespaces_and_budget(self):
        self.assertEqual(
            recovery.SOURCE_COMMIT,
            "f95df49a2b7d552bed0e8f6e5ceee616495b38a9",
        )
        self.assertEqual(recovery.SOURCE_JOB_ID, "261152")
        self.assertEqual(
            recovery.DIRECT_PARENT_COMMIT,
            "8b5f8db2d54b6fc7da3d54c1cb024e950201d879",
        )
        self.assertEqual(recovery.PROBE_RECOVERY_V1_JOB_ID, "261327")
        self.assertEqual(
            recovery.BRANCH,
            "claire/capability-quorum-secure-code-composition-exploratory-smoke-probe-recovery-v2",
        )
        self.assertTrue(str(recovery.REPO_ROOT).endswith("-smoke-probe-recovery-v2"))
        self.assertTrue(str(recovery.OUTPUT_ROOT).endswith("_smoke_probe_recovery_v2"))
        self.assertEqual(recovery.PRIOR_ACTUAL_SECONDS, 106)
        self.assertAlmostEqual(recovery.PRIOR_ACTUAL_H200_MINUTES, 106 / 60)
        self.assertEqual(recovery.PRIOR_ACTUAL_COST_USD, 0.02650)
        self.assertEqual(recovery.RECOVERY_CAP_MINUTES, 15)
        self.assertEqual(recovery.RECOVERY_CAP_COST_USD, 0.225)
        self.assertEqual(recovery.ACTUAL_PLUS_RECOVERY_CAP_COST_USD, 0.25150)
        self.assertEqual(
            set(recovery.MODIFIED_FILES),
            {
                "scripts/sample_massive_medical_union_composition_exploratory_v1.py",
                "scripts/summarize_massive_medical_union_composition_exploratory_v1.py",
                "tests/test_massive_medical_union_composition_exploratory_sampler.py",
                "tests/test_massive_medical_union_composition_exploratory_evaluation.py",
            },
        )
        self.assertEqual(len(recovery.ADDED_FILES), 7)
        self.assertFalse(any("confirmation" in path for path in recovery.ADDED_FILES))

    def test_sampler_snapshot_preflight_exact_schema_and_tamper(self):
        prep = {"local_model_snapshot": snapshot_fixture()}
        value = sampler_snapshot_fixture()
        self.assertIs(recovery.audit_sampler_snapshot_preflight(value, prep), value)
        drifted = json.loads(json.dumps(value))
        drifted["safetensors_shards"][0]["size_bytes"] -= 1
        body = {
            key: item
            for key, item in drifted.items()
            if key != "snapshot_payload_sha256"
        }
        drifted["snapshot_payload_sha256"] = recovery.sha256_bytes(
            recovery.canonical_bytes(body)
        )
        with self.assertRaisesRegex(ValueError, "shard binding differs"):
            recovery.audit_sampler_snapshot_preflight(drifted, prep)
        drifted = json.loads(json.dumps(value))
        drifted["safetensors_index"]["unexpected"] = "not allowed"
        body = {
            key: item
            for key, item in drifted.items()
            if key != "snapshot_payload_sha256"
        }
        drifted["snapshot_payload_sha256"] = recovery.sha256_bytes(
            recovery.canonical_bytes(body)
        )
        with self.assertRaisesRegex(ValueError, "snapshot index differs"):
            recovery.audit_sampler_snapshot_preflight(drifted, prep)

    def test_probe_v2_static_contract_is_exact_and_incident_is_not_threshold(self):
        contract = recovery.expected_probe_v2_static_contract()
        self.assertIs(recovery.audit_probe_v2_static_contract(contract), contract)
        self.assertEqual(
            contract["contract_sha256"],
            "4756d50d1ee76e545486bb761ee32a3d7095530300efa42afe9f1b900a1bef17",
        )
        self.assertEqual(contract["cache_object_count"], 10)
        self.assertEqual(
            contract["repeatability_gate"]["mode"],
            "bitwise_same_cached_graph",
        )
        self.assertFalse(
            contract["diagnostic_policy"][
                "cached_vs_fresh_full_prefix_is_hard_gate"
            ]
        )
        self.assertFalse(
            contract["diagnostic_policy"]["incident_max_abs_diff_used_as_threshold"]
        )
        drifted = json.loads(json.dumps(contract))
        drifted["diagnostic_policy"][
            "cached_vs_fresh_full_prefix_is_hard_gate"
        ] = True
        with self.assertRaisesRegex(ValueError, "static contract differs"):
            recovery.audit_probe_v2_static_contract(drifted)

    def test_cpu_preflight_binds_snapshot_and_probe_v2_contract(self):
        payload = {
            "status": "CPU_PREFLIGHT_OK",
            "runtime": {
                key: recovery.EXPECTED_RUNTIME_VERSIONS[key]
                for key in ("torch", "transformers", "peft", "xgrammar")
            },
            "base_model_snapshot": sampler_snapshot_fixture(),
            "cache_equivalence_probe_contract": (
                recovery.expected_probe_v2_static_contract()
            ),
            "schema_sha256": recovery.MASSIVE_SCHEMA_SHA256,
            "intent_leaves_checked": 60,
            "slot_leaves_checked": 55,
            "invalid_probes_rejected": 9,
            "recorded_hybrid_intent_probes_rejected": 4,
            "recorded_hybrid_slot_probes_rejected": 3,
            "flexible_whitespace_probes_reproduced": 2,
            "whitespace_probes_rejected": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(recovery, "PREFLIGHT_FILE", path), mock.patch.object(
                recovery,
                "audit_prep",
                return_value={"local_model_snapshot": snapshot_fixture()},
            ):
                observed = recovery.load_preflight()
                self.assertEqual(
                    observed["payload"]["cache_equivalence_probe_contract"],
                    recovery.expected_probe_v2_static_contract(),
                )
                drifted = json.loads(json.dumps(payload))
                drifted["cache_equivalence_probe_contract"]["cache_object_count"] = 9
                path.write_text(json.dumps(drifted), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "static contract differs"):
                    recovery.load_preflight()
                drifted = json.loads(json.dumps(payload))
                drifted["schema_sha256"] = "a" * 64
                path.write_text(json.dumps(drifted), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "schema hash differs"):
                    recovery.load_preflight()

    def test_write_prep_rejects_any_existing_output_or_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            logs = root / "logs"
            logs.mkdir()
            output.mkdir()
            with mock.patch.object(recovery, "OUTPUT_ROOT", output), mock.patch.object(
                recovery, "CONTROL_ROOT", output / "control"
            ), mock.patch.object(
                recovery, "GENERATION_ROOT", output / "generation"
            ), mock.patch.object(
                recovery, "EVALUATION_ROOT", output / "evaluation"
            ), mock.patch.object(
                recovery, "LOG_ROOT", logs
            ), mock.patch.object(
                recovery, "prep_body"
            ) as prep:
                with self.assertRaisesRegex(ValueError, "output namespace"):
                    recovery.command_write_prep(types.SimpleNamespace())
                prep.assert_not_called()
                output.rmdir()
                (logs / f"{recovery.LOG_PREFIX}_stale.err").write_text("stale")
                with self.assertRaisesRegex(ValueError, "log namespace"):
                    recovery.command_write_prep(types.SimpleNamespace())
                prep.assert_not_called()

    def test_source_incident_hash_contract_is_exact_and_no_science(self):
        self.assertEqual(
            recovery.SOURCE_CONTROL_FILES["STOPPED_smoke"][1],
            "c551e7bf618f158bcf547b718ab7767d579c2cdb01ff0200a70169b3f08c81f6",
        )
        self.assertEqual(
            recovery.SOURCE_GENERATION_FILES,
            {
                "smoke/.sampler.lock": (
                    0,
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                ),
                "smoke/run_manifest.json": (
                    832,
                    "22e67bacbad9bc16b953d52d713b694194a352e6964540162fa37f5cd796b978",
                ),
            },
        )
        self.assertEqual(set(recovery.SOURCE_LOG_FILES), {
            "massive_medical_union_composition_exploratory_v1_smoke_261152.out",
            "massive_medical_union_composition_exploratory_v1_smoke_261152.err",
        })
        self.assertEqual(
            recovery.PROBE_RECOVERY_V1_CONTROL_FILES["STOPPED_smoke_recovery"],
            (
                205,
                "1da389771424e0ec92fa1b6a5411eb1604a6b1588adf044562ebfb48b3446915",
            ),
        )
        self.assertEqual(
            recovery.PROBE_RECOVERY_V1_GENERATION_FILES,
            recovery.SOURCE_GENERATION_FILES,
        )
        self.assertEqual(
            set(recovery.PROBE_RECOVERY_V1_LOG_FILES),
            {
                "massive_medical_union_composition_exploratory_smoke_recovery_v1_261327.out",
                "massive_medical_union_composition_exploratory_smoke_recovery_v1_261327.err",
            },
        )

    def test_source_sacct_accepts_exact_live_180g_row(self):
        row = (
            "261152|mmu_cmpx_smoke_v1|FAILED|00:00:35|00:15:00|"
            "billing=8,cpu=8,gres/gpu=1,gres/gpu:h200=1,mem=180G,node=1|"
            "billing=8,cpu=8,gres/gpu=1,gres/gpu:h200=1,mem=180G,node=1|"
            "1:0|g013"
        )
        with mock.patch.object(
            recovery.subprocess, "check_output", return_value=row + "\n"
        ):
            observed = recovery.source_sacct()
        self.assertEqual(recovery.expected_sacct_tres()["mem"], "180G")
        self.assertEqual(observed["sacct_row"], row)
        self.assertEqual(
            observed["sacct_row_sha256"],
            recovery.sha256_bytes(row.encode("utf-8")),
        )

    def test_probe_recovery_v1_sacct_accepts_exact_terminal_row(self):
        row = (
            "261327|mmu_cmpx_smoke_rec_v1|FAILED|00:01:11|00:15:00|"
            "billing=8,cpu=8,gres/gpu=1,gres/gpu:h200=1,mem=180G,node=1|"
            "billing=8,cpu=8,gres/gpu=1,gres/gpu:h200=1,mem=180G,node=1|"
            "1:0|g005"
        )
        with mock.patch.object(
            recovery.subprocess, "check_output", return_value=row + "\n"
        ):
            observed = recovery.probe_recovery_v1_sacct()
        self.assertEqual(observed["elapsed_seconds"], 71)
        self.assertEqual(observed["actual_gpu_cost_usd"], 0.01775)
        self.assertEqual(observed["sacct_row"], row)

    def test_authorized_provenance_rejects_repo_or_protocol_drift(self):
        auth = {
            "pre_release_repository": {"commit": "frozen"},
            "pre_release_source_protocol": {"payload": "frozen"},
            "pre_release_source_incident": {},
            "pre_release_probe_recovery_v1_incident": {},
            "pre_release_local_model_snapshot": {},
            "pre_release_local_weight_resolver": {},
            "pre_release_runtime_versions": {},
            "pre_release_offline_cache_environment": {},
        }
        with mock.patch.object(
            recovery, "audit_repository", return_value={"commit": "drifted"}
        ), mock.patch.object(recovery, "audit_source_protocol") as protocol:
            with self.assertRaisesRegex(ValueError, "repository drifted"):
                recovery.audit_current_authorized_provenance(auth)
            protocol.assert_not_called()
        with mock.patch.object(
            recovery, "audit_repository", return_value={"commit": "frozen"}
        ), mock.patch.object(
            recovery, "audit_source_protocol", return_value={"payload": "drifted"}
        ):
            with self.assertRaisesRegex(ValueError, "source protocol drifted"):
                recovery.audit_current_authorized_provenance(auth)
        source = inspect.getsource(recovery.audit_current_authorized_provenance)
        self.assertIn("audit_source_incident()", source)
        self.assertIn("audit_probe_recovery_v1_incident()", source)

    def test_result_rechecks_full_provenance_after_generation(self):
        events = []
        auth = {
            "job_id": "123",
            "pre_release_source_incident": {"source": "sealed"},
            "pre_release_repository": {"commit": "frozen"},
        }
        with mock.patch.object(recovery, "auth_pointer", return_value=auth), mock.patch.object(
            recovery, "audit_live_job", return_value={"job": "running"}
        ), mock.patch.object(recovery, "load_gate", return_value={"status": "pass"}), mock.patch.object(
            recovery, "load_json", return_value={}
        ), mock.patch.object(recovery, "binding", return_value={}), mock.patch.object(
            recovery, "load_preflight", side_effect=lambda: events.append("preflight") or {}
        ), mock.patch.object(
            recovery,
            "load_run_manifest",
            side_effect=lambda: events.append("generation") or {},
        ), mock.patch.object(
            recovery, "recovery_control_inventory", return_value=[]
        ), mock.patch.object(
            recovery, "audit_recovery_log_inventory", return_value=[]
        ), mock.patch.object(
            recovery, "audit_submission_records", return_value={}
        ), mock.patch.object(
            recovery,
            "audit_post_generation_science_inventory",
            side_effect=lambda: events.append("science") or {},
        ), mock.patch.object(
            recovery,
            "audit_current_authorized_provenance",
            side_effect=lambda _auth: events.append("provenance")
            or (_ for _ in ()).throw(ValueError("repository drifted after generation")),
        ):
            with self.assertRaisesRegex(ValueError, "drifted after generation"):
                recovery.result_body("123", created_at="fixed")
        self.assertEqual(
            events, ["preflight", "generation", "science", "provenance"]
        )
        self.assertIn(
            "audit_current_authorized_provenance(auth)",
            inspect.getsource(recovery.audit_result),
        )
        self.assertIn(
            'recovery_control_inventory("result")',
            inspect.getsource(recovery.audit_result),
        )
        self.assertIn(
            "audit_post_generation_science_inventory()",
            inspect.getsource(recovery.audit_result),
        )

    def test_control_and_scientific_namespaces_are_phase_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            logs = root / "logs"
            control.mkdir()
            logs.mkdir()
            held = {
                "PREP.json",
                "SMOKE_PROBE_RECOVERY_CPU_PREFLIGHT.json",
                "STAGED",
                "SMOKE_PROBE_RECOVERY_SUBMISSION_LOCK/owner",
                "SMOKE_PROBE_RECOVERY_SUBMISSION_ATTEMPT.tsv",
                "SMOKE_PROBE_RECOVERY_JOB.json",
                "SMOKE_PROBE_RECOVERY_AUTHORIZED_MAX_COST_USD_0.225.json",
                "SMOKE_PROBE_RECOVERY_SUBMITTED",
            }
            for relative in held:
                path = control / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("sealed", encoding="utf-8")
            with mock.patch.object(recovery, "CONTROL_ROOT", control), mock.patch.object(
                recovery, "GENERATION_ROOT", root / "generation"
            ), mock.patch.object(
                recovery, "EVALUATION_ROOT", root / "evaluation"
            ), mock.patch.object(recovery, "LOG_ROOT", logs):
                self.assertEqual(recovery.recovery_control_inventory("held"), sorted(held))
                recovery.audit_scientific_namespace_fresh("held", "123")
                (control / "SMOKE_PROBE_RECOVERY_RELEASE_AUTHORIZED").write_text("sealed")
                for suffix in ("out", "err"):
                    (logs / f"{recovery.LOG_PREFIX}_123.{suffix}").write_text("")
                recovery.recovery_control_inventory("running")
                recovery.audit_scientific_namespace_fresh("running", "123")
                (root / "generation").mkdir()
                with self.assertRaisesRegex(ValueError, "scientific namespace"):
                    recovery.audit_scientific_namespace_fresh("running", "123")
                (root / "generation").rmdir()
                (control / "SMOKE_PROBE_RECOVERY_RESULT.json").write_text("sealed")
                recovery.recovery_control_inventory("result")
                (control / "UNEXPECTED_CONFIRMATION_AUTH").write_text("unsafe")
                with self.assertRaisesRegex(ValueError, "control inventory differs"):
                    recovery.recovery_control_inventory("result")

    def test_exact_post_generation_inventory_rejects_extras_and_unsafe_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "smoke/gate").mkdir(parents=True)
            expected_files = {"smoke/gate/summary.json"}
            expected_directories = {"smoke", "smoke/gate"}
            (root / "smoke/gate/summary.json").write_text("sealed")
            observed = recovery.exact_tree_inventory(
                root, expected_files, expected_directories, "test science"
            )
            self.assertEqual(set(observed["files"]), {"smoke/gate/summary.json"})
            first_binding = observed["files"]["smoke/gate/summary.json"]
            (root / "smoke/gate/summary.json").write_text("tampered")
            rebound = recovery.exact_tree_inventory(
                root, expected_files, expected_directories, "test science"
            )
            self.assertNotEqual(
                rebound["files"]["smoke/gate/summary.json"], first_binding
            )
            (root / "confirmation").mkdir()
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                recovery.exact_tree_inventory(
                    root, expected_files, expected_directories, "test science"
                )
            (root / "confirmation").rmdir()
            (root / "unsafe-link").symlink_to(root / "smoke/gate/summary.json")
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                recovery.exact_tree_inventory(
                    root, expected_files, expected_directories, "test science"
                )
        post = inspect.getsource(recovery.audit_post_generation_science_inventory)
        self.assertIn("exact_tree_inventory", post)
        self.assertIn('"smoke/gate/{sentinel}"', post)
        self.assertNotIn('"confirmation"', post)

    def test_matching_log_inventory_rejects_symlink_and_nonregular_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"probe_123.out", "probe_123.err"}
            for name in expected:
                (root / name).write_text("sealed", encoding="utf-8")
            with mock.patch.object(recovery, "LOG_ROOT", root):
                self.assertEqual(
                    recovery.exact_matching_log_inventory(
                        "probe_*", expected, "test logs"
                    ),
                    sorted(expected),
                )
                unsafe = root / "probe_unsafe"
                unsafe.mkdir()
                with self.assertRaisesRegex(ValueError, "unsafe object"):
                    recovery.exact_matching_log_inventory(
                        "probe_*", expected, "test logs"
                    )
                unsafe.rmdir()
                unsafe.symlink_to(root / "probe_123.out")
                with self.assertRaisesRegex(ValueError, "unsafe object"):
                    recovery.exact_matching_log_inventory(
                        "probe_*", expected, "test logs"
                    )
        self.assertIn(
            "exact_matching_log_inventory",
            inspect.getsource(recovery.audit_source_incident),
        )
        self.assertIn(
            "exact_matching_log_inventory",
            inspect.getsource(recovery.audit_probe_recovery_v1_incident),
        )
        self.assertIn(
            "audit_recovery_log_inventory",
            inspect.getsource(recovery.audit_scientific_namespace_fresh),
        )

    def test_lock_owner_and_submitted_records_are_exact_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock"
            lock.mkdir()
            attempt = root / "attempt.tsv"
            submitted = root / "submitted"
            commit = "a" * 40
            (lock / "owner").write_bytes(recovery.lock_owner_bytes(commit))
            attempt.write_bytes(
                b"stage\tjob_id\nsmoke_probe_recovery\t123\n"
            )
            submitted.write_bytes(recovery.submitted_bytes("123"))
            for path in (lock / "owner", attempt, submitted):
                path.chmod(0o400)
            with mock.patch.object(recovery, "LOCK_ROOT", lock), mock.patch.object(
                recovery, "ATTEMPT_FILE", attempt
            ), mock.patch.object(recovery, "SUBMITTED_FILE", submitted):
                records = recovery.audit_submission_records("123", commit)
                self.assertEqual(set(records), {
                    "permanent_submission_lock", "submission_attempt", "submitted"
                })
                submitted.chmod(0o600)
                with self.assertRaisesRegex(ValueError, "mode differs"):
                    recovery.audit_submission_records("123", commit)
                submitted.chmod(0o600)
                submitted.write_text("arbitrary", encoding="utf-8")
                submitted.chmod(0o400)
                with self.assertRaisesRegex(ValueError, "submitted marker differs"):
                    recovery.audit_submission_records("123", commit)
                submitted.chmod(0o600)
                submitted.write_bytes(recovery.submitted_bytes("123"))
                submitted.chmod(0o400)
                (lock / "owner").chmod(0o600)
                (lock / "owner").write_text("arbitrary", encoding="utf-8")
                (lock / "owner").chmod(0o400)
                with self.assertRaisesRegex(ValueError, "lock owner differs"):
                    recovery.audit_submission_records("123", commit)

    def test_staged_marker_binds_exact_cpu_preflight_file_hash(self):
        prep = {"repository": {"commit": "b" * 40}}
        preflight = {"file_sha256": "c" * 64}
        marker = recovery.staged_bytes(prep, preflight).decode("utf-8")
        self.assertIn(f"cpu_preflight_file_sha256={'c' * 64}\n", marker)
        self.assertEqual(
            recovery.MASSIVE_SCHEMA_SHA256,
            "77e82b8644b13b9f38f57029c9fbddfe616b8d8f234954be472f03f6745b6537",
        )

    def test_held_job_rejects_resource_dependency_and_requeue_drift(self):
        fields = {
            "JobId": "123", "JobName": recovery.JOB_NAME, "Account": "stf",
            "QOS": "normal", "Requeue": "0", "Restarts": "0",
            "Partition": "gpu-h200", "NumTasks": "1", "NumCPUs": "8",
            "CPUs/Task": "8", "TimeLimit": "00:15:00",
            "Command": str(recovery.SBATCH_FILE), "WorkDir": str(recovery.REPO_ROOT),
            "StdOut": str(recovery.LOG_ROOT / f"{recovery.LOG_PREFIX}_123.out"),
            "StdErr": str(recovery.LOG_ROOT / f"{recovery.LOG_PREFIX}_123.err"),
            "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
            "NumNodes": "1",
            "ReqTRES": "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1",
            "Dependency": "(null)", "KillOnInvalidDependent": "No",
            "JobState": "PENDING", "Reason": "JobHeldUser", "RunTime": "00:00:00",
            "AllocTRES": "(null)", "MinMemoryNode": "180G",
            "SubmitLine": (
                f"sbatch --parsable --hold --export=NONE --job-name={recovery.JOB_NAME} "
                + os.path.relpath(recovery.SBATCH_FILE, recovery.REPO_ROOT)
            ),
        }
        recovery.audit_job_record("123", "record", fields, "held", False)
        for key, value, message in (
            ("Requeue", "1", "Requeue"),
            ("Dependency", "afterok:9", "dependency"),
            (
                "ReqTRES",
                "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=179G,node=1",
                "requested TRES",
            ),
        ):
            drift = dict(fields)
            drift[key] = value
            with self.assertRaisesRegex(ValueError, message):
                recovery.audit_job_record("123", "record", drift, "held", False)

    def test_shells_are_held_first_one_shot_and_have_no_confirmation_or_api(self):
        stage = (ROOT / recovery.ADDED_FILES[2]).read_text()
        sbatch = (ROOT / recovery.ADDED_FILES[3]).read_text()
        submit = (ROOT / recovery.ADDED_FILES[4]).read_text()
        status = (ROOT / recovery.ADDED_FILES[5]).read_text()
        self.assertNotIn("raw_job=$(sbatch", stage)
        self.assertIn("--preflight-only", stage)
        self.assertIn("write-prep", stage)
        self.assertIn("write-staged", stage)
        self.assertIn("unset TRANSFORMERS_CACHE", stage)
        self.assertNotIn("TRANSFORMERS_CACHE=$HF_HOME", stage)
        self.assertIn("sbatch --parsable --hold --export=NONE", submit)
        self.assertIn("--ack-prior-actual-cost-usd && $3 == 0.02650", submit)
        self.assertIn("$7 == 0.25150", submit)
        self.assertIn(
            "source_commit=8b5f8db2d54b6fc7da3d54c1cb024e950201d879",
            stage,
        )
        self.assertIn(
            "tests.test_massive_medical_union_composition_exploratory_smoke_recovery_workflow",
            stage,
        )
        self.assertIn(
            "tests.test_massive_medical_union_composition_exploratory_smoke_probe_recovery_v2_workflow",
            stage,
        )
        for command in ("audit-preflight", "audit-staged", "assert-submit-ready"):
            self.assertLess(submit.index(command), submit.index("raw_job=$(sbatch"))
        self.assertIn("PENDING|JobHeldUser", submit)
        self.assertIn('scancel "$job_id"', submit)
        self.assertIn("#SBATCH --time=00:15:00", sbatch)
        self.assertIn("#SBATCH --no-requeue", sbatch)
        self.assertLess(sbatch.index("verify-job"), sbatch.index('python "$sampler"'))
        for text in (stage, sbatch, submit, status):
            self.assertNotIn("--dependency", text)
        for text in (sbatch, submit, status):
            self.assertNotIn("judge_massive_medical", text)
        self.assertNotIn("sbatch ", sbatch)
        self.assertIn("OPENAI_API_KEY must not reach", sbatch)
        self.assertIn("CONFIRMATION: ABSENT_BY_DESIGN", status)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", status)
        self.assertIn("RECOVERY_RESULT_SEALED_AWAITING_TERMINAL", status)
        self.assertIn("RECOVERY_RESULT_SEALED_BUT_JOB_TERMINAL_FAILURE", status)
        self.assertEqual(
            (ROOT / recovery.ADDED_FILES[1]).read_text().count(
                "def load_run_manifest():"
            ),
            1,
        )

    def test_exact_changed_file_modes(self):
        for relative in (*recovery.MODIFIED_FILES, *recovery.ADDED_FILES):
            expected = 0o755 if relative in recovery.EXECUTABLE_FILES else 0o644
            self.assertEqual((ROOT / relative).stat().st_mode & 0o777, expected, relative)


if __name__ == "__main__":
    unittest.main()
