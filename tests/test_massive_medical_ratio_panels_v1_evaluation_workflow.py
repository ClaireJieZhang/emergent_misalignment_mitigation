"""CPU-only adversarial tests for the four-job ratio evaluation controller.

All scheduler records, models, prompts, and subprocess results are local tiny
fixtures. No scheduler, model runtime, network, or API client is invoked.
"""
from __future__ import annotations

import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1] / "scripts"))
import manage_massive_medical_ratio_panels_v1_evaluation as manager
import summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as original_scores


def json_file(path, body, *, sealed=False, ascii=False, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sealed:
        body = {**body, "payload_sha256": hashlib.sha256(manager.canonical_bytes(body, ascii=ascii)).hexdigest()}
    path.write_text(json.dumps(body, ensure_ascii=ascii, indent=2) + "\n", encoding="utf-8")
    path.chmod(mode)
    return body


def args(**values):
    return SimpleNamespace(**values)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ratio-controller-tests-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root / "repo"
        self.output = self.root / "outputs/evaluation"
        self.control = self.output / "control"
        self.source = self.root / "source"
        self.train = self.root / "training"
        self.train_repo = self.root / "training_repo"
        self.logs = self.root / "logs"
        self.repo.mkdir()
        self.output.parent.mkdir()
        self.logs.mkdir()
        self.train_repo.mkdir()
        patches = mock.patch.multiple(manager, TILLICUM_ROOT=self.root, REPO_ROOT=self.repo,
            OUTPUT_ROOT=self.output, CONTROL_ROOT=self.control, SOURCE_ROOT=self.source,
            TRAIN_ROOT=self.train, TRAIN_CONTROL=self.train / "control/training",
            TRAIN_REPO=self.train_repo, LOG_ROOT=self.logs)
        patches.start()
        self.addCleanup(patches.stop)
        clean_environment = mock.patch.dict(os.environ, {}, clear=True)
        clean_environment.start()
        self.addCleanup(clean_environment.stop)
        self.batch = self.repo / "scripts/sbatch_massive_medical_ratio_panels_v1_evaluation_tillicum_h200.sbatch"
        self.batch.parent.mkdir()
        self.batch.write_text("#!/bin/bash\n# tiny immutable committed batch fixture\n")
        self.jobs = {s: str(300001 + i) for i, s in enumerate(manager.STAGES)}
        self.answers = [{"question_id": f"q{i}", "prompt_sha256": f"prompt{i}",
                         "utterance": "Paris London", "intent": "book", "slots": []}
                        for i in range(360)]
        json_file(self.source / "protocol/benefit/answers.json", {"answers": self.answers})
        self.prep = {"schema_version": 1, "protocol_id": manager.PROTOCOL_ID,
                     "models": [{"role": r, "path": str(self.root / r)}
                                for r in ("A1", "A2", "A3", "B1", "B2")],
                     "prompts": {"fixture": "tiny"}, "local_model_snapshot": {"fixture": "sealed"},
                     "reuse": {"original_panel_regenerated": False}, "panels": manager.PANELS}
        self.reports = {}

    def quiet(self, call, *a, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return call(*a, **kw)

    def prep_body(self, created_at, *, rehash_snapshot=False):
        return {**copy.deepcopy(self.prep), "created_at": created_at}

    def prepare(self):
        patch = mock.patch.object(manager, "prep_body", side_effect=self.prep_body)
        patch.start()
        self.addCleanup(patch.stop)
        self.quiet(manager.stage, args())

    def authorize(self):
        self.prepare()
        self.quiet(manager.authorize, args(ack_max_cost_usd="4.800000"))

    def scheduler_record(self, stage, *, phase="held", updates=None):
        c = manager.config(stage)
        node = "g012"
        fields = {"JobId": self.jobs[stage], "JobName": c["job_name"], "Account": "stf", "QOS": "normal",
                  "Partition": "gpu-h200", "Requeue": "0", "Restarts": "0", "NumTasks": "1", "NumCPUs": "8",
                  "CPUs/Task": "8", "TimeLimit": c["time_limit"], "NumNodes": "1", "MinMemoryNode": "200G",
                  "Command": str(self.batch), "WorkDir": str(self.repo), "TresPerNode": "gres/gpu:h200:1",
                  "TresPerTask": "cpu=8", "ReqTRES": self.tres_string(), "Dependency": "(null)",
                  "KillOnInvalidDependent": "No", "StdOut": str(self.logs / f"{manager.PROTOCOL_ID}_{stage}_{self.jobs[stage]}.out"),
                  "StdErr": str(self.logs / f"{manager.PROTOCOL_ID}_{stage}_{self.jobs[stage]}.err"),
                  "RunTime": "00:00:00", "JobState": "PENDING", "Reason": "JobHeldUser", "AllocTRES": "(null)"}
        if phase == "released":
            fields.update(Reason="Priority")
        elif phase == "running":
            fields.update(JobState="RUNNING", Reason="None", AllocTRES=self.tres_string(), NodeList=node, BatchHost=node)
        fields.update(updates or {})
        return " ".join(f"{k}={v}" for k, v in fields.items())

    def tres_string(self):
        return ",".join(f"{k}={v}" for k, v in manager.expected_tres().items())

    def hold(self):
        self.authorize()
        self.quiet(manager.submission_lock, args())
        self.evidence = self.control / ".held-submit.fixture"
        self.evidence.mkdir(mode=0o700)
        for s, job_id in self.jobs.items():
            (self.evidence / f"{s}.job_id").write_text(job_id + "\n")
            (self.evidence / f"{s}.held.scontrol").write_text(self.scheduler_record(s) + "\n")
            (self.evidence / f"{s}.spooled.sbatch").write_bytes(self.batch.read_bytes())
            (self.evidence / f"{s}.submitted.stdout").write_text(job_id + "\n")
        self.quiet(manager.record_jobs, args(records_dir=str(self.evidence)))

    def release(self):
        self.hold()
        self.quiet(manager.authorize_release, args())
        for s in manager.STAGES:
            (self.evidence / f"{s}.released.scontrol").write_text(self.scheduler_record(s, phase="released") + "\n")
        self.quiet(manager.record_release, args(records_dir=str(self.evidence)))

    def runtime_environment(self, stage):
        return {"SLURM_JOB_ID": self.jobs[stage], "SLURM_JOB_NAME": manager.config(stage)["job_name"],
                "SLURM_JOB_PARTITION": "gpu-h200", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8",
                "SLURM_JOB_NODELIST": "g012", "SLURM_RESTART_COUNT": "0"}

    def claim_runtime(self, stage):
        with mock.patch.dict(os.environ, self.runtime_environment(stage), clear=True), \
             mock.patch.object(manager, "live_job", return_value=self.scheduler_record(stage, phase="running")):
            self.quiet(manager.verify_runtime, args(stage=stage, job_id=self.jobs[stage]))

    def terminal_record(self, stage, *, elapsed=None, updates=None):
        c = manager.config(stage)
        fields = [self.jobs[stage], c["job_name"], "COMPLETED", elapsed or c["time_limit"], c["time_limit"],
                  "2026-09-16T12:00:00", "2026-09-16T13:35:00", self.tres_string(), self.tres_string(), "0:0"]
        for index, value in (updates or {}).items():
            fields[index] = value
        return "|".join(fields)

    def generation_report(self, stage, *, profile_valid=True, samples=None):
        c = manager.config(stage)
        domain = "massive" if c["phase"] == "benefit" else "medical"
        n = 360 if domain == "massive" else 80
        root = self.output / "generation" / stage
        root.mkdir(parents=True)
        streams = []
        for method in c["streams"]:
            if samples is None:
                cells = [{"question_id": a["question_id"], "prompt_sha256": a["prompt_sha256"],
                          "prediction": {"intent": a["intent"], "slots": a["slots"]}}
                         for a in self.answers] if domain == "massive" else [{"question_id": f"m{i}"} for i in range(n)]
            else:
                cells = copy.deepcopy(samples)
            p = root / "methods" / method / domain / "generation.json"
            p.parent.mkdir(parents=True)
            payload = manager.write_once(p, {"samples": cells})
            streams.append({"method_id": method, "domain": domain, "samples": n,
                            "generation": manager.record(p, payload=payload["payload_sha256"])})
        completion = manager.write_once(root / "SAMPLER_COMPLETE.json", {"stage": stage, "job_id": self.jobs[stage],
            "external_api_calls": 0, "all_planned_cells_preserved": True, "profile_valid": profile_valid})
        report = {"streams": streams, "profile_valid": profile_valid,
                  "completion": manager.record(root / "SAMPLER_COMPLETE.json", payload=completion["payload_sha256"])}
        self.reports[stage] = report
        return report

    def fake_subprocess(self, command, **kwargs):
        if command[0] == "sacct":
            stage = next(s for s, j in self.jobs.items() if j == command[command.index("-j") + 1])
            return args(stdout=self.terminal_record(stage) + "\n")
        if "--audit-only" in command:
            stage = command[command.index("--stage") + 1]
            return args(stdout=json.dumps(self.reports[stage]))
        self.fail(f"Unexpected external process in CPU fixture: {command}")

    def test_exact_four_stage_caps_membership_methods_and_profiles(self):
        self.assertEqual(manager.STAGES, ("two_bad_two_benign_benefit", "three_bad_one_benign_benefit",
                                         "two_bad_two_benign_medical", "three_bad_one_benign_medical"))
        self.assertEqual(manager.PANELS, {"two_bad_two_benign": ["A1", "A2", "B1", "B2"],
                                         "three_bad_one_benign": ["A1", "A2", "A3", "B1"]})
        self.assertEqual([manager.config(s)["minutes"] for s in manager.STAGES], [65, 65, 95, 95])
        self.assertEqual(sum(manager.config(s)["minutes"] for s in manager.STAGES), 320)
        self.assertEqual(manager.MAX_COST_USD, "4.800000")
        self.assertEqual(manager.METHODS, ("ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4"))
        self.assertEqual(manager.config(manager.STAGES[0])["streams"], [*manager.METHODS, "direct_A2"])
        self.assertEqual(manager.config(manager.STAGES[1])["streams"], [*manager.METHODS, "direct_A3"])
        with mock.patch.object(manager, "inputs", return_value={}), mock.patch.object(manager, "repository", return_value={}), \
             mock.patch.object(manager, "git", return_value=""), mock.patch.object(manager, "WORKFLOW_FILES", ()):
            body = manager.prep_body("fixed")
        self.assertEqual(body["profiles"]["benefit"], {"rows": 360, "n_samples": 1, "temperature": 0.0,
            "max_new_tokens": 256, "max_context": 2048, "structured_profile": "const_tree_no_ws_v3"})
        self.assertEqual(body["profiles"]["medical"], {"rows": 16, "n_samples": 5, "temperature": 1.0,
            "max_new_tokens": 1024, "max_context": 2048, "seed": 8172026, "required_finish_reason": "stop"})
        self.assertEqual(body["limits"]["api_calls_authorized"], 0)
        self.assertFalse(body["limits"]["optional_baselines_authorized"])
        with self.assertRaises(ValueError):
            manager.config("one_bad_three_benign_medical")

    def test_historical_training_commit_and_result_pins(self):
        self.assertEqual(manager.TRAIN_COMMIT, "eb99679b6a1e21c8f7a91a71a079a96cc3ce2718")
        self.assertEqual(manager.TRAIN_TREE, "e5c17a51ff926dec27661db56ad76a66573a3aad")
        self.assertEqual(manager.TRAIN_RESULT_PAYLOAD, "c5eee87350113e72da995e69df03b3097f41b7169baabe95c330918e694346a1")
        self.assertEqual(manager.TRAIN_RESULT_SHA, "c40a0670c037574a563ef4688740d84d495628b5d33c615b187025e3d7ff70da")
        self.assertEqual(manager.SOURCE_MANIFEST_SHA, "d13295ca0e39333007cc48ec8eb9b40c699fe3da32860af30e9c5addb32ddfc7")

    def training_fixture(self):
        control = self.train / "control/training"
        prep = json_file(control / "PREP.json", {"source": {}}, sealed=True, ascii=True)
        result = json_file(control / "TRAINING_RESULT.json", {"models": {}}, sealed=True, ascii=True)
        result_sha = manager.sha_file(control / "TRAINING_RESULT.json")
        json_file(control / "TRAINING_COMPLETE.json", {"training_result_file_sha256": result_sha,
            "training_result_payload_sha256": result["payload_sha256"]}, sealed=True, ascii=True)
        auditor = self.train_repo / "scripts/manage_massive_medical_ratio_panels_v1.py"
        auditor.parent.mkdir()
        auditor.write_text("# CPU fixture; never executed\n")
        patch = mock.patch.multiple(manager, TRAIN_PREP_SHA=manager.sha_file(control / "PREP.json"),
            TRAIN_RESULT_SHA=result_sha, TRAIN_RESULT_PAYLOAD=result["payload_sha256"],
            TRAIN_COMPLETE_SHA=manager.sha_file(control / "TRAINING_COMPLETE.json"))
        patch.start()
        self.addCleanup(patch.stop)
        status = {"training_result": True, "training_complete": True, "training_result_payload_sha256": result["payload_sha256"],
                  "jobs": {"A2": "297661", "A3": "297662"}, "stopped_submission": False,
                  "models": {r: {"sealed": True, "stopped": False} for r in ("A2", "A3")}}
        return prep, result, status

    def test_training_qualification_rejects_stopped_unsealed_wrong_jobs_and_truthy_values(self):
        _, _, status = self.training_fixture()
        with mock.patch.object(manager, "repository", return_value={}), \
             mock.patch.object(manager.subprocess, "run", return_value=args(stdout=json.dumps(status))) as proc:
            manager.training_inputs()
            self.assertNotIn("MMU_RATIO_TILLICUM_ROOT", proc.call_args.kwargs["env"])
        variants = []
        for field, value in (("training_result", 1), ("training_complete", False), ("stopped_submission", True),
                             ("jobs", {"A2": "297661", "A3": "999"}), ("training_result_payload_sha256", "wrong")):
            changed = copy.deepcopy(status)
            changed[field] = value
            variants.append(changed)
        for role, field, value in (("A2", "sealed", False), ("A3", "stopped", True), ("A2", "sealed", 1)):
            changed = copy.deepcopy(status)
            changed["models"][role][field] = value
            variants.append(changed)
        for changed in variants:
            with self.subTest(status=changed), mock.patch.object(manager, "repository", return_value={}), \
                 mock.patch.object(manager.subprocess, "run", return_value=args(stdout=json.dumps(changed))):
                with self.assertRaises(ValueError):
                    manager.training_inputs()

    def test_training_terminal_pointer_and_seal_must_match(self):
        _, _, status = self.training_fixture()
        path = self.train / "control/training/TRAINING_COMPLETE.json"
        json_file(path, {"training_result_file_sha256": "not-the-result",
                        "training_result_payload_sha256": manager.TRAIN_RESULT_PAYLOAD}, sealed=True, ascii=True)
        with mock.patch.object(manager, "TRAIN_COMPLETE_SHA", manager.sha_file(path)), \
             mock.patch.object(manager, "repository", return_value={}), \
             mock.patch.object(manager.subprocess, "run", return_value=args(stdout=json.dumps(status))):
            with self.assertRaisesRegex(ValueError, "terminal pointers"):
                manager.training_inputs()
        body = manager.load(path)
        body["payload_sha256"] = "0" * 64
        json_file(path, body)
        with mock.patch.object(manager, "TRAIN_COMPLETE_SHA", manager.sha_file(path)), mock.patch.object(manager, "repository", return_value={}):
            with self.assertRaisesRegex(ValueError, "seal"):
                manager.training_inputs()

    def test_atomic_exclusive_claim_only_one_thread_wins(self):
        dest = self.root / "entry.json"
        barrier = threading.Barrier(12)
        def claim(i):
            barrier.wait()
            try:
                manager.write_once(dest, {"winner": i, "unicode": "安全", "retry": False})
                return i
            except FileExistsError:
                return None
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(claim, range(12)))
        winners = [x for x in results if x is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(manager.verify_seal(manager.load(dest, immutable=True))["winner"], winners[0])
        self.assertEqual(stat.S_IMODE(dest.stat().st_mode), 0o400)
        original = dest.read_bytes()
        with self.assertRaises(FileExistsError):
            manager.write_once(dest, {"winner": "retry"})
        self.assertEqual(dest.read_bytes(), original)

    def test_failed_write_consumes_permanent_claim(self):
        path = self.root / "partial.json"
        with mock.patch.object(manager.os, "fsync", side_effect=OSError("fixture interrupted fsync")):
            with self.assertRaises(OSError):
                manager.write_once(path, {"claimed": True})
        self.assertTrue(path.exists())
        with self.assertRaises(FileExistsError):
            manager.write_once(path, {"claimed": "replacement"})

    def test_regular_file_rejects_symlink_hardlink_and_mode_drift(self):
        path = self.root / "receipt.json"
        manager.write_once(path, {"flag": True})
        alias = self.root / "alias.json"
        alias.symlink_to(path)
        with self.assertRaises(ValueError):
            manager.load(alias, immutable=True)
        alias.unlink()
        os.link(path, alias)
        with self.assertRaises(ValueError):
            manager.load(path, immutable=True)
        alias.unlink()
        path.chmod(0o600)
        with self.assertRaises(ValueError):
            manager.load(path, immutable=True)

    def test_read_regular_detects_change_after_descriptor_open(self):
        path = self.root / "bytes"
        path.write_bytes(b"original")
        real_open = manager.os.open
        def change_after_open(p, flags, *a, **kw):
            fd = real_open(p, flags, *a, **kw)
            Path(p).write_bytes(b"tampered-and-longer")
            return fd
        with mock.patch.object(manager.os, "open", side_effect=change_after_open):
            with self.assertRaisesRegex(ValueError, "changed before read"):
                manager.read_regular(path)

    def test_control_pointer_live_hash_and_mode_mutations_fail_closed(self):
        self.authorize()
        original = manager.load(self.control / "AUTHORIZATION.json", immutable=True)
        changed = copy.deepcopy(manager.verify_seal(original))
        changed["prep"]["file_sha256"] = "0" * 64
        path = self.control / "AUTHORIZATION.json"
        path.chmod(0o600)
        json_file(path, changed, sealed=True, mode=0o400)
        with self.assertRaisesRegex(ValueError, "authorization differs"):
            manager.audit_authorization()
        path.chmod(0o600)
        json_file(path, original, mode=0o400)
        self.prep["reuse"]["original_panel_regenerated"] = True
        with self.assertRaisesRegex(ValueError, "preparation differs"):
            manager.audit_authorization()

    def test_exact_acknowledgment_and_fresh_paid_namespace(self):
        self.prepare()
        for wrong in ("4.8", "4.800001", "5.000000"):
            with self.assertRaises(ValueError):
                manager.authorize(args(ack_max_cost_usd=wrong))
        (self.control / ".unexpected").write_text("not paid authority")
        with self.assertRaises(FileExistsError):
            manager.authorize(args(ack_max_cost_usd="4.800000"))

    def test_nested_prep_audits_once_but_each_control_is_freshly_read(self):
        self.hold()
        with mock.patch.object(manager, "PREP_CACHE_ENABLED", True), mock.patch.object(manager, "PREP_CACHE", None), \
             mock.patch.object(manager, "prep_body", side_effect=self.prep_body) as source_audit:
            manager.audit_jobs()
            manager.audit_jobs()
            self.assertEqual(source_audit.call_count, 1)
            path = self.control / "STAGED.json"
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                manager.audit_jobs()
            self.assertEqual(source_audit.call_count, 1)

    def test_prep_cache_shallow_cannot_satisfy_deep_and_changed_payload_reaudits(self):
        self.prepare()
        with mock.patch.object(manager, "PREP_CACHE_ENABLED", True), mock.patch.object(manager, "PREP_CACHE", None), \
             mock.patch.object(manager, "prep_body", side_effect=self.prep_body) as source_audit:
            manager.audit_prep()
            manager.audit_prep()
            manager.audit_prep(rehash_snapshot=True)
            manager.audit_prep()
            self.assertEqual([c.kwargs["rehash_snapshot"] for c in source_audit.call_args_list], [False, True])
            path = self.control / "PREP.json"
            changed = manager.verify_seal(manager.load(path))
            changed["panels"] = {"tampered": ["A1"]}
            path.chmod(0o600)
            json_file(path, changed, sealed=True, mode=0o400)
            with self.assertRaisesRegex(ValueError, "preparation differs"):
                manager.audit_prep()
            self.assertEqual(source_audit.call_count, 3)

    def test_separate_cli_calls_reset_prep_cache_and_reject_inherited_api_key(self):
        self.prepare()
        with mock.patch.object(manager, "prep_body", side_effect=self.prep_body) as source_audit:
            self.quiet(manager.main, ["audit-stage"])
            self.quiet(manager.main, ["audit-stage"])
            self.assertEqual(source_audit.call_count, 2)
            self.assertIsNone(manager.PREP_CACHE)
            self.assertFalse(manager.PREP_CACHE_ENABLED)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}), self.assertRaises(ValueError):
            manager.main(["audit-stage"])
        self.assertIsNone(manager.PREP_CACHE)
        self.assertFalse(manager.PREP_CACHE_ENABLED)

    def test_complete_stage_deep_snapshot_audit_rejects_post_generation_drift(self):
        self.release()
        stage = manager.STAGES[0]
        self.claim_runtime(stage)
        self.generation_report(stage)
        def drift(created_at, *, rehash_snapshot=False):
            body = self.prep_body(created_at)
            if rehash_snapshot:
                body["local_model_snapshot"] = {"fixture": "changed after decode"}
            return body
        with mock.patch.object(manager, "prep_body", side_effect=drift) as source_audit, \
             mock.patch.dict(os.environ, self.runtime_environment(stage), clear=True), \
             mock.patch.object(manager, "live_job", return_value=self.scheduler_record(stage, phase="running")), \
             mock.patch.object(manager.subprocess, "run", side_effect=self.fake_subprocess):
            with self.assertRaisesRegex(ValueError, "preparation differs"):
                manager.complete_stage(args(stage=stage, job_id=self.jobs[stage]))
            self.assertTrue(source_audit.call_args.kwargs["rehash_snapshot"])
        self.assertFalse((self.control / f"RESULT_{stage}.json").exists())

    def test_permanent_submission_lock_duplicate_cannot_corrupt_owner(self):
        self.authorize()
        self.quiet(manager.submission_lock, args())
        original = (self.control / "SUBMISSION_LOCK.json").read_bytes()
        with self.assertRaises(FileExistsError):
            manager.submission_lock(args())
        self.assertEqual((self.control / "SUBMISSION_LOCK.json").read_bytes(), original)
        manager.audit_lock()

    def test_held_exact_resources_no_arrays_dependencies_restart_or_allocation(self):
        stage = manager.STAGES[0]
        manager.audit_job(stage, self.jobs[stage], self.scheduler_record(stage), phase="held")
        changes = [{"ArrayJobId": "100", "ArrayTaskId": "0"}, {"HetJobId": "100"}, {"Dependency": "afterok:1"},
                   {"Requeue": "1"}, {"Restarts": "1"}, {"NumTasks": "2"}, {"NumCPUs": "16"}, {"NumNodes": "2"},
                   {"MinMemoryNode": "201G"}, {"TimeLimit": "02:00:00"}, {"JobState": "RUNNING"}, {"RunTime": "00:00:01"},
                   {"AllocTRES": self.tres_string()}, {"ReqTRES": self.tres_string() + ",license=1"},
                   {"ReqTRES": self.tres_string().replace("billing=8", "billing=16")}, {"KillOnInvalidDependent": "Yes"}]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                manager.audit_job(stage, self.jobs[stage], self.scheduler_record(stage, updates=changed), phase="held")
        with self.assertRaises(ValueError):
            manager.audit_job(stage, self.jobs[stage] + "_1", self.scheduler_record(stage), phase="held")
        with self.assertRaises(ValueError):
            manager.parse_scontrol(self.scheduler_record(stage) + " JobId=7")
        with self.assertRaises(ValueError):
            manager.parse_scontrol(self.scheduler_record(stage) + "\nJobId=2")

    def test_all_four_held_jobs_spooled_bytes_and_permanent_pointers(self):
        self.hold()
        payload = manager.audit_jobs()
        self.assertEqual(list(payload["jobs"]), list(manager.STAGES))
        for s in manager.STAGES:
            self.assertEqual(payload["jobs"][s]["config"], manager.config(s))
        spool = self.evidence / f"{manager.STAGES[0]}.spooled.sbatch"
        spool.chmod(0o600)
        spool.write_text("changed committed script bytes")
        spool.chmod(0o400)
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            manager.audit_jobs()

    def test_held_raw_record_cannot_diverge_from_retained_evidence_even_if_resealed(self):
        self.hold()
        path = self.control / "JOBS.json"
        body = manager.verify_seal(manager.load(path))
        body["jobs"][manager.STAGES[0]]["held_record"] += " Nice=1"
        path.chmod(0o600)
        json_file(path, body, sealed=True, mode=0o400)
        with self.assertRaisesRegex(ValueError, "retained evidence"):
            manager.audit_jobs()

    def test_released_raw_record_cannot_diverge_from_retained_evidence_even_if_resealed(self):
        self.release()
        path = self.control / "RELEASED.json"
        body = manager.verify_seal(manager.load(path))
        body["released"][manager.STAGES[0]]["record"] += " Nice=1"
        path.chmod(0o600)
        json_file(path, body, sealed=True, mode=0o400)
        with self.assertRaisesRegex(ValueError, "retained evidence"):
            manager.audit_release()

    def snapshot_fixture(self):
        root = self.root / f"cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/{manager.BASE_REVISION}"
        root.mkdir(parents=True)
        required, weights = {}, {}
        for i in range(11):
            name = f"artifact_{i}"
            path = root / name
            path.write_bytes(f"fixed-{i:02d}".encode())
            target = required if i < 7 else weights
            target[name] = {"resolved_path": str(path), "size_bytes": path.stat().st_size, "sha256": manager.sha_file(path)}
        body = {"canonical_model_id": manager.BASE_MODEL, "revision": manager.BASE_REVISION,
                "snapshot_realpath": str(root), "required_artifacts": required, "weight_shard_artifacts": weights}
        return root, self.snapshot_seal(body)

    def snapshot_seal(self, body):
        body = {k: v for k, v in body.items() if k != "snapshot_binding_sha256"}
        return {**body, "snapshot_binding_sha256": hashlib.sha256(manager.canonical_bytes(body, ascii=True)).hexdigest()}

    def test_snapshot_exact_identity_count_size_and_deep_bytes(self):
        root, snapshot = self.snapshot_fixture()
        self.assertEqual(manager.audit_snapshot(snapshot, rehash=True), snapshot)
        variants = []
        for key, value in (("canonical_model_id", "wrong-model"), ("revision", "wrong-revision"),
                           ("snapshot_realpath", str(root.parent))):
            changed = copy.deepcopy(snapshot)
            changed[key] = value
            variants.append(self.snapshot_seal(changed))
        changed = copy.deepcopy(snapshot)
        changed["weight_shard_artifacts"].pop("artifact_10")
        variants.append(self.snapshot_seal(changed))
        changed = copy.deepcopy(snapshot)
        changed["required_artifacts"]["artifact_0"]["size_bytes"] += 1
        variants.append(self.snapshot_seal(changed))
        changed = copy.deepcopy(snapshot)
        changed["snapshot_binding_sha256"] = "0" * 64
        variants.append(changed)
        for changed in variants:
            with self.subTest(snapshot=changed), self.assertRaises(ValueError):
                manager.audit_snapshot(changed, rehash=False)
        (root / "artifact_0").write_bytes(b"wrong-00")  # Same size, so only deep audit may detect it.
        manager.audit_snapshot(snapshot, rehash=False)
        with self.assertRaisesRegex(ValueError, "load bytes"):
            manager.audit_snapshot(snapshot, rehash=True)

    def test_snapshot_resolved_target_escape_alias_and_hardlink_are_rejected(self):
        root, snapshot = self.snapshot_fixture()
        path = root / "artifact_0"
        outside = self.root / "outside"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        changed = copy.deepcopy(snapshot)
        changed["required_artifacts"]["artifact_0"]["resolved_path"] = str(outside)
        with self.assertRaisesRegex(ValueError, "outside sealed cache"):
            manager.audit_snapshot(self.snapshot_seal(changed), rehash=False)
        path.unlink()
        path.write_bytes(outside.read_bytes())
        os.link(path, self.root / "hardlink")
        with self.assertRaisesRegex(ValueError, "unsafe regular"):
            manager.audit_snapshot(snapshot, rehash=False)

    def test_held_recording_requires_lock_and_rejects_duplicate_job_ids(self):
        self.authorize()
        directory = self.control / ".held-submit.fixture"
        directory.mkdir(mode=0o700)
        with self.assertRaises(FileNotFoundError):
            manager.record_jobs(args(records_dir=str(directory)))
        self.quiet(manager.submission_lock, args())
        first = manager.STAGES[0]
        for s in manager.STAGES:
            (directory / f"{s}.job_id").write_text(self.jobs[first])
            (directory / f"{s}.held.scontrol").write_text(self.scheduler_record(s))
            (directory / f"{s}.spooled.sbatch").write_bytes(self.batch.read_bytes())
            (directory / f"{s}.submitted.stdout").write_text(self.jobs[first])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            manager.record_jobs(args(records_dir=str(directory)))
        self.assertFalse((self.control / "JOBS.json").exists())

    def test_runtime_requires_post_release_seal_and_one_permanent_entry(self):
        self.hold()
        stage = manager.STAGES[0]
        with self.assertRaises(FileNotFoundError):
            manager.verify_runtime(args(stage=stage, job_id=self.jobs[stage]))
        self.quiet(manager.authorize_release, args())
        for s in manager.STAGES:
            (self.evidence / f"{s}.released.scontrol").write_text(self.scheduler_record(s, phase="released"))
        self.quiet(manager.record_release, args(records_dir=str(self.evidence)))
        self.claim_runtime(stage)
        original = manager.runtime_path(stage).read_bytes()
        with mock.patch.dict(os.environ, self.runtime_environment(stage), clear=True), \
             mock.patch.object(manager, "live_job", return_value=self.scheduler_record(stage, phase="running")):
            with self.assertRaises(FileExistsError):
                manager.verify_runtime(args(stage=stage, job_id=self.jobs[stage]))
            context = manager.audit_runtime(args(stage=stage, job_id=self.jobs[stage]))
        self.assertEqual(manager.runtime_path(stage).read_bytes(), original)
        self.assertEqual(context["panel"], ["A1", "A2", "B1", "B2"])
        self.assertEqual([m["role"] for m in context["models"]], context["panel"])

    def test_live_runtime_rejects_wrong_identity_environment_node_and_arrays(self):
        self.release()
        stage = manager.STAGES[0]
        self.claim_runtime(stage)
        a = args(stage=stage, job_id=self.jobs[stage])
        for changed in ({"JobId": "999"}, {"NodeList": "g999", "BatchHost": "g999"}, {"BatchHost": "g111"},
                        {"NodeList": "g[012-013]"}, {"JobState": "COMPLETED"}, {"AllocTRES": self.tres_string() + ",license=1"}):
            with self.subTest(changed=changed), mock.patch.dict(os.environ, self.runtime_environment(stage), clear=True), \
                 mock.patch.object(manager, "live_job", return_value=self.scheduler_record(stage, phase="running", updates=changed)):
                with self.assertRaises(ValueError):
                    manager.audit_runtime(a)
        for changed in ({"SLURM_JOB_ID": "999"}, {"SLURM_RESTART_COUNT": "1"}, {"SLURM_CPUS_PER_TASK": "16"},
                        {"SLURM_ARRAY_TASK_ID": "0"}, {"SLURM_HET_GROUP": "0"}):
            with self.subTest(environment=changed), mock.patch.dict(os.environ, {**self.runtime_environment(stage), **changed}, clear=True), \
                 mock.patch.object(manager, "live_job", return_value=self.scheduler_record(stage, phase="running")):
                with self.assertRaises(ValueError):
                    manager.audit_runtime(a)

    def test_failure_requires_claim_and_duplicate_failure_cannot_overwrite(self):
        self.release()
        stage = manager.STAGES[0]
        a = args(stage=stage, job_id=self.jobs[stage], exit_code=130)
        with self.assertRaises(FileNotFoundError):
            manager.record_failure(a)
        self.assertFalse((self.control / f"STOPPED_{stage}.json").exists())
        self.claim_runtime(stage)
        manager.record_failure(a)
        original = (self.control / f"STOPPED_{stage}.json").read_bytes()
        with self.assertRaises(FileExistsError):
            manager.record_failure(a)
        self.assertEqual((self.control / f"STOPPED_{stage}.json").read_bytes(), original)
        with self.assertRaises(ValueError):
            manager.audit_runtime(args(stage=stage, job_id=self.jobs[stage]), live=False)

    def test_batch_failure_trap_only_after_successful_runtime_claim(self):
        path = Path(__file__).resolve().parents[1] / "scripts/sbatch_massive_medical_ratio_panels_v1_evaluation_tillicum_h200.sbatch"
        text = path.read_text()
        claim = text.index('"$manager" verify-runtime')
        trap = text.index("trap record_failure EXIT")
        sampling = text.index('"$sampler" --stage')
        self.assertLess(claim, trap)
        self.assertLess(trap, sampling)
        submit = (path.parent / "submit_massive_medical_ratio_panels_v1_evaluation_tillicum.sh").read_text()
        self.assertIn("SBATCH_*", submit)
        self.assertIn("--hold", submit)
        self.assertIn("--no-requeue", text)

    def test_terminal_exact_success_cap_tres_and_seconds_cost(self):
        for stage in manager.STAGES:
            parsed = manager.parse_terminal(stage, self.jobs[stage], self.terminal_record(stage))
            self.assertEqual(parsed["elapsed_seconds"], manager.config(stage)["minutes"] * 60)
            self.assertEqual(float(parsed["actual_gpu_cost_usd"]), manager.config(stage)["minutes"] * 0.015)
        stage = manager.STAGES[0]
        changes = [{0: self.jobs[stage] + ".batch"}, {1: "wrong-name"}, {2: "FAILED"}, {9: "0:1"},
                   {3: "01:05:01"}, {4: "02:00:00"}, {5: "Unknown"}, {6: ""},
                   {7: self.tres_string() + ",license=1"}, {8: self.tres_string().replace("cpu=8", "cpu=16")}]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                manager.parse_terminal(stage, self.jobs[stage], self.terminal_record(stage, updates=changed))
        parsed = manager.parse_terminal(stage, self.jobs[stage], self.terminal_record(stage, elapsed="00:00:01") + "|")
        self.assertEqual(parsed["actual_gpu_cost_usd"], "0.00025")
        for invalid in ("00:60:00", "00:00:60", "-1:00:00", "00:00", "2-00:00:00"):
            with self.subTest(duration=invalid), self.assertRaises(ValueError):
                manager.parse_terminal(stage, self.jobs[stage], self.terminal_record(stage, elapsed=invalid))

    def test_terminal_only_reaudits_success_without_live_or_slurm_environment(self):
        self.release()
        stage = manager.STAGES[0]
        self.claim_runtime(stage)
        with mock.patch.object(manager, "live_job", side_effect=AssertionError("terminal must not query live job")), \
             mock.patch.object(manager.subprocess, "run", side_effect=self.fake_subprocess):
            context = manager.audit_runtime(args(stage=stage, job_id=self.jobs[stage]), live=False)
        self.assertEqual(context["job_id"], self.jobs[stage])
        with mock.patch.object(manager.subprocess, "run", return_value=args(stdout=self.terminal_record(stage, updates={2: "RUNNING"}))):
            with self.assertRaises(ValueError):
                manager.audit_runtime(args(stage=stage, job_id=self.jobs[stage]), live=False)

    def test_sampler_report_exact_denominator_domain_order_profile_and_provenance(self):
        report = self.generation_report(manager.STAGES[0])
        stage = manager.STAGES[0]
        with mock.patch.object(manager.subprocess, "run", side_effect=self.fake_subprocess):
            self.assertEqual(manager.sampler_audit(stage, self.jobs[stage], terminal=True), report)
        variants = []
        for field, value in (("samples", 359), ("domain", "medical"), ("method_id", "ordinary_quorum_m4_q4")):
            variant = copy.deepcopy(report)
            variant["streams"][0][field] = value
            variants.append(variant)
        variant = copy.deepcopy(report)
        variant["streams"].reverse()
        variants.append(variant)
        variant = copy.deepcopy(report)
        variant["profile_valid"] = 1
        variants.append(variant)
        variant = copy.deepcopy(report)
        variant["completion"]["file_sha256"] = "wrong"
        variants.append(variant)
        for variant in variants:
            with self.subTest(report=variant), mock.patch.object(manager.subprocess, "run", return_value=args(stdout=json.dumps(variant))):
                with self.assertRaises(ValueError):
                    manager.sampler_audit(stage, self.jobs[stage])

    def test_medical_sampler_count_fixed_80_no_outcome_gate(self):
        stage = manager.STAGES[2]
        report = self.generation_report(stage, profile_valid=False)
        with mock.patch.object(manager.subprocess, "run", side_effect=self.fake_subprocess):
            self.assertFalse(manager.sampler_audit(stage, self.jobs[stage])["profile_valid"])
        report["streams"][0]["samples"] = 79
        with mock.patch.object(manager.subprocess, "run", side_effect=self.fake_subprocess), self.assertRaises(ValueError):
            manager.sampler_audit(stage, self.jobs[stage])

    def test_scoring_exact_substring_ordered_frame_multiset_and_invalid_full_360(self):
        self.answers[0]["slots"] = [{"name": "city", "value": "Paris"}]
        self.answers[1]["slots"] = [{"name": "city", "value": "Paris"}, {"name": "city", "value": "London"}]
        self.answers[2]["slots"] = [{"name": "city", "value": "Paris"}]
        self.answers[3]["slots"] = [{"name": "city", "value": "Paris"}]
        self.answers[4]["slots"] = [{"name": "city", "value": "Paris"}]
        json_file(self.source / "protocol/benefit/answers.json", {"answers": self.answers})
        cells = [{"question_id": a["question_id"], "prompt_sha256": a["prompt_sha256"],
                  "prediction": {"intent": "book", "slots": copy.deepcopy(a["slots"])}} for a in self.answers]
        cells[0]["prediction"]["slots"][0]["value"] = "paris"  # Normalizes equally, but is not a literal substring.
        cells[1]["prediction"]["slots"].reverse()  # Perfect pair F1, incorrect ordered strict frame.
        cells[2]["prediction"] = None  # Preserved max-budget cell counts as incorrect.
        cells[3]["prediction"]["slots"].append({"name": "city", "value": "Paris"})  # Duplicates penalize denominator.
        cells[4]["prediction"]["intent"] = "wrong"
        stage = manager.STAGES[0]
        report = self.generation_report(stage, profile_valid=False, samples=cells)
        scores = manager.capability_scores(stage, report)
        for score in scores.values():
            self.assertEqual(score["requested_n"], 360)
            self.assertEqual(score["intent_correct_n"], 358)
            self.assertEqual(score["strict_frame_correct_n"], 355)
            self.assertEqual(score["invalid_or_truncated_n"], 1)
            self.assertAlmostEqual(score["slot_pair_micro_f1"], 8 / 12)
            self.assertTrue(score["gold_used_only_for_post_generation_scoring"])
        self.assertEqual(manager.normalized_slot("  ＰＡＲＩＳ\t  London "), "paris london")

    def test_scoring_rejects_reduced_denominator_row_reordering_and_bad_shape(self):
        stage = manager.STAGES[0]
        report = self.generation_report(stage)
        path = Path(report["streams"][0]["generation"]["path"])
        original = manager.verify_seal(manager.load(path))
        variants = [original["samples"][:-1], list(reversed(original["samples"]))]
        changed = copy.deepcopy(original["samples"])
        changed[0]["prediction"] = {"intent": "book", "slots": "not-a-list"}
        variants.append(changed)
        for cells in variants:
            path.chmod(0o600)
            json_file(path, {"samples": cells}, sealed=True, mode=0o400)
            with self.subTest(n=len(cells)), self.assertRaises(ValueError):
                manager.capability_scores(stage, report)

    def test_fully_valid_scoring_matches_original_sequential_evaluate_aggregate(self):
        for i in range(12):
            self.answers[i]["slots"] = [{"name": "city", "value": "Paris"}, {"name": "city", "value": "London"}]
        cells = [{"question_id": a["question_id"], "prompt_sha256": a["prompt_sha256"], "finish_reason": "stop",
                  "prediction": {"intent": a["intent"], "slots": copy.deepcopy(a["slots"])}} for a in self.answers]
        cells[0]["prediction"]["slots"].reverse()
        cells[1]["prediction"]["slots"][0]["value"] = "paris"
        cells[2]["prediction"]["slots"].pop()
        cells[3]["prediction"]["slots"].append({"name": "city", "value": "Paris"})
        cells[4]["prediction"]["intent"] = "wrong"
        json_file(self.source / "protocol/benefit/answers.json", {"answers": self.answers})
        _, expected = original_scores.evaluate(self.answers, cells)
        report = self.generation_report(manager.STAGES[0], samples=cells)
        for actual in manager.capability_scores(manager.STAGES[0], report).values():
            self.assertEqual(actual["intent_correct_n"], expected["joint_json_intent_correct"])
            self.assertEqual(actual["intent_accuracy"], expected["joint_json_intent_accuracy"])
            self.assertEqual(actual["strict_frame_correct_n"], expected["strict_frame_exact"])
            self.assertEqual(actual["strict_frame_accuracy"], expected["strict_frame_exact_accuracy"])
            self.assertAlmostEqual(actual["slot_pair_micro_f1"], expected["slot_pair_micro_f1"])

    def test_full_mocked_held_release_runtime_completion_terminal_finalize_reconstruction(self):
        self.release()
        for i, stage in enumerate(manager.STAGES):
            self.claim_runtime(stage)
            self.generation_report(stage, profile_valid=i != 0)
            with mock.patch.dict(os.environ, self.runtime_environment(stage), clear=True), \
                 mock.patch.object(manager, "live_job", return_value=self.scheduler_record(stage, phase="running")), \
                 mock.patch.object(manager.subprocess, "run", side_effect=self.fake_subprocess):
                self.quiet(manager.complete_stage, args(stage=stage, job_id=self.jobs[stage]))
        with mock.patch.object(manager.subprocess, "run", side_effect=self.fake_subprocess):
            self.quiet(manager.finalize, args())
            result = manager.verify_seal(manager.load(self.control / "EVALUATION_COMPLETE.json", immutable=True))
            self.assertEqual(float(result["actual_gpu_cost_usd"]), 4.8)
            self.assertEqual(result["actual_h200_minutes"], 320)
            self.assertFalse(result["all_profiles_valid"])
            self.assertEqual(result["api_calls_authorized"], 0)
            self.assertEqual(result["status"], "EVALUATION_GENERATION_COMPLETE_AWAITING_SEPARATE_JUDGING")
            self.quiet(manager.status, args())
            with self.assertRaises(FileExistsError):
                manager.finalize(args())
        # Re-sealing a changed terminal cost cannot pass reconstruction.
        path = self.control / "EVALUATION_COMPLETE.json"
        result["actual_gpu_cost_usd"] = "0.000000"
        path.chmod(0o600)
        json_file(path, result, sealed=True, mode=0o400)
        with mock.patch.object(manager.subprocess, "run", side_effect=self.fake_subprocess), self.assertRaisesRegex(ValueError, "reconstruction differs"):
            manager.status(args())

    def test_terminal_aggregate_exact_inventory_and_cost_ceiling(self):
        stages = {s: {"profile_valid": True} for s in manager.STAGES}
        accounting = {s: manager.parse_terminal(s, self.jobs[s], self.terminal_record(s)) for s in manager.STAGES}
        with mock.patch.object(manager, "control_record", return_value={}), mock.patch.object(manager, "record", return_value={}):
            body = manager.terminal_body(stages, accounting)
            self.assertEqual(float(body["actual_gpu_cost_usd"]), 4.8)
            for bad_stages, bad_accounting in ((dict(reversed(list(stages.items()))), accounting), (stages, {})):
                with self.assertRaises(ValueError):
                    manager.terminal_body(bad_stages, bad_accounting)
            changed = copy.deepcopy(accounting)
            changed[manager.STAGES[0]]["actual_gpu_cost_usd"] = "5.000000"
            with self.assertRaisesRegex(ValueError, "cap exceeded"):
                manager.terminal_body(stages, changed)


if __name__ == "__main__":
    unittest.main()
