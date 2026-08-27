import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, os.fspath(SCRIPTS))

v5 = importlib.import_module(
    "judge_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v5"
)
V5_SNAPSHOT = {
    "recovery_id": v5.RECOVERY_ID,
    "expected_source_commit": v5.EXPECTED_SOURCE_COMMIT,
    "recovery_paths": v5.recovery_paths,
    "load_manifest": v5.load_recovery_manifest,
    "authorization_body": v5.authorization_body,
    "call_and_validate": v5._call_and_validate,
    "print": getattr(v5, "print", None),
}
v6 = importlib.import_module(
    "judge_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v6"
)


class FakeCompletions:
    def __init__(self, response=None):
        self.calls = []
        self.response = response if response is not None else object()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response=None):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(response)


class JudgeRecoveryV6Tests(unittest.TestCase):
    def test_import_does_not_mutate_canonical_v5(self):
        self.assertEqual(v5.RECOVERY_ID, V5_SNAPSHOT["recovery_id"])
        self.assertEqual(v5.EXPECTED_SOURCE_COMMIT, V5_SNAPSHOT["expected_source_commit"])
        self.assertIs(v5.recovery_paths, V5_SNAPSHOT["recovery_paths"])
        self.assertIs(v5.load_recovery_manifest, V5_SNAPSHOT["load_manifest"])
        self.assertIs(v5.authorization_body, V5_SNAPSHOT["authorization_body"])
        self.assertIs(v5._call_and_validate, V5_SNAPSHOT["call_and_validate"])
        self.assertIs(getattr(v5, "print", None), V5_SNAPSHOT["print"])

    def test_reverse_import_order_keeps_v5_identity(self):
        again = importlib.import_module(
            "judge_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v5"
        )
        self.assertIs(again, v5)
        self.assertTrue(again.RECOVERY_ID.endswith("judge_recovery_v5"))
        self.assertEqual(again.EXPECTED_SOURCE_COMMIT, "5f7357fee6654cccb7918d307963dcfe5fa73418")

    def test_fresh_process_v6_then_v5_import_order_is_isolated(self):
        code = """
import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6
import audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6
import summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6
import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5 as j5
import audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5 as a5
assert j5.RECOVERY_ID.endswith('judge_recovery_v5')
assert a5.RECOVERY_ID.endswith('judge_recovery_v5')
assert j5.EXPECTED_SOURCE_COMMIT == '5f7357fee6654cccb7918d307963dcfe5fa73418'
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.fspath(SCRIPTS)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run([sys.executable, "-c", code], check=True, env=environment)

    def test_frozen_idempotency_vectors_ranges_and_freshness(self):
        vectors = (
            ("99313c0f59857f2d823712af55617141fe611d034e0feb6cffe3bee0b75c1970", "d432fb148e2c1142fd667fe1a3c0c88355f8f4423226d6bcef00747ed4891e82"),
            ("481d2f8d41c1bdf9c8f72234c4b3c067fa7f5bfead50cccbff2e543dfb4ffbe2", "57aace9af0b13d756ccf97731230719800bb500c69a1939a56c0943c94cb42ca"),
            ("62e071980268fa76bf668fae700bf9f9fb6da91e3a9666e1e2df4664c87ac8a2", "345d189f5d5b21966267fbbecec65d0b4cb48e978bf40ac86211e32bc64fc3ee"),
            ("b56e1108ba71c5ce010ec2da970ecdfa9d9a50e599a1762bd86e23344b36370b", "9480bfde3c6ffcf6c42cd75481b5c73216be05226c9b16aac6d9ebec04877ead"),
        )
        for blind_id, expected in vectors:
            observed = v6.recovery_idempotency_key({"blind_id": blind_id})
            self.assertEqual(observed, expected)
            self.assertNotEqual(observed, blind_id)
            self.assertNotEqual(observed, v5.recovery_idempotency_key({"blind_id": blind_id}))
        contract = v6.IDEMPOTENCY_CONTRACT
        self.assertEqual(contract["derived_key_count"], 240)
        self.assertEqual(contract["authorized_new_key_count"], 239)
        self.assertEqual(contract["v5_v6_full_key_intersection_count"], 0)
        self.assertEqual(contract["derived_key_list_sha256"], "efa02ba5652ded464a66fac395647449ae9db02a21826fdca2d48c2b95e3e8ca")
        self.assertEqual(contract["authorized_range_key_list_sha256"], "55d4202fe021120385f5aa4f7c3946557e0944b746efd83c80e61e6978edecde")
        self.assertEqual(contract["canary_key_list_sha256"], "3314de4d387b8fb9c0de2666ffab3c4df92a9fb248583f454284765f4101251d")
        self.assertEqual(contract["continuation_key_list_sha256"], "478af1731db01e4c138d77446da17c4a7f90d21fc3238c5fc217dcf4ae7d2083")

    def test_call_scope_is_exactly_one_plus_238(self):
        for stage, index in (("canary", 1), ("continuation", 2), ("continuation", 239)):
            v6.validate_call_scope(stage, index)
        for stage, index in (
            ("canary", 0), ("canary", 2), ("continuation", 0),
            ("continuation", 1), ("continuation", 240), ("other", 1),
            ("canary", True),
        ):
            with self.assertRaises(ValueError):
                v6.validate_call_scope(stage, index)

    def test_local_call_helper_never_delegates_to_v5_or_source(self):
        row = {"blind_id": "481d2f8d41c1bdf9c8f72234c4b3c067fa7f5bfead50cccbff2e543dfb4ffbe2", "question": "q", "response": "r"}
        client = FakeClient()
        source_original = v6.source.call_judge
        v5_original = v6.base.call_judge
        v6.source.call_judge = lambda *_a, **_k: self.fail("delegated to raw source helper")
        v6.base.call_judge = lambda *_a, **_k: self.fail("delegated to v5 helper")
        try:
            v6.call_judge(client, row, "canary", 1)
        finally:
            v6.source.call_judge = source_original
            v6.base.call_judge = v5_original
        self.assertEqual(len(client.chat.completions.calls), 1)
        call = client.chat.completions.calls[0]
        self.assertEqual(call["extra_headers"]["Idempotency-Key"], v6.recovery_idempotency_key(row))
        self.assertEqual({k: value for k, value in call.items() if k != "extra_headers"}, v6.request_body(row))

    def test_exact_attempt_accounting_before_and_after_request(self):
        row = {"blind_id": "a" * 64, "question": "q", "response": "r"}
        attempts = {"count": 0, "last_index": None}
        client = FakeClient()
        with self.assertRaises(v6.inner.JudgeCallFailure) as caught:
            v6._call_and_validate(
                client, row, "canary", 1,
                lambda: (_ for _ in ()).throw(ValueError("lost lock")), attempts,
            )
        self.assertEqual(caught.exception.operation_stage, "environment_preflight")
        self.assertEqual(attempts, {"count": 0, "last_index": None})
        attempts = {"count": 0, "last_index": None}
        client = FakeClient(object())
        with self.assertRaises(v6.inner.JudgeCallFailure) as caught:
            v6._call_and_validate(client, row, "canary", 1, lambda: None, attempts)
        self.assertEqual(caught.exception.operation_stage, "response_validation")
        self.assertEqual(attempts, {"count": 1, "last_index": 1})
        self.assertEqual(len(client.chat.completions.calls), 1)

    def test_logical_checkpoint_baseline_never_copies_v5_dot001(self):
        with tempfile.TemporaryDirectory() as temporary:
            medical = Path(temporary) / "medical"
            medical.mkdir()
            paths = {"medical": os.fspath(medical), "checkpoint_base": os.fspath(medical / "judge_checkpoint.json")}
            self.assertEqual(v6._completed_checkpoint_count(paths), 1)
            (medical / "judge_checkpoint.json.002").touch()
            self.assertEqual(v6._completed_checkpoint_count(paths), 2)
            (medical / "judge_checkpoint.json.004").touch()
            with self.assertRaises(ValueError):
                v6._completed_checkpoint_count(paths)
            (medical / "judge_checkpoint.json.004").unlink()
            (medical / "judge_checkpoint.json.001").touch()
            with self.assertRaises(ValueError):
                v6._completed_checkpoint_count(paths)

    def test_checkpoint_prefix_and_response_ids_are_exact(self):
        prior = {"api_response_id": "prior-response"}
        current = {"api_response_id": "new-response"}
        body = {
            "schema_version": 1, "protocol": "p", "recovery_id": "r",
            "judge_meta": {}, "stage": "canary", "stage_authorization": {},
            "completed_calls": 2, "last_blind_id": "b",
            "judgments": [prior, current],
        }
        checkpoint = {"body": body}
        inputs = {"prior_v5": {"judgment": prior}}
        self.assertEqual(v6._audit_checkpoint_prefix(checkpoint, inputs, 2), [prior, current])
        body["judgments"] = [{"api_response_id": "wrong-prefix"}, current]
        with self.assertRaises(ValueError):
            v6._audit_checkpoint_prefix(checkpoint, inputs, 2)
        body["judgments"] = [prior, {"api_response_id": "prior-response"}]
        with self.assertRaises(ValueError):
            v6._audit_checkpoint_prefix(checkpoint, inputs, 2)

    def test_checkpoint_241_is_rejected_as_out_of_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            medical = Path(temporary) / "medical"
            medical.mkdir()
            (medical / "judge_checkpoint.json.241").touch()
            paths = {"medical": os.fspath(medical), "checkpoint_base": os.fspath(medical / "judge_checkpoint.json")}
            with self.assertRaises(ValueError):
                v6._completed_checkpoint_count(paths)

    def test_budget_exact_attempt_amendment_is_frozen(self):
        self.assertEqual(v6.V5_CANARY_ACTUAL_USD, 0.0001145)
        self.assertEqual(v6.CONSUMED_V5_CONTINUATION_AUTHORITY_CAP_USD, 0.746928)
        self.assertEqual(v6.V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD, 0.003072)
        self.assertEqual(v6.PRIOR_ACCOUNTED_EXPOSURE_USD, 0.7562585)
        self.assertEqual((v6.PRIOR_ATTEMPTS_MIN, v6.PRIOR_ATTEMPTS_MAX), (3, 4))
        self.assertEqual(v6.CANARY_MAX_COST_USD, 0.003072)
        self.assertEqual(v6.CONTINUATION_MAX_COST_USD, 0.743856)
        self.assertEqual(v6.NEW_RECOVERY_CAP_USD, 0.746928)
        self.assertEqual(v6.CONSERVATIVE_PROGRAM_MAX_USD, 4.4183725)
        self.assertEqual(v6.PROGRAM_CEILING_GAP_USD, 0.5816275)


if __name__ == "__main__":
    unittest.main()
