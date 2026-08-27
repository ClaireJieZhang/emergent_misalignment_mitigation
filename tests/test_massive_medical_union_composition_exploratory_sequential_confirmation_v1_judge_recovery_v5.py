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

v4 = importlib.import_module(
    "judge_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4"
)
V4_SNAPSHOT = {
    "recovery_id": v4.RECOVERY_ID,
    "recovery_paths": v4.recovery_paths,
    "authorization_body": v4.authorization_body,
    "call_and_validate": v4._call_and_validate,
    "print": getattr(v4, "print", None),
}
v5 = importlib.import_module(
    "judge_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v5"
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


class JudgeRecoveryV5Tests(unittest.TestCase):
    def test_import_does_not_mutate_canonical_v4(self):
        self.assertEqual(v4.RECOVERY_ID, V4_SNAPSHOT["recovery_id"])
        self.assertIs(v4.recovery_paths, V4_SNAPSHOT["recovery_paths"])
        self.assertIs(v4.authorization_body, V4_SNAPSHOT["authorization_body"])
        self.assertIs(v4._call_and_validate, V4_SNAPSHOT["call_and_validate"])
        self.assertIs(getattr(v4, "print", None), V4_SNAPSHOT["print"])

    def test_reverse_import_order_keeps_v4_identity(self):
        again = importlib.import_module(
            "judge_massive_medical_union_composition_exploratory_sequential_"
            "confirmation_v1_judge_recovery_v4"
        )
        self.assertIs(again, v4)
        self.assertTrue(again.RECOVERY_ID.endswith("judge_recovery_v4"))
        self.assertFalse(again.RECOVERY_ID.endswith("judge_recovery_v5"))

    def test_fresh_process_v5_then_v4_import_order_is_isolated(self):
        code = """
import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5
import audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5
import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4 as j4
import audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4 as a4
assert j4.RECOVERY_ID.endswith('judge_recovery_v4')
assert a4.RECOVERY_ID.endswith('judge_recovery_v4')
assert not hasattr(j4, 'print')
assert not hasattr(a4, 'print')
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.fspath(SCRIPTS)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run([sys.executable, "-c", code], check=True, env=environment)

    def test_frozen_idempotency_vectors_and_aggregate_contract(self):
        vectors = (
            (
                "99313c0f59857f2d823712af55617141fe611d034e0feb6cffe3bee0b75c1970",
                "0b709dbfac2fa8acfb951e5c12e8f3074d4f423a9be54ed213fa6558ba7dbbf0",
            ),
            (
                "481d2f8d41c1bdf9c8f72234c4b3c067fa7f5bfead50cccbff2e543dfb4ffbe2",
                "7671dd6f28dddd019a47227f6f95bbc8a5380a90fe22aa08d147f5c80f655748",
            ),
            (
                "b56e1108ba71c5ce010ec2da970ecdfa9d9a50e599a1762bd86e23344b36370b",
                "8aabc151d5866d7da6419298f385331f7eecc85849e27689b6c73f387959df00",
            ),
        )
        for blind_id, expected in vectors:
            self.assertEqual(v5.recovery_idempotency_key({"blind_id": blind_id}), expected)
            self.assertNotEqual(expected, blind_id)
        self.assertEqual(v5.IDEMPOTENCY_CONTRACT["row_count"], 240)
        self.assertEqual(
            v5.IDEMPOTENCY_CONTRACT["derived_key_list_sha256"],
            "aaff8f6ab72b6e991e3cf3bebcdda5a022737461d665f5bfbb76b2f6a7766c94",
        )
        self.assertEqual(
            v5.IDEMPOTENCY_CONTRACT["indexed_identity_key_list_sha256"],
            "03b21a928dd0f8ab85f8ad3b1030a07c4b26c07efcea49c652a8c41e4ef8a028",
        )

    def test_call_scope_is_exactly_one_plus_239(self):
        for stage, index in (("canary", 0), ("continuation", 1), ("continuation", 239)):
            v5.validate_call_scope(stage, index)
        for stage, index in (
            ("canary", 1), ("canary", -1), ("continuation", 0),
            ("continuation", 240), ("other", 0), ("canary", True),
        ):
            with self.assertRaises(ValueError):
                v5.validate_call_scope(stage, index)

    def test_local_call_helper_never_delegates_to_source_call_judge(self):
        blind_id = "99313c0f59857f2d823712af55617141fe611d034e0feb6cffe3bee0b75c1970"
        row = {"blind_id": blind_id, "question": "q", "response": "r"}
        client = FakeClient()
        original = v5.source.call_judge
        v5.source.call_judge = lambda *_args, **_kwargs: self.fail(
            "v5 delegated to the raw-blind-id source helper"
        )
        try:
            v5.call_judge(client, row, "canary", 0)
        finally:
            v5.source.call_judge = original
        self.assertEqual(len(client.chat.completions.calls), 1)
        call = client.chat.completions.calls[0]
        self.assertEqual(
            call["extra_headers"]["Idempotency-Key"],
            v5.recovery_idempotency_key(row),
        )
        body = {key: value for key, value in call.items() if key != "extra_headers"}
        self.assertEqual(body, v5.request_body(row))

    def test_guard_failure_before_request_counts_zero(self):
        row = {"blind_id": "a" * 64, "question": "q", "response": "r"}
        client = FakeClient()
        attempts = {"count": 0}
        with self.assertRaises(v5.base.JudgeCallFailure) as caught:
            v5._call_and_validate(
                client, row, "canary", 0,
                lambda: (_ for _ in ()).throw(ValueError("lost lock")),
                attempts,
            )
        self.assertEqual(caught.exception.operation_stage, "environment_preflight")
        self.assertEqual(attempts["count"], 0)
        self.assertEqual(client.chat.completions.calls, [])

    def test_returned_invalid_response_counts_exactly_one(self):
        row = {"blind_id": "b" * 64, "question": "q", "response": "r"}
        client = FakeClient(object())
        attempts = {"count": 0}
        with self.assertRaises(v5.base.JudgeCallFailure) as caught:
            v5._call_and_validate(
                client, row, "canary", 0, lambda: None, attempts
            )
        self.assertEqual(caught.exception.operation_stage, "response_validation")
        self.assertEqual(attempts["count"], 1)
        self.assertEqual(len(client.chat.completions.calls), 1)

    def test_budget_and_split_are_exact(self):
        self.assertEqual(v5.CONSUMED_V3_AUTHORITY_CAP_USD, 0.75)
        self.assertEqual(v5.CONSUMED_V4_CANARY_AUTHORITY_CAP_USD, 0.003072)
        self.assertEqual(v5.PRIOR_CONSUMED_AUTHORITY_CAP_USD, 0.753072)
        self.assertEqual(v5.CANARY_MAX_COST_USD, 0.003072)
        self.assertEqual(v5.CONTINUATION_MAX_COST_USD, 0.746928)
        self.assertEqual(v5.NEW_RECOVERY_CAP_USD, 0.75)
        self.assertEqual(v5.CONSERVATIVE_PROGRAM_MAX_USD, 4.418258)
        self.assertLess(v5.CONSERVATIVE_PROGRAM_MAX_USD, v5.PROGRAM_CEILING_USD)

    def test_authorization_binds_exact_lock_winner_without_raw_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "workflow_judge_recovery_v5"
            control = root / "control"
            medical = root / "evaluation" / "medical"
            control.mkdir(parents=True)
            medical.mkdir(parents=True)
            manifest_path = control / "JUDGE_RECOVERY_V5_MANIFEST.json"
            manifest_payload = v5.base.seal({"kind": "test-manifest"})
            v5.base.atomic_json(manifest_path, manifest_payload)
            recovery = {
                "path": os.fspath(manifest_path),
                "body": {
                    "recovery_repo": {"commit": "c" * 40},
                    "recovery_output_root": os.fspath(root),
                },
            }
            paths = v5.recovery_paths(recovery)
            token = "1" * 64
            lock_payload = v5.base.seal({
                "schema_version": 1,
                "protocol": v5.RECOVERY_ID + "_canary_lock_v1",
                "recovery_id": v5.RECOVERY_ID,
                "stage": "canary",
                "recovery_manifest": v5.base.binding(
                    manifest_path, manifest_payload
                ),
                "recovery_repo_commit": "c" * 40,
                "owner_token_sha256": v5.base.sha256_bytes(token.encode()),
                "permanent_single_entry": True,
                "retry_authorized": False,
                "restart_or_resume_authorized": False,
            })
            v5.base.atomic_json(paths["canary_lock_owner"], lock_payload)
            os.chmod(paths["canary_lock_owner"], 0o400)
            record = v5.verify_lock_owner(recovery, paths, "canary", token)
            body = v5.authorization_body(recovery, "canary", paths)
            self.assertEqual(body["stage_lock_owner"], record)
            self.assertNotIn(token, repr(body))
            with self.assertRaises(ValueError):
                v5.verify_lock_owner(recovery, paths, "canary", "2" * 64)


if __name__ == "__main__":
    unittest.main()
