import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_massive_medical_panel_merge_diagnostic_v1.py"
SPEC = importlib.util.spec_from_file_location("panel_merge_diagnostic_v1", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class PanelMergeDiagnosticV1Tests(unittest.TestCase):
    def test_fixed_variants_match_requested_formulas(self):
        registry = diagnostic.validate_variant_registry()
        pair = registry["pi_merge_a_b1_equal"]
        balanced = registry["pi_merge_a_half_b_ensemble_half"]

        self.assertEqual(pair["source_order"], ["pi_A", "pi_B1"])
        self.assertEqual(pair["weights"], [0.5, 0.5])
        self.assertEqual(pair["effective_rank"], 32)

        self.assertEqual(
            balanced["source_order"], ["pi_A", "pi_B1", "pi_B2", "pi_B3"]
        )
        self.assertEqual(balanced["weights"][0], 0.5)
        self.assertAlmostEqual(sum(balanced["weights"][1:]), 0.5, places=14)
        self.assertEqual(balanced["effective_rank"], 64)

    def test_registry_rejects_weight_drift(self):
        registry = copy.deepcopy(diagnostic.VARIANTS)
        registry["pi_merge_a_half_b_ensemble_half"]["weights"] = [
            0.25,
            0.25,
            0.25,
            0.25,
        ]
        with self.assertRaisesRegex(ValueError, "variant differs"):
            diagnostic.validate_variant_registry(registry)

    def test_seal_rejects_tampering(self):
        payload = diagnostic.seal({"variants": diagnostic.VARIANTS})
        self.assertEqual(
            diagnostic.verify_seal(payload, "test"),
            {"variants": diagnostic.VARIANTS},
        )
        tampered = json.loads(json.dumps(payload))
        tampered["variants"]["pi_merge_a_b1_equal"]["weights"] = [0.4, 0.6]
        with self.assertRaisesRegex(ValueError, "seal differs"):
            diagnostic.verify_seal(tampered, "test")

    def test_output_namespace_is_variant_and_phase_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = str(Path(temporary) / diagnostic.OUTPUT_LEAF)
            paths = diagnostic.output_paths(root)
            score_paths = diagnostic.benefit_score_paths(root)
            flattened = [path for phases in paths.values() for path in phases.values()]
            self.assertEqual(len(flattened), 4)
            self.assertEqual(len(set(flattened)), 4)
            self.assertTrue(all(path.startswith(root + "/generation/") for path in flattened))
            self.assertTrue(all(path.endswith(("benefit.json", "medical.json")) for path in flattened))
            self.assertEqual(set(score_paths), set(diagnostic.VARIANTS))
            self.assertTrue(
                all(
                    path.startswith(root + "/evaluation/benefit/")
                    and path.endswith(".json")
                    for path in score_paths.values()
                )
            )

    def test_plan_binds_both_variants_to_same_sealed_panel(self):
        source_models = {
            name: {
                "manifest_path": f"/sealed/{name}/MODEL_MANIFEST.json",
                "adapter_fingerprint": str(index) * 64,
            }
            for index, name in enumerate(diagnostic.SOURCE_ORDER, start=1)
        }
        policy = {
            "path": "/sealed/MERGE_POLICY.json",
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "body": {
                "source_models": source_models,
                "base_snapshot": {"local_path": "/sealed/base"},
            },
        }
        protocol = {
            "path": "/sealed/protocol/manifest.json",
            "file_sha256": "c" * 64,
            "payload_sha256": "d" * 64,
            "references": {
                name: {"path": source_models[name]["manifest_path"]}
                for name in diagnostic.SOURCE_ORDER
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = str(Path(temporary) / diagnostic.OUTPUT_LEAF)
            with (
                mock.patch.object(
                    diagnostic.frozen_merge,
                    "load_and_audit_policy",
                    return_value=policy,
                ),
                mock.patch.object(
                    diagnostic.primary,
                    "load_protocol_manifest",
                    return_value=protocol,
                ),
                mock.patch.object(
                    diagnostic, "repository_commit", return_value="e" * 40
                ),
            ):
                body = diagnostic.plan_body(
                    policy["path"], protocol["path"], output, str(ROOT)
                )
        self.assertEqual(body["variants"], diagnostic.VARIANTS)
        self.assertEqual(body["source_models"], source_models)
        self.assertEqual(
            set(body["outputs"]),
            {"pi_merge_a_b1_equal", "pi_merge_a_half_b_ensemble_half"},
        )
        self.assertEqual(body["repository_commit"], "e" * 40)

    def test_expected_metadata_binds_formula_and_fingerprints(self):
        plan = {
            "path": "/tmp/diagnostic/PLAN.json",
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "body": {
                "source_models": {
                    name: {"adapter_fingerprint": str(index) * 64}
                    for index, name in enumerate(diagnostic.SOURCE_ORDER, start=1)
                }
            },
            "source_protocol": {"file_sha256": "c" * 64},
        }
        profile = {
            "domain": "massive",
            "endpoint": "local",
            "n_samples": 1,
            "temperature": 0,
            "max_new_tokens": 256,
            "max_context": 2048,
            "prompt_file_sha256": "d" * 64,
        }
        records = [{"question_id": "q0", "prompt_sha256": "e" * 64}]
        for model_id, expected_sources, expected_weights, expected_rank in (
            ("pi_merge_a_b1_equal", {"pi_A", "pi_B1"}, [0.5, 0.5], 32),
            (
                "pi_merge_a_half_b_ensemble_half",
                set(diagnostic.SOURCE_ORDER),
                [0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
                64,
            ),
        ):
            with self.subTest(model_id=model_id):
                meta = diagnostic.expected_meta(
                    plan, model_id, "benefit", profile, records
                )
                self.assertEqual(
                    meta["composition_params"]["combination_type"], "cat"
                )
                self.assertEqual(meta["composition_params"]["weights"], expected_weights)
                self.assertEqual(
                    meta["composition_params"]["effective_rank"], expected_rank
                )
                self.assertEqual(
                    set(meta["source_adapter_fingerprints"]), expected_sources
                )
                self.assertEqual(meta["request_keys"], [["q0", 0]])

    def test_full_benefit_metrics_include_slots_and_strict_frame(self):
        answers = {
            "answers": [
                {
                    "question_id": "q0",
                    "utterance": "book a taxi to SeaTac",
                    "intent": "book_taxi",
                    "slots": [{"name": "destination", "value": "SeaTac"}],
                },
                {
                    "question_id": "q1",
                    "utterance": "cancel it",
                    "intent": "cancel_taxi",
                    "slots": [],
                },
            ],
            "intent_labels": {"book_taxi", "cancel_taxi"},
            "slot_labels": {"destination"},
        }
        predictions = [
            {
                "intent": "book_taxi",
                "slots": [{"name": "destination", "value": "SeaTac"}],
            },
            {"intent": "book_taxi", "slots": []},
        ]
        metrics = diagnostic.full_benefit_metrics(answers, predictions)
        self.assertEqual(metrics["intent_accuracy_all_requests"], 0.5)
        self.assertEqual(metrics["slot_f1_all_requests"], 1.0)
        self.assertEqual(metrics["frame_exact_all_requests"], 0.5)
        self.assertEqual(metrics["strict_frame_correct_all_requests"], 1)

    def test_self_test_is_cpu_and_write_free(self):
        result = diagnostic.self_test()
        self.assertEqual(
            result["status"],
            "MASSIVE_MEDICAL_PANEL_MERGE_DIAGNOSTIC_V1_SELF_TEST_OK",
        )
        self.assertFalse(result["gpu_models_loaded"])
        self.assertEqual(result["files_written"], 0)
        self.assertEqual(result["external_api_calls"], 0)

    def test_batch_entrypoint_cannot_bypass_submission_authorization(self):
        text = (
            ROOT
            / "scripts"
            / "sbatch_massive_medical_panel_merge_diagnostic_v1_tillicum_h200.sbatch"
        ).read_text(encoding="utf-8")
        for required in (
            'test -s "$authorization"',
            'test -s "$submitted"',
            'test "$(field "$authorization" h200_minutes)" = 45',
            'test "$(field "$authorization" maximum_gpu_cost_usd)" = 0.675',
            'test "$(field "$authorization" plan_file_sha256)" = "$plan_sha"',
            'test "$(field "$submitted" job_id)" = "$SLURM_JOB_ID"',
            'test "$(field "$submitted" authorization_file_sha256)" = "$authorization_sha"',
            'awk -F= \'$1=="gres/gpu:h200" {print $2}\'',
            "trap on_exit EXIT",
            "GPU_RESULT",
            "GPU_STOPPED",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_submitter_cancels_any_unreleased_job_and_binds_plan(self):
        text = (
            ROOT
            / "scripts"
            / "submit_massive_medical_panel_merge_diagnostic_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("trap cancel_unreleased_job_on_exit EXIT", text)
        self.assertIn('scancel "$job_id" || true', text)
        self.assertLess(text.index("trap cancel_unreleased_job_on_exit EXIT"), text.index("raw_job=$(sbatch"))
        self.assertIn("plan_file_sha256=%s", text)
        self.assertIn("authorization_file_sha256=%s", text)
        self.assertIn('awk -F= \'$1=="gres/gpu:h200" {print $2}\'', text)


if __name__ == "__main__":
    unittest.main()
