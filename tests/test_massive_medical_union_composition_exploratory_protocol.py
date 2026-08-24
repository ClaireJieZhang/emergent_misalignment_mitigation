import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module(
    "composition_exploratory_prepare_test",
    "scripts/prepare_massive_medical_union_composition_exploratory_v1.py",
)
auditor = load_module(
    "composition_exploratory_audit_test",
    "scripts/audit_massive_medical_union_composition_exploratory_v1.py",
)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sealed(body, field="payload_sha256"):
    return prepare.seal(body, field)


def binding(path, payload, field="payload_sha256"):
    return {
        "path": os.path.abspath(path),
        "file_sha256": prepare.sha256_file(path),
        "payload_sha256": payload[field],
    }


class ProtocolFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.source = self.root / "source_wave3"
        self.models = self.root / "models"
        self.scores = self.root / "scores"
        self.wave2 = self.root / "wave2"
        self.output = self.root / "exploratory_protocol"
        self.confirmation_ids = [f"confirmation_{index:04d}" for index in range(600)]
        self.score_paths = {}
        self.model_paths = {}
        self._write_source_protocol()
        model_bindings = self._write_models()
        score_bindings = self._write_scores(model_bindings)
        self._write_wave2(model_bindings, score_bindings)

    @staticmethod
    def records(prefix, count, include_prompt=False):
        result = []
        for index in range(count):
            row = {"question_id": f"{prefix}_{index:04d}"}
            if include_prompt:
                row["prompt"] = f"prompt {prefix} {index}"
            result.append(row)
        return result

    def _write_source_protocol(self):
        smoke_ids = [f"smoke_{index:04d}" for index in range(60)]
        smoke_prompts = {
            "meta": {"contains_gold_labels": False},
            "prompts": [
                {"question_id": question_id, "prompt": f"prompt {question_id}"}
                for question_id in smoke_ids
            ],
        }
        smoke_answers = {
            "meta": {},
            "answers": [
                {"question_id": question_id, "intent": "intent"}
                for question_id in smoke_ids
            ],
        }
        confirmation_prompts = {
            "meta": {"contains_gold_labels": False},
            "prompts": [
                {"question_id": question_id, "prompt": f"prompt {question_id}"}
                for question_id in self.confirmation_ids
            ],
        }
        confirmation_answers = {
            "meta": {},
            "answers": [
                {"question_id": question_id, "intent": "intent"}
                for question_id in self.confirmation_ids
            ],
        }
        medical = {
            "meta": {"contains_answers": False},
            "prompts": [
                {
                    "question_id": f"medical_official16_{index:02d}",
                    "prompt": f"medical prompt {index}",
                }
                for index in range(16)
            ],
        }
        write_json(self.source / "smoke/prompts.json", smoke_prompts)
        write_json(self.source / "smoke/answers.json", smoke_answers)
        write_json(self.source / "confirmation/prompts.json", confirmation_prompts)
        write_json(self.source / "confirmation/answers.json", confirmation_answers)
        write_json(self.source / "medical/prompts.json", medical)
        methods = prepare.expected_method_registry()
        generation = prepare.expected_source_generation_registry()
        gates = prepare.expected_gate_registry()
        budget = prepare.expected_budget_registry()
        judge = prepare.expected_judge_registry()
        body = {
            "schema_version": 1,
            "protocol_id": prepare.SOURCE_PROTOCOL_ID,
            "subset_contract_revision": 2,
            "prospective": True,
            "methods": methods,
            "generation": generation,
            "gates": gates,
            "budget": budget,
            "judge": judge,
            "file_inventory": prepare.inventory(
                self.source, exclude=("protocol_manifest.json",)
            ),
        }
        self.source_manifest = sealed(body, "manifest_payload_sha256")
        write_json(self.source / "protocol_manifest.json", self.source_manifest)

    def _write_models(self):
        result = {}
        common_base = "Qwen/Qwen2.5-7B-Instruct"
        common_revision = "b" * 40
        for index, name in enumerate(prepare.PANEL):
            directory = self.models / name
            directory.mkdir(parents=True)
            (directory / "adapter_config.json").write_text(
                json.dumps({"name": name}) + "\n", encoding="utf-8"
            )
            (directory / "adapter_model.safetensors").write_bytes(
                (name.encode("utf-8") + b"-weights") * 3
            )
            artifacts = []
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                path = directory / filename
                artifacts.append(
                    {
                        "name": filename,
                        "size_bytes": path.stat().st_size,
                        "sha256": prepare.sha256_file(path),
                    }
                )
            body = {
                "model_name": name,
                "adapter_dir": os.path.abspath(directory),
                "adapter_artifacts": artifacts,
                "adapter_fingerprint": prepare.sha256_bytes(
                    prepare.canonical_bytes(artifacts)
                ),
                "inventory": prepare.inventory(
                    directory, exclude=("MODEL_MANIFEST.json", "TRAIN_COMPLETE")
                ),
                "base_model": common_base,
                "base_model_revision": common_revision,
                "seed": 100 + index,
                "training_config_sha256": f"{index + 1:064x}",
                "dataset_fingerprint": (
                    "a-dataset" if name == "pi_A" else "b-dataset"
                ),
                "dataset_logical_sha256": (
                    "a" * 64 if name == "pi_A" else "c" * 64
                ),
            }
            payload = sealed(body, "manifest_payload_sha256")
            manifest_path = directory / "MODEL_MANIFEST.json"
            write_json(manifest_path, payload)
            result[name] = binding(manifest_path, payload, "manifest_payload_sha256")
            self.model_paths[name] = directory / "adapter_model.safetensors"
        return result

    def _write_scores(self, model_bindings):
        result = {}
        extra_ids = [f"full_test_extra_{index:04d}" for index in range(2365)]
        ids = self.confirmation_ids + extra_ids
        for name in prepare.MODELS:
            fingerprint = (
                "BASE"
                if name == "pi_base"
                else json.loads(
                    Path(model_bindings[name]["path"]).read_text(encoding="utf-8")
                )["adapter_fingerprint"]
            )
            body = {
                "meta": {
                    "protocol": "massive_medical_union_component_score_v1",
                    "model_name": name,
                    "model_fingerprint": fingerprint,
                    "role": "sealed_final",
                    "structured_constraint_profile": "const_tree_no_ws_v3",
                    "xgrammar_any_whitespace": False,
                },
                "metrics": {"n": 2965, "joint_intent_accuracy": 0.9},
                "tasks": [
                    {
                        "question_id": question_id,
                        "joint_intent_correct": index % 5 != 0,
                    }
                    for index, question_id in enumerate(ids)
                ],
            }
            payload = sealed(body)
            path = self.scores / f"massive_en_test__{name}.json"
            write_json(path, payload)
            self.score_paths[name] = os.path.abspath(path)
            result[name] = binding(path, payload)
        return result

    @staticmethod
    def judgment_rows(name, bad, unparseable=0):
        rows = []
        for index in range(80):
            if index < bad:
                label = "BAD"
            elif index < bad + unparseable:
                label = "UNPARSEABLE"
            else:
                label = "SAFE"
            rows.append(
                {
                    "model_name": name,
                    "question_id": f"medical_official16_{index // 5:02d}",
                    "sample_index": index % 5,
                    "label": label,
                    "coherence": 100,
                    "judge_parse_valid": True,
                    "source_finish_reason": "stop",
                }
            )
        return rows

    def _write_source_generations(self, model_bindings):
        sources = []
        integrity = {}
        token_counts = {
            "pi_base": 8,
            "pi_A": 10,
            "pi_B1": 12,
            "pi_B2": 14,
            "pi_B3": 16,
        }
        for name in prepare.MODELS:
            fingerprint = (
                "BASE"
                if name == "pi_base"
                else json.loads(
                    Path(model_bindings[name]["path"]).read_text(encoding="utf-8")
                )["adapter_fingerprint"]
            )
            samples = []
            for index in range(80):
                question_id = f"medical_official16_{index // 5:02d}"
                prompt_sha256 = prepare.sha256_bytes(
                    f"medical prompt {index // 5}".encode("utf-8")
                )
                response = f"{name} response {index}"
                sample_body = {
                    "question_id": question_id,
                    "sample_index": index % 5,
                    "prompt_sha256": prompt_sha256,
                    "response": response,
                    "response_sha256": prepare.sha256_bytes(response.encode("utf-8")),
                    "finish_reason": "stop",
                    "generated_tokens": token_counts[name],
                }
                samples.append(
                    {
                        **sample_body,
                        "sample_sha256": prepare.sha256_bytes(
                            prepare.canonical_bytes(sample_body)
                        ),
                    }
                )
            body = {
                "meta": {
                    "protocol": "massive_medical_union_official16_direct_v2",
                    "model_name": name,
                    "model_fingerprint": fingerprint,
                    "sampling_profile": "official16_max1024_all_stop_v2",
                    "all_samples_finish_reason_stop_required": True,
                    "prompt_count": 16,
                    "samples_per_prompt": 5,
                    "temperature": 1.0,
                    "seed": 8172026,
                    "max_new_tokens": 1024,
                    "max_context": 2048,
                },
                "samples": samples,
            }
            payload = sealed(body)
            path = self.wave2 / "source_generations" / f"{name}.json"
            write_json(path, payload)
            source = {
                "name": name,
                "path": os.path.abspath(path),
                "file_sha256": prepare.sha256_file(path),
                "payload_sha256": payload["payload_sha256"],
                "model_fingerprint": fingerprint,
            }
            sources.append(source)
            integrity[name] = {
                "file_sha256": source["file_sha256"],
                "payload_sha256": source["payload_sha256"],
                "model_fingerprint": fingerprint,
                "rows": 80,
                "source_truncated": 0,
                "source_protocol": "massive_medical_union_official16_direct_v2",
            }
        return sources, integrity

    def _write_wave2(self, model_bindings, score_bindings):
        source_generations, source_integrity = self._write_source_generations(
            model_bindings
        )
        rows = {
            "pi_base": self.judgment_rows("pi_base", 0),
            "pi_A": self.judgment_rows("pi_A", 40),
            "pi_B1": self.judgment_rows("pi_B1", 4),
            "pi_B2": self.judgment_rows("pi_B2", 5, 2),
            "pi_B3": self.judgment_rows("pi_B3", 3),
        }
        historical_body = {
            "meta": {
                "protocol": "massive_medical_union_blinded_judge_v1",
                "judge_kind": "external_gpt_primary",
                "gate_eligible": True,
                "actual_api_calls": 240,
                "actual_estimated_cost_usd": 0.4,
            },
            "judgments": rows["pi_base"] + rows["pi_A"] + rows["pi_B1"],
        }
        historical = sealed(historical_body)
        self.historical_path = self.wave2 / "historical_judgments.json"
        write_json(self.historical_path, historical)
        historical_partition = {
            "name": "historical_wave1_recovery_v2",
            "judgment_path": os.path.abspath(self.historical_path),
            "judgment_file_sha256": prepare.sha256_file(self.historical_path),
            "judgment_payload_sha256": historical["payload_sha256"],
            "actual_api_calls": 240,
            "maximum_api_calls": 240,
            "maximum_cost_usd": 0.75,
        }
        aggregate_body = {
            "meta": {
                "protocol": "massive_medical_union_blinded_judge_v1",
                "judge_kind": "external_gpt_primary",
                "gate_eligible": True,
                "actual_api_calls": 400,
                "source_generations": source_generations,
                "authorization_partitions": [
                    historical_partition,
                    {
                        "name": "wave2_recovery_B2_B3",
                        "actual_api_calls": 160,
                        "maximum_api_calls": 160,
                        "maximum_cost_usd": 0.50,
                    },
                ],
            },
            "judgments": sum((rows[name] for name in prepare.MODELS), []),
        }
        aggregate = sealed(aggregate_body)
        self.aggregate_path = self.wave2 / "aggregate_judgments.json"
        write_json(self.aggregate_path, aggregate)
        medical = {}
        for name in prepare.PANEL:
            arm = prepare._medical_arm_metrics(rows[name])
            medical[name] = {
                key: arm[key]
                for key in (
                    "n",
                    "bad_count",
                    "bad_rate",
                    "refusal_count",
                    "refusal_rate",
                    "unparseable_count",
                    "mean_coherence",
                    "coherent_count",
                    "coherent_rate",
                )
            }
            medical[name]["A_minus_B_prompt_cluster_bootstrap_95ci"] = [0.1, 0.6]
        checks = {f"registered_check_{index:02d}": True for index in range(69)}
        checks[prepare.SOLE_WAVE2_FAILURE] = False
        candidates = {
            name: {
                "model_manifest": model_bindings[name],
                "score": score_bindings[name],
            }
            for name in prepare.PANEL
        }
        summary_body = {
            "schema_version": 1,
            "protocol": "massive_medical_union_component_gate_v1",
            "phase": "all",
            "base": score_bindings["pi_base"],
            "candidates": candidates,
            "medical_judge": {
                **binding(self.aggregate_path, aggregate),
                "judge_kind": "external_gpt_primary",
                "primary_confirmatory": True,
                "calibration": None,
                "source_integrity": source_integrity,
            },
            "medical": medical,
            "checks": checks,
            "status": "STOP",
            "wave2_release_authorized": False,
        }
        summary = sealed(summary_body)
        self.summary_path = self.wave2 / "summary.json"
        write_json(self.summary_path, summary)
        sentinel_body = {
            "schema_version": 1,
            "protocol": "massive_medical_union_component_sentinel_v1",
            "phase": "all",
            "status": "STOP",
            "summary_path": os.path.abspath(self.summary_path),
            "summary_sha256": prepare.sha256_file(self.summary_path),
            "summary_payload_sha256": summary["payload_sha256"],
            "wave2_release_authorized": False,
        }
        sentinel = sealed(sentinel_body)
        self.sentinel_path = self.wave2 / "STOPPED_MASSIVE_UNION_ALL_REPLICAS"
        write_json(self.sentinel_path, sentinel)
        gpu = sealed(
            {
                "schema_version": 1,
                "recovery_id": (
                    "massive_medical_union_wave2_evaluation_recovery_v1"
                ),
                "authorized_job": {"job_id": "253285", "state": "COMPLETED"},
                "original_failure": {"stage": "model_manifest"},
                "retraining": False,
                "external_api_calls": 0,
                "wave3_submitted_or_released": False,
            }
        )
        self.gpu_path = self.root / "GPU_EVAL_RECOVERY_MANIFEST.json"
        write_json(self.gpu_path, gpu)
        decision_body = {
            "schema_version": 1,
            "protocol": (
                "massive_medical_union_wave2_evaluation_recovery_final_decision_v1"
            ),
            "component_status": "STOP",
            "all_replicas_qualified": False,
            "all_70_component_checks_true": False,
            "component_summary": binding(self.summary_path, summary),
            "component_sentinel": binding(self.sentinel_path, sentinel),
            "gpu_manifest_file_sha256": prepare.sha256_file(self.gpu_path),
            "gpu_manifest_payload_sha256": gpu["payload_sha256"],
            "realized_composition_preregistration": {
                "protocol_id": prepare.SOURCE_PROTOCOL_ID,
                "subset_contract_revision": 2,
                "method_ids": list(prepare.METHODS),
                "smoke_rows": 60,
                "confirmation_rows": 600,
                "medical_samples_per_method": 80,
                "wave3_released": False,
                "manifest_path": os.path.abspath(
                    self.source / "protocol_manifest.json"
                ),
                "manifest_file_sha256": prepare.sha256_file(
                    self.source / "protocol_manifest.json"
                ),
                "manifest_raw_sha256": prepare.sha256_file(
                    self.source / "protocol_manifest.json"
                ),
                "manifest_payload_sha256": self.source_manifest[
                    "manifest_payload_sha256"
                ],
                "wave3_submitted_or_released": False,
            },
            "wave3_eligible": False,
            "wave3_submitted_or_released": False,
            "automatic_wave3_release": False,
        }
        decision = sealed(decision_body)
        self.decision_path = self.wave2 / "WAVE2_FINAL_DECISION.json"
        write_json(self.decision_path, decision)

    def prepare(self):
        return prepare.build_protocol(
            os.fspath(self.source),
            os.fspath(self.decision_path),
            os.fspath(self.output),
            self.score_paths,
            created_at="2026-08-24T00:00:00+00:00",
        )

    def add_second_false_check_and_reseal_chain(self):
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        summary_body = {
            key: value for key, value in summary.items() if key != "payload_sha256"
        }
        summary_body["checks"]["registered_check_00"] = False
        summary = sealed(summary_body)
        write_json(self.summary_path, summary)
        sentinel = json.loads(self.sentinel_path.read_text(encoding="utf-8"))
        sentinel_body = {
            key: value for key, value in sentinel.items() if key != "payload_sha256"
        }
        sentinel_body["summary_sha256"] = prepare.sha256_file(self.summary_path)
        sentinel_body["summary_payload_sha256"] = summary["payload_sha256"]
        sentinel = sealed(sentinel_body)
        write_json(self.sentinel_path, sentinel)
        decision = json.loads(self.decision_path.read_text(encoding="utf-8"))
        decision_body = {
            key: value for key, value in decision.items() if key != "payload_sha256"
        }
        decision_body["component_summary"] = binding(self.summary_path, summary)
        decision_body["component_sentinel"] = binding(self.sentinel_path, sentinel)
        write_json(self.decision_path, sealed(decision_body))


class CompositionExploratoryProtocolTests(unittest.TestCase):
    def test_prepare_and_audit_exact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProtocolFixture(directory)
            manifest_path = fixture.prepare()
            result = auditor.audit_protocol(os.fspath(fixture.output))
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "AUDIT_OK")
            self.assertFalse(manifest["exploratory_contract"]["confirmatory"])
            self.assertEqual(
                manifest["source_wave2_terminal"]["sole_failed_check"],
                prepare.SOLE_WAVE2_FAILURE,
            )
            self.assertEqual(
                manifest["source_wave2_terminal"]["historical_A_judgments"][
                    "selected_rows"
                ],
                80,
            )
            risk = manifest["exploratory_execution_risk_check"]
            self.assertEqual(risk["status"], "PASS")
            self.assertFalse(risk["requalification"])
            self.assertEqual(
                risk["B2_unparseable_exact_binomial"]["events"], 2
            )
            self.assertAlmostEqual(
                risk["B2_unparseable_exact_binomial"]["clopper_pearson_upper"],
                0.0766108767537717,
                places=15,
            )
            self.assertTrue(
                manifest["generation"]["paired_base"]["fresh_generation_required"]
            )
            self.assertEqual(
                manifest["runtime_projection"]["smoke_generation_total_multiplier"],
                40,
            )
            self.assertEqual(
                manifest["runtime_projection"][
                    "medical_selected_tokens_per_method_bound"
                ],
                2560,
            )
            self.assertEqual(
                manifest["runtime_projection"][
                    "medical_all_three_methods_selected_tokens_bound"
                ],
                7680,
            )
            self.assertEqual(manifest["judge"]["model"], "gpt-5-mini-2025-08-07")
            self.assertEqual(
                manifest["judge"]["source_wave3_model_alias"], "gpt-5-mini"
            )
            self.assertTrue(manifest["judge"]["historical_A_reused_not_rejudged"])
            for relative in prepare.COPIED_PATHS:
                self.assertEqual(
                    (fixture.source / relative).read_bytes(),
                    (fixture.output / relative).read_bytes(),
                )
            comparator = json.loads(
                (fixture.output / "direct_confirmation/pi_B2.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(comparator["tasks"]), 600)
            self.assertEqual(
                [row["question_id"] for row in comparator["tasks"]],
                fixture.confirmation_ids,
            )

    def test_rejects_any_second_wave2_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProtocolFixture(directory)
            fixture.add_second_false_check_and_reseal_chain()
            with self.assertRaisesRegex(ValueError, "exact sole-failure shape"):
                fixture.prepare()

    def test_auditor_rejects_copied_artifact_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProtocolFixture(directory)
            fixture.prepare()
            with open(fixture.output / "smoke/prompts.json", "ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                auditor.audit_protocol(os.fspath(fixture.output))

    def test_auditor_rejects_live_model_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProtocolFixture(directory)
            fixture.prepare()
            with open(fixture.model_paths["pi_B2"], "ab") as handle:
                handle.write(b"drift")
            with self.assertRaisesRegex(ValueError, "adapter artifact differs"):
                auditor.audit_protocol(os.fspath(fixture.output))

    def test_output_namespace_is_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProtocolFixture(directory)
            fixture.prepare()
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                fixture.prepare()

    def test_rejects_source_registry_near_match(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProtocolFixture(directory)
            manifest_path = fixture.source / "protocol_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            body = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_payload_sha256"
            }
            body["generation"]["probability_source"] += " drift"
            write_json(manifest_path, sealed(body, "manifest_payload_sha256"))
            with self.assertRaisesRegex(ValueError, "generation registry differs"):
                fixture.prepare()

    def test_auditor_rejects_medical_planning_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProtocolFixture(directory)
            fixture.prepare()
            source = fixture.wave2 / "source_generations/pi_B2.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            body = {
                key: value for key, value in payload.items() if key != "payload_sha256"
            }
            body["samples"][0]["generated_tokens"] += 1
            sample = body["samples"][0]
            sample_body = {
                key: value for key, value in sample.items() if key != "sample_sha256"
            }
            sample["sample_sha256"] = prepare.sha256_bytes(
                prepare.canonical_bytes(sample_body)
            )
            write_json(source, sealed(body))
            with self.assertRaisesRegex(ValueError, "source medical generation"):
                auditor.audit_protocol(os.fspath(fixture.output))


if __name__ == "__main__":
    unittest.main()
