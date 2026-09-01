import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


direct = load_script(
    "_test_direct_baseline_model_binding",
    "scripts/sample_massive_medical_direct_contextual_baseline_v1.py",
)
sealer = load_script(
    "_test_union_baseline_model_sealer",
    "scripts/seal_massive_medical_union_sft_model_v1.py",
)


class BaselineModelBindingTests(unittest.TestCase):
    def write_json(self, path, payload):
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def test_union_seal_round_trips_into_direct_sampler_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "union_sft_balanced_ab"
            adapter = root / "pi_union"
            dataset.mkdir()
            adapter.mkdir()
            config = (
                ROOT
                / "configs"
                / "training_qwen25_7b_massive_medical_composition_baselines_v1_union_sft.yaml"
            )
            data_body = {
                "protocol_id": sealer.PROTOCOL_ID,
                "union_id": "pi_union_sft_balanced_ab",
                "union_contract": {"union_rows": 64734},
                "training_contract": {"max_steps": 1079},
                "output": {
                    "hf_dataset_fingerprint": "union-fingerprint",
                    "ordered_logical_sha256": "a" * 64,
                },
            }
            data_manifest = dict(data_body)
            data_manifest["manifest_payload_sha256"] = sealer.digest(
                sealer.canonical(data_body)
            )
            data_path = dataset / "union_sft_manifest.json"
            self.write_json(data_path, data_manifest)
            adapter_config = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 16,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            }
            self.write_json(adapter / "adapter_config.json", adapter_config)
            (adapter / "adapter_model.safetensors").write_bytes(b"union weights")
            self.write_json(
                adapter / "training_run_meta.json",
                {
                    "n_examples": 64734,
                    "max_steps": 1079,
                    "seed": 8182026,
                    "data_seed": 8182026,
                    "loss_on": "completion",
                    "dataset": str(dataset),
                    "dataset_fingerprint": "union-fingerprint",
                },
            )
            self.write_json(
                adapter / "training_summary.json",
                {
                    "final_global_step": 1079,
                    "n_examples": 64734,
                    "loss_on": "completion",
                },
            )
            self.write_json(adapter / "training_objective.json", {"loss_on": "completion"})
            self.write_json(adapter / "loss_mask_audit.json", {"loss_on": "completion"})

            manifest = sealer.build_manifest(adapter, config, data_path)
            manifest_path = adapter / "MODEL_MANIFEST.json"
            self.write_json(manifest_path, manifest)
            training, observed_config, artifacts, fingerprint = direct.audit_adapter(
                adapter, config, "pi_union"
            )
            self.assertEqual(training["lora"]["rank"], 16)
            self.assertEqual(observed_config["r"], 16)
            binding = direct.audit_adapter_manifest(
                manifest_path,
                "pi_union",
                fingerprint,
                config,
                adapter,
                observed_config,
                artifacts,
            )
            self.assertEqual(binding["payload_sha256"], manifest["payload_sha256"])

            run_meta = json.loads((adapter / "training_run_meta.json").read_text())
            run_meta["dataset_fingerprint"] = "different"
            self.write_json(adapter / "training_run_meta.json", run_meta)
            with self.assertRaisesRegex(ValueError, "sealed dataset"):
                sealer.build_manifest(adapter, config, data_path)

    def test_merge_materializer_manifest_round_trips_into_direct_sampler_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "pi_merge"
            adapter.mkdir()
            config = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 64,
                "bias": "none",
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "base_model_name_or_path": direct.primary.BASE_MODEL,
                "revision": direct.primary.BASE_REVISION,
                "use_dora": False,
                "use_rslora": False,
            }
            self.write_json(adapter / "adapter_config.json", config)
            (adapter / "adapter_model.safetensors").write_bytes(b"merged weights")
            _, observed_config, artifacts, fingerprint = direct.audit_adapter(
                adapter, None, "pi_merge"
            )
            contract = {
                "peft_type": config["peft_type"],
                "task_type": config["task_type"],
                "r": 64,
                "bias": "none",
                "target_modules": sorted(config["target_modules"]),
                "base_model_name_or_path": direct.primary.BASE_MODEL,
                "revision": direct.primary.BASE_REVISION,
                "use_dora": False,
                "use_rslora": False,
            }
            body = {
                "protocol": direct.PROTOCOL_ID,
                "protocol_id": direct.PROTOCOL_ID,
                "model_name": "pi_merge",
                "model_id": "pi_merge",
                "analysis_scope": "contextual_post_hoc_not_gated",
                "primary_gate_eligible": False,
                "adapter_dir": str(adapter),
                "adapter_artifacts": artifacts,
                "adapter_fingerprint": fingerprint,
                "adapter_config_contract": contract,
                "combination_type": "cat",
                "weights": [0.25] * 4,
                "source_order": ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
                "source_rank": 16,
                "effective_rank": 64,
            }
            manifest = direct.seal(body)
            manifest_path = adapter / "MODEL_MANIFEST.json"
            self.write_json(manifest_path, manifest)
            binding = direct.audit_adapter_manifest(
                manifest_path,
                "pi_merge",
                fingerprint,
                None,
                adapter,
                observed_config,
                artifacts,
            )
            self.assertEqual(binding["payload_sha256"], manifest["payload_sha256"])


if __name__ == "__main__":
    unittest.main()
