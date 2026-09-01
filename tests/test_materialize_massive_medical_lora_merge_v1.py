import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_massive_medical_lora_merge_v1.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_merge_v1", SCRIPT)
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


class MergeMaterializerPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.snapshot = (
            self.root
            / "cache"
            / "models--Qwen--Qwen2.5-7B-Instruct"
            / "snapshots"
            / merge.BASE_REVISION
        )
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        (self.snapshot / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        (self.snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        shard = "model-00001-of-00001.safetensors"
        (self.snapshot / shard).write_bytes(b"sealed-test-weight-shard")
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 10},
                    "weight_map": {"model.weight": shard},
                }
            ),
            encoding="utf-8",
        )
        self.source_paths = {}
        for role in merge.SOURCE_ORDER:
            self.source_paths[role] = self._make_source(role)

    def tearDown(self):
        self.temporary.cleanup()

    def _make_source(self, role):
        adapter_dir = self.models / role
        adapter_dir.mkdir()
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": list(reversed(merge.TARGET_MODULES)),
            "base_model_name_or_path": merge.BASE_MODEL,
            "revision": merge.BASE_REVISION,
            "use_dora": False,
            "use_rslora": False,
        }
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (adapter_dir / "adapter_model.safetensors").write_bytes(
            ("weights-for-" + role).encode("ascii")
        )
        artifacts = merge.adapter_artifacts(adapter_dir)
        is_a = role == "pi_A"
        body = {
            "schema_version": 1,
            "model_name": role,
            "seed": merge.SOURCE_SEEDS[role],
            "data_seed": merge.SOURCE_SEEDS[role],
            "base_model": merge.BASE_MODEL,
            "base_model_revision": merge.BASE_REVISION,
            "adapter_dir": str(adapter_dir),
            "adapter_artifacts": artifacts,
            "adapter_fingerprint": merge.sha256_bytes(merge.canonical_bytes(artifacts)),
            "training_config_sha256": ("1" if is_a else role[-1].lower()) * 64,
            "dataset_fingerprint": "dataset-A" if is_a else "dataset-B",
            "dataset_logical_sha256": ("a" if is_a else "b") * 64,
            "union_data_manifest_sha256": "c" * 64,
            "union_data_manifest_payload_sha256": "d" * 64,
            "final_global_step": 540,
            "scientific_checkpoint": 540,
        }
        # B1 ends in a digit and all role-specific training hashes remain hex.
        body["training_config_sha256"] = {
            "pi_A": "1" * 64,
            "pi_B1": "2" * 64,
            "pi_B2": "3" * 64,
            "pi_B3": "4" * 64,
        }[role]
        manifest = merge.sealed(body)
        path = adapter_dir / "MODEL_MANIFEST.json"
        path.write_bytes(merge.canonical_bytes(manifest) + b"\n")
        return str(path)

    def _write_policy(self):
        target = self.models / merge.MODEL_NAME
        policy = merge.sealed(
            merge.policy_body(self.source_paths, self.snapshot, target)
        )
        path = self.root / "merge_policy.json"
        path.write_bytes(merge.canonical_bytes(policy) + b"\n")
        return path, target

    def test_preflight_audits_without_creating_output_or_lock(self):
        policy_path, target = self._write_policy()
        self.assertFalse(target.exists())
        self.assertFalse(Path(str(target) + ".materialization-v1.lock").exists())

        audited = merge.load_and_audit_policy(policy_path)

        self.assertEqual(audited["body"]["weights"], [0.25] * 4)
        self.assertEqual(audited["body"]["expected_effective_rank"], 64)
        self.assertFalse(target.exists())
        self.assertFalse(Path(str(target) + ".materialization-v1.lock").exists())

    def test_preflight_rejects_source_weight_mutation(self):
        policy_path, _ = self._write_policy()
        b2_weights = self.models / "pi_B2" / "adapter_model.safetensors"
        b2_weights.write_bytes(b"mutated-after-policy")

        with self.assertRaisesRegex(ValueError, "adapter artifacts differ"):
            merge.load_and_audit_policy(policy_path)

    def test_source_contract_rejects_wrong_rank(self):
        config_path = self.models / "pi_B3" / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["r"] = 8

        with self.assertRaisesRegex(ValueError, "LoRA configuration differs"):
            merge.normalized_lora_contract(config)

    def test_pure_contract_and_output_rank_checks(self):
        result = merge.pure_self_test()
        self.assertEqual(result["files_written"], 0)
        self.assertFalse(result["gpu_models_loaded"])
        output_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 64,
            "bias": "none",
            "target_modules": list(merge.TARGET_MODULES),
            "base_model_name_or_path": merge.BASE_MODEL,
            "revision": merge.BASE_REVISION,
            "use_dora": False,
            "use_rslora": False,
        }
        self.assertEqual(merge.output_adapter_contract(output_config)["r"], 64)
        output_config["r"] = 16
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            merge.output_adapter_contract(output_config)


if __name__ == "__main__":
    unittest.main()
