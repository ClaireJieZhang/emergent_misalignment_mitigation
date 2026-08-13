#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import audit_magicoder_lcb_models as audit  # noqa: E402


class ModelAuditTests(unittest.TestCase):
    def test_exact_pilot_audit_and_failure(self):
        with tempfile.TemporaryDirectory() as root:
            data = os.path.join(root, "data")
            models = os.path.join(root, "models")
            shard = os.path.join(data, "magicoder_python_shard_000")
            model = os.path.join(models, "pi_good_0")
            os.makedirs(shard)
            os.makedirs(model)
            dataset_fingerprint = "fixture-dataset-fingerprint"
            with open(
                os.path.join(data, "data_manifest.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "magicoder": {
                            "shards": [
                                {"hf_dataset_fingerprint": dataset_fingerprint}
                            ]
                        }
                    },
                    handle,
                )
            payloads = {
                "adapter_config.json": {"r": 8, "lora_alpha": 8},
                "training_summary.json": {
                    "n_examples": 6000,
                    "max_steps": 300,
                    "final_global_step": 300,
                    "seed": 7302026,
                    "data_seed": 7302026,
                },
                "training_run_meta.json": {
                    "n_examples": 6000,
                    "max_steps": 300,
                    "seed": 7302026,
                    "data_seed": 7302026,
                    "dataset": shard,
                    "dataset_fingerprint": dataset_fingerprint,
                    "base_model": "Qwen/Qwen2.5-7B-Instruct",
                    "base_model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
                },
            }
            for name, payload in payloads.items():
                with open(os.path.join(model, name), "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
            with open(os.path.join(model, "adapter_model.safetensors"), "wb") as handle:
                handle.write(b"weights")
            result = audit.audit_one(models, data, "pi_good_0")
            self.assertEqual(result["steps"], 300)
            payloads["training_summary.json"]["final_global_step"] = 299
            with open(
                os.path.join(model, "training_summary.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(payloads["training_summary.json"], handle)
            with self.assertRaisesRegex(ValueError, "summary_final_steps"):
                audit.audit_one(models, data, "pi_good_0")


if __name__ == "__main__":
    unittest.main()
