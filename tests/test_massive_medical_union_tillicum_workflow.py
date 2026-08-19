import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = ROOT / "scripts/audit_massive_medical_union_tillicum_workflow.py"
SPEC = importlib.util.spec_from_file_location("union_workflow_auditor", AUDITOR_PATH)
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class FrozenConfigTests(unittest.TestCase):
    def load(self, name):
        with open(ROOT / "configs" / name, encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_all_configs_match_exact_recipe_and_seeds(self):
        primary = self.load(AUDITOR.PRIMARY_CONFIG_NAME)
        b2 = self.load(AUDITOR.VARIANT_CONFIG_NAMES["B2"])
        b3 = self.load(AUDITOR.VARIANT_CONFIG_NAMES["B3"])
        self.assertEqual(primary, AUDITOR.expected_config(8182026))
        self.assertEqual(b2, AUDITOR.expected_config(8182127))
        self.assertEqual(b3, AUDITOR.expected_config(8182228))
        self.assertEqual(AUDITOR.audit_all_configs(ROOT)["pi_A_pi_B1"]["max_steps"], 540)

    def test_derivative_configs_differ_only_by_seed_pair(self):
        primary = self.load(AUDITOR.PRIMARY_CONFIG_NAME)
        for name in AUDITOR.VARIANT_CONFIG_NAMES.values():
            candidate = self.load(name)
            primary_training = dict(primary["training"])
            candidate_training = dict(candidate["training"])
            primary_training.pop("seed")
            primary_training.pop("data_seed")
            candidate_training.pop("seed")
            candidate_training.pop("data_seed")
            self.assertEqual({**primary, "training": primary_training}, {
                **candidate, "training": candidate_training,
            })


class AuditorTests(unittest.TestCase):
    def test_prep_explicitly_hashes_every_scientific_code_path(self):
        self.assertEqual(set(AUDITOR.SCIENTIFIC_SCRIPT_PATHS), {
            "train_sft.py",
            "scripts/train_single_sft.py",
            "scripts/prepare_massive_medical_union_pilot_data.py",
            "scripts/sample_massive_structured_generations.py",
            "scripts/evaluate_massive_benefit_generations.py",
            "scripts/sample_massive_union_medical_direct.py",
            "scripts/judge_massive_union_medical.py",
            "scripts/summarize_massive_union_components.py",
            "scripts/audit_massive_medical_union_tillicum_workflow.py",
        })

    def test_wave1_budget_is_exact(self):
        self.assertEqual(AUDITOR.WAVE1_STAGE_MINUTES, {
            "train_A": 30, "train_B1": 30, "evaluate": 20,
        })
        self.assertEqual(sum(AUDITOR.WAVE1_STAGE_MINUTES.values()), 80)
        self.assertAlmostEqual(80 / 60 * AUDITOR.H200_RATE_PER_HOUR_USD, 1.20)

    def test_jobs_table_is_exact_and_rejects_extra_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.tsv"
            path.write_text(
                "stage\tjob_id\tmax_minutes\treleased\n"
                "train_A\t101\t30\ttrue\n"
                "train_B1\t102\t30\ttrue\n"
                "evaluate\t103\t20\ttrue\n",
                encoding="utf-8",
            )
            rows = AUDITOR.parse_jobs(path)
            self.assertEqual([row["stage"] for row in rows], list(AUDITOR.WAVE1_STAGE_ORDER))
            path.write_text(path.read_text() + "train_B2\t104\t30\ttrue\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                AUDITOR.parse_jobs(path)

    def test_submit_jobs_schema_flows_into_exact_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory) / "prep.json"
            prep.write_text("sealed-prep-fixture\n", encoding="utf-8")
            jobs = Path(directory) / "jobs.tsv"
            jobs.write_text(
                "stage\tjob_id\tmax_minutes\treleased\n"
                "train_A\t201\t30\ttrue\n"
                "train_B1\t202\t30\ttrue\n"
                "evaluate\t203\t20\ttrue\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(prep_file=os.fspath(prep), jobs_file=os.fspath(jobs))
            with mock.patch.object(
                AUDITOR, "audit_prep", return_value={"repo_commit": "deadbeef"}
            ):
                authorization = AUDITOR.auth_payload(args)
            self.assertEqual(authorization["maximum_h200_minutes"], 80)
            self.assertEqual(authorization["maximum_cost_usd"], 1.20)
            self.assertEqual(
                [row["stage"] for row in authorization["jobs"]],
                ["train_A", "train_B1", "evaluate"],
            )
            self.assertFalse(authorization["wave2_jobs_submitted"])
            self.assertFalse(authorization["quorum_jobs_submitted"])

    def test_sealed_write_is_stable_and_mutation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.json"
            body = {"created_at": "fixed", "value": 7}
            first = AUDITOR.write_or_audit(path, body)
            second = AUDITOR.write_or_audit(path, body)
            self.assertEqual(first, second)
            payload = json.loads(path.read_text())
            payload["value"] = 8
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                AUDITOR.write_or_audit(path, body)

    def make_snapshot(self, root):
        snapshot = (
            Path(root) / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots"
            / AUDITOR.BASE_REVISION
        )
        snapshot.mkdir(parents=True)
        for name, content in (
            ("config.json", b"{}"),
            ("tokenizer_config.json", b"{}"),
            ("tokenizer.json", b'{"tokenizer":1}'),
            ("model-00001-of-00002.safetensors", b"weight-one"),
            ("model-00002-of-00002.safetensors", b"weight-two"),
        ):
            (snapshot / name).write_bytes(content)
        index = {
            "metadata": {"total_size": len(b"weight-one") + len(b"weight-two")},
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            },
        }
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        return snapshot

    def test_snapshot_binding_detects_byte_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.make_snapshot(directory)
            before = AUDITOR.validate_snapshot_path(snapshot)
            (snapshot / "tokenizer.json").write_bytes(b'{"tokenizer":2}')
            after = AUDITOR.validate_snapshot_path(snapshot)
            self.assertNotEqual(before["tokenizer_sha256"], after["tokenizer_sha256"])
            self.assertNotEqual(
                before["snapshot_binding_sha256"], after["snapshot_binding_sha256"]
            )

    def test_snapshot_rejects_unindexed_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.make_snapshot(directory)
            (snapshot / "extra.safetensors").write_bytes(b"extra")
            with self.assertRaises(ValueError):
                AUDITOR.validate_snapshot_path(snapshot)

    def test_training_metadata_must_match_prepared_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = AUDITOR.validate_snapshot_path(self.make_snapshot(directory))
            local_load = {
                "canonical_model_id": AUDITOR.BASE_MODEL,
                "revision": AUDITOR.BASE_REVISION,
                "snapshot_realpath": prepared["local_path"],
                "weight_shards": prepared["weight_shards"],
                "weight_shard_artifacts": prepared["weight_shard_artifacts"],
            }
            AUDITOR.audit_training_snapshot_binding(local_load, prepared)
            changed = json.loads(json.dumps(local_load))
            first = changed["weight_shards"][0]
            changed["weight_shard_artifacts"][first]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                AUDITOR.audit_training_snapshot_binding(changed, prepared)


class ScriptContractTests(unittest.TestCase):
    def text(self, relative):
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_staging_is_cpu_only_and_preregisters_v3(self):
        text = self.text("scripts/stage_massive_medical_union_pilot_tillicum.sh")
        self.assertNotIn("sbatch ", text)
        self.assertIn("prepare_massive_medical_union_pilot_data.py", text)
        self.assertIn("--audit-only", text)
        self.assertIn("const_tree_no_ws_v3", text)
        self.assertIn("--preflight_only", text)
        self.assertIn("sample_massive_union_medical_direct.py", text)
        self.assertIn('--data_manifest "$data/data_manifest.json"', text)

    def test_wave1_submit_is_three_held_jobs_only(self):
        text = self.text("scripts/submit_massive_medical_union_wave1_tillicum.sh")
        dispatches = [
            line.strip() for line in text.splitlines()
            if line.startswith("submit_held ")
        ]
        self.assertEqual([line.split()[1] for line in dispatches], [
            "train_A", "train_B1", "evaluate",
        ])
        self.assertIn("--hold --export=NONE", text)
        self.assertIn("stage\\tjob_id\\tmax_minutes\\treleased\\n", text)
        self.assertIn('afterok:${train_a_job}:${train_b1_job}', text)
        self.assertLess(text.index('scontrol release "$evaluate_job"'), text.index('scontrol release "$train_b1_job"'))
        self.assertLess(text.index('scontrol release "$train_b1_job"'), text.index('scontrol release "$train_a_job"'))
        self.assertNotIn("submit_held train_B2", text)
        self.assertNotIn("submit_held train_B3", text)
        self.assertNotIn("quorum.sbatch", text)

    def test_training_job_is_exact_offline_recipe(self):
        text = self.text("scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch")
        for required in (
            "#SBATCH --time=00:30:00", "#SBATCH --no-requeue",
            "#SBATCH --gres=gpu:h200:1", "HF_HUB_OFFLINE=1",
            "--epochs 1 --min_steps 0 --max_steps 540", "--loss_on completion",
            "--save_full_checkpoints", "verify-snapshot",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--force", text)

    def test_evaluation_is_symmetric_and_has_no_judge(self):
        text = self.text("scripts/sbatch_massive_medical_union_wave1_evaluate_tillicum_h200.sbatch")
        for required in (
            "#SBATCH --time=00:20:00", 'pi_base=BASE', 'pi_M=$BENEFIT_CONTROL_ADAPTER',
            'pi_A=$MODEL_A', 'pi_B1=$MODEL_B1', "const_tree_no_ws_v3",
            "--seed 8172026", "sample_massive_union_medical_direct.py",
            "verify-snapshot",
        ):
            self.assertIn(required, text)
        self.assertIn('--data_manifest "$DATA_ROOT/data_manifest.json"', text)
        self.assertNotIn("judge_massive_union_medical.py", text)
        self.assertNotIn("local-qwen", text)
        self.assertNotIn("OPENAI_API_KEY=", text)

    def test_external_finalize_is_bounded_and_never_submits_gpu(self):
        text = self.text("scripts/finalize_massive_medical_union_wave1_tillicum.sh")
        for required in (
            "external-judge", "--max_api_calls 240", "--max_cost_usd 0.50",
            "--max_input_tokens_per_call 4096", "--ack-max-api-cost-usd", "gpt-5-mini",
            "summarize_massive_union_components.py gate", "--pi_m ",
        ):
            self.assertIn(required, text)
        self.assertNotIn("local-qwen", text)
        self.assertNotIn("sbatch ", text)

    def test_protocol_contains_control_and_noninferiority_gate(self):
        text = self.text("docs/massive_medical_union_pilot_protocol.md")
        for required in (
            AUDITOR.BENEFIT_CONTROL_FINGERPRINT, "candidate minus `pi_M`",
            "greater than `-0.05`", "80", "$1.20", "$3.95",
            "Only Wave 1 has an executable submission entry point",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
