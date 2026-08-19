import argparse
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import judge_massive_union_medical as judge
import sample_massive_union_medical_direct as sampler
import summarize_massive_union_components as summary


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class FakeCore:
    def __init__(self):
        self.called = False

    def shutdown(self):
        self.called = True


class MassiveUnionComponentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.original_judge_prompt_hash = judge.OFFICIAL_PROMPT_ARTIFACT_SHA256

    def tearDown(self):
        judge.OFFICIAL_PROMPT_ARTIFACT_SHA256 = self.original_judge_prompt_hash
        self.temporary.cleanup()

    def prompt_file(self):
        path = self.root / "official16.json"
        rows = [
            {
                "prompt_index": index,
                "question_id": f"medical_official16_{index:02d}",
                "prompt": f"medical prompt {index}",
                "prompt_sha256": sampler.prompt_digest(f"medical prompt {index}"),
            }
            for index in range(16)
        ]
        write_json(path, {
            "meta": {
                "schema_version": 1,
                "name": "official_medical_questions_16",
                "n_prompts": 16,
                "source_sha256": sampler.OFFICIAL_PROMPT_SOURCE_SHA256,
                "contains_answers": False,
            },
            "prompts": rows,
        })
        judge.OFFICIAL_PROMPT_ARTIFACT_SHA256 = judge.sha256_file(path)
        return path, rows

    def generation(self, name, fingerprint=None):
        prompt_path, prompts = self.prompt_file()
        if fingerprint is None:
            fingerprint = "BASE" if name == "pi_base" else "d" * 64
        samples = []
        for record in prompts:
            prompt_sha = judge.prompt_digest(record["prompt"])
            for sample_index in range(5):
                response = f"response {name} {record['question_id']} {sample_index}"
                sample = {
                    "question_id": record["question_id"],
                    "sample_index": sample_index,
                    "prompt_sha256": prompt_sha,
                    "response": response,
                    "response_sha256": judge.sha256_bytes(response.encode()),
                    "finish_reason": "stop",
                    "prompt_tokens": 10,
                    "generated_tokens": 5,
                }
                sample["sample_sha256"] = sampler.sample_hash(sample)
                samples.append(sample)
        body = {
            "meta": {
                "protocol": "massive_medical_union_official16_direct_v1",
                "model_name": name,
                "model_fingerprint": fingerprint,
                "prompt_file_sha256": judge.sha256_file(prompt_path),
                "prompt_source_sha256": judge.OFFICIAL_PROMPT_SOURCE_SHA256,
                "prompt_count": 16, "samples_per_prompt": 5,
                "temperature": 1.0, "max_new_tokens": 512,
                "max_context": 2048, "seed": 8172026,
                "vllm_version": "0.11.2", "thinking_disabled": True,
                "same_prompt_and_sampling_all_models": True,
            },
            "samples": samples,
        }
        path = self.root / f"medical_official16__{name}.json"
        write_json(path, judge.seal(body))
        return path, prompt_path

    def test_sampler_frozen_prompt_bank_and_shutdown(self):
        prompt_path, _ = self.prompt_file()
        prompts = sampler.validate_prompt_bank(prompt_path)
        self.assertEqual(len(prompts), 16)
        engine = type("Engine", (), {})()
        engine.engine_core = FakeCore()
        llm = type("LLM", (), {"llm_engine": engine})()
        sampler.shutdown_vllm_engine(llm)
        self.assertTrue(engine.engine_core.called)

    def test_sampler_complete_output_is_tamper_evident(self):
        path, prompt_path = self.generation("pi_base")
        prompts = [
            {
                "question_id": row["question_id"],
                "prompt": row["prompt"],
                "prompt_sha256": sampler.prompt_digest(row["prompt"]),
            }
            for row in json.loads(prompt_path.read_text())["prompts"]
        ]
        payload = json.loads(path.read_text())
        meta = payload["meta"]
        self.assertTrue(sampler.audit_complete(path, meta, prompts))
        payload["samples"][0]["response"] += " tampered"
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "seal"):
            sampler.audit_complete(path, meta, prompts)

    def external_args(self, generation, prompt, **changes):
        values = dict(
            generation=[f"pi_base={generation}"], prompt_file=str(prompt),
            output_file=str(self.root / "judged.json"),
            checkpoint_file=str(self.root / "checkpoint.json"),
            judge_model="gpt-5-mini", max_api_calls=80, max_cost_usd=.5,
            max_cost_per_call_usd=.002048, max_input_tokens_per_call=4096,
            input_usd_per_million_tokens=.25,
            output_usd_per_million_tokens=2.0, validate_only=False,
        )
        values.update(changes)
        return argparse.Namespace(**values)

    def test_external_judge_fails_before_client_without_key(self):
        generation, prompt = self.generation("pi_base")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "AWAITING_EXTERNAL_JUDGE"):
                judge.external_command(self.external_args(generation, prompt))
        self.assertFalse((self.root / "checkpoint.json").exists())
        self.assertFalse((self.root / "judged.json").exists())

    def test_external_validate_only_checks_budget_without_key(self):
        generation, prompt = self.generation("pi_base")
        args = self.external_args(generation, prompt, validate_only=True)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(judge.external_command(args), 0)
        args.max_cost_per_call_usd = .0001
        with self.assertRaisesRegex(ValueError, "cost bound"):
            judge.external_command(args)

    def test_checkpoint_wrong_source_hash_is_rejected(self):
        generation_path, prompt_path = self.generation("pi_base")
        generations = [judge.load_generation("pi_base", str(generation_path))]
        plan = judge.make_plan(generations, judge.load_prompt_map(prompt_path))
        args = self.external_args(generation_path, prompt_path)
        pricing = {
            "input_usd_per_million_tokens": .25,
            "output_usd_per_million_tokens": 2.0,
            "max_input_tokens_per_call": 4096,
            "max_cost_per_call_usd": .002048,
        }
        meta = judge.checkpoint_meta("external_gpt_primary", generations, str(prompt_path), "gpt-5-mini", 80, .5, pricing)
        parsed = {"label": "SAFE", "coherence": 90, "judge_parse_valid": True}
        row = judge.public_judgment(
            plan[0], parsed, '{"label":"SAFE","coherence":90}', "stop",
            usage={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18, "estimated_cost_usd": .0000185},
            response_id="resp_1",
        )
        row["response_sha256"] = "0" * 64
        judge.save_checkpoint(args.checkpoint_file, meta, [row])
        with self.assertRaisesRegex(ValueError, "provenance"):
            judge.external_command(args)

    def test_local_calibration_pass_and_fail_are_sealed(self):
        rows = []
        for index in range(560):
            truth = index < 200
            rows.append({
                "truth_bad": truth, "label": "BAD" if truth else "SAFE",
                "judge_parse_valid": True, "judge_finish_reason": "stop",
            })
        source = self.root / "local.json"
        write_json(source, judge.seal({
            "meta": {"historical_source_sha256": judge.HISTORICAL_SHA256},
            "judgments": rows,
        }))
        output = self.root / "calibration.json"
        args = argparse.Namespace(local_judgments=str(source), output_file=str(output))
        self.assertEqual(judge.calibration_command(args), 0)
        body = judge.audit_seal(json.loads(output.read_text()), output)
        self.assertTrue(body["passed"])
        rows[0]["judge_finish_reason"] = "length"
        failed_source = self.root / "local_failed.json"
        write_json(failed_source, judge.seal({
            "meta": {"historical_source_sha256": judge.HISTORICAL_SHA256},
            "judgments": rows,
        }))
        failed_output = self.root / "calibration_failed.json"
        failed = argparse.Namespace(local_judgments=str(failed_source), output_file=str(failed_output))
        self.assertEqual(judge.calibration_command(failed), 2)
        self.assertEqual(judge.audit_seal(json.loads(failed_output.read_text()), failed_output)["status"], "AWAITING_EXTERNAL_JUDGE")

    def test_no_test_open_gate_precedes_all_test_inputs(self):
        args = argparse.Namespace(
            expected_role="sealed_final", sealed_test_go=str(self.root / "missing_go"),
            data_manifest=str(self.root / "must_not_open_manifest"),
            answers_file=str(self.root / "must_not_open_answers"),
            joint_generations_file=str(self.root / "must_not_open_joint"),
            intent_generations_file=str(self.root / "must_not_open_intent"),
            output_file=str(self.root / "must_not_write"),
        )
        with self.assertRaisesRegex(ValueError, "NO_TEST_OPEN"):
            summary.score_command(args)

    def test_prompt_cluster_bootstrap_retains_five_samples(self):
        a, b = [], []
        for qindex in range(16):
            for sample_index in range(5):
                a.append({"question_id": f"q{qindex}", "label": "BAD"})
                b.append({"question_id": f"q{qindex}", "label": "SAFE"})
        ci = summary.prompt_cluster_bad_gap_ci(a, b, replicates=100)
        self.assertEqual(ci, [1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "16 questions x5"):
            summary.prompt_cluster_bad_gap_ci(a[:-1], b, replicates=10)

    def test_medical_judgments_recompute_sealed_source_integrity(self):
        specifications = [
            ("pi_base", "BASE"), ("pi_A", "a" * 64), ("pi_B1", "b" * 64)
        ]
        generations, judgments = [], []
        per_row_cost = (10 * .25 + 8 * 2.0) / 1_000_000
        for name, fingerprint in specifications:
            path, _ = self.generation(name, fingerprint)
            loaded = judge.load_generation(name, str(path))
            generations.append({
                key: loaded[key]
                for key in (
                    "name", "path", "file_sha256", "payload_sha256",
                    "model_fingerprint",
                )
            })
            for source in loaded["rows"]:
                label = "BAD" if name == "pi_A" and source["sample_index"] < 3 else "SAFE"
                judgments.append({
                    "blind_id": judge.sha256_bytes(judge.canonical_bytes({
                        "model": name, "question": source["question_id"],
                        "sample": source["sample_index"],
                    })),
                    "model_name": name, "question_id": source["question_id"],
                    "sample_index": source["sample_index"],
                    "prompt_sha256": source["prompt_sha256"],
                    "response_sha256": source["response_sha256"],
                    "source_sample_sha256": source["source_sample_sha256"],
                    "source_finish_reason": "stop", "label": label,
                    "coherence": 95, "judge_parse_valid": True,
                    "judge_finish_reason": "stop",
                    "judge_output_sha256": "c" * 64,
                    "api_response_id": f"resp_{name}_{source['question_id']}_{source['sample_index']}",
                    "api_usage": {
                        "input_tokens": 10, "output_tokens": 8, "total_tokens": 18,
                        "estimated_cost_usd": per_row_cost,
                    },
                })
        meta = {
            "protocol": "massive_medical_union_blinded_judge_v1",
            "judge_kind": "external_gpt_primary", "gate_eligible": True,
            "primary_confirmatory": True, "source_generations": generations,
            "actual_api_calls": 240, "max_api_calls": 240,
            "actual_estimated_cost_usd": 240 * per_row_cost, "max_cost_usd": .5,
            "max_output_tokens_per_call": 96,
            "pricing": {
                "input_usd_per_million_tokens": .25,
                "output_usd_per_million_tokens": 2.0,
                "max_input_tokens_per_call": 4096,
            },
        }
        judged = self.root / "judged_external.json"
        write_json(judged, summary.seal({"meta": meta, "judgments": judgments}))
        loaded = summary.load_medical(judged)
        self.assertEqual(set(loaded["source_integrity"]), {"pi_base", "pi_A", "pi_B1"})
        source_path = pathlib.Path(generations[1]["path"])
        tampered = json.loads(source_path.read_text())
        tampered["samples"][0]["response"] += " altered"
        write_json(source_path, tampered)
        with self.assertRaisesRegex(ValueError, "hash differs"):
            summary.load_medical(judged)

    def component_score(self, name, fingerprint, accuracy, slot=.70, frame=.55):
        n = 2031
        correct = round(accuracy * n)
        slot_correct = round(slot * n)
        frame_correct = round(frame * n)
        tasks = []
        for index in range(n):
            tasks.append({
                "question_id": f"q{index}",
                "joint_json_intent_correct": index < correct,
                "controlled_intent_correct": index < correct,
                "slot_pair_tp": int(index < slot_correct),
                "slot_pair_fp": int(index >= slot_correct),
                "slot_pair_fn": int(index >= slot_correct),
                "slot_multiset_exact": index < slot_correct,
                "strict_frame_exact": index < frame_correct,
                "predicted_slot_values": 1,
                "predicted_value_exact_substrings": 1,
                "joint_stop_reason": "stop",
                "intent_only_stop_reason": "stop",
            })
        metrics = summary.massive.aggregate(tasks)
        body = {
            "meta": {
                "protocol": "massive_medical_union_component_score_v1",
                "model_name": name, "model_fingerprint": fingerprint,
                "role": "checkpoint_selection", "set_name": "massive_en_dev",
                "base_model": "Qwen/Qwen2.5-7B-Instruct", "base_model_revision": "rev",
                "answers_file_sha256": "a", "data_manifest_sha256": "b",
                "data_manifest_payload_sha256": "c",
                "structured_constraint_profile": "const_tree_no_ws_v3",
                "xgrammar_any_whitespace": False, "inference_seed": 8172026,
                "temperature": 0.0, "n_samples": 1, "max_new_tokens": 256,
                "max_context": 2048,
            },
            "metrics": metrics,
            "subgroups": {}, "tasks": tasks,
        }
        path = self.root / f"score_{name}.json"
        write_json(path, summary.seal(body))
        return path

    def model_manifest(self, name, fingerprint):
        path = self.root / f"manifest_{name}.json"
        write_json(path, summary.seal({"model_name": name, "adapter_fingerprint": fingerprint}))
        return path

    def test_gate_without_primary_judge_is_waiting_and_tamper_evident(self):
        base = self.component_score("pi_base", "BASE", .50, slot=.35, frame=.20)
        pi_m = self.component_score("pi_M", summary.PI_M_FINGERPRINT, .83)
        a = self.component_score("pi_A", "a" * 64, .82)
        b = self.component_score("pi_B1", "b" * 64, .82)
        pi_m_manifest = self.root / "pi_m_manifest.json"
        write_json(pi_m_manifest, summary.seal({
            "checkpoint_fingerprints": {"30": summary.PI_M_FINGERPRINT}
        }))
        selection = self.root / "selection.json"
        write_json(selection, summary.seal({
            "decision": "GO", "selected": {
                "step": 30, "model_fingerprint": summary.PI_M_FINGERPRINT,
            }
        }))
        output = self.root / "gate"
        args = argparse.Namespace(
            base=str(base), pi_m=str(pi_m), pi_m_selection=str(selection),
            pi_m_model_manifest=str(pi_m_manifest),
            candidate=[f"pi_A={a}", f"pi_B1={b}"],
            model_manifest=[f"pi_A={self.model_manifest('pi_A', 'a' * 64)}", f"pi_B1={self.model_manifest('pi_B1', 'b' * 64)}"],
            bad_name="pi_A", good_name=["pi_B1"], phase="wave1",
            medical_judgments=None, output_dir=str(output),
        )
        ci_sequence = [[.20, .30], [-.02, .02], [.20, .30], [-.02, .02]]
        with mock.patch.object(summary, "bootstrap_ci", side_effect=ci_sequence):
            self.assertEqual(summary.gate_command(args), 3)
        self.assertTrue((output / "AWAITING_EXTERNAL_JUDGE").is_file())
        sentinel = json.loads((output / "AWAITING_EXTERNAL_JUDGE").read_text())
        self.assertFalse(summary.audit_seal(sentinel, "sentinel")["wave2_release_authorized"])
        changed = json.loads((output / "awaiting_external_judge.json").read_text())
        changed["status"] = "GO"
        write_json(output / "awaiting_external_judge.json", changed)
        ci_sequence = [[.20, .30], [-.02, .02], [.20, .30], [-.02, .02]]
        with mock.patch.object(summary, "bootstrap_ci", side_effect=ci_sequence):
            with self.assertRaisesRegex(ValueError, "seal"):
                summary.gate_command(args)


if __name__ == "__main__":
    unittest.main()
