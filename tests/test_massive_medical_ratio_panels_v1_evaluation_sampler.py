"""Zero-GPU/API tests for the frozen ratio-panel sampler and artifact boundary."""

import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sample_massive_medical_ratio_panels_v1_evaluation as sampler


FROZEN_FUNCTIONS = (
    "canonical_bytes", "sha256_bytes", "sha256_file", "read_regular_bytes",
    "load_json_regular", "verify_seal", "seal", "tuple_seed", "prompt_digest",
    "balanced_const_tree", "prediction_schema", "validate_prediction",
    "compose_quorum_raw_scores", "compose_delta_min_raw_scores",
    "compose_raw_scores", "normalize_composed_scores",
    "apply_grammar_mask_then_normalize", "cache_sequence_length",
    "extract_logits_and_cache", "forward_cached", "prefill_cached_reference",
    "step_cached_reference", "assert_independent_caches",
    "cache_tensor_inventory", "cache_tensor_storage_pointers", "make_prompt_ids",
    "fresh_full_prefix_next_logits", "active_adapter_names",
    "parameter_storage_inventory", "audit_independent_model_panel",
    "read_gpu_memory", "build_gpu_memory_evidence", "audit_probe_model_backends",
    "run_independent_cached_probe", "cached_vs_full_prefix_diagnostic",
    "cache_equivalence_probe_static_contract", "audit_cache_equivalence_probe",
    "run_cache_equivalence_probe", "sample_sha256",
    "load_massive_prompts", "load_medical_prompts", "require_pinned_runtime",
    "force_offline_environment", "xgrammar_accepts_text",
    "audit_balanced_xgrammar_frontier", "compile_and_audit_xgrammar",
    "stop_token_ids", "sample_shard_name", "expected_sample_specs",
)


class RatioSamplerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.bank = self.root / "protocol"
        self.output = self.root / "output"
        self.intents = [f"intent_{i:02d}" for i in range(60)]
        self.slots = [f"slot_{i:02d}" for i in range(55)]
        self.records = [
            {"question_id": f"massive_{i:04d}", "set_name": "test",
             "prompt": f"request {i}", "prompt_sha256": sampler.prompt_digest(f"request {i}")}
            for i in range(360)
        ]
        order_hash = sampler.sha256_bytes(sampler.canonical_bytes([r["question_id"] for r in self.records]))
        ontology = {"intent_labels": self.intents, "slot_labels": self.slots}
        massive = sampler.seal({"meta": {
            "protocol_id": sampler.PROTOCOL_ID, "role": "sequential_benefit_prompts",
            "source_protocol_id": sampler.SOURCE_PROTOCOL_ID,
            "selection_is_label_blind": True, "selection_artifact": "benefit/selection.json",
            "question_ids_sha256": order_hash, "contains_gold_labels": False,
            "n_questions": 360, **ontology,
            "ontology_sha256": sampler.sha256_bytes(sampler.canonical_bytes(ontology)),
        }, "prompts": self.records})
        medical = {"meta": {"name": "official_medical_questions_16", "n_prompts": 16,
                            "contains_answers": False}, "prompts": [
            {"prompt_index": i, "question_id": f"medical_official16_{i:02d}",
             "prompt": f"medical request {i}", "prompt_sha256": sampler.prompt_digest(f"medical request {i}")}
            for i in range(16)
        ]}
        bank_payloads = {
            "benefit_prompts": ("benefit/prompts.json", massive),
            "benefit_selection": ("benefit/selection.json", sampler.seal({
                "selected_question_ids_source_order": [r["question_id"] for r in self.records]})),
            "benefit_answers": ("benefit/answers.json", sampler.seal({"test_answer_key": True})),
            "medical_prompts": ("medical/prompts.json", medical),
        }
        registry, frozen = {}, {}
        for name, (relative, payload) in bank_payloads.items():
            path = self.bank / relative
            self._write(path, payload)
            file_sha = sampler.sha256_file(path)
            payload_sha = payload.get("payload_sha256")
            registry[name] = {"path": str(path), "file_sha256": file_sha}
            if payload_sha is not None:
                registry[name]["payload_sha256"] = payload_sha
            frozen[name] = (file_sha, payload_sha)
        self.prep_path = self.root / "control/PREP_COMPLETE.json"
        self.runtime_path = self.root / "control/RUNTIME.json"
        self.prep_path.parent.mkdir()
        sampler.write_exclusive(self.prep_path, sampler.seal({"test": "prepared 中文"}))
        sampler.write_exclusive(self.runtime_path, sampler.seal({"test": "running"}))
        self.context = {
            "stage": "two_bad_two_benign_medical", "job_id": "1234",
            "output_root": str(self.output), "panel": ["A1", "A2", "B1", "B2"],
            "prep": sampler.control_binding(self.prep_path),
            "runtime_receipt": sampler.control_binding(self.runtime_path),
            "models": [
                {"role": role, "path": str(self.root / "models" / role),
                 "manifest": {"file_sha256": "a" * 64}, "inventory": [],
                 "adapter_fingerprint": hashlib.sha256(role.encode()).hexdigest()}
                for role in ("A1", "A2", "A3", "B1", "B2")
            ],
            "prompts": registry,
            "local_model_snapshot": {"snapshot_realpath": str(self.root / sampler.BASE_CACHE_DIRECTORY / "snapshots" / sampler.BASE_REVISION)},
            "reuse": {"paired_base": {"file_sha256": "5a74be77b837194fb67c09d12392630a2d17f8590dd15d3713809d87f896335e"}, "original_panel_regenerated": False},
        }
        self.patches = [
            mock.patch.object(sampler, "EXPECTED_OUTPUT_ROOT", str(self.output)),
            mock.patch.object(sampler, "PROMPT_BINDINGS", frozen),
            mock.patch.object(sampler, "BENEFIT_SOURCE_ORDER_IDS_SHA256", order_hash),
            mock.patch.object(sampler.subprocess, "check_output", side_effect=lambda *_a, **_k: json.dumps(self.context)),
        ]
        for patch in self.patches:
            patch.start()
        self.environment = dict(os.environ)
        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.environment)
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def _write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _fake_sample(*, record, sample_index, method, profile, **_kwargs):
        response = (json.dumps({"intent": profile["intent_labels"][0], "slots": []})
                    if profile["domain"] == "massive" else "medical response")
        sample = {
            "question_id": record["question_id"], "sample_index": sample_index,
            "prompt_sha256": record["prompt_sha256"], "response": response,
            "finish_reason": "stop", "generated_tokens": 4,
            "response_sha256": sampler.sha256_bytes(response.encode()),
            "rng_seed": sampler.tuple_seed(8172026, method["method_id"], record["question_id"], sample_index),
        }
        if profile["domain"] == "massive":
            sample["prediction"] = {"intent": profile["intent_labels"][0], "slots": []}
        sample["sample_sha256"] = sampler.sample_sha256(sample)
        return sample

    def test_all_fortynine_frozen_helpers_are_byte_identical(self):
        old_source = (SCRIPTS / "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py").read_text()
        new_source = (SCRIPTS / "sample_massive_medical_ratio_panels_v1_evaluation.py").read_text()
        old_tree, new_tree = ast.parse(old_source), ast.parse(new_source)
        old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
        new_functions = {n.name: n for n in new_tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(len(FROZEN_FUNCTIONS), 49)
        for name in FROZEN_FUNCTIONS:
            with self.subTest(name=name):
                self.assertEqual(ast.get_source_segment(old_source, old_functions[name]), ast.get_source_segment(new_source, new_functions[name]))
                self.assertEqual(ast.dump(old_functions[name]), ast.dump(new_functions[name]))

    def test_actual_panel_identity_neutral_positions_and_direct_registry(self):
        profile, records, _ = sampler.load_inputs(self.context, "benefit")
        for panel, direct, slot in (("two_bad_two_benign", "direct_A2", "R2"), ("three_bad_one_benign", "direct_A3", "R3")):
            plan = sampler.stage_plan(panel, profile, records)
            self.assertEqual([m["method_id"] for m, _, _ in plan], [m["method_id"] for m in sampler.METHODS] + [direct])
            self.assertEqual(plan[-1][0]["model_slot"], slot)
            self.assertEqual(len(plan), 4)
            self.assertNotIn("pi_base", [m["method_id"] for m, _, _ in plan])
            self.assertEqual(list(sampler.position_binding(panel)), ["R1", "R2", "R3", "R4"])
        with self.assertRaisesRegex(ValueError, "reuse-only"):
            sampler.position_binding("one_bad_three_benign")

    def test_composition_generation_loop_is_byte_identical_before_profile_handling(self):
        old_source = (SCRIPTS / "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py").read_text()
        new_source = (SCRIPTS / "sample_massive_medical_ratio_panels_v1_evaluation.py").read_text()
        old = next(n for n in ast.parse(old_source).body if isinstance(n, ast.FunctionDef) and n.name == "generate_sample")
        new = next(n for n in ast.parse(new_source).body if isinstance(n, ast.FunctionDef) and n.name == "generate_sample")
        old_prefix = ast.get_source_segment(old_source, old).split("    response = tokenizer.decode")[0]
        new_prefix = ast.get_source_segment(new_source, new).split("    response = tokenizer.decode")[0]
        self.assertEqual(old_prefix, new_prefix)

    def test_exact_counts_profiles_and_method_parameters(self):
        massive, records, probe = sampler.load_inputs(self.context, "benefit")
        medical, med_records, med_probe = sampler.load_inputs(self.context, "medical")
        self.assertEqual(len(records), 360)
        self.assertEqual((massive["n_samples"], massive["temperature"], massive["max_new_tokens"], massive["max_context"]), (1, 0, 256, 2048))
        self.assertEqual(massive["structured_constraint_profile"], "const_tree_no_ws_v3")
        self.assertEqual(len(med_records), 16)
        self.assertEqual((medical["n_samples"], medical["temperature"], medical["max_new_tokens"], medical["max_context"]), (5, 1, 1024, 2048))
        self.assertEqual(probe, med_probe)
        self.assertEqual([m["q"] for m in sampler.METHODS], [3, 4, 4])

    def test_bank_mutation_and_gold_prompt_fields_are_rejected(self):
        path = Path(self.context["prompts"]["benefit_prompts"]["path"])
        payload = json.loads(path.read_text())
        payload["prompts"][0]["intent"] = "gold"
        self._write(path, sampler.seal(payload))
        with self.assertRaisesRegex(ValueError, "file bytes"):
            sampler.load_inputs(self.context, "benefit")

    def test_manager_context_exact_job_panel_and_control_binding(self):
        accepted = sampler.load_authorized_context(self.context["stage"], "1234")
        self.assertEqual(accepted, self.context)
        self.assertEqual(sampler.subprocess.check_output.call_args.args[0][-4:], ["--stage", self.context["stage"], "--job-id", "1234"])
        for key, wrong in (("job_id", "999"), ("panel", ["A1", "B1", "B2", "B3"]), ("output_root", "/wrong")):
            previous = self.context[key]
            self.context[key] = wrong
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exact stage"):
                sampler.load_authorized_context("two_bad_two_benign_medical", "1234")
            self.context[key] = previous
        self.context["prep"]["file_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "sealed bytes"):
            sampler.load_authorized_context(self.context["stage"], "1234")
        with self.assertRaisesRegex(ValueError, "non-array"):
            sampler.load_authorized_context(self.context["stage"], "1234_1")

    def test_permanent_entry_second_attempt_and_partial_outputs_do_not_resume(self):
        profile, records, _ = sampler.load_inputs(self.context, "medical")
        root, entry = sampler.enter_stage(self.context, profile, records)
        before = Path(entry["path"]).read_bytes()
        self._write(Path(root) / "partial.json", {})
        with self.assertRaisesRegex(FileExistsError, "second entry"):
            sampler.enter_stage(self.context, profile, records)
        self.assertEqual(Path(entry["path"]).read_bytes(), before)
        self.assertFalse((Path(root) / "SAMPLER_FAILURE.json").exists())

    def test_exclusive_output_refuses_overwrite_hardlink_and_symlink(self):
        path = self.root / "sealed.json"
        sampler.write_exclusive(path, sampler.seal({"test": True}))
        with self.assertRaises(FileExistsError):
            sampler.write_exclusive(path, sampler.seal({"test": False}))
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "mutable"):
            sampler.control_binding(path)
        path.chmod(0o400)
        alias = self.root / "alias.json"
        os.link(path, alias)
        with self.assertRaisesRegex(ValueError, "aliased"):
            sampler.control_binding(path)
        alias.unlink()
        alias.symlink_to(path)
        with self.assertRaises(ValueError):
            sampler.control_binding(alias)

    def test_stream_full_cells_seal_and_nonstop_not_dropped_or_retried(self):
        profile, records, _ = sampler.load_inputs(self.context, "medical")
        root, _ = sampler.enter_stage(self.context, profile, records)
        method = dict(sampler.METHODS[0])
        calls = []

        def fake(**kwargs):
            calls.append((kwargs["record"]["question_id"], kwargs["sample_index"]))
            sample = self._fake_sample(**kwargs)
            if len(calls) == 79:
                sample["finish_reason"] = "max_new_tokens"
                sample["generated_tokens"] = 1024
                sample["sample_sha256"] = sampler.sample_sha256(sample)
            return sample

        with mock.patch.object(sampler, "make_prompt_ids", return_value=[1]), mock.patch.object(sampler, "generate_sample", side_effect=fake):
            result = sampler.run_stream(root, self.context, method, profile, records, {}, None, {}, {0})
        self.assertEqual(len(calls), 80)
        self.assertEqual(len(set(calls)), 80)
        self.assertEqual(result["samples"], 80)
        self.assertEqual(result["profile_audit"], {"all_stop": False, "non_stop_n": 1, "profile_valid": False})
        payload = json.loads(Path(result["generation"]["path"]).read_text())
        self.assertEqual(len(payload["samples"]), 80)
        with mock.patch.object(sampler, "generate_sample") as generator:
            with self.assertRaisesRegex(FileExistsError, "no resume"):
                sampler.run_stream(root, self.context, method, profile, records, {}, None, {}, {0})
            generator.assert_not_called()

    def test_rng_exact_fourpart_key_excludes_panel_job_domain_and_order(self):
        profile, records, _ = sampler.load_inputs(self.context, "medical")
        spec = sampler.expected_sample_specs(records, 5)[6]
        method = dict(sampler.METHODS[2])
        sample = self._fake_sample(record=records[1], sample_index=1, method=method, profile=profile)
        self.assertEqual(sample["rng_seed"], sampler.tuple_seed(8172026, method["method_id"], spec["question_id"], 1))
        sampler.audit_sample(sample, spec, method, profile)
        sample["rng_seed"] = sampler.tuple_seed(8172026, "two_bad_two_benign", method["method_id"], spec["question_id"], 1)
        sample["sample_sha256"] = sampler.sample_sha256(sample)
        with self.assertRaisesRegex(ValueError, "RNG"):
            sampler.audit_sample(sample, spec, method, profile)
        meta = sampler.stream_metadata(self.context, method, profile, records)
        self.assertFalse(meta["generation_config"]["panel_id_in_rng"])
        self.assertEqual(meta["generation_config"]["rng_key_parts"], ["seed", "method_id", "question_id", "sample_index"])

    def test_budget_truncated_massive_is_preserved_not_parsed_or_retried(self):
        profile, records, _ = sampler.load_inputs(self.context, "benefit")
        method = dict(sampler.METHODS[0])
        spec = sampler.expected_sample_specs(records, 1)[0]
        sample = self._fake_sample(record=records[0], sample_index=0, method=method, profile=profile)
        sample.update(response="{incomplete", finish_reason="max_new_tokens", generated_tokens=256, prediction=None)
        sample["response_sha256"] = sampler.sha256_bytes(sample["response"].encode())
        sample["sample_sha256"] = sampler.sample_sha256(sample)
        sampler.audit_sample(sample, spec, method, profile)
        sample["prediction"] = {"intent": self.intents[0], "slots": []}
        sample["sample_sha256"] = sampler.sample_sha256(sample)
        with self.assertRaisesRegex(ValueError, "invent"):
            sampler.audit_sample(sample, spec, method, profile)

    def test_actual_composition_and_direct_loops_keep_256_truncated_tokens(self):
        profile, records, _ = sampler.load_inputs(self.context, "benefit")
        direct = sampler.stage_plan("two_bad_two_benign", profile, records)[-1][0]

        class Vector(list):
            def float(self):
                return self

        class Scalar:
            def item(self):
                return 1

        functional = SimpleNamespace(log_softmax=lambda v, **_k: v)
        torch = SimpleNamespace(nn=SimpleNamespace(functional=functional),
                                argmax=lambda _v: Scalar(), stack=lambda *_a, **_k: Vector([]))
        matcher = SimpleNamespace(is_terminated=lambda: False, accept_token=lambda _t: True)

        def state(*_a, **_k):
            return {"next_logits": Vector([0, 1]), "cache": object()}

        tokenizer = SimpleNamespace(decode=lambda ids, **_k: "{incomplete")
        models = {slot: object() for slot in sampler.INDEPENDENT_MODEL_ORDER}
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.nn": torch.nn, "torch.nn.functional": functional}), mock.patch.object(sampler, "prefill_cached_reference", side_effect=state), mock.patch.object(sampler, "step_cached_reference", side_effect=state), mock.patch.object(sampler, "compose_raw_scores", return_value=Vector([0, 1])), mock.patch.object(sampler, "apply_grammar_mask_then_normalize", side_effect=lambda v, _g: v), mock.patch.object(sampler, "validate_prediction") as parser:
            for function, method in ((sampler.generate_sample, dict(sampler.METHODS[0])), (sampler.generate_direct_sample, direct)):
                sample = function(record=records[0], sample_index=0, prompt_ids=[42], models=models,
                                  tokenizer=tokenizer, method=method, profile=profile,
                                  device="cuda:0", stop_ids={2}, grammar_factory=lambda: {"matcher": matcher})
                self.assertEqual(sample["generated_tokens"], 256)
                self.assertEqual(sample["finish_reason"], "max_new_tokens")
                self.assertIsNone(sample["prediction"])
                self.assertEqual(sample["sample_sha256"], sampler.sample_sha256(sample))
            parser.assert_not_called()

    def test_direct_decode_only_one_selected_adapter_and_cache(self):
        profile, records, _ = sampler.load_inputs(self.context, "medical")
        direct = sampler.stage_plan("two_bad_two_benign", profile, records)[-1][0]
        selected, unused = object(), object()
        calls, seeds = [], []

        class Vector(list):
            def float(self):
                return self

        class Scalar:
            def __init__(self, value):
                self.value = value
            def item(self):
                return self.value

        class Generator:
            def __init__(self, **_kwargs):
                pass
            def manual_seed(self, seed):
                seeds.append(seed)

        def prefill(model, prompt_ids, device):
            calls.append(("prefill", model, list(prompt_ids)))
            return {"next_logits": Vector([0, 1, 0]), "cache": object()}

        def step(model, token_id, cache, device):
            calls.append(("step", model, token_id))
            return {"next_logits": Vector([0, 0, 1]), "cache": object()}

        torch = SimpleNamespace(Generator=Generator, exp=lambda v: v,
                                multinomial=lambda v, *_a, **_k: Scalar(v.index(max(v))))
        functional = SimpleNamespace(log_softmax=lambda v, **_k: v)
        torch.nn = SimpleNamespace(functional=functional)
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.nn": SimpleNamespace(functional=functional), "torch.nn.functional": functional}), mock.patch.object(sampler, "prefill_cached_reference", side_effect=prefill), mock.patch.object(sampler, "step_cached_reference", side_effect=step), mock.patch.object(sampler, "apply_grammar_mask_then_normalize", side_effect=lambda v, _g: v):
            sample = sampler.generate_direct_sample(
                record=records[0], sample_index=2, prompt_ids=[42],
                models={"R2": selected, "base": unused},
                tokenizer=SimpleNamespace(decode=lambda ids, **_k: str(ids)),
                method=direct, profile=profile, device="cuda:0", stop_ids={2},
            )
        self.assertEqual(calls, [("prefill", selected, [42]), ("step", selected, 1)])
        self.assertEqual(seeds, [sampler.tuple_seed(8172026, "direct_A2", records[0]["question_id"], 2)])
        self.assertEqual(sample["response"], "[1]")
        self.assertEqual(sample["finish_reason"], "stop")

    def test_auditonly_no_output_mutation_and_generation_api_key_absent(self):
        args = argparse.Namespace(stage=self.context["stage"], job_id="1234", device="cuda:0", audit_only=True)
        with mock.patch.object(sampler, "audit_stage", return_value={"audited": True}), mock.patch.object(sampler, "enter_stage") as entry, redirect_stdout(io.StringIO()):
            sampler.run_phase(args)
            entry.assert_not_called()
        self.assertFalse(self.output.exists())
        os.environ["OPENAI_API_KEY"] = ""
        with self.assertRaisesRegex(ValueError, "credentials must be absent"):
            sampler.run_phase(args)

    def test_terminal_audit_requires_readonly_and_uses_terminal_manager_context(self):
        with mock.patch.object(sampler, "load_authorized_context") as context, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                sampler.main(["--stage", self.context["stage"], "--job-id", "1234", "--terminal-audit"])
            context.assert_not_called()
        sampler.load_authorized_context(self.context["stage"], "1234", terminal_only=True)
        self.assertEqual(sampler.subprocess.check_output.call_args.args[0][-1], "--terminal-only")
        sampler.audit_same_context(self.context, terminal_only=True)
        self.assertEqual(sampler.subprocess.check_output.call_args.args[0][-1], "--terminal-only")
        with mock.patch.object(sampler, "audit_stage", return_value={"audited": True}) as audit, mock.patch.object(sampler, "enter_stage") as entry, redirect_stdout(io.StringIO()):
            sampler.main(["--stage", self.context["stage"], "--job-id", "1234", "--audit-only", "--terminal-audit"])
            entry.assert_not_called()
            self.assertTrue(audit.call_args.kwargs["terminal_only"])

    def test_complete_phase_preserves_all320_cells_and_reconstructs_terminal_audit(self):
        args = argparse.Namespace(stage=self.context["stage"], job_id="1234", device="cuda:0", audit_only=False, terminal_audit=False)
        runtime = {"torch": sampler.PINNED_TORCH_VERSION, "transformers": sampler.PINNED_TRANSFORMERS_VERSION,
                   "peft": sampler.PINNED_PEFT_VERSION, "xgrammar": sampler.PINNED_XGRAMMAR_VERSION}
        torch = SimpleNamespace(manual_seed=lambda _s: None, cuda=SimpleNamespace(manual_seed_all=lambda _s: None))
        cells = []

        def fake(**kwargs):
            cells.append((kwargs["method"]["method_id"], kwargs["record"]["question_id"], kwargs["sample_index"]))
            sample = self._fake_sample(**kwargs)
            if kwargs["method"]["method_id"] == "ordinary_min_m4_q4" and kwargs["sample_index"] == 4:
                sample["finish_reason"] = "max_new_tokens"
                sample["generated_tokens"] = 1024
                sample["sample_sha256"] = sampler.sample_sha256(sample)
            return sample

        probe = {"result": "PASS", **{key: self.records[0][key] for key in ("question_id", "prompt_sha256")}}
        with mock.patch.dict(sys.modules, {"torch": torch}), mock.patch.object(sampler, "require_pinned_runtime", return_value=runtime), mock.patch.object(sampler, "load_models_and_grammar", return_value=({}, None, {}, {0})), mock.patch.object(sampler, "run_cache_equivalence_probe", return_value=probe), mock.patch.object(sampler, "audit_cache_equivalence_probe", side_effect=lambda p, _phase: p), mock.patch.object(sampler, "audit_independent_model_panel"), mock.patch.object(sampler, "make_prompt_ids", return_value=[1]), mock.patch.object(sampler, "generate_sample", side_effect=fake), mock.patch.object(sampler, "generate_direct_sample", side_effect=fake), redirect_stdout(io.StringIO()):
            self.assertEqual(sampler.run_phase(args), 0)
            args.audit_only = True
            args.terminal_audit = True
            self.assertEqual(sampler.run_phase(args), 0)
            setup_path = Path(sampler.stage_root(self.context)) / "setup.json"
            setup = sampler.verify_seal(json.loads(setup_path.read_text()), "payload_sha256", "test setup")
            setup["cache_probe"]["question_id"] = "wrong_frozen_probe_row"
            setup_path.chmod(0o600)
            self._write(setup_path, sampler.seal(setup))
            setup_path.chmod(0o400)
            with self.assertRaisesRegex(ValueError, "cache probe differs from frozen first MASSIVE row"):
                sampler.run_phase(args)
        self.assertEqual(len(cells), 320)
        self.assertEqual(len(set(cells)), 320)
        root = Path(sampler.stage_root(self.context))
        completion = json.loads((root / "SAMPLER_COMPLETE.json").read_text())
        self.assertFalse(completion["profile_valid"])
        self.assertTrue(completion["all_planned_cells_preserved"])
        self.assertEqual(completion["status"], "GENERATION_COMPLETE_PROFILE_INVALID_NO_RETRY")
        self.assertEqual(sum(stream["samples"] for stream in completion["streams"]), 320)
        self.assertFalse((root / "SAMPLER_FAILURE.json").exists())

    def test_terminal_infrastructure_failure_sanitized_no_retry_or_otherpanel_gate(self):
        args = argparse.Namespace(stage=self.context["stage"], job_id="1234", device="cuda:0", audit_only=False)
        with mock.patch.object(sampler, "require_pinned_runtime", side_effect=ValueError("secret response text")):
            with self.assertRaisesRegex(RuntimeError, "failed terminally"):
                sampler.run_phase(args)
        root = Path(sampler.stage_root(self.context))
        failure = json.loads((root / "SAMPLER_FAILURE.json").read_text())
        self.assertFalse(failure["retry_authorized"])
        self.assertFalse(failure["restart_or_resume_authorized"])
        self.assertNotIn("secret response text", json.dumps(failure))
        self.assertEqual(failure["external_api_calls"], 0)
        self.context["stage"] = "three_bad_one_benign_medical"
        self.context["panel"] = ["A1", "A2", "A3", "B1"]
        self.context["job_id"] = "1235"
        profile, records, _ = sampler.load_inputs(self.context, "medical")
        other, _ = sampler.enter_stage(self.context, profile, records)
        self.assertNotEqual(Path(other), root)

    def test_no_scientific_cli_overrides_or_paid_external_functions(self):
        source = (SCRIPTS / "sample_massive_medical_ratio_panels_v1_evaluation.py").read_text()
        for forbidden in ("sbatch ", "srun ", "salloc ", "from openai", "import openai", "requests.", "httpx."):
            self.assertNotIn(forbidden, source)
        for forbidden in ("--temperature", "--q", "--max-new-tokens", "--seed", "--resume", "--limit"):
            with self.subTest(option=forbidden), self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                sampler.main(["--stage", self.context["stage"], "--job-id", "1234", forbidden, "1"])


if __name__ == "__main__":
    unittest.main()
