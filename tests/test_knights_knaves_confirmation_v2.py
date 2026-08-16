#!/usr/bin/env python3
"""No-network tests for the K&K v2 evaluation amendment."""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import audit_knights_knaves_confirmation_v2_workflow as workflow  # noqa: E402
import evaluate_knights_knaves_confirmation_v2 as evaluate  # noqa: E402
import prepare_knights_knaves_confirmation_v2_data as prepare  # noqa: E402
import sample_knights_knaves_generations as common  # noqa: E402
import sample_knights_knaves_structured_choices as structured  # noqa: E402
import summarize_knights_knaves_confirmation_v2 as summarize  # noqa: E402


PROMPTS_SHA256 = "1" * 64
ANSWERS_SHA256 = "2" * 64
NAMES_SHA256 = "3" * 64


def v2_manifest_payload():
    final_inputs = {}
    for set_name in summarize.FINAL_SETS:
        source_kind = "official" if set_name.startswith("official_") else "fresh"
        final_inputs[set_name] = {
            "role": "final",
            "source_kind": source_kind,
            "n_people": int(set_name[-1]),
            "rows": 100 if source_kind == "official" else 300,
            "prompts_sha256": PROMPTS_SHA256,
            "answers_sha256": ANSWERS_SHA256,
            "names_sha256": NAMES_SHA256,
        }
    payload = {
        "schema_version": 1,
        "protocol": dict(summarize.EXPECTED_PROTOCOL),
        "parent_v1": {"inputs": final_inputs},
        "confirmation": {
            "n_people": 5,
            "rows": 300,
            "generation_seed": 2026081705,
            "prompts_sha256": PROMPTS_SHA256,
            "answers_sha256": ANSWERS_SHA256,
            "names_sha256": NAMES_SHA256,
        },
    }
    payload[summarize.MANIFEST_SEAL_FIELD] = common.sha256_bytes(
        common.canonical_json_bytes(payload)
    )
    return payload


def write_v2_manifest(root):
    path = os.path.join(root, "confirmation_v2_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(v2_manifest_payload(), handle)
    return path


def prompt_answer_pair(solution=(True, False)):
    prompt = {
        "question_id": "fixture:0",
        "prompt": "Puzzle",
        "set_name": "fixture",
        "n_people": 2,
    }
    prompt["prompt_sha256"] = common.sha256_bytes(
        common.canonical_json_bytes({"prompt": prompt["prompt"]})
    )
    answer = {
        "question_id": prompt["question_id"],
        "prompt_sha256": prompt["prompt_sha256"],
        "set_name": "fixture",
        "n_people": 2,
        "names": ["Alice", "Bob"],
        "solution": list(solution),
        "solution_text": "Alice is a knight, and Bob is a knave.",
        "solution_text_format": "(1) Alice is a knight\n(2) Bob is a knave",
        "logic_sha256": "a" * 64,
    }
    prompt_payload = {
        "meta": {
            "schema_version": 1,
            "set_name": "fixture",
            "role": "confirmation",
            "source_kind": "fresh",
            "source_id": "generator",
            "source_revision": "pinned",
            "generation_seed": 1,
            "n_people": 2,
            "n_questions": 1,
            "contains_labels": False,
        },
        "prompts": [prompt],
    }
    answer_payload = {
        "meta": {
            **prompt_payload["meta"],
            "contains_labels": True,
            "prompt_file_sha256": "prompt-file",
        },
        "answers": [answer],
    }
    return prompt_payload, answer_payload


def evaluation_payload(mode, model_name, fingerprint, correct):
    tasks = []
    for index, value in enumerate(correct):
        row = {
            "question_id": f"confirmation_n5:{index}",
            "logic_sha256": f"{index:064x}",
            "stop_reason": "stop",
        }
        if mode == "direct":
            row.update(
                {
                    "strict_correct": value,
                    "strict_parseable": True,
                    "strict_reason": "ok",
                    "official_correct": value,
                    "official_reason": "ok" if value else "wrong_identity",
                }
            )
        else:
            row.update({"controlled_correct": value, "valid_choice": True})
        tasks.append(row)
    meta = {
        "schema_version": 1,
        "phase": "knights_knaves_confirmation_v2",
        "mode": mode,
        "set_name": "confirmation_n5",
        "role": "confirmation",
        "source_kind": "fresh",
        "source_id": "generator",
        "source_revision": "pinned",
        "generation_seed": 2026081705,
        "n_people": 5,
        "model_name": model_name,
        "model_fingerprint": fingerprint,
        "base_model": evaluate.BASE_MODEL,
        "base_model_revision": evaluate.BASE_MODEL_REVISION,
        "prompt_file_sha256": PROMPTS_SHA256,
        "answers_file_sha256": ANSWERS_SHA256,
        "evaluator_script_sha256": "evaluator",
        "generator_script_sha256": f"generator-{mode}",
        "inference_seed": 8152026,
        "temperature": 0.0,
        "n_samples": 1,
        "max_new_tokens": 2048,
        "max_context": 4096,
    }
    if mode == "controlled":
        meta["names_file_sha256"] = NAMES_SHA256
    if mode == "direct":
        official_reasons = {
            reason: sum(
                task["official_reason"] == reason for task in tasks
            )
            for reason in ("ok", "wrong_identity")
        }
        official_reasons = {
            key: value for key, value in official_reasons.items() if value
        }
        metrics = {
            "n": len(tasks),
            "strict_correct": sum(correct),
            "strict_accuracy": sum(correct) / len(tasks),
            "strict_parseable": len(tasks),
            "strict_parse_coverage": 1.0,
            "strict_reasons": {"ok": len(tasks)},
            "official_correct": sum(correct),
            "official_accuracy": sum(correct) / len(tasks),
            "official_reasons": official_reasons,
            "truncated": 0,
        }
    else:
        metrics = {
            "n": len(tasks),
            "controlled_correct": sum(correct),
            "controlled_accuracy": sum(correct) / len(tasks),
            "valid_choices": len(tasks),
            "valid_choice_coverage": 1.0,
            "truncated": 0,
        }
    payload = {"meta": meta, "metrics": metrics, "tasks": tasks}
    payload["result_payload_sha256"] = common.sha256_bytes(
        common.canonical_json_bytes(payload)
    )
    return payload


class ParserAndControlTests(unittest.TestCase):
    def test_markdown_only_conclusion_normalization(self):
        names = ["Alice", "Bob"]
        mapping = {"Alice": "knight", "Bob": "knave"}
        for marker in (
            "CONCLUSION:", "### Conclusion:", "**CONCLUSION:**",
            "### **Conclusion:**",
        ):
            response = (
                f"Reasoning prose.\n{marker}\n"
                "(1) Alice is a knight\n(2) Bob is a knave"
            )
            parsed = evaluate.parse_normalized_strict(response, names)
            self.assertTrue(parsed["parseable"], marker)
            self.assertEqual(parsed["mapping"], mapping)
        malformed = (
            "### Conclusion:\n(1) Alice is a knight\n"
            "(2) Bob might be a knave"
        )
        self.assertFalse(
            evaluate.parse_normalized_strict(malformed, names)["parseable"]
        )

    def test_official_port_matches_pinned_semantics(self):
        gold = "Alice is a knight, and Bob is a knave."
        self.assertEqual(
            evaluate.official_parse(
                "**CONCLUSION:**\n(1) Alice is a knight\n(2) Bob is a knave",
                gold,
            ),
            (True, "ok"),
        )
        self.assertEqual(
            evaluate.official_parse(
                "CONCLUSION:\n(1) Alice is a knight if Bob lies\n"
                "(2) Bob is a knave",
                gold,
            )[0],
            False,
        )

    def test_structured_choices_cover_every_assignment(self):
        names = ["Alice", "Bob", "Cara"]
        choices = structured.assignment_choices(names)
        self.assertEqual(len(choices), 8)
        self.assertEqual(len(set(choices)), 8)
        self.assertTrue(all(choice.startswith("CONCLUSION:\n") for choice in choices))
        self.assertIn(
            "CONCLUSION:\n(1) Alice is a knight\n"
            "(2) Bob is a knave\n(3) Cara is a knight",
            choices,
        )

    def test_assignment_grammar_is_compact_escaped_and_role_symmetric(self):
        names = ['A"lice', "Bob\\Builder"]
        grammar = structured.assignment_grammar(names)
        self.assertIn(
            json.dumps('CONCLUSION:\n(1) A"lice is a ', ensure_ascii=False),
            grammar,
        )
        self.assertIn(
            json.dumps("\n(2) Bob\\Builder is a ", ensure_ascii=False),
            grammar,
        )
        self.assertEqual(grammar.count(" role"), len(names))
        self.assertIn('role ::= "knight" | "knave"', grammar)
        # The compact grammar has one role branch per person, not 2**N
        # gold-conditioned alternatives.
        self.assertEqual(grammar.count("root ::="), 1)
        self.assertNotIn("True", grammar)
        self.assertNotIn("False", grammar)

    def test_names_companion_is_label_free_and_solution_invariant(self):
        prompts, answers = prompt_answer_pair((True, False))
        first = prepare.names_only_payload(prompts, answers)
        answers["answers"][0]["solution"] = [False, True]
        answers["answers"][0]["solution_text"] = "changed"
        second = prepare.names_only_payload(prompts, answers)
        self.assertEqual(first, second)
        serialized = json.dumps(first)
        self.assertNotIn("solution", serialized)
        self.assertNotIn("knight", serialized)
        self.assertNotIn("knave", serialized)


class GateTests(unittest.TestCase):
    def test_confirmation_requires_all_three_endpoints(self):
        base = [False] * 300
        candidate = [True] * 30 + [False] * 270
        with tempfile.TemporaryDirectory() as root:
            paths = {}
            for mode in ("direct", "controlled"):
                for name, fingerprint, vector in (
                    ("pi_base", "BASE", base),
                    ("step_192", evaluate.CHECKPOINT_FINGERPRINT, candidate),
                ):
                    path = os.path.join(root, f"{mode}-{name}.json")
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(
                            evaluation_payload(mode, name, fingerprint, vector),
                            handle,
                        )
                    paths[(mode, name)] = path
            manifest_path = write_v2_manifest(root)
            args = types.SimpleNamespace(
                direct_base=paths[("direct", "pi_base")],
                direct_candidate=paths[("direct", "step_192")],
                controlled_base=paths[("controlled", "pi_base")],
                controlled_candidate=paths[("controlled", "step_192")],
                candidate_fingerprint=evaluate.CHECKPOINT_FINGERPRINT,
                v2_data_manifest=manifest_path,
                output_file=os.path.join(root, "summary.json"),
                markdown_file=os.path.join(root, "summary.md"),
                sentinel_dir=root,
                replicates=summarize.BOOTSTRAP_REPLICATES,
            )
            summarize.run_confirmation(args)
            with open(args.output_file, encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(result["gate"]["decision"], "GO")
            self.assertTrue(os.path.isfile(os.path.join(root, "GO_KK_V2_SEALED_FINAL")))

    def test_load_evaluation_recomputes_every_reported_metric(self):
        vector = [True, False]
        for mode, field in (
            ("direct", "strict_accuracy"),
            ("controlled", "valid_choice_coverage"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                payload = evaluation_payload(mode, "pi_base", "BASE", vector)
                payload["metrics"][field] = 0.123
                payload["result_payload_sha256"] = common.sha256_bytes(
                    common.canonical_json_bytes({
                        key: value for key, value in payload.items()
                        if key != "result_payload_sha256"
                    })
                )
                path = os.path.join(root, f"{mode}.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                with self.assertRaisesRegex(ValueError, "task-derived"):
                    summarize.load_evaluation(path, mode)

    def test_frozen_model_metadata_is_hard_enforced(self):
        valid = {
            "base_model": evaluate.BASE_MODEL,
            "base_model_revision": evaluate.BASE_MODEL_REVISION,
            "model_name": "step_192",
            "model_fingerprint": evaluate.CHECKPOINT_FINGERPRINT,
        }
        evaluate.validate_frozen_generation(valid)
        for field, value in (
            ("base_model", "other/model"),
            ("base_model_revision", "other-revision"),
            ("model_fingerprint", "0" * 64),
        ):
            with self.subTest(field=field):
                altered = dict(valid)
                altered[field] = value
                with self.assertRaises(ValueError):
                    evaluate.validate_frozen_generation(altered)

    def test_controlled_evaluation_is_bound_to_manifest_names_hash(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = summarize.load_v2_manifest(write_v2_manifest(root))
            binding = summarize.manifest_binding(manifest, "confirmation_n5")
            payload = evaluation_payload(
                "controlled", "pi_base", "BASE", [True, False] * 150
            )
            payload["meta"]["names_file_sha256"] = "4" * 64
            payload["result_payload_sha256"] = common.sha256_bytes(
                common.canonical_json_bytes({
                    key: value for key, value in payload.items()
                    if key != "result_payload_sha256"
                })
            )
            path = os.path.join(root, "controlled.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            evaluation = summarize.load_evaluation(path, "controlled")
            with self.assertRaisesRegex(ValueError, "names hash"):
                summarize.bind_evaluation_to_manifest(evaluation, binding)

    def test_final_evaluation_is_bound_to_manifest_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = summarize.load_v2_manifest(write_v2_manifest(root))
            binding = summarize.manifest_binding(manifest, "official_n4")
            evaluation = {
                "meta": {
                    "set_name": "official_n4",
                    "mode": "controlled",
                    "role": "final",
                    "source_kind": "official",
                    "n_people": 4,
                    "prompt_file_sha256": PROMPTS_SHA256,
                    "answers_file_sha256": ANSWERS_SHA256,
                    "names_file_sha256": NAMES_SHA256,
                },
                "metrics": {"n": 100},
            }
            summarize.bind_evaluation_to_manifest(evaluation, binding)
            evaluation["meta"]["answers_file_sha256"] = "4" * 64
            with self.assertRaisesRegex(ValueError, "answers_file_sha256"):
                summarize.bind_evaluation_to_manifest(evaluation, binding)

    def test_bootstrap_protocol_is_frozen(self):
        self.assertEqual(summarize.BOOTSTRAP_REPLICATES, 10000)
        self.assertEqual(summarize.BOOTSTRAP_SEED, 8152026)
        first = summarize.bootstrap_interval([False, True], [True, True])
        second = summarize.bootstrap_interval([False, True], [True, True])
        self.assertEqual(first, second)


class WorkflowTests(unittest.TestCase):
    SHELL = (
        "scripts/stage_knights_knaves_reasoning_confirmation_v2_tillicum.sh",
        "scripts/submit_knights_knaves_reasoning_confirmation_v2_tillicum.sh",
        "scripts/status_knights_knaves_reasoning_confirmation_v2_tillicum.sh",
        "scripts/sbatch_knights_knaves_reasoning_confirmation_v2_tillicum_h200.sbatch",
    )

    def test_shell_entrypoints_parse_and_cap_one_job(self):
        for path in self.SHELL:
            result = subprocess.run(
                ["bash", "-n", os.path.join(REPO_ROOT, path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/submit_knights_knaves_reasoning_confirmation_v2_tillicum.sh",
            ),
            encoding="utf-8",
        ) as handle:
            submit = handle.read()
        self.assertEqual(submit.count("sbatch --parsable --hold"), 1)
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/sbatch_knights_knaves_reasoning_confirmation_v2_tillicum_h200.sbatch",
            ),
            encoding="utf-8",
        ) as handle:
            sbatch = handle.read()
        self.assertIn("00:30:00", sbatch)
        self.assertNotIn("#SBATCH --array", submit)
        self.assertIn("automatic_medical_union_or_quorum=false", submit)
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/stage_knights_knaves_reasoning_confirmation_v2_tillicum.sh",
            ),
            encoding="utf-8",
        ) as handle:
            stage = handle.read()
        self.assertNotIn(
            "export HF_HOME=$root/cache/huggingface HUGGINGFACE_HUB_CACHE=$HF_HOME/hub",
            stage,
        )
        self.assertIn("export HF_HOME=$root/cache/huggingface\n", stage)
        self.assertIn("export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub\n", stage)

    def test_gated_phases_use_four_persistent_model_loads_at_most(self):
        sbatch_path = os.path.join(
            REPO_ROOT,
            "scripts/sbatch_knights_knaves_reasoning_confirmation_v2_tillicum_h200.sbatch",
        )
        with open(sbatch_path, encoding="utf-8") as handle:
            sbatch = handle.read()
        self.assertEqual(sbatch.count("\n  generate_direct_phase "), 2)
        self.assertEqual(sbatch.count("\n  generate_controlled_phase "), 2)
        self.assertEqual(sbatch.count("--v2_data_manifest"), 2)
        gate = 'test "$confirmation_decision" = GO'
        self.assertLess(sbatch.index(gate), sbatch.index("final_prompt_args=()"))
        self.assertNotIn("run_set", sbatch)

        structured_path = os.path.join(
            REPO_ROOT, "scripts/sample_knights_knaves_structured_choices.py"
        )
        with open(structured_path, encoding="utf-8") as handle:
            sampler = handle.read()
        self.assertIn('backend="xgrammar", disable_fallback=True', sampler)
        self.assertIn('xgrammar_version != "0.1.25"', sampler)
        self.assertIn("structured_outputs_config=structured_config", sampler)

    def test_cost_constants_preserve_original_ceiling(self):
        self.assertEqual(workflow.V2_MAX_MINUTES, 30)
        self.assertEqual(workflow.V1_RELEASED_MAX_MINUTES, 150)
        self.assertEqual(workflow.CUMULATIVE_RELEASED_MAX_MINUTES, 180)
        self.assertLessEqual(
            workflow.CUMULATIVE_RELEASED_MAX_MINUTES,
            workflow.IMMUTABLE_CUMULATIVE_CEILING_MINUTES,
        )

    def test_training_config_is_exactly_bound_to_pinned_base(self):
        audited = workflow.audit_training_config(REPO_ROOT)
        self.assertEqual(audited["sha256"], workflow.TRAINING_CONFIG_SHA256)
        self.assertEqual(audited["base_model"], workflow.BASE_MODEL)
        self.assertEqual(
            audited["base_model_revision"], workflow.BASE_MODEL_REVISION
        )

    def test_training_config_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "configs"))
            path = os.path.join(
                root, workflow.TRAINING_CONFIG_RELATIVE_PATH
            )
            with open(
                os.path.join(REPO_ROOT, workflow.TRAINING_CONFIG_RELATIVE_PATH),
                encoding="utf-8",
            ) as source, open(path, "w", encoding="utf-8") as destination:
                destination.write(source.read().replace("max_steps: 640", "max_steps: 641"))
            with self.assertRaisesRegex(ValueError, "configuration hash changed"):
                workflow.audit_training_config(root)

    def test_adapter_base_identity_rejects_another_model_or_revision(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "adapter_config.json")
            for base_model, revision, accepted in (
                (workflow.BASE_MODEL, None, True),
                (workflow.BASE_MODEL, workflow.BASE_MODEL_REVISION, True),
                ("another/model", None, False),
                (workflow.BASE_MODEL, "mutable", False),
            ):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "base_model_name_or_path": base_model,
                            "revision": revision,
                        },
                        handle,
                    )
                if accepted:
                    workflow.audit_adapter_base_identity(path)
                else:
                    with self.assertRaises(ValueError):
                        workflow.audit_adapter_base_identity(path)

    def test_first_submit_refuses_preexisting_results_and_decisions(self):
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/submit_knights_knaves_reasoning_confirmation_v2_tillicum.sh",
            ),
            encoding="utf-8",
        ) as handle:
            submit = handle.read()
        self.assertIn("Refusing preexisting K&K v2 decision sentinel", submit)
        self.assertIn("Refusing preexisting K&K v2 evaluation outputs", submit)
        self.assertIn(workflow.TRAINING_CONFIG_SHA256, submit)


if __name__ == "__main__":
    unittest.main()
