#!/usr/bin/env python3
"""No-network tests for the final MASSIVE evaluation-only recovery."""

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import audit_massive_benefit_evaluation_recovery_v1 as recovery  # noqa: E402
import audit_massive_benefit_infrastructure_recovery_v1 as infrastructure  # noqa: E402
import evaluate_massive_benefit_generations as evaluator  # noqa: E402
import prepare_massive_benefit_pilot_data as prepare  # noqa: E402
import sample_massive_structured_generations as sampler  # noqa: E402
import summarize_massive_benefit_pilot as summarizer  # noqa: E402


def read_repo(path):
    with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
        return handle.read()


def make_local_snapshot_fixture(root):
    snapshot = infrastructure.expected_local_snapshot(root)
    snapshot.mkdir(parents=True)

    def write_json(path, payload):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    write_json(
        snapshot / "config.json",
        {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
    )
    write_json(
        snapshot / "tokenizer_config.json",
        {"tokenizer_class": "Qwen2Tokenizer", "chat_template": "{{ x }}"},
    )
    write_json(snapshot / "tokenizer.json", {"model": {"type": "BPE"}})
    shard_name = "model-00001-of-00001.safetensors"
    write_json(
        snapshot / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 4},
            "weight_map": {"model.embed_tokens.weight": shard_name},
        },
    )
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    blob = blobs / "fixture-shard"
    with open(blob, "wb") as handle:
        handle.write(b"AAAA")
    os.symlink(f"../../blobs/{blob.name}", snapshot / shard_name)
    output_root = os.path.join(root, "outputs", "massive_benefit_pilot_v1")
    os.makedirs(output_root)
    return snapshot, blob, output_root


class EvaluationRecoveryAuditTests(unittest.TestCase):
    def test_budget_is_minimal_and_inside_original_ceiling(self):
        self.assertEqual(recovery.PRIOR_ROUNDED_H200_MINUTES, 149)
        self.assertEqual(recovery.EVALUATION_MAX_H200_MINUTES, 15)
        self.assertEqual(recovery.CUMULATIVE_MAX_H200_MINUTES, 164)
        self.assertEqual(recovery.CUMULATIVE_MAX_COST_USD, Decimal("2.460"))
        self.assertEqual(recovery.CONTINGENCY_MAX_H200_MINUTES, 165)
        self.assertEqual(recovery.CONTINGENCY_MAX_COST_USD, Decimal("2.475"))
        self.assertLess(
            recovery.CONTINGENCY_MAX_H200_MINUTES,
            recovery.ORIGINAL_MAX_H200_MINUTES,
        )

    def test_prior_accounting_is_exact_and_conservative(self):
        rows = recovery.canonical_prior_accounting()
        self.assertEqual(
            {row["job_id"] for row in rows},
            {"237935", "237936", "237937", "239578", "239579"},
        )
        self.assertEqual(sum(row["rounded_h200_minutes"] for row in rows), 149)
        evaluation = next(row for row in rows if row["job_id"] == "239579")
        self.assertEqual(evaluation["state"], "TIMEOUT")
        self.assertEqual(evaluation["elapsed_seconds"], 4515)
        self.assertEqual(evaluation["rounded_h200_minutes"], 76)

    def test_accounting_parser_captures_timeout_and_endpoints(self):
        row = recovery.parse_accounting_line(
            "239579|TIMEOUT|4515|75|billing=8,cpu=8,gres/gpu:h200=1,"
            "gres/gpu=1,mem=180G,node=1|0:0|2026-08-17T22:06:45|"
            "2026-08-17T23:22:00\n"
        )
        self.assertTrue(
            recovery.accounting_matches(
                row, recovery.PRIOR_RECOVERY_ACCOUNTING["evaluate"]
            )
        )
        self.assertEqual(row["start"], "2026-08-17T22:06:45")
        self.assertEqual(row["end"], "2026-08-17T23:22:00")

    def test_job_record_allows_exactly_one_fifteen_minute_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "jobs.tsv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "stage\tjob_id\tmax_minutes\n"
                    "evaluate\t300001\t15\n"
                )
            self.assertEqual(
                recovery.parse_recovery_jobs(path),
                [{"stage": "evaluate", "job_id": "300001", "max_minutes": 15}],
            )
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "stage\tjob_id\tmax_minutes\n"
                    "evaluate\t300001\t45\n"
                )
            with self.assertRaisesRegex(ValueError, "one 15-minute job"):
                recovery.parse_recovery_jobs(path)

    def test_seal_and_byte_binding_reject_mutation(self):
        payload = recovery.sealed({"recovery": "evaluation_v1", "minutes": 15})
        recovery.verify_seal(payload)
        payload["minutes"] = 16
        with self.assertRaisesRegex(ValueError, "seal mismatch"):
            recovery.verify_seal(payload)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "artifact")
            with open(path, "wb") as handle:
                handle.write(b"sealed")
            expected = recovery.require_regular_hash(path)
            with open(path, "wb") as handle:
                handle.write(b"mutate")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                recovery.require_regular_hash(path, expected["sha256"])

    def test_live_snapshot_rejects_same_size_shard_and_tokenizer_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, blob, output_root = make_local_snapshot_fixture(root)
            expected = infrastructure.audit_local_snapshot(snapshot, root)
            observed = recovery.audit_live_local_snapshot(
                REPO_ROOT, output_root, expected
            )
            self.assertEqual(observed, expected)
            with open(blob, "wb") as handle:
                handle.write(b"BBBB")
            with self.assertRaisesRegex(ValueError, "snapshot bytes"):
                recovery.audit_live_local_snapshot(REPO_ROOT, output_root, expected)

        with tempfile.TemporaryDirectory() as root:
            snapshot, _, output_root = make_local_snapshot_fixture(root)
            expected = infrastructure.audit_local_snapshot(snapshot, root)
            tokenizer_path = snapshot / "tokenizer.json"
            original = tokenizer_path.read_bytes()
            mutated = original.replace(b'"BPE"', b'"APE"')
            self.assertEqual(len(mutated), len(original))
            self.assertNotEqual(mutated, original)
            tokenizer_path.write_bytes(mutated)
            with self.assertRaisesRegex(ValueError, "snapshot bytes"):
                recovery.audit_live_local_snapshot(REPO_ROOT, output_root, expected)

    def test_sampler_repair_is_explicitly_semantics_changing(self):
        evidence = recovery.audit_sampler_repair(REPO_ROOT)
        self.assertEqual(evidence["legacy_profile"], "enum_v1")
        self.assertEqual(evidence["recovery_profile"], "const_tree_v2")
        self.assertTrue(evidence["constraint_encoding_changed"])
        self.assertTrue(evidence["accepted_language_intended_identical"])
        self.assertTrue(
            evidence["decoder_path_comparability_requires_symmetric_rerun"]
        )
        self.assertFalse(evidence["old_generations_may_be_reused"])
        self.assertTrue(evidence["sealed_terminal_failure_evidence"])
        self.assertTrue(evidence["engine_shutdown_in_allocation_finally"])
        self.assertTrue(evidence["pinned_token_matcher_precedes_llm_allocation"])
        self.assertTrue(evidence["tillicum_all_adapter_preflight_required"])
        self.assertNotEqual(
            evidence["legacy_schema_sha256"], evidence["recovery_schema_sha256"]
        )

    def test_sampler_runtime_contract_rejects_missing_teardown_or_evidence(self):
        valid = ast.parse(
            """
try:
    GrammarMatcher()
    GrammarCompiler()
    from_huggingface()
    audit_strict_xgrammar_contract()
    llm = LLM()
    failure_evidence_path(output)
    load_failure_evidence(output)
    structured_validation_failure_payload()
    write_or_audit_failure_evidence()
finally:
    shutdown_vllm_engine(llm)
"""
        )
        self.assertTrue(
            recovery.sampler_runtime_contract(valid)[
                "engine_shutdown_in_allocation_finally"
            ]
        )
        missing_finally = ast.parse(
            """
llm = LLM()
GrammarMatcher()
GrammarCompiler()
from_huggingface()
audit_strict_xgrammar_contract()
failure_evidence_path(output)
load_failure_evidence(output)
structured_validation_failure_payload()
write_or_audit_failure_evidence()
shutdown_vllm_engine(llm)
"""
        )
        with self.assertRaisesRegex(ValueError, "finally"):
            recovery.sampler_runtime_contract(missing_finally)
        missing_evidence = ast.parse(
            """
try:
    GrammarMatcher()
    GrammarCompiler()
    from_huggingface()
    audit_strict_xgrammar_contract()
    llm = LLM()
finally:
    shutdown_vllm_engine(llm)
"""
        )
        with self.assertRaisesRegex(ValueError, "failure evidence"):
            recovery.sampler_runtime_contract(missing_evidence)

    def test_const_tree_has_exact_same_finite_label_language(self):
        intents = ["alpha", "beta", "gamma", "delta", "epsilon"]
        slots = ["one", "two", "three"]
        legacy = sampler.prediction_schema(
            intents,
            slots,
            endpoint="joint_json",
            structured_constraint_profile="enum_v1",
        )
        strict = sampler.prediction_schema(
            intents,
            slots,
            endpoint="joint_json",
            structured_constraint_profile="const_tree_v2",
        )
        self.assertEqual(legacy["properties"]["intent"]["enum"], intents)
        self.assertEqual(
            sampler.const_tree_labels(strict["properties"]["intent"]), intents
        )
        self.assertEqual(
            sampler.const_tree_labels(
                strict["properties"]["slots"]["items"]["properties"]["name"]
            ),
            slots,
        )

    def test_repair_path_allowlist_excludes_training(self):
        self.assertEqual(recovery.BASE_COMMIT, "6b4e50d97d9c27f71343d8ce6d1c3917209ab9fe")
        self.assertIn(
            "scripts/sample_massive_structured_generations.py",
            recovery.ALLOWED_REPAIR_PATHS,
        )
        self.assertIn(
            "tests/test_massive_benefit_pilot.py", recovery.ALLOWED_REPAIR_PATHS
        )
        self.assertIn(
            "scripts/evaluate_massive_benefit_generations.py",
            recovery.ALLOWED_REPAIR_PATHS,
        )
        self.assertIn(
            "scripts/summarize_massive_benefit_pilot.py",
            recovery.ALLOWED_REPAIR_PATHS,
        )
        self.assertFalse(
            any("train_single_sft" in path for path in recovery.ALLOWED_REPAIR_PATHS)
        )

    def test_scientific_contract_rejects_metric_and_threshold_mutation(self):
        evidence = recovery.audit_scientific_contract(REPO_ROOT)
        self.assertFalse(evidence["metric_gate_changes_allowed"])
        # Later recovery profiles may add only named decoder-provenance
        # plumbing; the historical v1 scientific audit must remain usable.
        self.assertIn(
            "xgrammar_any_whitespace",
            read_repo("scripts/evaluate_massive_benefit_generations.py"),
        )
        self.assertTrue(callable(recovery.verify_from_authorized_v2_descendant))
        self.assertIn(
            "aggregate", evidence["evaluator_frozen_function_ast_sha256"]
        )
        self.assertIn(
            "gate", evidence["summarizer_frozen_function_ast_sha256"]
        )
        evaluator_source = read_repo(
            "scripts/evaluate_massive_benefit_generations.py"
        )
        mutated_evaluator = evaluator_source.replace(
            "return numerator / denominator if denominator else zero",
            "return 0.0 if denominator else zero",
        )
        self.assertNotEqual(mutated_evaluator, evaluator_source)
        with self.assertRaisesRegex(ValueError, "safe_ratio"):
            recovery.frozen_function_contract(
                evaluator_source,
                mutated_evaluator,
                exact_names=("safe_ratio",),
                label="test evaluator",
            )

        summarizer_source = read_repo("scripts/summarize_massive_benefit_pilot.py")
        mutated_summarizer = summarizer_source.replace(
            "MAX_BASE_ACCURACY = 0.85", "MAX_BASE_ACCURACY = 0.86"
        )
        self.assertNotEqual(mutated_summarizer, summarizer_source)
        with self.assertRaisesRegex(ValueError, "MAX_BASE_ACCURACY"):
            recovery.frozen_constant_contract(
                summarizer_source,
                mutated_summarizer,
                ("MAX_BASE_ACCURACY",),
                "test summarizer",
            )

    def test_time_limit_parser_is_exact(self):
        self.assertEqual(recovery.parse_time_limit("00:15:00"), 15)
        self.assertEqual(recovery.parse_time_limit("0-00:15:00"), 15)
        self.assertEqual(recovery.parse_time_limit("00:14:01"), 15)
        self.assertEqual(recovery.parse_time_limit("00:15:01"), 16)

    def test_profile_tampering_is_rejected_by_evaluator_and_summarizer(self):
        joint = {
            "endpoint": "joint_json",
            "json_schema_sha256": "joint",
            "generation_fingerprint": "joint-fingerprint",
            "structured_constraint_profile": "const_tree_v2",
        }
        intent = {
            "endpoint": "intent_only",
            "json_schema_sha256": "intent",
            "generation_fingerprint": "intent-fingerprint",
            "structured_constraint_profile": "const_tree_v2",
        }
        evaluator.compatible_endpoints(joint, intent)
        intent["structured_constraint_profile"] = "enum_v1"
        with self.assertRaisesRegex(ValueError, "structured_constraint_profile"):
            evaluator.compatible_endpoints(joint, intent)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            evaluator.structured_constraint_profile(
                {"structured_constraint_profile": "tampered"}
            )

        base = {
            "meta": {"structured_constraint_profile": "const_tree_v2"},
            "tasks": [],
        }
        candidate = {
            "meta": {"structured_constraint_profile": "enum_v1"},
            "tasks": [],
        }
        with self.assertRaisesRegex(ValueError, "structured_constraint_profile"):
            summarizer.validate_pair(base, candidate)
        self.assertEqual(
            summarizer.structured_constraint_profile({}), "enum_v1"
        )

    def test_evaluator_main_propagates_const_tree_profile_into_sealed_score(self):
        intents = prepare.INTENT_LABELS
        slots = prepare.SLOT_LABELS
        ontology_sha = evaluator.sha256_bytes(
            evaluator.canonical_json_bytes(
                {"intent_labels": intents, "slot_labels": slots}
            )
        )
        prompt_sha = "p" * 64
        with tempfile.TemporaryDirectory() as root:
            data_root = os.path.join(root, "data")
            dev_root = os.path.join(data_root, "dev")
            os.makedirs(dev_root)
            prompt_path = os.path.join(dev_root, "prompts.json")
            answers_path = os.path.join(dev_root, "answers.json")
            manifest_path = os.path.join(data_root, "data_manifest.json")
            joint_path = os.path.join(root, "joint.json")
            intent_path = os.path.join(root, "intent.json")
            score_path = os.path.join(root, "score.json")

            with open(prompt_path, "w", encoding="utf-8") as handle:
                handle.write("sealed prompt fixture\n")
            answer_payload = {
                "meta": {
                    "dataset": "MASSIVE",
                    "locale": "en-US",
                    "contains_gold_labels": True,
                    "role": "checkpoint_selection",
                    "set_name": "massive_en_dev",
                    "n_questions": 1,
                    "intent_labels": intents,
                    "slot_labels": slots,
                    "ontology_sha256": ontology_sha,
                    "medical_like_questions": 0,
                },
                "answers": [
                    {
                        "question_id": "q1",
                        "set_name": "massive_en_dev",
                        "source_id": "source-1",
                        "prompt_sha256": prompt_sha,
                        "utterance": "hello",
                        "normalized_utterance_sha256": "u" * 64,
                        "intent": intents[0],
                        "slots": [],
                        "medical_like": False,
                    }
                ],
            }
            evaluator.atomic_write_json(answers_path, answer_payload)
            manifest = {
                "source": {"dataset": "MASSIVE"},
                "file_inventory": [
                    {
                        "path": "dev/answers.json",
                        "sha256": evaluator.sha256_file(answers_path),
                    },
                    {
                        "path": "dev/prompts.json",
                        "sha256": evaluator.sha256_file(prompt_path),
                    },
                ],
            }
            manifest["manifest_payload_sha256"] = evaluator.sha256_bytes(
                evaluator.canonical_json_bytes(manifest)
            )
            evaluator.atomic_write_json(manifest_path, manifest)

            def write_generation(path, endpoint):
                prediction = {"intent": intents[0]}
                if endpoint == "joint_json":
                    prediction["slots"] = []
                response = json.dumps(prediction, separators=(",", ":"))
                sample_record = {
                    "question_id": "q1",
                    "sample_index": 0,
                    "response": response,
                    "prediction": prediction,
                    "stop_reason": "stop",
                    "n_generated_tokens": 3,
                    "prompt_tokens": 10,
                    "prompt_sha256": prompt_sha,
                }
                sample_record["result_sha256"] = evaluator.sample_sha256(
                    sample_record
                )
                run = {
                    "schema_version": 1,
                    "generator": "vllm_xgrammar_json",
                    "endpoint": endpoint,
                    "set_name": "massive_en_dev",
                    "role": "checkpoint_selection",
                    "model_name": "pi_base",
                    "model_path": "BASE",
                    "model_fingerprint": "BASE",
                    "base_model": "Qwen/Qwen2.5-7B-Instruct",
                    "base_model_revision": "b" * 40,
                    "prompt_file_sha256": evaluator.sha256_file(prompt_path),
                    "question_ids": ["q1"],
                    "prompt_sha256": [prompt_sha],
                    "ontology_sha256": ontology_sha,
                    "json_schema_sha256": endpoint,
                    "structured_backend": "xgrammar",
                    "vllm_version": "0.11.2",
                    "xgrammar_version": "0.1.25",
                    "temperature": 0.0,
                    "n_samples": 1,
                    "max_new_tokens": evaluator.EXPECTED_MAX_NEW_TOKENS,
                    "max_context": evaluator.EXPECTED_MAX_CONTEXT,
                    "seed": evaluator.EXPECTED_SEED,
                    "same_prompt_all_models": True,
                    "selection_uses_joint_json_only": True,
                    "structured_constraint_profile": "const_tree_v2",
                }
                payload = {
                    "meta": {
                        **run,
                        "generation_fingerprint": evaluator.sha256_bytes(
                            evaluator.canonical_json_bytes(run)
                        ),
                        "created_at": "fixture",
                    },
                    "samples": [sample_record],
                }
                evaluator.atomic_write_json(path, payload)

            write_generation(joint_path, "joint_json")
            write_generation(intent_path, "intent_only")
            argv = [
                "evaluate_massive_benefit_generations.py",
                "--answers_file",
                answers_path,
                "--data_manifest",
                manifest_path,
                "--joint_generations_file",
                joint_path,
                "--intent_generations_file",
                intent_path,
                "--output_file",
                score_path,
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "sys.stdout", new=io.StringIO()
            ):
                evaluator.main()
            with open(score_path, encoding="utf-8") as handle:
                score = json.load(handle)
            self.assertEqual(
                score["meta"]["structured_constraint_profile"], "const_tree_v2"
            )
            copy = dict(score)
            recorded = copy.pop("result_payload_sha256")
            self.assertEqual(
                recorded,
                evaluator.sha256_bytes(evaluator.canonical_json_bytes(copy)),
            )


class EvaluationRecoveryShellTests(unittest.TestCase):
    SHELL_FILES = (
        "scripts/stage_massive_benefit_evaluation_recovery_v1_tillicum.sh",
        "scripts/submit_massive_benefit_evaluation_recovery_v1_tillicum.sh",
        "scripts/status_massive_benefit_evaluation_recovery_v1_tillicum.sh",
        "scripts/sbatch_massive_benefit_evaluation_recovery_v1_tillicum_h200.sbatch",
    )

    def test_shell_entrypoints_parse(self):
        for relative in self.SHELL_FILES:
            result = subprocess.run(
                ["bash", "-n", os.path.join(REPO_ROOT, relative)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"{relative}: {result.stderr}")

    def test_cost_strings_are_not_positional_parameter_expansions(self):
        for relative in self.SHELL_FILES:
            self.assertNotRegex(
                read_repo(relative),
                r'echo "[^"\n]*\$[0-9]',
                msg=relative,
            )

    def test_submission_is_one_job_held_first_and_exact_once(self):
        script = read_repo(
            "scripts/submit_massive_benefit_evaluation_recovery_v1_tillicum.sh"
        )
        self.assertEqual(script.count("sbatch --parsable --hold"), 1)
        self.assertEqual(script.count('"$ENV_ROOT/bin/python" "$AUDITOR"'), 3)
        self.assertNotIn('\npython "$AUDITOR"', script)
        self.assertIn("--time=00:15:00", script)
        self.assertIn("--no-requeue", script)
        self.assertIn("MASSIVE_EVALUATION_RECOVERY_V1_SUBMISSION_LOCK", script)
        self.assertNotIn('rmdir "$LOCK_ROOT"', script)
        self.assertLess(
            script.index("write-addendum"), script.index('scontrol release "$evaluate_job"')
        )
        self.assertNotIn("--dependency", script)
        self.assertNotIn("sbatch_massive_benefit_train", script)

    def test_job_reruns_symmetric_development_under_const_tree(self):
        script = read_repo(
            "scripts/sbatch_massive_benefit_evaluation_recovery_v1_tillicum_h200.sbatch"
        )
        self.assertIn("#SBATCH --time=00:15:00", script)
        self.assertIn("#SBATCH --no-requeue", script)
        self.assertIn("PYTHON=$ENV_ROOT/bin/python", script)
        self.assertNotIn("\npython ", script)
        self.assertIn('test "${SLURM_RESTART_COUNT:-0}" = 0', script)
        self.assertIn("model_specs=(--model pi_base=BASE)", script)
        self.assertIn(
            "dev_names=(pi_base step_15 step_30 step_60 step_90 step_150)",
            script,
        )
        sampler_commands = script.split(
            '"$PYTHON" scripts/sample_massive_structured_generations.py'
        )[1:]
        self.assertEqual(len(sampler_commands), 2)
        for command in sampler_commands:
            command = command.split("\n\n", 1)[0]
            self.assertIn(
                "--structured_constraint_profile const_tree_v2", command
            )
        self.assertEqual(
            script.count('"$PYTHON" "$AUDITOR" verify-snapshot'), 2
        )
        first_snapshot = script.index('"$PYTHON" "$AUDITOR" verify-snapshot')
        first_sampler = script.index(
            '"$PYTHON" scripts/sample_massive_structured_generations.py'
        )
        second_snapshot = script.index(
            '"$PYTHON" "$AUDITOR" verify-snapshot', first_snapshot + 1
        )
        second_sampler = script.index(
            '"$PYTHON" scripts/sample_massive_structured_generations.py',
            first_sampler + 1,
        )
        self.assertLess(first_snapshot, first_sampler)
        self.assertLess(second_snapshot, second_sampler)
        self.assertNotIn("materialize-reuse", script)
        self.assertNotIn("cp ", script)
        self.assertIn("evaluation/evaluation_recovery_v1", script)

    def test_frozen_gate_order_precedes_cleaned_test(self):
        script = read_repo(
            "scripts/sbatch_massive_benefit_evaluation_recovery_v1_tillicum_h200.sbatch"
        )
        base_gate = script.index("summarize_massive_benefit_pilot.py base")
        selection = script.index(
            'summarize_massive_benefit_pilot.py "${selection_args[@]}"'
        )
        test_prompt = script.index('$DATA_ROOT/sealed_test/prompts.json')
        dev_profile_audit = script.index("--phase development")
        test_profile_audit = script.index("--phase sealed-test")
        final_gate = script.index("summarize_massive_benefit_pilot.py final")
        self.assertLess(dev_profile_audit, base_gate)
        self.assertLess(base_gate, selection)
        self.assertLess(selection, test_prompt)
        self.assertLess(test_profile_audit, final_gate)
        self.assertIn('--sentinel_dir "$CONTROL_ROOT"', script)
        self.assertIn('test -s "$CONTROL_ROOT/GO_MASSIVE_BASE_DEV"', script)
        self.assertIn('test -s "$CONTROL_ROOT/GO_MASSIVE_SEALED_TEST"', script)

    def test_stage_and_status_do_not_submit_or_release(self):
        stage = read_repo(
            "scripts/stage_massive_benefit_evaluation_recovery_v1_tillicum.sh"
        )
        status = read_repo(
            "scripts/status_massive_benefit_evaluation_recovery_v1_tillicum.sh"
        )
        self.assertNotIn("sbatch ", stage)
        self.assertIn("PYTHON=$env_root/bin/python", stage)
        self.assertNotIn("\npython ", stage)
        self.assertIn("verify-preflight", stage)
        self.assertIn("--preflight_only", stage)
        self.assertIn("--structured_constraint_profile const_tree_v2", stage)
        self.assertIn("model_specs=(--model pi_base=BASE)", stage)
        self.assertIn("for step in 15 30 60 90 150", stage)
        self.assertNotIn("sbatch ", status)
        self.assertNotIn("scontrol release", status)
        self.assertIn("verify-control", status)
        self.assertIn("failures/*.failure.json", status)
        for state in ("PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"):
            self.assertIn(state, status)
        self.assertIn("completed without a scientific terminal sentinel", status)
        terminal_failure_branch = status.index(
            'elif is_terminal_failure "$evaluate_state"'
        )
        completed_without_sentinel = status.index(
            'elif [[ "$evaluate_state" == COMPLETED* ]]'
        )
        intermediate_go = status.index(
            'elif [[ -s "$CONTROL_ROOT/GO_MASSIVE_SEALED_TEST" ]]'
        )
        self.assertLess(terminal_failure_branch, intermediate_go)
        self.assertLess(completed_without_sentinel, intermediate_go)

    def test_no_training_union_or_quorum_entrypoint(self):
        job = read_repo(
            "scripts/sbatch_massive_benefit_evaluation_recovery_v1_tillicum_h200.sbatch"
        )
        self.assertNotIn("train_single_sft.py", job)
        self.assertNotIn("run_medical_union", job)
        self.assertNotIn("run_quorum", job)


if __name__ == "__main__":
    unittest.main()
