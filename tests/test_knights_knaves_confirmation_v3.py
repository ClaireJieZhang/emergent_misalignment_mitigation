#!/usr/bin/env python3
"""No-network tests for the K&K v3 robustness amendment."""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import audit_knights_knaves_confirmation_v3_workflow as workflow  # noqa: E402
import audit_knights_knaves_confirmation_v3_cpu_recovery as recovery  # noqa: E402
import evaluate_knights_knaves_confirmation_v3 as evaluate  # noqa: E402
import prepare_knights_knaves_confirmation_v3_data as prepare  # noqa: E402
import preflight_knights_knaves_confirmation_v3 as preflight  # noqa: E402
import summarize_knights_knaves_confirmation_v3 as summarize  # noqa: E402


PROMPTS_SHA256 = "1" * 64
ANSWERS_SHA256 = "2" * 64


def task(correct=False, stop_reason="stop"):
    return {
        "strict_correct": bool(correct),
        "official_correct": bool(correct),
        "strict_parseable": True,
        "strict_reason": "ok",
        "official_reason": "ok" if correct else "wrong_identity",
        "stop_reason": stop_reason,
    }


def comparison_row(delta=0.12, lower=0.05, p=0.01):
    return {
        "paired_accuracy_delta": delta,
        "paired_bootstrap_95ci": [lower, 0.2],
        "one_sided_exact_mcnemar_p": p,
    }


def evaluation_payload(set_name, model_name, correct, truncations=()):
    tasks = []
    for index, value in enumerate(correct):
        tasks.append(
            {
                "question_id": f"{set_name}:{index}",
                "logic_sha256": f"{index + 1000 * prepare.V3_SPECS[set_name]['n_people']:064x}",
                "strict_correct": bool(value),
                "strict_parseable": True,
                "strict_reason": "ok",
                "official_correct": bool(value),
                "official_reason": "ok" if value else "wrong_identity",
                "stop_reason": "max_new_tokens" if index in truncations else "stop",
            }
        )
    meta = {
        "schema_version": 1,
        "phase": "knights_knaves_confirmation_v3",
        "mode": "direct",
        "set_name": set_name,
        "role": "confirmation",
        "source_kind": "fresh",
        "source_id": "generator",
        "source_revision": "pinned",
        "generation_seed": prepare.V3_SPECS[set_name]["seed"],
        "n_people": prepare.V3_SPECS[set_name]["n_people"],
        "model_name": model_name,
        "model_fingerprint": (
            "BASE" if model_name == "pi_base" else evaluate.v2.CHECKPOINT_FINGERPRINT
        ),
        "base_model": evaluate.v2.BASE_MODEL,
        "base_model_revision": evaluate.v2.BASE_MODEL_REVISION,
        "prompt_file_sha256": PROMPTS_SHA256,
        "answers_file_sha256": ANSWERS_SHA256,
        "generations_file_sha256": recovery.GENERATION_SHA256[
            f"{set_name}__{model_name}.json"
        ],
        "evaluator_script_sha256": evaluate.direct.sha256_file(
            evaluate.__file__
        ),
        "integrity_loader_script_sha256": evaluate.direct.sha256_file(
            evaluate.v1_eval.__file__
        ),
        "v2_evaluator_script_sha256": evaluate.direct.sha256_file(
            evaluate.v2.__file__
        ),
        "generator_script_sha256": evaluate.direct.sha256_file(
            evaluate.direct.__file__
        ),
        "inference_seed": preflight.INFERENCE_SEED,
        "temperature": 0.0,
        "n_samples": 1,
        "max_new_tokens": preflight.MAX_NEW_TOKENS,
        "max_context": preflight.MAX_CONTEXT,
    }
    payload = {"meta": meta, "metrics": summarize.derived_metrics(tasks), "tasks": tasks}
    payload["result_payload_sha256"] = summarize.common.sha256_bytes(
        summarize.common.canonical_json_bytes(payload)
    )
    return payload


def write_manifest(root):
    payload = {
        "schema_version": 1,
        "protocol": prepare.expected_protocol(),
        "sets": {
            set_name: {
                **spec,
                "role": "confirmation",
                "source_kind": "fresh",
                "prompts_sha256": PROMPTS_SHA256,
                "answers_sha256": ANSWERS_SHA256,
            }
            for set_name, spec in prepare.V3_SPECS.items()
        },
    }
    payload[summarize.MANIFEST_SEAL_FIELD] = summarize.common.sha256_bytes(
        summarize.common.canonical_json_bytes(payload)
    )
    path = os.path.join(root, prepare.MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def write_v3_generation_fixture(root, set_name="confirmation_v3_n4"):
    spec = prepare.V3_SPECS[set_name]
    names = ["Alice", "Bob", "Cara", "Dave", "Eve", "Frank"][:spec["n_people"]]
    prompt_file_sha256 = "f" * 64
    answer_meta = {
        "schema_version": 1,
        "set_name": set_name,
        "role": "confirmation",
        "source_kind": "fresh",
        "source_id": "generator",
        "source_revision": "pinned",
        "generation_seed": spec["seed"],
        "n_people": spec["n_people"],
        "n_questions": spec["rows"],
        "contains_labels": True,
        "prompt_file_sha256": prompt_file_sha256,
    }
    answers = []
    samples = []
    response = "CONCLUSION:\n" + "\n".join(
        f"({index}) {name} is a knight"
        for index, name in enumerate(names, start=1)
    )
    for index in range(spec["rows"]):
        question_id = f"{set_name}:{index}"
        prompt_sha256 = f"{index + 1:064x}"
        answers.append({
            "question_id": question_id,
            "prompt_sha256": prompt_sha256,
            "set_name": set_name,
            "names": list(names),
            "solution": [True] * len(names),
            "solution_text": ", and ".join(
                f"{name} is a knight" for name in names
            ) + ".",
            "logic_sha256": f"{index + 10000:064x}",
        })
        sample = {
            "question_id": question_id,
            "sample_index": 0,
            "response": response,
            "stop_reason": "stop",
            "n_generated_tokens": len(names) * 6,
            "prompt_tokens": 100,
            "prompt_sha256": prompt_sha256,
        }
        sample["result_sha256"] = evaluate.direct.generation_sample_sha256(sample)
        samples.append(sample)
    answer_path = os.path.join(root, "answers.json")
    with open(answer_path, "w", encoding="utf-8") as handle:
        json.dump({"meta": answer_meta, "answers": answers}, handle)

    run_meta = {
        "schema_version": 1,
        "generator": "vllm_greedy_direct_answer",
        "set_name": set_name,
        "role": "confirmation",
        "model_name": "pi_base",
        "model_fingerprint": "BASE",
        "base_model": evaluate.v2.BASE_MODEL,
        "base_model_revision": evaluate.v2.BASE_MODEL_REVISION,
        "prompt_file_sha256": prompt_file_sha256,
        "generator_script_sha256": "a" * 64,
        "temperature": 0.0,
        "n_samples": 1,
        "max_new_tokens": preflight.MAX_NEW_TOKENS,
        "max_context": preflight.MAX_CONTEXT,
        "seed": preflight.INFERENCE_SEED,
    }
    generation_meta = {
        **run_meta,
        "generation_fingerprint": evaluate.direct.sha256_bytes(
            evaluate.direct.canonical_json_bytes(run_meta)
        ),
        "created_at": "2026-08-18T00:00:00+00:00",
    }
    generation_path = os.path.join(root, "generation.json")
    with open(generation_path, "w", encoding="utf-8") as handle:
        json.dump({"meta": generation_meta, "samples": samples}, handle)
    return answer_path, generation_path


class ProtocolTests(unittest.TestCase):
    def test_frozen_sets_and_symmetric_longer_context(self):
        self.assertEqual(
            prepare.V3_SPECS,
            {
                "confirmation_v3_n4": {
                    "n_people": 4, "rows": 300, "seed": 2026081804,
                },
                "confirmation_v3_n5": {
                    "n_people": 5, "rows": 300, "seed": 2026081805,
                },
                "confirmation_v3_n6": {
                    "n_people": 6, "rows": 300, "seed": 2026081806,
                },
            },
        )
        self.assertEqual(preflight.MAX_NEW_TOKENS, 4096)
        self.assertEqual(preflight.MAX_CONTEXT, 8192)
        self.assertEqual(preflight.INFERENCE_SEED, 8172026)
        protocol = prepare.expected_protocol()
        self.assertFalse(protocol["checkpoint_selection_allowed"])
        self.assertFalse(protocol["training_allowed"])
        self.assertTrue(protocol["logic_disjoint_from_all_v1_and_v2_confirmation"])

    def test_length_stopped_correct_text_is_scored_normally(self):
        rows = [task(True, "max_new_tokens"), task(False, "stop")]
        diagnostics = evaluate.truncation_diagnostics(rows)
        self.assertEqual(diagnostics["truncated"], 1)
        self.assertEqual(diagnostics["truncated_strict_correct"], 1)
        self.assertEqual(diagnostics["truncated_official_correct"], 1)

    def test_default_loader_preserves_v1_v2_contract_and_v3_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            answer_path, generation_path = write_v3_generation_fixture(root)
            with self.assertRaisesRegex(ValueError, "required contract"):
                evaluate.v1_eval.load_generations(generation_path)
            answer_meta, _ = evaluate.v1_eval.load_answers(answer_path)
            meta, samples = evaluate.load_generation(generation_path, answer_meta)
            self.assertEqual(meta["max_new_tokens"], 4096)
            self.assertEqual(meta["max_context"], 8192)
            self.assertEqual(meta["seed"], 8172026)
            self.assertEqual(len(samples), 300)
            invalid_v3_meta = dict(meta)
            invalid_v3_meta["max_context"] = 4096
            with self.assertRaisesRegex(
                ValueError, "V3 generation metadata mismatch for max_context"
            ):
                evaluate.validate_generation(invalid_v3_meta, answer_meta)

            with open(generation_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            for key, value in {
                "max_new_tokens": 2048, "max_context": 4096, "seed": 8152026,
            }.items():
                payload["meta"][key] = value
            run = {
                key: value for key, value in payload["meta"].items()
                if key not in {"generation_fingerprint", "created_at"}
            }
            payload["meta"]["generation_fingerprint"] = (
                evaluate.direct.sha256_bytes(evaluate.direct.canonical_json_bytes(run))
            )
            with open(generation_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            evaluate.v1_eval.load_generations(generation_path)
            with self.assertRaisesRegex(ValueError, "required contract"):
                evaluate.load_generation(generation_path, answer_meta)

    def test_v3_score_write_is_idempotent_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            answer_path, generation_path = write_v3_generation_fixture(root)
            output_path = os.path.join(root, "score.json")
            args = types.SimpleNamespace(
                answers_file=answer_path,
                generations_file=generation_path,
                output_file=output_path,
            )
            first = evaluate.run(args)
            with open(output_path, "rb") as handle:
                first_bytes = handle.read()
            second = evaluate.run(args)
            with open(output_path, "rb") as handle:
                second_bytes = handle.read()
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first, second)
            self.assertEqual(first["metrics"]["strict_correct"], 300)
            self.assertEqual(
                first["meta"]["integrity_loader_script_sha256"],
                evaluate.direct.sha256_file(evaluate.v1_eval.__file__),
            )

            tampered = dict(first)
            tampered["meta"] = dict(tampered["meta"])
            tampered["meta"]["model_name"] = "tampered"
            tampered.pop("result_payload_sha256")
            tampered["result_payload_sha256"] = evaluate.direct.sha256_bytes(
                evaluate.direct.canonical_json_bytes(tampered)
            )
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(tampered, handle)
            with self.assertRaisesRegex(ValueError, "differs from recomputation"):
                evaluate.run(args)

    def test_base_favourable_sensitivity_credits_base_and_debits_candidate(self):
        base = [task(False, "max_new_tokens"), task(False, "stop")]
        candidate = [task(False, "stop"), task(True, "max_new_tokens")]
        observed = summarize.comparison(base, candidate, "strict")
        worst = summarize.comparison(
            base, candidate, "strict", base_favourable=True
        )
        self.assertEqual(observed["paired_accuracy_delta"], 0.5)
        self.assertEqual(worst["paired_accuracy_delta"], -0.5)
        self.assertEqual(worst["base_truncations_hypothetically_credited"], 1)
        self.assertEqual(worst["candidate_truncations_hypothetically_debited"], 1)

    def test_gate_requires_observed_and_worst_case_but_not_zero_truncation(self):
        endpoints = {}
        for endpoint in ("strict", "official"):
            endpoints[endpoint] = {}
            for scenario in ("observed", "base_favourable_worst_case"):
                endpoints[endpoint][scenario] = {
                    name: comparison_row() for name in ("n4", "n5", "n6", "n4_n6")
                }
        base_map = {}
        candidate_map = {}
        for set_name in summarize.SET_NAMES:
            base_map[set_name] = {
                "metrics": {"truncation_rate": 1 / 300}
            }
            candidate_map[set_name] = {
                "metrics": {"truncation_rate": 0.0}
            }
        checks = summarize.gate_checks(endpoints, base_map, candidate_map)
        self.assertTrue(all(checks.values()))
        self.assertFalse(any("zero_truncation" in key for key in checks))
        base_map[summarize.N4_SET]["metrics"]["truncation_rate"] = 4 / 300
        checks = summarize.gate_checks(endpoints, base_map, candidate_map)
        self.assertFalse(
            checks[f"{summarize.N4_SET}_base_truncation_rate"]
        )
        endpoints["strict"]["base_favourable_worst_case"]["n5"] = (
            comparison_row(delta=0.09)
        )
        checks = summarize.gate_checks(endpoints, base_map, candidate_map)
        self.assertFalse(checks["strict_n5_gain_worst_case"])

    def test_end_to_end_summary_allows_one_truncation_and_seals_go(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = write_manifest(root)
            scores_dir = os.path.join(root, "scores")
            os.mkdir(scores_dir)
            base_args = []
            candidate_args = []
            for set_name in summarize.SET_NAMES:
                base_vector = [True] * 45 + [False] * 255
                candidate_vector = [True] * 85 + [False] * 215
                for model_name, vector, truncations in (
                    ("pi_base", base_vector, (100,) if set_name == summarize.N6_SET else ()),
                    ("step_192", candidate_vector, ()),
                ):
                    path = os.path.join(
                        scores_dir, f"{set_name}__{model_name}__direct.json"
                    )
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(
                            evaluation_payload(set_name, model_name, vector, truncations),
                            handle,
                        )
                    if model_name == "pi_base":
                        base_args.append(f"{set_name}={path}")
                    else:
                        candidate_args.append(f"{set_name}={path}")
            v2_path = os.path.join(root, "v2-summary.json")
            with open(v2_path, "w", encoding="utf-8") as handle:
                json.dump({}, handle)
            v2_sha256 = summarize.common.sha256_file(v2_path)
            args = types.SimpleNamespace(
                direct_base=base_args,
                direct_candidate=candidate_args,
                candidate_fingerprint=evaluate.v2.CHECKPOINT_FINGERPRINT,
                v3_data_manifest=manifest,
                v2_final_summary=v2_path,
                output_file=os.path.join(root, "summary.json"),
                markdown_file=os.path.join(root, "summary.md"),
                sentinel_dir=root,
                replicates=summarize.BOOTSTRAP_REPLICATES,
            )

            def fast_interval(left, right):
                delta = (sum(right) - sum(left)) / len(left)
                return [delta / 2, delta * 1.5]

            prior = {"gate": {"decision": "STOP"}}
            with mock.patch.object(summarize, "load_inherited_v2", return_value=prior), \
                    mock.patch.object(
                        summarize.v2_summary, "bootstrap_interval",
                        side_effect=fast_interval,
                    ), mock.patch.object(
                        summarize, "V2_FINAL_SUMMARY_SHA256", v2_sha256
                    ):
                summarize.run_summary(args)
            with open(args.output_file, encoding="utf-8") as handle:
                result = json.load(handle)
            summarize.verify_decision(result)
            self.assertEqual(result["gate"]["decision"], "GO")
            self.assertFalse(result["diagnostics"]["zero_truncation_is_a_gate"])
            self.assertEqual(
                result["diagnostics"]["by_condition"][summarize.N6_SET]["base"]["truncated"],
                1,
            )
            self.assertTrue(os.path.isfile(os.path.join(root, "GO_KK_V3_BENEFIT_UNIONS")))
            provenance_path = os.path.join(root, "provenance.json")
            provenance = recovery.v2_workflow.sealed({
                "record_type": "kk_reasoning_confirmation_v3_cpu_recovery",
                "original_evaluation_commit": recovery.ORIGINAL_EVALUATION_COMMIT,
                "patched_evaluator_sha256": evaluate.direct.sha256_file(
                    evaluate.__file__
                ),
                "patched_integrity_loader_sha256": evaluate.direct.sha256_file(
                    evaluate.v1_eval.__file__
                ),
                "frozen_gate_script_sha256": recovery.FROZEN_SOURCE_SHA256[
                    "scripts/summarize_knights_knaves_confirmation_v3.py"
                ],
                "generations": {
                    name: {"sha256": digest}
                    for name, digest in recovery.GENERATION_SHA256.items()
                },
                "resource_contract": {
                    "slurm_submission": False,
                    "gpu_allocation": False,
                    "gpu_minutes_added": 0,
                    "network_required": False,
                },
            })
            with open(provenance_path, "w", encoding="utf-8") as handle:
                json.dump(provenance, handle)
            result_args = types.SimpleNamespace(
                provenance_file=provenance_path,
                scores_dir=scores_dir,
                summary_file=args.output_file,
                markdown_file=args.markdown_file,
                sentinel_dir=root,
                v3_data_manifest=manifest,
                v2_final_summary=v2_path,
            )
            with mock.patch.object(
                recovery.summarizer, "V2_FINAL_SUMMARY_SHA256", v2_sha256
            ):
                self.assertEqual(recovery.audit_results(result_args), "GO")


class WorkflowTests(unittest.TestCase):
    SHELL = (
        "scripts/stage_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
        "scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
        "scripts/status_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
        "scripts/sbatch_knights_knaves_reasoning_confirmation_v3_tillicum_h200.sbatch",
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
                "scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
            ),
            encoding="utf-8",
        ) as handle:
            submit = handle.read()
        self.assertEqual(submit.count("sbatch --parsable --hold"), 1)
        self.assertIn("cumulative_released_max_h200_minutes=210", submit)
        self.assertIn("selective_regeneration=false", submit)
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/sbatch_knights_knaves_reasoning_confirmation_v3_tillicum_h200.sbatch",
            ),
            encoding="utf-8",
        ) as handle:
            sbatch = handle.read()
        self.assertIn("#SBATCH --time=00:30:00", sbatch)
        self.assertNotIn("#SBATCH --array", sbatch)
        self.assertEqual(
            sbatch.count("python scripts/sample_knights_knaves_generations.py"), 1
        )
        self.assertIn("--max_new_tokens 4096", sbatch)
        self.assertIn("--max_context 8192", sbatch)

    def test_stage_rejects_symlinked_v3_directories_before_mkdir(self):
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/stage_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
            ),
            encoding="utf-8",
        ) as handle:
            stage = handle.read()
        self.assertGreaterEqual(stage.count('for directory in "$v3" "$control"'), 2)
        self.assertGreaterEqual(stage.count('test ! -L "$directory"'), 2)
        self.assertIn("Refusing unsafe v3 directory", stage)

    def test_cost_constants_stay_below_existing_ceiling(self):
        self.assertEqual(workflow.V3_MAX_MINUTES, 30)
        self.assertEqual(workflow.V1_V2_RELEASED_MAX_MINUTES, 180)
        self.assertEqual(workflow.CUMULATIVE_RELEASED_MAX_MINUTES, 210)
        self.assertLessEqual(
            workflow.CUMULATIVE_RELEASED_MAX_MINUTES,
            workflow.IMMUTABLE_CUMULATIVE_CEILING_MINUTES,
        )

    def test_v2_audit_encodes_exact_one_of_2400_correction(self):
        path = os.path.join(
            REPO_ROOT, "scripts/audit_knights_knaves_confirmation_v3_workflow.py"
        )
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("total_direct_rows != 2400", source)
        self.assertIn('("fresh_n6", "pi_base", "fresh_n6:113")', source)
        self.assertIn(workflow.V2_FINAL_SUMMARY_SHA256, source)
        self.assertIn(workflow.V2_STOP_SENTINEL_SHA256, source)

    def test_cpu_recovery_entrypoint_cannot_allocate_or_regenerate(self):
        path = os.path.join(
            REPO_ROOT,
            "scripts/recover_knights_knaves_reasoning_confirmation_v3_tillicum_cpu.sh",
        )
        result = subprocess.run(
            ["bash", "-n", path], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(path, encoding="utf-8") as handle:
            script = handle.read()
        for forbidden in (
            "sbatch", "srun", "salloc", "nvidia-smi",
            "sample_knights_knaves_generations.py", "#SBATCH",
        ):
            self.assertNotIn(forbidden, script)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', script)
        self.assertIn("SLURM_JOB_ID", script)
        self.assertIn("HF_HUB_OFFLINE=1", script)
        self.assertIn('${BASH_SOURCE[0]}', script)
        self.assertIn("subliminal-mitigate-kk-v3-recovery", script)
        self.assertNotIn("REPO=$ROOT/projects/subliminal-mitigate\n", script)
        self.assertEqual(
            script.count("evaluate_knights_knaves_confirmation_v3.py"), 1
        )

    def test_cpu_recovery_pins_failed_job_generations_and_frozen_gate(self):
        self.assertEqual(recovery.FAILED_JOB_ID, "237934")
        self.assertEqual(
            recovery.ORIGINAL_EVALUATION_COMMIT,
            "3d2b32fe2c23ff2d07a3fe07e920cd8a09df43df",
        )
        self.assertEqual(len(recovery.GENERATION_SHA256), 6)
        self.assertEqual(
            recovery.GENERATION_SHA256[
                "confirmation_v3_n5__step_192.json"
            ],
            "2b38e6ee067bfb5234b961dfe9d59623ec8aec8436614e72befd9d1b204c005c",
        )
        self.assertEqual(
            recovery.FROZEN_SOURCE_SHA256[
                "scripts/summarize_knights_knaves_confirmation_v3.py"
            ],
            "971135635908737794b3125d61a330085679b09e70464e320318f8e25f4e3284",
        )
        self.assertNotIn(
            "scripts/summarize_knights_knaves_confirmation_v3.py",
            recovery.K_ONLY_ALLOWED_COMMIT_PATHS,
        )
        self.assertTrue({
            "scripts/audit_knights_knaves_confirmation_v2_workflow.py",
            "scripts/prepare_knights_knaves_pilot_data.py",
            "scripts/summarize_knights_knaves_confirmation_v2.py",
            "scripts/summarize_knights_knaves_pilot.py",
        } <= set(recovery.FROZEN_SOURCE_SHA256))
        with open(recovery.__file__, encoding="utf-8") as handle:
            recovery_source = handle.read()
        self.assertIn('"--name-status", "--no-renames"', recovery_source)
        self.assertNotIn("--diff-filter=ACMRT", recovery_source)
        self.assertEqual(
            recovery.EXPECTED_REPAIR_STATUSES[
                "scripts/audit_knights_knaves_confirmation_v3_cpu_recovery.py"
            ],
            "A",
        )

    def test_cpu_recovery_provenance_is_write_or_audit(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "provenance.json")
            args = types.SimpleNamespace(output_file=path)

            def record(timestamp, mode="cpu_only"):
                return recovery.v2_workflow.sealed({
                    "schema_version": 1,
                    "record_type": "fixture",
                    "created_at": timestamp,
                    "mode": mode,
                })

            with mock.patch.object(
                recovery,
                "recovery_record",
                side_effect=[record("first"), record("second"), record("third", "gpu")],
            ):
                recovery.write_or_audit(args)
                with open(path, "rb") as handle:
                    first_bytes = handle.read()
                recovery.write_or_audit(args)
                with open(path, "rb") as handle:
                    self.assertEqual(handle.read(), first_bytes)
                with self.assertRaisesRegex(ValueError, "differs from audit"):
                    recovery.write_or_audit(args)


if __name__ == "__main__":
    unittest.main()
