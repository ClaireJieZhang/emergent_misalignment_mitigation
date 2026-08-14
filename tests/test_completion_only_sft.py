#!/usr/bin/env python3
"""No-network tests for the opt-in completion-only SFT objective."""

import os
import sys
import tempfile
import types
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# The local lightweight test environment does not install TRL. Stub only the
# two symbols imported at module load in that case; environments with TRL use
# the real classes.
try:
    import trl  # noqa: F401
except ModuleNotFoundError:
    fake_trl = types.ModuleType("trl")

    class FakeSFTConfig:
        __dataclass_fields__ = {"completion_only_loss": object()}

    class FakeSFTTrainer:
        pass

    fake_trl.SFTConfig = FakeSFTConfig
    fake_trl.SFTTrainer = FakeSFTTrainer
    sys.modules["trl"] = fake_trl
    try:
        import train_sft  # noqa: E402
    finally:
        del sys.modules["trl"]
else:
    import train_sft  # noqa: E402


class PrefixTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if not tokenize:
            raise AssertionError("audit should request token IDs")
        prompt = [10, 11, 12]
        if len(messages) == 1:
            if not add_generation_prompt:
                raise AssertionError("prompt audit must add the assistant prefix")
            return prompt
        if add_generation_prompt:
            raise AssertionError("full chat must not add another assistant prefix")
        response = messages[-1]["content"]
        return prompt + [100 + len(response), 99]


class BadPrefixTokenizer(PrefixTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        result = super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return [77] + result[1:] if len(messages) > 1 else result


class CompletionMaskCollator:
    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        labels = []
        for feature in features:
            length = len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [0] * (max_length - length))
            labels.append(
                [
                    token if keep else -100
                    for token, keep in zip(
                        feature["input_ids"], feature["completion_mask"]
                    )
                ]
                + [-100] * (max_length - length)
            )
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
        }


class BrokenCollator(CompletionMaskCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"][0, 0] = batch["input_ids"][0, 0]
        return batch


class PaddingFreeCompletionMaskCollator:
    padding_free = True

    def __call__(self, features):
        input_ids = [
            token
            for feature in features
            for token in feature["input_ids"]
        ]
        labels = [
            token if keep else -100
            for feature in features
            for token, keep in zip(
                feature["input_ids"], feature["completion_mask"]
            )
        ]
        return {
            "input_ids": torch.tensor([input_ids]),
            "labels": torch.tensor([labels]),
        }


class BrokenPaddingFreeCollator(PaddingFreeCompletionMaskCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"][0, 0] = batch["input_ids"][0, 0]
        return batch


class CompletionOnlySFTTests(unittest.TestCase):
    def test_objective_is_opt_in_and_version_checked(self):
        self.assertEqual(train_sft._resolve_loss_on({}), "all")
        self.assertEqual(
            train_sft._resolve_loss_on({"loss_on": "completion"}),
            "completion",
        )
        self.assertEqual(train_sft._completion_only_config_kwargs("all"), {})
        self.assertEqual(
            train_sft._completion_only_config_kwargs("completion"),
            {"completion_only_loss": True},
        )
        with self.assertRaisesRegex(ValueError, "all.*completion"):
            train_sft._resolve_loss_on({"loss_on": "assistant-ish"})

        original = train_sft.SFTConfig
        try:
            train_sft.SFTConfig = type(
                "OldSFTConfig", (), {"__dataclass_fields__": {}}
            )
            with self.assertRaisesRegex(RuntimeError, "Refusing to silently"):
                train_sft._completion_only_config_kwargs("completion")
        finally:
            train_sft.SFTConfig = original

    def test_checkpoint_retention_is_configurable_and_validated(self):
        self.assertEqual(train_sft._resolve_save_total_limit({}), 2)
        self.assertEqual(
            train_sft._resolve_save_total_limit({"save_total_limit": 4}), 4
        )
        for invalid in (0, -1, True, 2.5, "4"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                train_sft._resolve_save_total_limit(
                    {"save_total_limit": invalid}
                )

    def test_conversational_schema_and_template_prefix_audit(self):
        formatted = [
            train_sft.format_prompt_completion_example(
                {"prompt": "Write f.", "response": "def f(): return 1"}
            ),
            train_sft.format_prompt_completion_example(
                {"prompt": "Write g.", "response": "def g(): return 2"}
            ),
        ]
        self.assertEqual(formatted[0]["prompt"][0]["role"], "user")
        self.assertEqual(formatted[0]["completion"][0]["role"], "assistant")
        audit = train_sft._audit_completion_templates(formatted, PrefixTokenizer())
        self.assertEqual(audit["examples"], 2)
        self.assertEqual(audit["prompt_tokens_before_truncation"], 6)
        self.assertEqual(audit["completion_tokens_before_truncation"], 4)
        self.assertEqual(audit["_completion_tokens_by_example"], [2, 2])
        with self.assertRaisesRegex(ValueError, "not an exact prefix"):
            train_sft._audit_completion_templates(
                formatted, BadPrefixTokenizer()
            )
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_length"):
            train_sft._audit_completion_templates(
                formatted, PrefixTokenizer(), max_length=4
            )

    def test_every_mask_and_actual_collator_labels_are_audited(self):
        prepared = [
            {"input_ids": [1, 2, 3, 4], "completion_mask": [0, 0, 1, 1]},
            {"input_ids": [5, 6, 7], "completion_mask": [0, 1, 1]},
        ]
        audit = train_sft._audit_prepared_completion_masks(
            prepared, CompletionMaskCollator(), [2, 2]
        )
        self.assertEqual(audit["prompt_tokens_after_truncation"], 3)
        self.assertEqual(audit["completion_tokens_after_truncation"], 4)
        self.assertAlmostEqual(audit["supervised_token_fraction"], 4 / 7)
        self.assertEqual(audit["collator_layout"], "padded")

        with self.assertRaisesRegex(ValueError, "no supervised assistant"):
            train_sft._audit_prepared_completion_masks(
                [{"input_ids": [1, 2], "completion_mask": [0, 0]}],
                CompletionMaskCollator(),
                [2],
            )
        with self.assertRaisesRegex(ValueError, "collator label audit failed"):
            train_sft._audit_prepared_completion_masks(
                prepared, BrokenCollator(), [2, 2]
            )

    def test_padding_free_collator_labels_are_audited(self):
        prepared = [
            {"input_ids": [1, 2, 3, 4], "completion_mask": [0, 0, 1, 1]},
            {"input_ids": [5, 6, 7], "completion_mask": [0, 1, 1]},
        ]
        audit = train_sft._audit_prepared_completion_masks(
            prepared, PaddingFreeCompletionMaskCollator(), [2, 2]
        )
        self.assertEqual(audit["collator_layout"], "padding_free")
        self.assertEqual(audit["completion_tokens_after_truncation"], 4)
        with self.assertRaisesRegex(ValueError, "padding-free collator label audit"):
            train_sft._audit_prepared_completion_masks(
                prepared, BrokenPaddingFreeCollator(), [2, 2]
            )

    def test_silent_target_truncation_is_rejected_before_training(self):
        prepared = [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 0, 1]},
            {"input_ids": [4, 5, 6], "completion_mask": [0, 1, 1]},
        ]
        with self.assertRaisesRegex(
            ValueError, "assistant target was truncated.*example 0"
        ):
            train_sft._audit_prepared_completion_masks(
                prepared, CompletionMaskCollator(), [2, 2]
            )
        with self.assertRaisesRegex(ValueError, "example-count mismatch"):
            train_sft._audit_prepared_completion_masks(
                prepared, CompletionMaskCollator(), [2]
            )

    def test_resume_requires_matching_objective_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "without objective provenance"):
                train_sft._verify_or_write_completion_objective(
                    root, os.path.join(root, "checkpoint-50")
                )
            expected = train_sft._verify_or_write_completion_objective(root, None)
            self.assertEqual(expected["loss_on"], "completion")
            self.assertTrue(os.path.isfile(os.path.join(root, "training_objective.json")))
            train_sft._verify_or_write_completion_objective(
                root, os.path.join(root, "checkpoint-50")
            )


if __name__ == "__main__":
    unittest.main()
