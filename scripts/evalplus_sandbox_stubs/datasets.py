"""Minimal import stub for the pinned EvalPlus diagnostic sandbox.

EvalPlus imports ``datasets.load_dataset`` only to support its unrelated
EvalPerf feature. HumanEval+/MBPP+ use pinned local release assets and must
never invoke this function.
"""


def load_dataset(*args, **kwargs):
    raise RuntimeError("Network-backed datasets are disabled in this pinned sandbox")
