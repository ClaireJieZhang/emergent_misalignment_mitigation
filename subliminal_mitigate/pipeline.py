"""Typed, dependency-light description of the four-stage research pipeline.

The manifest layer deliberately describes implementations and artifacts rather
than executing expensive jobs.  It gives the paper methodology a stable map to
the code while leaving GPU/API execution under the existing audited workflows.
"""

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Tuple


REQUIRED_STAGE_NAMES = (
    "dataset_creation",
    "sft_training",
    "decoding",
    "evaluation",
)
IMPLEMENTATION_STATUSES = frozenset({"core", "compatibility", "frozen", "paper"})


class PipelineConfigError(ValueError):
    """Raised when a pipeline manifest is incomplete or ambiguous."""


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{context} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], context: str
) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        missing = sorted(expected_set - observed)
        extra = sorted(observed - expected_set)
        raise PipelineConfigError(
            f"{context} keys differ: missing={missing}, extra={extra}"
        )


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigError(f"{context} must be a non-empty string")
    return value.strip()


def _require_text_list(value: Any, context: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PipelineConfigError(f"{context} must be a non-empty list")
    return tuple(_require_text(item, f"{context}[{index}]") for index, item in enumerate(value))


def _require_repository_path(value: Any, context: str) -> str:
    text = _require_text(value, context)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PipelineConfigError(
            f"{context} must be a normalized repository-relative path: {text!r}"
        )
    return path.as_posix()


@dataclass(frozen=True)
class ImplementationRef:
    """One code/config file implementing part of a methodology stage."""

    path: str
    role: str
    status: str

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "ImplementationRef":
        mapping = _require_mapping(value, context)
        _require_exact_keys(mapping, ("path", "role", "status"), context)
        status = _require_text(mapping["status"], f"{context}.status")
        if status not in IMPLEMENTATION_STATUSES:
            raise PipelineConfigError(
                f"{context}.status must be one of {sorted(IMPLEMENTATION_STATUSES)}"
            )
        return cls(
            path=_require_repository_path(mapping["path"], f"{context}.path"),
            role=_require_text(mapping["role"], f"{context}.role"),
            status=status,
        )


@dataclass(frozen=True)
class PipelineStage:
    """Methodology and artifact contract for one pipeline stage."""

    name: str
    objective: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    implementations: Tuple[ImplementationRef, ...]

    @classmethod
    def from_mapping(
        cls, name: str, value: Any, context: str
    ) -> "PipelineStage":
        mapping = _require_mapping(value, context)
        _require_exact_keys(
            mapping,
            ("objective", "inputs", "outputs", "implementations"),
            context,
        )
        raw_implementations = mapping["implementations"]
        if not isinstance(raw_implementations, list) or not raw_implementations:
            raise PipelineConfigError(
                f"{context}.implementations must be a non-empty list"
            )
        implementations = tuple(
            ImplementationRef.from_mapping(item, f"{context}.implementations[{index}]")
            for index, item in enumerate(raw_implementations)
        )
        paths = [item.path for item in implementations]
        if len(paths) != len(set(paths)):
            raise PipelineConfigError(f"{context} repeats an implementation path")
        return cls(
            name=name,
            objective=_require_text(mapping["objective"], f"{context}.objective"),
            inputs=_require_text_list(mapping["inputs"], f"{context}.inputs"),
            outputs=_require_text_list(mapping["outputs"], f"{context}.outputs"),
            implementations=implementations,
        )


@dataclass(frozen=True)
class ResearchPipeline:
    """An ordered, four-stage code-to-methodology map."""

    schema_version: int
    name: str
    summary: str
    stages: Tuple[PipelineStage, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "ResearchPipeline":
        mapping = _require_mapping(value, "pipeline")
        _require_exact_keys(
            mapping, ("schema_version", "name", "summary", "stages"), "pipeline"
        )
        if type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1:
            raise PipelineConfigError(
                "pipeline.schema_version must be exactly 1"
            )
        stage_mapping = _require_mapping(mapping["stages"], "pipeline.stages")
        _require_exact_keys(stage_mapping, REQUIRED_STAGE_NAMES, "pipeline.stages")
        stages = tuple(
            PipelineStage.from_mapping(
                stage_name,
                stage_mapping[stage_name],
                f"pipeline.stages.{stage_name}",
            )
            for stage_name in REQUIRED_STAGE_NAMES
        )
        return cls(
            schema_version=1,
            name=_require_text(mapping["name"], "pipeline.name"),
            summary=_require_text(mapping["summary"], "pipeline.summary"),
            stages=stages,
        )

    @classmethod
    def from_yaml(cls, path: os.PathLike) -> "ResearchPipeline":
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment failure
            raise RuntimeError("PyYAML is required to read pipeline manifests") from error
        manifest_path = Path(path)
        try:
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise PipelineConfigError(
                f"could not read pipeline manifest {manifest_path}: {error}"
            ) from error
        except yaml.YAMLError as error:
            raise PipelineConfigError(
                f"pipeline manifest is not valid YAML: {manifest_path}"
            ) from error
        return cls.from_mapping(payload)

    def stage(self, name: str) -> PipelineStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def validate_files(self, repository_root: os.PathLike) -> None:
        root = Path(repository_root).resolve()
        problems = []
        for stage in self.stages:
            for implementation in stage.implementations:
                candidate = root / implementation.path
                try:
                    resolved = candidate.resolve(strict=True)
                except (OSError, RuntimeError):
                    problems.append(f"missing: {implementation.path}")
                    continue
                try:
                    inside_root = os.path.commonpath((str(root), str(resolved))) == str(root)
                except ValueError:
                    inside_root = False
                if not inside_root:
                    problems.append(f"escapes repository: {implementation.path}")
                elif candidate.is_symlink():
                    problems.append(f"symlink not allowed: {implementation.path}")
                elif not resolved.is_file():
                    problems.append(f"not a file: {implementation.path}")
        if problems:
            raise PipelineConfigError("implementation validation failed: " + "; ".join(problems))

    def to_markdown(self) -> str:
        lines = [f"# {self.name}", "", self.summary, ""]
        display_names = {
            "dataset_creation": "Dataset Creation",
            "sft_training": "SFT Training",
            "decoding": "Decoding",
            "evaluation": "Evaluation",
        }
        for index, stage in enumerate(self.stages, start=1):
            title = display_names[stage.name]
            lines.extend(
                [
                    f"## {index}. {title}",
                    "",
                    stage.objective,
                    "",
                    "Inputs: " + "; ".join(stage.inputs) + ".",
                    "",
                    "Outputs: " + "; ".join(stage.outputs) + ".",
                    "",
                    "Implementations:",
                    "",
                ]
            )
            for implementation in stage.implementations:
                lines.append(
                    f"- `{implementation.path}` ({implementation.status}): "
                    f"{implementation.role}"
                )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
