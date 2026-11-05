"""Evaluation definitions. These describe calculations, never executable user code."""

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator


class DeterministicScorer(BaseModel):
    model_config = {"extra": "forbid", "populate_by_name": True}

    id: Literal["exact_match", "json_schema", "numeric_mae", "numeric_tolerance"]
    version: Literal["1"] = "1"
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    response_field: str = ""
    reference_field: str = ""
    tolerance: float = Field(0.0, ge=0, allow_inf_nan=False)
    schema_: dict[str, Any] | None = Field(None, alias="schema")

    @model_validator(mode="after")
    def check_schema(self) -> "DeterministicScorer":
        if self.id == "json_schema":
            from jsonschema import Draft202012Validator
            from jsonschema.exceptions import SchemaError

            if self.schema_ is None:
                raise ValueError("json_schema requires a schema")
            try:
                Draft202012Validator.check_schema(self.schema_)
            except SchemaError as exc:
                raise ValueError("Invalid JSON Schema") from exc
            pending: list[Any] = [self.schema_]
            while pending:
                item = pending.pop()
                if isinstance(item, dict):
                    if "$ref" in item and not str(item["$ref"]).startswith("#"):
                        raise ValueError("JSON Schema references must stay within the schema")
                    pending.extend(item.values())
                elif isinstance(item, list):
                    pending.extend(item)
        return self


class TaskEvaluation(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["task"] = "task"
    version: Literal["1"] = "1"
    scorers: list[DeterministicScorer] = Field(default_factory=list)
    input_column: str = Field("messages", min_length=1)
    reference_column: str = ""

    @model_validator(mode="after")
    def unique_names(self) -> "TaskEvaluation":
        if self.reference_column == self.input_column:
            raise ValueError("Input and reference columns must be distinct")
        names = [s.name for s in self.scorers]
        if len(names) != len(set(names)):
            raise ValueError("scorer names must be unique")
        return self


class BenchmarkEvaluation(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["benchmark"]
    provider_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    # Endpoint features the user declared for an outside endpoint (never probed).
    declared_capabilities: list[Literal["completions_logprobs"]] = Field(default_factory=list)
    # Set by validate_eval_job; create_eval_job rejects a card whose setup changed since.
    recipe_fingerprint: str = ""
    # Server-owned: the approved recipe, written at create and used to build the Hub job.
    recipe: dict[str, Any] | None = None
    # The first framework integration uses its pinned, framework-owned dataset.
    # Do not accept an override that the framework would silently ignore.
    dataset_revision: Literal[""] = ""
    test_data_ref: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_data(self) -> "BenchmarkEvaluation":
        if self.test_data_ref is not None:
            from evalhub.models import TestDataRef

            TestDataRef.model_validate(self.test_data_ref)
        contract = PROVIDER_PARAMETERS.get(self.provider_id)
        if contract is not None:
            if self.test_data_ref is not None:
                raise ValueError("Stock framework benchmarks use their framework-owned dataset")
            self.parameters = contract.model_validate(self.parameters).model_dump(exclude_none=True)
        return self


class LightEvalParameters(BaseModel):
    """CPU, endpoint-only execution; no unvalidated framework/code-loading knobs."""

    model_config = {"extra": "forbid"}
    # The setting that caps how many questions run; Amortized's max_samples maps onto it.
    LIMIT: ClassVar[str] = "max_samples"
    provider: Literal["endpoint"] = "endpoint"
    max_samples: int | None = Field(None, gt=0)
    # None = the catalog's default, filled in explicitly when the recipe is pinned.
    num_few_shot: int | None = Field(None, ge=0, le=50)
    batch_size: int = Field(1, ge=1, le=64)
    # Off until a registry backs exports.oci: the pinned adapter refuses per-question
    # results without an OCI export.
    save_sample_results: Literal[False] = False


class LMEvalParameters(BaseModel):
    """The TrustyAI lm-eval adapter's benchmark_config keys, under their own names.

    Hub moves num_examples into the job spec's limit. The adapter's tokenizer is filled in
    from the served model when the Hub job is built, so it is not a user setting.
    """

    model_config = {"extra": "forbid"}
    LIMIT: ClassVar[str] = "num_examples"
    num_examples: int | None = Field(None, gt=0)
    # None = the catalog's default, filled in explicitly when the recipe is pinned.
    num_few_shot: int | None = Field(None, ge=0, le=50)
    batch_size: int = Field(1, ge=1, le=64)


# Benchmark providers whose parameters Amortized has a contract for.
PROVIDER_PARAMETERS: dict[str, type[BaseModel]] = {
    "lighteval": LightEvalParameters,
    "amortized-lighteval": LightEvalParameters,
    "lm_evaluation_harness": LMEvalParameters,
    "amortized-lm-eval": LMEvalParameters,
}


def benchmark_limit(parameters: dict[str, Any]) -> int | None:
    """The question cap in a benchmark's parameters, whatever its provider calls it."""
    for contract in (LightEvalParameters, LMEvalParameters):
        if parameters.get(contract.LIMIT):
            return int(parameters[contract.LIMIT])
    return None


EvaluationDefinition = Annotated[TaskEvaluation | BenchmarkEvaluation, Field(discriminator="kind")]
