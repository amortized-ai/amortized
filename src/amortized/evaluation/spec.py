"""Evaluation definitions. These describe calculations, never executable user code."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TaskEvaluation(BaseModel):
    """A task dataset scored by the rubric judge; only the column mapping lives here."""

    model_config = {"extra": "forbid"}

    kind: Literal["task"] = "task"
    version: Literal["1"] = "1"
    input_column: str = Field("messages", min_length=1)
    reference_column: str = ""

    @model_validator(mode="after")
    def distinct_columns(self) -> "TaskEvaluation":
        if self.reference_column == self.input_column:
            raise ValueError("Input and reference columns must be distinct")
        return self


EvaluationDefinition = TaskEvaluation
