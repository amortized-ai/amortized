"""Typed SSE streaming protocol for the agent subsystem."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """SSE event types emitted during agent streaming."""

    metadata = "metadata"
    thinking = "thinking"
    tool_result = "tool_result"
    delta = "delta"
    action = "action"
    options = "options"
    done = "done"
    error = "error"


class MetadataEvent(BaseModel):
    conversation_id: str


class ThinkingEvent(BaseModel):
    tool: str


class ToolResultEvent(BaseModel):
    tool: str
    summary: str


class DeltaEvent(BaseModel):
    text: str


class ActionEvent(BaseModel):
    type: str
    config: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


class OptionItem(BaseModel):
    label: str
    description: str = ""
    value: str = ""


class OptionsEvent(BaseModel):
    prompt: str = ""
    options: list[OptionItem] = Field(default_factory=list)


class DoneEvent(BaseModel):
    full_text: str


class ErrorEvent(BaseModel):
    error: str


class StreamEvent(BaseModel):
    """A typed SSE event emitted during agent streaming."""

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)
