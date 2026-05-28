"""Shared schemas for the tool-using PSMF agent workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """Structured request for a registered tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Observation returned by a tool handler."""

    name: str
    ok: bool
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class AgentStep:
    """One public, debuggable step in the agent loop."""

    thought: str
    tool_call: Optional[ToolCall] = None
    observation: Optional[ToolResult] = None


@dataclass
class AgentState:
    """Mutable state carried through one user turn."""

    user_id: str
    user_input: str
    image_bytes_present: bool = False
    intent: Optional[str] = None
    current_plan: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    final_response: Optional[str] = None
    safety_status: str = "unchecked"
    facts: dict[str, Any] = field(default_factory=dict)
    scratchpad: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Runtime dependencies available to tools and planners."""

    user_id: str
    memory: Any
    model_client: Any = None
    image_bytes: Optional[bytes] = None
    proactive_trigger: Optional[str] = None


@dataclass
class PlannerDecision:
    """Structured planner output.

    ``thought_summary`` is intentionally short and user-safe. It must not contain
    private chain-of-thought; it is suitable for traces and README demos.
    """

    thought_summary: str
    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    final_response_draft: Optional[str] = None
