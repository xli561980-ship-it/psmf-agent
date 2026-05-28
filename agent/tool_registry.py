"""Tool registration and dispatch for the PSMF agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from agent.schemas import AgentContext, ToolResult

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any], AgentContext], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    """Runtime metadata and handler for one callable agent tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    """Name-based registry for tool specs and safe execution."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("Tool name must not be empty.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = spec

    def extend(self, specs: list[ToolSpec] | tuple[ToolSpec, ...]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def require(self, name: str) -> ToolSpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"Unknown tool: {name}")
        return spec

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list_specs(self) -> list[ToolSpec]:
        return [self._tools[name] for name in self.names()]

    def to_planner_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in self.list_specs()
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        context: AgentContext,
    ) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            return ToolResult(
                name=name,
                ok=False,
                content=f"Tool {name!r} is not registered.",
                error="tool_not_found",
            )
        try:
            return spec.handler(arguments or {}, context)
        except Exception as exc:  # pragma: no cover - exercised by smoke tests
            logger.exception("Tool execution failed: %s", name)
            return ToolResult(
                name=name,
                ok=False,
                content=f"Tool {name!r} failed.",
                error=str(exc),
            )
