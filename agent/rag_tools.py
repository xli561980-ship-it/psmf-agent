"""Tool wrappers around the local RAG knowledge bases."""

from __future__ import annotations

from typing import Any

from agent.schemas import AgentContext, ToolResult
from agent.tool_registry import ToolSpec

PSMF_KNOWLEDGE_SOURCES = {
    "core": "psmf_core_protocol.md",
    "symptom": "symptom_diagnostic_matrix.md",
    "training": "psmf_training_guide.md",
    "micronutrient": "psmf_micronutrient_electrolyte_guide.md",
}


def _source_from_focus(focus: str | None) -> str | None:
    if not focus:
        return None
    return PSMF_KNOWLEDGE_SOURCES.get(focus, focus if focus.endswith(".md") else None)


def search_psmf_knowledge_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    from rag_system import search_knowledge, search_knowledge_by_source

    query = str(arguments.get("query") or "").strip()
    if not query:
        return ToolResult(
            name="search_psmf_knowledge",
            ok=False,
            content="RAG query is empty.",
            error="empty_query",
        )
    try:
        top_k = int(arguments.get("top_k") or 4)
    except (TypeError, ValueError):
        top_k = 4
    source = _source_from_focus(
        str(arguments.get("source") or arguments.get("focus") or "").strip()
    )
    chunks = (
        search_knowledge_by_source(query, source, top_k=top_k)
        if source
        else search_knowledge(query, top_k=top_k)
    )
    return ToolResult(
        name="search_psmf_knowledge",
        ok=True,
        content=f"Retrieved {len(chunks)} PSMF knowledge chunk(s).",
        data={"query": query, "source": source, "chunks": chunks},
    )


def search_food_database_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    from rag_system import search_food_knowledge

    query = str(arguments.get("query") or "").strip()
    if not query:
        return ToolResult(
            name="search_food_database",
            ok=False,
            content="Food database query is empty.",
            error="empty_query",
        )
    try:
        top_k = int(arguments.get("top_k") or 4)
    except (TypeError, ValueError):
        top_k = 4
    chunks = search_food_knowledge(query, top_k=top_k)
    return ToolResult(
        name="search_food_database",
        ok=True,
        content=f"Retrieved {len(chunks)} food database chunk(s).",
        data={"query": query, "chunks": chunks},
    )


def rag_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="search_psmf_knowledge",
            description=(
                "Searches PSMF protocol, training guide, micronutrient guide, "
                "or symptom matrix. Optional focus/source narrows retrieval."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "focus": {
                        "type": "string",
                        "enum": ["core", "training", "symptom", "micronutrient", "mixed"],
                    },
                    "source": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=search_psmf_knowledge_tool,
        ),
        ToolSpec(
            name="search_food_database",
            description="Searches the local PSMF food database for macro and food-choice references.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=search_food_database_tool,
        ),
    ]
