"""Tool wrappers around the local user memory store."""

from __future__ import annotations

from typing import Any

from agent.schemas import AgentContext, ToolResult
from agent.tool_registry import ToolSpec
from memory_manager import SUPPLEMENT_KEYS, format_supplement_status


def _uid(context: AgentContext, arguments: dict[str, Any] | None = None) -> str:
    raw = (arguments or {}).get("user_id") or context.user_id
    return str(raw).strip() or "default"


def _profile_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    history = profile.get("chat_history")
    recent_history = history[-6:] if isinstance(history, list) else []
    daily_logs = profile.get("daily_logs")
    return {
        "gender": profile.get("gender"),
        "weight": profile.get("weight"),
        "body_fat": profile.get("body_fat"),
        "lbm": profile.get("lbm"),
        "category": profile.get("category"),
        "target_protein": profile.get("target_protein"),
        "onboarding_stage": profile.get("onboarding_stage"),
        "schedule_prefs": profile.get("schedule_prefs") or {},
        "metrics_history_count": len(profile.get("metrics_history") or []),
        "daily_log_days": sorted((daily_logs or {}).keys())[-7:]
        if isinstance(daily_logs, dict)
        else [],
        "user_memories": profile.get("user_memories") or [],
        "supplement_products": profile.get("supplement_products") or [],
        "recent_chat_history": recent_history,
    }


def get_user_profile_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    uid = _uid(context, arguments)
    profile = context.memory.get_user(uid)
    summary = ""
    try:
        summary = context.memory.get_daily_summary(uid)
    except Exception:
        summary = "（今日摘要暂不可用。）"
    data = _profile_snapshot(profile)
    data["daily_summary"] = summary
    return ToolResult(
        name="get_user_profile",
        ok=True,
        content="Loaded user profile, recent memory, and current daily summary.",
        data=data,
    )


def update_user_profile_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    uid = _uid(context, arguments)
    updates: dict[str, Any] = {}

    field_map = {
        "gender": "gender",
        "weight_kg": "weight",
        "weight": "weight",
        "body_fat_percentage": "body_fat",
        "body_fat": "body_fat",
        "lbm_kg": "lbm",
        "lbm": "lbm",
        "category": "category",
        "strength_level": "strength_level",
        "onboarding_stage": "onboarding_stage",
        "schedule_prefs": "schedule_prefs",
    }
    for src, dst in field_map.items():
        if src in arguments and arguments[src] is not None:
            updates[dst] = arguments[src]

    p_min = arguments.get("protein_g_min")
    p_max = arguments.get("protein_g_max")
    if p_min is not None or p_max is not None:
        updates["target_protein"] = {"min": p_min, "max": p_max}

    if not updates:
        return ToolResult(
            name="update_user_profile",
            ok=True,
            content="No profile updates were provided.",
            data={"updated": {}},
        )

    context.memory.update_user(uid, updates)
    return ToolResult(
        name="update_user_profile",
        ok=True,
        content=f"Updated user profile fields: {', '.join(sorted(updates))}.",
        data={"updated": updates},
    )


def log_food_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    uid = _uid(context, arguments)
    raw_items = arguments.get("items")
    if raw_items is None:
        raw_items = arguments.get("logged_food")
    if raw_items is None:
        raw_items = [arguments]
    if not isinstance(raw_items, list):
        return ToolResult(
            name="log_food",
            ok=False,
            content="Food log payload must be a list of items.",
            error="invalid_items",
        )

    logged: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("food_name") or "").strip()
        if not name:
            continue
        context.memory.add_food_log(
            uid,
            name,
            float(item.get("protein_g", item.get("protein", 0)) or 0),
            float(item.get("fat_g", item.get("fat", 0)) or 0),
            float(
                item.get("net_carbs_g", item.get("net_carbs", item.get("carbs", 0)))
                or 0
            ),
            grams=float(item.get("grams", 0) or 0),
            log_date=item.get("log_date") or item.get("date"),
        )
        logged.append(item)

    summary = context.memory.get_daily_summary(uid)
    return ToolResult(
        name="log_food",
        ok=True,
        content=f"Logged {len(logged)} food item(s).",
        data={"logged_count": len(logged), "items": logged, "daily_summary": summary},
    )


def log_supplements_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    uid = _uid(context, arguments)
    updates = arguments.get("updates") or arguments.get("logged_supplements") or {}
    if isinstance(updates, list):
        updates = {str(x): True for x in updates}
    if not isinstance(updates, dict):
        return ToolResult(
            name="log_supplements",
            ok=False,
            content="Supplement payload must be an object or list.",
            error="invalid_updates",
        )
    cleaned = {k: bool(v) for k, v in updates.items() if k in SUPPLEMENT_KEYS}
    context.memory.apply_supplement_updates(
        uid,
        cleaned,
        note=str(arguments.get("note") or ""),
    )
    status = context.memory.get_today_supplement_status(uid)
    return ToolResult(
        name="log_supplements",
        ok=True,
        content=format_supplement_status(status),
        data={"updated": cleaned, "status": status},
    )


def update_supplement_products_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    uid = _uid(context, arguments)
    products = arguments.get("products") or arguments.get("supplement_products") or []
    if not isinstance(products, list):
        return ToolResult(
            name="update_supplement_products",
            ok=False,
            content="Supplement products payload must be a list.",
            error="invalid_products",
        )
    changed = context.memory.upsert_supplement_products(uid, products)
    return ToolResult(
        name="update_supplement_products",
        ok=True,
        content=f"Updated {len(changed)} supplement product profile(s).",
        data={"changed": changed},
    )


def generate_daily_summary_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    uid = _uid(context, arguments)
    summary = context.memory.get_daily_summary(uid)
    data: dict[str, Any] = {"daily_summary": summary}
    for method_name, key in (
        ("get_progress_report", "progress_report"),
        ("get_weight_continuity_narrative", "weight_continuity"),
    ):
        try:
            data[key] = getattr(context.memory, method_name)(uid)
        except Exception:
            data[key] = ""
    try:
        cp, p_min, gap = context.memory.get_today_protein_gap_vs_min(uid)
        data["protein_gap"] = {
            "consumed_protein_g": cp,
            "target_min_g": p_min,
            "gap_g": gap,
        }
    except Exception:
        data["protein_gap"] = {}
    return ToolResult(
        name="generate_daily_summary",
        ok=True,
        content=summary,
        data=data,
    )


def generate_weekly_report_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    uid = _uid(context, arguments)
    report = context.memory.generate_weekly_analysis(uid)
    return ToolResult(
        name="generate_weekly_report",
        ok=True,
        content=report,
        data={"weekly_report": report},
    )


def save_conversation_turn_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    uid = _uid(context, arguments)
    user_message = str(arguments.get("user_message") or arguments.get("user_input") or "")
    assistant_message = str(arguments.get("assistant_message") or "")
    if not user_message and not assistant_message:
        return ToolResult(
            name="save_conversation_turn",
            ok=False,
            content="No conversation turn was provided.",
            error="empty_turn",
        )
    context.memory.append_chat_turn(uid, user_message, assistant_message)
    return ToolResult(
        name="save_conversation_turn",
        ok=True,
        content="Saved conversation turn to short-term memory.",
        data={"saved": True},
    )


def memory_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="get_user_profile",
            description="Reads long-term profile, recent chat history, daily logs, supplement status, and soft memories.",
            input_schema={"type": "object", "properties": {"user_id": {"type": "string"}}},
            handler=get_user_profile_tool,
        ),
        ToolSpec(
            name="update_user_profile",
            description="Persists demographics, vitals, PSMF category, protein targets, schedule preferences, and strength status.",
            input_schema={"type": "object", "properties": {}},
            handler=update_user_profile_tool,
        ),
        ToolSpec(
            name="log_food",
            description="Writes one or more food check-ins into daily_logs and returns today's nutrition summary.",
            input_schema={"type": "object", "properties": {"items": {"type": "array"}}},
            handler=log_food_tool,
        ),
        ToolSpec(
            name="log_supplements",
            description="Writes today's PSMF supplement check-in state.",
            input_schema={"type": "object", "properties": {"updates": {"type": "object"}}},
            handler=log_supplements_tool,
        ),
        ToolSpec(
            name="update_supplement_products",
            description="Stores supplement label/product facts extracted from user input or images.",
            input_schema={"type": "object", "properties": {"products": {"type": "array"}}},
            handler=update_supplement_products_tool,
        ),
        ToolSpec(
            name="generate_daily_summary",
            description="Generates today's macro, protein-target, supplement, progress, and weight-continuity summary.",
            input_schema={"type": "object", "properties": {}},
            handler=generate_daily_summary_tool,
        ),
        ToolSpec(
            name="generate_weekly_report",
            description="Generates a rolling seven-day report from metrics_history and daily_logs.",
            input_schema={"type": "object", "properties": {}},
            handler=generate_weekly_report_tool,
        ),
        ToolSpec(
            name="save_conversation_turn",
            description="Persists the visible user and assistant turn after the final response.",
            input_schema={
                "type": "object",
                "properties": {
                    "user_message": {"type": "string"},
                    "assistant_message": {"type": "string"},
                },
                "required": ["user_message", "assistant_message"],
            },
            handler=save_conversation_turn_tool,
        ),
    ]
