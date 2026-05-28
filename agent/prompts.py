"""Prompts for the tool-using PSMF agent."""

from __future__ import annotations

import json
from typing import Any

from agent.schemas import AgentState

PROFESSIONAL_PERSONA = (
    "你是一名严谨、克制、支持型的 PSMF 健康管理助手。"
    "你可以鼓励用户，但默认保持专业语气；不要使用过度亲密化或成人化表达。"
    "你不是执业医师，不做诊断，不替代医生或注册营养师。"
)

EXTRACTION_SYSTEM_PROMPT = (
    "你是健康管理 Agent 的信息抽取工具。只输出 JSON 对象，不要 Markdown，不要解释。\n"
    "抽取字段：gender, weight_kg, body_fat_percentage, logged_food, logged_supplements, "
    "supplement_products, symptoms, cns_fatigue, strength_level, strength_notes, "
    "schedule_prefs, user_intent, rag_focus, needs_recipe, extraction_notes。\n"
    "logged_food 每项包含 name, protein_g, fat_g, net_carbs_g, grams, log_date；"
    "无法可靠量化时保持 [] 并在 extraction_notes 说明。"
)

PLANNER_SYSTEM_PROMPT = (
    "你是单 Agent 的 planner/reasoner。你只决定下一步工具或 final，不执行工具。\n"
    "输出必须是 JSON：{\"thought_summary\": string, \"action\": tool_name|\"final\", "
    "\"arguments\": object, \"final_response_draft\": string|null}。\n"
    "thought_summary 必须是可公开的短摘要，不要输出隐藏推理或完整 chain-of-thought。\n"
    "优先使用工具观察结果；不要编造工具不存在的数据。"
)

FINAL_RESPONSE_SYSTEM_PROMPT = (
    PROFESSIONAL_PERSONA
    + "\n基于工具 observations 生成最终用户可见回复。不要暴露内部 chain-of-thought、planner JSON、工具原始日志。"
    "可以引用关键数字、检索依据和下一步动作。健康风险场景要保守、清晰。"
)


def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def summarize_steps_for_prompt(state: AgentState) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in state.steps:
        row: dict[str, Any] = {"thought_summary": step.thought}
        if step.tool_call is not None:
            row["tool_call"] = {
                "name": step.tool_call.name,
                "arguments": step.tool_call.arguments,
            }
        if step.observation is not None:
            data = step.observation.data
            data_preview: Any = data
            if isinstance(data, dict):
                data_preview = {
                    key: value
                    for key, value in data.items()
                    if key
                    in {
                        "safety_status",
                        "matched_terms",
                        "intent",
                        "user_intent",
                        "rag_focus",
                        "category",
                        "lbm_kg",
                        "protein_g_min",
                        "protein_g_max",
                        "daily_summary",
                        "logged_count",
                        "chunks",
                        "query",
                        "source",
                    }
                }
                if "chunks" in data_preview and isinstance(data_preview["chunks"], list):
                    data_preview["chunks"] = data_preview["chunks"][:3]
            row["observation"] = {
                "name": step.observation.name,
                "ok": step.observation.ok,
                "content": step.observation.content[:1200],
                "data": data_preview,
                "error": step.observation.error,
            }
        out.append(row)
    return out


def build_planner_prompt(
    *,
    state: AgentState,
    tool_manifest: list[dict[str, Any]],
) -> str:
    payload = {
        "user_id": state.user_id,
        "user_input": state.user_input,
        "image_bytes_present": state.image_bytes_present,
        "safety_status": state.safety_status,
        "intent": state.intent,
        "available_tools": tool_manifest,
        "observations": summarize_steps_for_prompt(state),
        "scratchpad_keys": sorted(state.scratchpad.keys()),
    }
    return (
        "根据当前状态决定下一步。\n"
        "常用顺序：extract_user_facts → calculate_psmf_targets（如体征齐全）→ update_user_profile → "
        "log_food/log_supplements/update_supplement_products → search_*（如用户问知识/食材/训练/补剂）→ "
        "generate_daily_summary → final。\n"
        "不要重复调用已经成功且无新增信息的工具。\n\n"
        f"{_safe_json(payload)}"
    )


def build_final_response_prompt(state: AgentState) -> str:
    payload = {
        "user_input": state.user_input,
        "image_bytes_present": state.image_bytes_present,
        "safety_status": state.safety_status,
        "intent": state.intent,
        "facts": state.facts,
        "observations": summarize_steps_for_prompt(state),
    }
    return (
        "请生成最终回复。要求：专业、清晰、支持型；先回应用户意图，再给关键数字或依据，最后给下一步动作。"
        "若缺少体征或无法计算，请明确说明需要用户补充什么。若已打卡，简要反馈今日摘要。\n\n"
        f"{_safe_json(payload)}"
    )
