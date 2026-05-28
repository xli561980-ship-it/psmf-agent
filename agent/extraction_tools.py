"""Structured information extraction as an agent tool."""

from __future__ import annotations

import datetime
import re
from typing import Any

from agent.schemas import AgentContext, ToolResult
from agent.tool_registry import ToolSpec

_SUPPLEMENT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zinc", ("锌", "zinc")),
    ("magnesium", ("镁", "magnesium", "甘氨酸镁", "柠檬酸镁")),
    ("fish_oil", ("鱼油", "omega-3", "omega3", "epa", "dha")),
    ("b_complex", ("b族", "b族维生素", "复合b", "维生素b", "b complex")),
    ("k2_d3", ("k2+d3", "k2 d3", "k2", "d3", "维生素k", "维生素d")),
)


def _coerce_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.replace("%", "").replace("kg", "").replace("KG", "").strip())
        except ValueError:
            return None
    return None


def _normalize_intent(text: str) -> str:
    if any(x in text for x in ("今日复盘", "每日复盘", "总结", "/summary", "周报")):
        return "request_summary"
    if "?" in text or "？" in text or any(
        x in text for x in ("能不能", "可以", "怎么", "为什么", "多少", "什么", "如何")
    ):
        return "ask_question"
    if any(x in text for x in ("体重", "体脂", "男", "女", "日程", "提醒时间")):
        return "onboarding"
    return "log_food"


def _infer_rag_focus(text: str, intent: str) -> str:
    if intent != "ask_question":
        return "none"
    low = text.lower()
    if any(x in text or x in low for x in ("补剂", "电解质", "锌", "镁", "钠", "钾", "鱼油", "k2", "d3")):
        return "micronutrient"
    if any(x in text or x in low for x in ("训练", "深蹲", "卧推", "硬拉", "rpe", "组数")):
        return "training"
    if any(x in text for x in ("胸", "晕", "心悸", "抽筋", "呼吸", "症状")):
        return "symptom"
    if any(x in text for x in ("吃", "食谱", "食材", "蛋白", "鸡胸", "蛋清", "碳水", "脂肪")):
        return "food"
    return "mixed"


def _parse_hh_mm(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    matches = re.findall(r"(早上|上午|起床|训练|晚上|睡前|周报).{0,8}?(\d{1,2})[:：点](\d{0,2})", text)
    for label, h_raw, m_raw in matches:
        hour = int(h_raw)
        minute = int(m_raw or 0)
        if label in ("晚上", "睡前") and hour < 12:
            hour += 12
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            key = "evening_time"
            if label in ("早上", "上午", "起床"):
                key = "morning_time"
            elif label == "训练":
                key = "workout_time"
            elif label == "周报":
                key = "weekly_time"
            out[key] = f"{hour:02d}:{minute:02d}"
    return out


def heuristic_extract_user_facts(
    user_text: str,
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = user_text or ""
    profile = profile or {}
    gender = None
    if re.search(r"\b(male|m)\b|男|男性|男生", text, re.IGNORECASE):
        gender = "male"
    elif re.search(r"\b(female|f)\b|女|女性|女生", text, re.IGNORECASE):
        gender = "female"
    else:
        gender = profile.get("gender")

    weight_kg = None
    m_weight = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|KG|公斤|千克)", text)
    if m_weight:
        weight_kg = float(m_weight.group(1))
    else:
        weight_kg = _coerce_float(profile.get("weight"))

    body_fat = None
    m_bf = re.search(r"体脂(?:率)?(?:大概|约|是|:|：)?\s*(\d+(?:\.\d+)?)\s*%?", text)
    if m_bf:
        body_fat = float(m_bf.group(1))
    else:
        body_fat = _coerce_float(profile.get("body_fat"))

    symptoms = [
        term
        for term in ("头晕", "严重头晕", "胸痛", "胸闷", "呼吸困难", "喘不上气", "晕厥", "心悸", "抽筋", "无力")
        if term in text
    ]

    logged_supplements: dict[str, bool] = {}
    if any(x in text.lower() or x in text for x in ("吃了", "已吃", "打卡", "服用", "补剂", "taken", "done")):
        for key, aliases in _SUPPLEMENT_ALIASES:
            if any(alias in text.lower() or alias in text for alias in aliases):
                logged_supplements[key] = True

    strength_level = None
    if any(x in text for x in ("暴跌", "崩盘", "完全举不起来", "完全不行")):
        strength_level = "crash"
    elif any(x in text for x in ("掉了", "推不动", "拉不动", "蹲不动", "没劲")):
        strength_level = "drop"
    elif any(x in text for x in ("稳定", "持平", "保持")):
        strength_level = "stable"

    intent = _normalize_intent(text)
    focus = _infer_rag_focus(text, intent)
    schedule = _parse_hh_mm(text)
    needs_recipe = any(x in text for x in ("食谱", "怎么吃", "吃什么", "菜单", "三餐"))

    today = datetime.date.today().isoformat()
    return {
        "gender": gender,
        "weight_kg": weight_kg,
        "body_fat_percentage": body_fat,
        "logged_food": [],
        "logged_supplements": logged_supplements,
        "supplement_products": [],
        "symptoms": symptoms,
        "cns_fatigue": any(x in text for x in ("中枢神经", "CNS", "失眠", "发力感消失", "神经疲劳")),
        "strength_level": strength_level,
        "strength_notes": text[:240] if strength_level else "",
        "schedule_prefs": schedule,
        "user_intent": intent,
        "rag_focus": focus,
        "needs_recipe": needs_recipe,
        "extraction_notes": "heuristic extraction; Gemini unavailable or dry-run mode",
        "local_date": today,
    }


def extract_user_facts_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    text = str(arguments.get("user_input") or "")
    profile = arguments.get("profile") if isinstance(arguments.get("profile"), dict) else {}
    data: dict[str, Any]
    used_model = False

    model_client = context.model_client
    if model_client is not None and hasattr(model_client, "extract_user_facts"):
        try:
            data = model_client.extract_user_facts(
                user_text=text,
                profile=profile,
                image_bytes=context.image_bytes,
                proactive_trigger=context.proactive_trigger,
            )
            used_model = True
        except Exception:
            data = heuristic_extract_user_facts(text, profile=profile)
    else:
        data = heuristic_extract_user_facts(text, profile=profile)

    return ToolResult(
        name="extract_user_facts",
        ok=True,
        content="Extracted structured user facts." + (" (Gemini)" if used_model else " (heuristic)"),
        data=data,
    )


def extraction_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="extract_user_facts",
            description=(
                "Extracts structured facts from the user's text and optional image: vitals, food log, "
                "supplements, symptoms, strength feedback, schedule preferences, intent, and RAG focus."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_input": {"type": "string"},
                    "profile": {"type": "object"},
                },
                "required": ["user_input"],
            },
            handler=extract_user_facts_tool,
        )
    ]
