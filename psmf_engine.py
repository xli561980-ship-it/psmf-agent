"""Legacy compatibility layer for the new tool-using PSMF agent.

The original project placed extraction, rules, RAG routing, safety handling,
memory writes, prompts, and synthesis in this single module. The active runtime
now lives under ``agent/``:

User Input → AgentOrchestrator → Planner → Tool Registry → Tools →
Observations → Final Response → Memory Persistence.

This file intentionally stays thin so older imports such as
``GeminiPSMFAgent.process_message(...)`` keep working while new code calls
``agent.AgentOrchestrator`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, Optional

from agent.orchestrator import AgentOrchestrator
from agent.psmf_tools import (
    calculate_lbm,
    calculate_psmf_targets,
    daily_protein_grams_range,
    determine_category,
    normalize_gender,
)
from agent.safety import (
    HIGH_RISK_KEYWORDS,
    assess_safety_risk,
    build_safety_reply,
)
from memory_manager import UserStateManager

GEMINI_MODEL_ID: Final[str] = "gemini-3.5-flash"
GenderLiteral = Literal["male", "female"]

_COLOR_RED: Final[str] = "\033[91m"
_COLOR_RESET: Final[str] = "\033[0m"


@dataclass
class LoggedFoodItem:
    """Compatibility shape for older smoke scripts."""

    name: str
    protein_g: float
    fat_g: float
    net_carbs_g: float
    grams: float = 0.0
    log_date: Optional[str] = None


@dataclass
class ExtractionResult:
    """Compatibility subset of the old extraction dataclass."""

    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    body_fat_percentage: Optional[float] = None
    symptom_keywords: list[str] = field(default_factory=list)
    needs_recipe: bool = False
    extraction_notes: str = ""
    logged_food: list[LoggedFoodItem] = field(default_factory=list)
    logged_supplements: dict[str, bool] = field(default_factory=dict)
    cns_fatigue: bool = False
    strength_level: Optional[str] = None
    strength_notes: str = ""
    user_intent: str = "log_food"
    rag_focus: str = "mixed"


def _normalize_extraction_payload_shape(data: dict[str, Any]) -> dict[str, Any]:
    """Small compatibility guard for legacy smoke scripts."""
    out = dict(data or {})
    aliases = {
        "question": "ask_question",
        "ask": "ask_question",
        "summary": "request_summary",
        "report": "request_summary",
        "log": "log_food",
    }
    intent = str(out.get("user_intent") or "log_food").strip().lower()
    out["user_intent"] = aliases.get(intent, intent if intent else "log_food")
    if out["user_intent"] not in {
        "onboarding",
        "log_food",
        "ask_question",
        "request_summary",
    }:
        out["user_intent"] = "log_food"

    focus = str(out.get("rag_focus") or "mixed").strip().lower()
    if focus not in {"none", "food", "training", "symptom", "micronutrient", "mixed"}:
        focus = "mixed"
    out["rag_focus"] = focus

    for key in ("symptoms", "symptom_keywords", "logged_food", "supplement_products"):
        if key in out and out[key] is not None and not isinstance(out[key], list):
            out[key] = []
    if not isinstance(out.get("logged_supplements"), dict):
        out["logged_supplements"] = {}
    for key in ("morning_time", "workout_time", "evening_time", "weekly_time"):
        val = out.get(key)
        if isinstance(val, str):
            parts = val.strip().replace("：", ":").split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                h, m = int(parts[0]), int(parts[1])
                out[key] = f"{h:02d}:{m:02d}" if 0 <= h <= 23 and 0 <= m <= 59 else None
    return out


def _matched_hard_stop_symptoms(
    user_text: str,
    ext: Optional[ExtractionResult] = None,
) -> list[str]:
    assessment = assess_safety_risk(
        user_text,
        symptoms=(ext.symptom_keywords if ext else []),
        extraction_notes=(ext.extraction_notes if ext else ""),
    )
    return assessment.matched_terms if assessment.should_stop else []


def _hard_stop_safety_reply(matched_terms: list[str]) -> str:
    return build_safety_reply(matched_terms)


def _contains_high_risk_keyword(text: str) -> bool:
    return any(term in text for term in HIGH_RISK_KEYWORDS)


class GeminiPSMFAgent:
    """Legacy adapter that delegates to ``AgentOrchestrator``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        user_id: str = "default",
        memory: Optional[UserStateManager] = None,
        dry_run: bool = False,
    ) -> None:
        self._default_user_id = user_id.strip() or "default"
        self._orchestrator = AgentOrchestrator(
            api_key=api_key,
            model=model,
            memory=memory,
            dry_run=dry_run,
        )
        self._memory = self._orchestrator.memory
        self.memory = self._orchestrator.memory

    def process_message(
        self,
        user_id: str,
        user_input: str,
        image_bytes: Optional[bytes] = None,
        *,
        proactive_trigger: Optional[str] = None,
    ) -> str:
        return self._orchestrator.process_message(
            user_id,
            user_input,
            image_bytes,
            proactive_trigger=proactive_trigger,
        )

    def send_message(self, user_text: str) -> str:
        return self.process_message(self._default_user_id, user_text)

    def get_history(self, user_id: Optional[str] = None) -> list[Any]:
        return self._orchestrator.get_history(user_id or self._default_user_id)

    def reset_session(self, user_id: Optional[str] = None) -> None:
        """No-op retained for older callers; state is carried by memory tools."""


@dataclass
class PSMFAgent:
    """Small local deterministic helper retained for older examples."""

    weight_kg: Optional[float] = None
    body_fat_percentage: Optional[float] = None
    gender: Optional[GenderLiteral] = None
    lbm_kg: Optional[float] = None
    category: Optional[int] = None
    protein_g_min: Optional[float] = None
    protein_g_max: Optional[float] = None
    checkin_history: list[str] = field(default_factory=list)

    def onboarding(
        self,
        weight_kg: float,
        body_fat_percentage: float,
        gender: str,
    ) -> None:
        targets = calculate_psmf_targets(
            gender=gender,
            weight_kg=weight_kg,
            body_fat_percentage=body_fat_percentage,
        )
        self.weight_kg = targets["weight_kg"]
        self.body_fat_percentage = targets["body_fat_percentage"]
        self.gender = targets["gender"]
        self.lbm_kg = targets["lbm_kg"]
        self.category = targets["category"]
        self.protein_g_min = targets["protein_g_min"]
        self.protein_g_max = targets["protein_g_max"]

        print("—— PSMF 建档结果 ——")
        print(f"性别（规范化）: {self.gender}")
        print(f"体重: {self.weight_kg:.2f} kg")
        print(f"体脂率: {self.body_fat_percentage:.2f} %")
        print(f"估算瘦体重 LBM: {self.lbm_kg:.2f} kg")
        print(f"PSMF Category: {self.category}")
        print(
            f"每日蛋白质目标区间: {self.protein_g_min:.1f} – {self.protein_g_max:.1f} g"
        )

    def daily_checkin(self, feeling: str) -> None:
        text = feeling.strip()
        self.checkin_history.append(text)
        assessment = assess_safety_risk(text)
        if assessment.should_stop:
            print(f"{_COLOR_RED}{assessment.message}{_COLOR_RESET}")
            return
        print("—— 今日打卡 ——")
        print(f"您的描述：{text}")
        if self.category is not None and self.lbm_kg is not None:
            print(f"Category {self.category}，LBM 约 {self.lbm_kg:.2f} kg。")
