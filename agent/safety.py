"""Deterministic health-safety guardrails for the PSMF agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from agent.schemas import AgentContext, ToolResult
from agent.tool_registry import ToolSpec

HIGH_RISK_KEYWORDS: Final[tuple[str, ...]] = (
    "头晕",
    "严重头晕",
    "心悸",
    "抽筋",
    "无力",
    "胸痛",
    "胸闷",
    "呼吸困难",
    "喘不上气",
    "晕厥",
    "昏厥",
    "眼前发黑",
    "心律不齐",
    "心跳很乱",
    "手脚麻木",
    "意识模糊",
)

HARD_STOP_SYMPTOM_KEYWORDS: Final[tuple[str, ...]] = (
    "胸痛",
    "胸闷",
    "呼吸困难",
    "喘不上气",
    "晕厥",
    "昏厥",
    "眼前发黑",
    "严重头晕",
    "心悸",
    "心律不齐",
    "心跳很乱",
    "手脚麻木",
    "意识模糊",
)


@dataclass(frozen=True)
class SafetyAssessment:
    """Deterministic pre-check result."""

    status: str
    matched_terms: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def should_stop(self) -> bool:
        return self.status == "hard_stop"


def _unique_ordered(items: list[str]) -> list[str]:
    return list(dict.fromkeys(x for x in items if x))


def build_safety_reply(matched_terms: list[str]) -> str:
    terms = "、".join(_unique_ordered(matched_terms)) or "高风险症状"
    return (
        "【安全警告】我先不继续给饮食、热量或训练建议。\n\n"
        f"你提到了可能需要优先处理的症状：{terms}。在 PSMF / 极低热量状态下，这类信号不能当作普通疲劳处理。\n\n"
        "请现在先停止节食推进和剧烈运动，坐下或躺下休息，补充水分；如果你有医生给过电解质/补盐方案，"
        "请按既定安全方案执行。如果症状正在发生、持续不缓解、加重，或伴随胸痛、呼吸困难、晕厥/眼前发黑、"
        "明显心律异常，请尽快就医或急诊。\n\n"
        "等你确认症状已经解除、没有急性风险后，我再继续帮你复盘今天的饮食和训练。"
    )


def assess_safety_risk(
    user_text: str,
    *,
    symptoms: list[str] | None = None,
    extraction_notes: str = "",
) -> SafetyAssessment:
    """Rule-first health safety check.

    This deliberately runs outside the LLM/planner loop. A hard-stop result
    prevents RAG retrieval, dieting advice, training advice, and final synthesis.
    """

    blob = " ".join([user_text or "", " ".join(symptoms or []), extraction_notes or ""])
    hard_terms = [term for term in HARD_STOP_SYMPTOM_KEYWORDS if term in blob]
    if hard_terms:
        matched = _unique_ordered(hard_terms)
        return SafetyAssessment(
            status="hard_stop",
            matched_terms=matched,
            message=build_safety_reply(matched),
        )

    caution_terms = [term for term in HIGH_RISK_KEYWORDS if term in blob]
    if caution_terms:
        return SafetyAssessment(
            status="caution",
            matched_terms=_unique_ordered(caution_terms),
            message="检测到需要谨慎处理的身体反馈，后续建议应保持保守并提醒用户关注症状变化。",
        )

    return SafetyAssessment(status="clear")


def check_safety_risk_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    text = str(arguments.get("user_input") or arguments.get("text") or "")
    if not text:
        text = ""
    symptoms_raw = arguments.get("symptoms") or []
    symptoms = [str(x) for x in symptoms_raw] if isinstance(symptoms_raw, list) else []
    notes = str(arguments.get("extraction_notes") or "")
    assessment = assess_safety_risk(text, symptoms=symptoms, extraction_notes=notes)
    return ToolResult(
        name="check_safety_risk",
        ok=True,
        content=assessment.message or f"safety_status={assessment.status}",
        data={
            "safety_status": assessment.status,
            "matched_terms": assessment.matched_terms,
            "should_stop": assessment.should_stop,
        },
    )


def safety_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="check_safety_risk",
            description=(
                "Deterministically checks high-risk symptoms and returns hard_stop "
                "for acute safety signals before diet/training advice continues."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_input": {"type": "string"},
                    "symptoms": {"type": "array", "items": {"type": "string"}},
                    "extraction_notes": {"type": "string"},
                },
                "required": ["user_input"],
            },
            handler=check_safety_risk_tool,
        )
    ]
