"""Planner and LLM adapter for the ReAct-style PSMF agent."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from agent.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    FINAL_RESPONSE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_final_response_prompt,
    build_planner_prompt,
)
from agent.schemas import AgentState, PlannerDecision


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", t, re.IGNORECASE)
    return match.group(1).strip() if match else t


def _guess_image_mime(data: bytes) -> str:
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


class GeminiAgentModel:
    """Small Gemini adapter used by planner, extraction tool, and final response."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fast_model: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.model = (model or os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash").strip()
        self.fast_model = (
            fast_model
            or os.environ.get("GEMINI_FAST_MODEL")
            or os.environ.get("GEMINI_LITE_MODEL")
            or "gemini-3.1-flash-lite"
        ).strip()
        self.dry_run = dry_run or (os.environ.get("PSMF_AGENT_DRY_RUN") or "").lower() in {
            "1",
            "true",
            "yes",
        }
        self._client: Any = None
        self._types: Any = None
        if self.dry_run:
            return
        key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
        if not key:
            return
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            return
        proxy = (os.environ.get("GEMINI_PROXY") or "").strip()
        http_options = (
            genai_types.HttpOptions(client_args={"proxy": proxy}) if proxy else None
        )
        self._types = genai_types
        self._client = (
            genai.Client(api_key=key, http_options=http_options)
            if http_options
            else genai.Client(api_key=key)
        )

    @property
    def available(self) -> bool:
        return self._client is not None and not self.dry_run

    def _generate_text(
        self,
        *,
        system_instruction: str,
        prompt: str,
        model: str,
        temperature: float,
        json_mode: bool = False,
        image_bytes: bytes | None = None,
    ) -> str:
        if not self.available:
            raise RuntimeError("Gemini model is unavailable.")
        cfg_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": temperature,
        }
        if json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
        config = self._types.GenerateContentConfig(**cfg_kwargs)
        contents: Any = prompt
        if image_bytes:
            contents = [
                self._types.Content(
                    role="user",
                    parts=[
                        self._types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=_guess_image_mime(image_bytes),
                        ),
                        self._types.Part.from_text(text=prompt),
                    ],
                )
            ]
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return (getattr(response, "text", None) or "").strip()

    def generate_json(
        self,
        *,
        system_instruction: str,
        prompt: str,
        image_bytes: bytes | None = None,
        temperature: float = 0.05,
        fast: bool = True,
    ) -> dict[str, Any]:
        raw = self._generate_text(
            system_instruction=system_instruction,
            prompt=prompt,
            model=self.fast_model if fast else self.model,
            temperature=temperature,
            json_mode=True,
            image_bytes=image_bytes,
        )
        if not raw:
            raise RuntimeError("Model returned empty JSON text.")
        data = json.loads(_strip_json_fence(raw))
        if not isinstance(data, dict):
            raise ValueError("Model JSON must be an object.")
        return data

    def extract_user_facts(
        self,
        *,
        user_text: str,
        profile: dict[str, Any],
        image_bytes: bytes | None = None,
        proactive_trigger: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_input": user_text,
            "profile": profile,
            "image_bytes_present": bool(image_bytes),
            "proactive_trigger": proactive_trigger,
        }
        prompt = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return self.generate_json(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            prompt=prompt,
            image_bytes=image_bytes,
            temperature=0.05,
            fast=not bool(image_bytes),
        )

    def generate_final_response(self, state: AgentState) -> str:
        return self._generate_text(
            system_instruction=FINAL_RESPONSE_SYSTEM_PROMPT,
            prompt=build_final_response_prompt(state),
            model=self.model,
            temperature=0.45,
        )


class AgentPlanner:
    """Planner that can use Gemini, with a deterministic fallback for tests."""

    def __init__(self, model_client: GeminiAgentModel | None = None) -> None:
        self.model_client = model_client or GeminiAgentModel()

    def decide(
        self,
        state: AgentState,
        tool_manifest: list[dict[str, Any]],
    ) -> PlannerDecision:
        if self.model_client.available:
            try:
                data = self.model_client.generate_json(
                    system_instruction=PLANNER_SYSTEM_PROMPT,
                    prompt=build_planner_prompt(state=state, tool_manifest=tool_manifest),
                    temperature=0.1,
                    fast=True,
                )
                return self._decision_from_dict(data)
            except Exception:
                pass
        return self._fallback_decide(state)

    def finalize(self, state: AgentState, draft: str | None = None) -> str:
        if self.model_client.available:
            try:
                return self.model_client.generate_final_response(state)
            except Exception:
                pass
        if draft and draft.strip():
            return draft.strip()
        return self._fallback_final_response(state)

    @staticmethod
    def _decision_from_dict(data: dict[str, Any]) -> PlannerDecision:
        thought = str(data.get("thought_summary") or "planner_decision").strip()
        action = str(data.get("action") or "final").strip()
        args = data.get("arguments")
        if not isinstance(args, dict):
            args = {}
        draft = data.get("final_response_draft")
        return PlannerDecision(
            thought_summary=thought[:500],
            action=action,
            arguments=args,
            final_response_draft=str(draft) if draft is not None else None,
        )

    @staticmethod
    def _tool_ok(state: AgentState, tool_name: str) -> bool:
        return any(
            step.observation is not None
            and step.observation.name == tool_name
            and step.observation.ok
            for step in state.steps
        )

    def _fallback_decide(self, state: AgentState) -> PlannerDecision:
        facts = state.facts or {}
        if not self._tool_ok(state, "extract_user_facts"):
            profile = state.scratchpad.get("get_user_profile") or {}
            return PlannerDecision(
                thought_summary="extract_user_facts",
                action="extract_user_facts",
                arguments={"user_input": state.user_input, "profile": profile},
            )

        if (
            facts.get("gender")
            and facts.get("weight_kg") is not None
            and facts.get("body_fat_percentage") is not None
            and not self._tool_ok(state, "calculate_psmf_targets")
        ):
            return PlannerDecision(
                thought_summary="calculate_psmf_targets",
                action="calculate_psmf_targets",
                arguments={
                    "gender": facts.get("gender"),
                    "weight_kg": facts.get("weight_kg"),
                    "body_fat_percentage": facts.get("body_fat_percentage"),
                },
            )

        targets = state.scratchpad.get("calculate_psmf_targets") or {}
        if targets and not self._tool_ok(state, "update_user_profile"):
            args = dict(targets)
            if facts.get("strength_level"):
                args["strength_level"] = facts.get("strength_level")
            if facts.get("schedule_prefs"):
                args["schedule_prefs"] = facts.get("schedule_prefs")
            return PlannerDecision(
                thought_summary="update_user_profile",
                action="update_user_profile",
                arguments=args,
            )

        if facts.get("logged_food") and not self._tool_ok(state, "log_food"):
            return PlannerDecision(
                thought_summary="log_food",
                action="log_food",
                arguments={"items": facts.get("logged_food")},
            )

        if facts.get("logged_supplements") and not self._tool_ok(state, "log_supplements"):
            return PlannerDecision(
                thought_summary="log_supplements",
                action="log_supplements",
                arguments={"updates": facts.get("logged_supplements")},
            )

        if facts.get("supplement_products") and not self._tool_ok(
            state, "update_supplement_products"
        ):
            return PlannerDecision(
                thought_summary="update_supplement_products",
                action="update_supplement_products",
                arguments={"products": facts.get("supplement_products")},
            )

        intent = str(facts.get("user_intent") or "")
        focus = str(facts.get("rag_focus") or "")
        needs_rag = intent == "ask_question" or facts.get("needs_recipe")
        if needs_rag and focus == "food" and not self._tool_ok(state, "search_food_database"):
            return PlannerDecision(
                thought_summary="search_food_database",
                action="search_food_database",
                arguments={"query": state.user_input, "top_k": 4},
            )
        if needs_rag and focus != "food" and not self._tool_ok(state, "search_psmf_knowledge"):
            return PlannerDecision(
                thought_summary="search_psmf_knowledge",
                action="search_psmf_knowledge",
                arguments={"query": state.user_input, "focus": focus or "mixed", "top_k": 4},
            )

        if not self._tool_ok(state, "generate_daily_summary"):
            return PlannerDecision(
                thought_summary="generate_daily_summary",
                action="generate_daily_summary",
                arguments={},
            )

        return PlannerDecision(
            thought_summary="ready_for_final",
            action="final",
            final_response_draft=None,
        )

    def _fallback_final_response(self, state: AgentState) -> str:
        facts = state.facts or {}
        targets = state.scratchpad.get("calculate_psmf_targets") or {}
        summary = state.scratchpad.get("generate_daily_summary") or {}
        chunks = []
        for key in ("search_psmf_knowledge", "search_food_database"):
            data = state.scratchpad.get(key) or {}
            for chunk in data.get("chunks") or []:
                if isinstance(chunk, str):
                    chunks.append(chunk)
        lines: list[str] = []
        if targets:
            lines.append(
                f"我已根据你的体征估算：LBM 约 {targets.get('lbm_kg')} kg，"
                f"PSMF Category {targets.get('category')}，每日蛋白质目标约 "
                f"{targets.get('protein_g_min')}–{targets.get('protein_g_max')} g。"
            )
        elif facts.get("weight_kg") is None or facts.get("body_fat_percentage") is None:
            lines.append("我还缺少完整体征数据。请补充性别、体重 kg 和体脂率 %，我才能计算 LBM、Category 和蛋白质目标。")
        if summary.get("daily_summary"):
            lines.append(str(summary["daily_summary"]))
        if chunks:
            lines.append("我也检索了相关 PSMF 知识库。建议以高蛋白、低脂、低净碳水为底线，并优先关注身体反馈。")
        if not lines:
            lines.append("已记录你的信息。下一步请继续补充体征、饮食打卡或具体问题，我会基于工具结果继续处理。")
        lines.append("以上为健康管理辅助信息，不替代医生或注册营养师的专业评估。")
        return "\n\n".join(lines)
