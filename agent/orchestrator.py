"""Agent orchestrator implementing a bounded ReAct-style tool loop."""

from __future__ import annotations

import logging
from typing import Optional

from agent.extraction_tools import extraction_tool_specs
from agent.memory_tools import memory_tool_specs
from agent.planner import AgentPlanner, GeminiAgentModel
from agent.psmf_tools import psmf_tool_specs
from agent.rag_tools import rag_tool_specs
from agent.safety import safety_tool_specs
from agent.schemas import AgentContext, AgentState, AgentStep, ToolCall, ToolResult
from agent.tool_registry import ToolRegistry
from memory_manager import UserStateManager

logger = logging.getLogger(__name__)


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.extend(safety_tool_specs())
    registry.extend(extraction_tool_specs())
    registry.extend(psmf_tool_specs())
    registry.extend(memory_tool_specs())
    registry.extend(rag_tool_specs())
    return registry


class AgentOrchestrator:
    """Single-agent orchestrator with planner, tools, observations, and memory."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fast_model: Optional[str] = None,
        memory: UserStateManager | None = None,
        planner: AgentPlanner | None = None,
        registry: ToolRegistry | None = None,
        max_steps: int = 6,
        dry_run: bool = False,
    ) -> None:
        self.memory = memory or UserStateManager()
        self.model_client = GeminiAgentModel(
            api_key=api_key,
            model=model,
            fast_model=fast_model,
            dry_run=dry_run,
        )
        self.planner = planner or AgentPlanner(self.model_client)
        self.registry = registry or build_default_tool_registry()
        self.max_steps = max(1, max_steps)

    def process_message(
        self,
        user_id: str,
        user_input: str,
        image_bytes: bytes | None = None,
        *,
        proactive_trigger: str | None = None,
    ) -> str:
        uid = user_id.strip() or "default"
        text = user_input.strip()
        if not text and image_bytes:
            text = "（用户上传了食品营养成分表或补剂包装/标签照片，请结合图片分析。）"
        if not text and not image_bytes:
            return ""

        state = AgentState(
            user_id=uid,
            user_input=text,
            image_bytes_present=bool(image_bytes),
        )
        context = AgentContext(
            user_id=uid,
            memory=self.memory,
            model_client=self.model_client,
            image_bytes=image_bytes,
            proactive_trigger=proactive_trigger,
        )

        # Deterministic safety pre-check runs before RAG, diet advice, or training advice.
        safety = self._execute_tool(
            state,
            context,
            "precheck_safety",
            ToolCall("check_safety_risk", {"user_input": text}),
        )
        state.safety_status = str(safety.data.get("safety_status") or "clear")
        if safety.data.get("should_stop"):
            state.final_response = safety.content
            self._save_turn(context, text, state.final_response)
            return state.final_response

        profile = self._execute_tool(
            state,
            context,
            "load_user_profile",
            ToolCall("get_user_profile", {}),
        )
        if profile.ok:
            state.scratchpad["get_user_profile"] = profile.data

        final_draft: str | None = None
        for _ in range(self.max_steps):
            decision = self.planner.decide(
                state,
                self.registry.to_planner_manifest(),
            )
            state.current_plan.append(decision.thought_summary)
            if decision.action == "final":
                final_draft = decision.final_response_draft
                break

            result = self._execute_tool(
                state,
                context,
                decision.thought_summary,
                ToolCall(decision.action, decision.arguments),
            )
            self._absorb_observation(state, result)
            if state.final_response and state.safety_status == "hard_stop":
                self._save_turn(context, text, state.final_response)
                return state.final_response
            if not result.ok and result.error == "tool_not_found":
                final_draft = "我尝试调用一个未注册工具，已停止本轮工具循环。请稍后重试或联系维护者检查工具注册表。"
                break
        else:
            final_draft = "本轮工具调用已达到上限。我会基于已经取得的信息先给出保守答复。"

        state.final_response = self.planner.finalize(state, final_draft)
        self._save_turn(context, text, state.final_response)
        return state.final_response

    def send_message(self, user_text: str) -> str:
        return self.process_message("default", user_text)

    def get_history(self, user_id: str | None = None) -> list[object]:
        uid = (user_id or "default").strip() or "default"
        try:
            return list(self.memory.get_user(uid).get("chat_history") or [])
        except Exception:
            logger.exception("Failed to read chat history for user_id=%s", uid)
            return []

    def _execute_tool(
        self,
        state: AgentState,
        context: AgentContext,
        thought: str,
        tool_call: ToolCall,
    ) -> ToolResult:
        result = self.registry.execute(tool_call.name, tool_call.arguments, context)
        state.steps.append(
            AgentStep(
                thought=thought,
                tool_call=tool_call,
                observation=result,
            )
        )
        return result

    def _absorb_observation(self, state: AgentState, result: ToolResult) -> None:
        if result.ok:
            state.scratchpad[result.name] = result.data
        if result.name == "extract_user_facts" and result.ok:
            state.facts.update(result.data)
            state.intent = str(
                result.data.get("user_intent") or result.data.get("intent") or ""
            ) or None
            symptoms = result.data.get("symptoms")
            if symptoms:
                safety = self.registry.execute(
                    "check_safety_risk",
                    {
                        "user_input": state.user_input,
                        "symptoms": symptoms,
                        "extraction_notes": result.data.get("extraction_notes") or "",
                    },
                    AgentContext(user_id=state.user_id, memory=self.memory),
                )
                state.steps.append(
                    AgentStep(
                        thought="post_extraction_safety_check",
                        tool_call=ToolCall(
                            "check_safety_risk",
                            {
                                "user_input": state.user_input,
                                "symptoms": symptoms,
                            },
                        ),
                        observation=safety,
                    )
                )
                state.safety_status = str(safety.data.get("safety_status") or state.safety_status)
                if safety.data.get("should_stop"):
                    state.final_response = safety.content
        elif result.name == "calculate_psmf_targets" and result.ok:
            state.facts.update(result.data)

    def _save_turn(self, context: AgentContext, user_text: str, reply: str) -> None:
        result = self.registry.execute(
            "save_conversation_turn",
            {"user_message": user_text, "assistant_message": reply},
            context,
        )
        if not result.ok:
            logger.warning("Failed to save conversation turn: %s", result.error)
