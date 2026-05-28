#!/usr/bin/env python3
"""Local smoke tests for the tool-using PSMF agent.

The checks intentionally avoid Gemini API calls and RAG index downloads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.orchestrator import AgentOrchestrator, build_default_tool_registry
from agent.extraction_tools import heuristic_extract_user_facts
from agent.planner import AgentPlanner
from agent.schemas import AgentState, PlannerDecision
from memory_manager import UserStateManager


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_tool_registry_core_tools() -> None:
    registry = build_default_tool_registry()
    names = set(registry.names())
    expected = {
        "check_safety_risk",
        "get_user_profile",
        "extract_user_facts",
        "calculate_psmf_targets",
        "update_user_profile",
        "log_food",
        "log_supplements",
        "search_psmf_knowledge",
        "search_food_database",
        "generate_daily_summary",
        "save_conversation_turn",
    }
    missing = expected - names
    _assert(not missing, f"missing core tools: {sorted(missing)}")
    print(f"tool_registry_ok tools={len(names)}")


def check_psmf_tool_calculation() -> None:
    registry = build_default_tool_registry()
    result = registry.execute(
        "calculate_psmf_targets",
        {"gender": "male", "weight_kg": 85, "body_fat_percentage": 24},
        context=_test_context(),
    )
    _assert(result.ok, f"calculate_psmf_targets failed: {result.error}")
    data = result.data
    _assert(round(float(data["lbm_kg"]), 1) == 64.6, "LBM calculation mismatch")
    _assert(data["category"] == 2, "Category calculation mismatch")
    _assert(round(float(data["protein_g_min"]), 1) == 177.6, "protein min mismatch")
    _assert(round(float(data["protein_g_max"]), 1) == 213.2, "protein max mismatch")
    print("psmf_tool_calculation_ok")


def check_compact_vitals_extraction() -> None:
    facts = heuristic_extract_user_facts("男，85kg，30%")
    _assert(facts["gender"] == "male", "compact vitals should extract gender")
    _assert(facts["weight_kg"] == 85.0, "compact vitals should extract weight")
    _assert(facts["body_fat_percentage"] == 30.0, "compact vitals should extract body fat")
    print("compact_vitals_extraction_ok")


def check_safety_hard_stop() -> None:
    registry = build_default_tool_registry()
    result = registry.execute(
        "check_safety_risk",
        {"user_input": "我胸闷，喘不上气，还有点心悸"},
        context=_test_context(),
    )
    _assert(result.ok, f"check_safety_risk failed: {result.error}")
    _assert(result.data["safety_status"] == "hard_stop", "safety should hard stop")
    _assert(result.data["should_stop"] is True, "should_stop should be true")
    _assert("就医" in result.content, "safety content should recommend care")
    print("safety_hard_stop_ok")


class _FailingPlanner(AgentPlanner):
    def __init__(self) -> None:
        self.calls = 0

    def decide(
        self,
        state: AgentState,
        tool_manifest: list[dict[str, object]],
    ) -> PlannerDecision:
        self.calls += 1
        if self.calls == 1:
            return PlannerDecision(
                thought_summary="try_missing_tool",
                action="nonexistent_tool",
                arguments={},
            )
        return PlannerDecision(thought_summary="final", action="final")

    def finalize(self, state: AgentState, draft: str | None = None) -> str:
        return draft or "fallback final"


def check_orchestrator_tool_failure_fallback() -> None:
    tmp_path = Path(f"/private/tmp/psmf-agent-smoke-memory-{os.getpid()}.json")
    memory = UserStateManager(tmp_path)
    agent = AgentOrchestrator(
        memory=memory,
        planner=_FailingPlanner(),
        registry=build_default_tool_registry(),
        dry_run=True,
        max_steps=2,
    )
    reply = agent.process_message("smoke_failure", "测试工具失败兜底")
    _assert("未注册工具" in reply or "fallback" in reply, "orchestrator did not fallback")
    print("orchestrator_tool_failure_fallback_ok")


def check_readme_startup_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commands = {
        "streamlit run app.py": ROOT / "app.py",
        "python telegram_bot.py": ROOT / "telegram_bot.py",
        "python main.py": ROOT / "main.py",
        "python scripts/smoke_test.py": ROOT / "scripts" / "smoke_test.py",
    }
    for command, path in commands.items():
        _assert(command in readme, f"README missing startup command: {command}")
        _assert(path.exists(), f"README command target missing: {path}")
    print("readme_startup_commands_ok")


def _test_context():
    from agent.schemas import AgentContext

    return AgentContext(
        user_id="smoke",
        memory=UserStateManager(Path(f"/private/tmp/psmf-agent-smoke-context-{os.getpid()}.json")),
    )


def main() -> None:
    check_tool_registry_core_tools()
    check_psmf_tool_calculation()
    check_compact_vitals_extraction()
    check_safety_hard_stop()
    check_orchestrator_tool_failure_fallback()
    check_readme_startup_commands()
    print("all_agent_smoke_tests_ok")


if __name__ == "__main__":
    main()
