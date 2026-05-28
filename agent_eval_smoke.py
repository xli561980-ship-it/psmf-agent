#!/usr/bin/env python3
"""Small local smoke checks for the PSMF agent architecture.

This script intentionally avoids Gemini API calls. It checks deterministic rules,
RAG chunking, retrieval sanity, and safety hard-stop behavior.
"""

from __future__ import annotations

from rag_system import _chunk_text, search_food_knowledge, search_knowledge_by_source
from psmf_engine import (
    ExtractionResult,
    _hard_stop_safety_reply,
    _matched_hard_stop_symptoms,
    _normalize_extraction_payload_shape,
    calculate_lbm,
    determine_category,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_chunking() -> None:
    sample = """
# PSMF 测试

普通段落一，解释原则。

| 症状 | 机制 | 指令 |
| :-- | :-- | :-- |
| 心悸 | 电解质异常 | 停止运动并评估 |
| 抽筋 | 镁/钠不足 | 补水并复查 |

## 训练

低容量、高强度、长休息。
""".strip()
    chunks = _chunk_text(sample, chunk_size=180, overlap=40)
    joined = "\n---\n".join(chunks)
    _assert("| 心悸 | 电解质异常 |" in joined, "table content missing")
    _assert(all(len(c) <= 220 for c in chunks), "chunk unexpectedly large")
    print(f"chunking_ok chunks={len(chunks)}")


def check_deterministic_rules() -> None:
    lbm = calculate_lbm(85, 24)
    cat = determine_category(24, "male")
    _assert(round(lbm, 1) == 64.6, "LBM calculation mismatch")
    _assert(cat == 2, "category mismatch")
    print("deterministic_rules_ok")


def check_extraction_shape_guard() -> None:
    payload = _normalize_extraction_payload_shape(
        {
            "weight_kg": "85",
            "body_fat_percentage": 24,
            "user_intent": "question",
            "rag_focus": "training",
            "symptoms": "not-a-list",
            "logged_supplements": [],
            "morning_time": "8:00",
        }
    )
    _assert(payload["user_intent"] == "ask_question", "intent normalization failed")
    _assert(payload["symptoms"] == [], "symptoms should be list-normalized")
    _assert(payload["logged_supplements"] == {}, "supplements should be dict-normalized")
    _assert(payload["morning_time"] == "08:00", "time normalization failed")
    print("extraction_shape_guard_ok")


def check_safety_hard_stop() -> None:
    ext = ExtractionResult(symptom_keywords=["心悸"])
    terms = _matched_hard_stop_symptoms("今天训练后胸闷，有点喘不上气", ext)
    _assert("胸闷" in terms and "心悸" in terms, "hard-stop terms missing")
    reply = _hard_stop_safety_reply(terms)
    _assert("不继续给饮食" in reply and "就医" in reply, "safety reply missing action")
    print("safety_hard_stop_ok")


def check_retrieval() -> None:
    cases = [
        (
            "symptom_diagnostic_matrix.md",
            "头晕 心悸 胸闷 PSMF 电解质 风险",
        ),
        (
            "psmf_training_guide.md",
            "PSMF 深蹲力量下降 训练容量 RPE 减量",
        ),
    ]
    for source, query in cases:
        chunks = search_knowledge_by_source(query, source, top_k=2)
        _assert(chunks, f"no retrieval results for {source}")
        print(f"retrieval_ok source={source} hits={len(chunks)} first={chunks[0][:80]!r}")
    food = search_food_knowledge("鸡胸 蛋清 低脂 高蛋白 晚上缺口80g", top_k=2)
    _assert(food, "no food retrieval results")
    print(f"retrieval_ok source=food_db hits={len(food)} first={food[0][:80]!r}")


def main() -> None:
    check_chunking()
    check_deterministic_rules()
    check_extraction_shape_guard()
    check_safety_hard_stop()
    check_retrieval()
    print("all_smoke_checks_ok")


if __name__ == "__main__":
    main()
