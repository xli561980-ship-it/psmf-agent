#!/usr/bin/env python3
"""
PSMF Agent 终端闭环入口：

1. 初始化本地 RAG（Chroma 为空则自动 ingest）
2. 校验 GEMINI_API_KEY
3. 自然语言多轮对话 → AgentOrchestrator（Planner + Tools + RAG）
4. 对回复中的安全警告段落使用红色终端高亮
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 路径与 .env（先于业务模块加载，便于 GEMINI_API_KEY 生效）
# -----------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent
_ENV_FILE: Path = _PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=_ENV_FILE)

from agent import AgentOrchestrator  # noqa: E402
from rag_system import ensure_all_local_indexes_ready  # noqa: E402

# 终端颜色（与安全护栏一致）
_COLOR_RED: str = "\033[91m"
_COLOR_BOLD: str = "\033[1m"
_COLOR_RESET: str = "\033[0m"

# 若模型回复中出现下列「高危干预」语义，整段以红色输出（可按需扩展）
_SAFETY_CRITICAL_SUBSTRINGS: tuple[str, ...] = (
    "立即就医",
    "尽快就医",
    "马上就医",
    "急诊",
    "急救",
    "拨打120",
    "拨打 120",
    "呼叫急救",
    "停止节食",
    "停止禁食",
    "终止节食",
    "必须终止",
    "立即停止",
    "马上停止",
    "严禁继续",
    "不得继续",
    "禁止训练",
    "禁止剧烈运动",
    "生命危险",
    "极度危险",
    "严重风险",
    "严重不适",
    "晕厥",
    "意识丧失",
    "胸痛",
    "呼吸困难",
    "心肌梗死",
    "心脏骤停",
    "【安全",
    "安全护栏",
    "安全警告",
    "⚠️",
    "紧急",
    "强行终止",
    "必须中断",
    "请勿继续",
    "不得延误",
)

# Markdown 小节：以 ## 开头且标题含安全/紧急/警告 时整节标红（直到下一个 ## 级标题）
_SECTION_SAFETY_TITLE: re.Pattern[str] = re.compile(
    r"^#{1,3}\s*.*(安全警告|紧急|⚠|危险干预|医疗)",
    re.MULTILINE | re.IGNORECASE,
)


def _paragraph_requires_red_alert(block: str) -> bool:
    """判断该文本块是否属于须红色强调的安全警告内容。"""
    b: str = block.strip()
    if not b:
        return False
    first_line: str = b.split("\n", 1)[0].strip()
    if _SECTION_SAFETY_TITLE.search(first_line):
        return True
    if first_line.startswith("#") and (
        "安全" in first_line or "紧急" in first_line or "⚠" in first_line
    ):
        return True
    lower_hit: bool = any(s in b for s in _SAFETY_CRITICAL_SUBSTRINGS)
    return lower_hit


def _split_reply_into_blocks(text: str) -> list[str]:
    """
    将模型回复拆成若干块以便按块着色。

    优先按「行首 # / ## / ### 标题」切分章节；无标题部分再按空行分段。
    """
    t: str = text.strip()
    if not t:
        return []

    sections: list[str] = re.split(r"(?m)(?=^#{1,3}\s)", t)
    out: list[str] = []
    for sec in sections:
        s: str = sec.strip()
        if not s:
            continue
        if s.lstrip().startswith("#"):
            out.append(s)
        else:
            for sub in re.split(r"\n\n+", s):
                sub2: str = sub.strip()
                if sub2:
                    out.append(sub2)
    return out if out else [t]


def print_agent_markdown_with_safety_highlight(reply: str) -> None:
    """
    打印 Agent 的 Markdown 回复；命中安全语义的块使用红色字体。

    说明：终端无法渲染完整 Markdown，但保留 ``##``、列表等字符以便阅读。
    """
    blocks: list[str] = _split_reply_into_blocks(reply)
    if not blocks:
        print("(无输出)")
        return

    for i, block in enumerate(blocks):
        if i:
            print()
        if _paragraph_requires_red_alert(block):
            print(f"{_COLOR_RED}{block}{_COLOR_RESET}")
        else:
            print(block)


def _init_rag_or_exit() -> dict[str, tuple[bool, int]]:
    """启动时确保主知识库与食谱库（food_db）就绪；失败则退出。"""
    try:
        return ensure_all_local_indexes_ready()
    except FileNotFoundError as exc:
        print(f"RAG 初始化失败：缺少知识库文件。\n{exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(
            f"RAG 初始化失败（ChromaDB / 嵌入模型等）。请检查 chromadb 与网络后重试。\n"
            f"详情：{exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _ensure_gemini_api_key_interactive() -> str:
    """
    从环境读取 GEMINI_API_KEY；若缺失则提示用户在项目 .env 中配置并支持重试。
    """
    key: str = os.environ.get("GEMINI_API_KEY", "").strip()
    while not key:
        print(
            f"\n{_COLOR_BOLD}未检测到 GEMINI_API_KEY{_COLOR_RESET}。\n"
            f"请在项目目录创建或编辑：{_ENV_FILE}\n"
            "添加一行（无引号）：\n"
            "  GEMINI_API_KEY=你的Google_AI_Studio密钥\n"
            "获取密钥：https://aistudio.google.com/apikey\n"
        )
        try:
            input("配置保存后，按 Enter 重新加载…")
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            sys.exit(1)
        load_dotenv(dotenv_path=_ENV_FILE, override=True)
        key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(
        f"\n{_COLOR_BOLD}========== PSMF Agent（Gemini + RAG 闭环）=========={_COLOR_RESET}\n"
    )

    idx_stats: dict[str, tuple[bool, int]] = _init_rag_or_exit()
    pk: tuple[bool, int] = idx_stats["psmf_knowledge"]
    fk: tuple[bool, int] = idx_stats["food_db"]
    print(
        f"主知识库 psmf_knowledge："
        f"{'已自动 ingest' if pk[0] else '已就绪'}（约 {pk[1]} 块）"
    )
    print(
        f"食谱库 food_db："
        f"{'已自动 ingest' if fk[0] else '已就绪'}（约 {fk[1]} 块）"
    )

    _ensure_gemini_api_key_interactive()

    try:
        agent: AgentOrchestrator = AgentOrchestrator()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"无法启动 Gemini 客户端：{exc}", file=sys.stderr)
        sys.exit(1)

    print(
        "\n"
        "你可以用一段自然语言同时说明性别、体重、体脂、感受等，无需逐步「查户口」。\n"
        "示例：我是男生，85 公斤，体脂大概 25%，今天第一天，稍微有点饿但精神很好。\n"
        "输入 exit / quit / 退出 结束；空行退出。\n"
    )

    while True:
        try:
            user_line: str = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_COLOR_BOLD}再见。{_COLOR_RESET}")
            break

        if not user_line:
            print("结束。")
            break
        if user_line.lower() in ("exit", "quit", "q", "退出"):
            print(f"{_COLOR_BOLD}再见。{_COLOR_RESET}")
            break

        reply: str = agent.send_message(user_line)
        print(f"\n{_COLOR_BOLD}── 专家（Markdown）──{_COLOR_RESET}\n")
        print_agent_markdown_with_safety_highlight(reply)


if __name__ == "__main__":
    main()
