#!/usr/bin/env python3
"""
PSMF Agent — Streamlit 网页 Demo。

与 ``telegram_bot.py`` 共用同一套 ``AgentOrchestrator`` + ``user_profiles.json``；
网页用户使用独立 ``user_id``（``web_<登录名>``），与 Telegram ``chat_id`` 不冲突。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

_ROOT: Path = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_ROOT / ".env")

import streamlit as st  # noqa: E402

from memory_manager import (  # noqa: E402
    SUPPLEMENT_KEYS,
    SUPPLEMENT_LABELS_ZH,
    format_supplement_status,
)
from agent import AgentOrchestrator  # noqa: E402
from rag_system import ensure_all_local_indexes_ready  # noqa: E402

# --- 演示登录（可通过 .env 覆盖，便于公开作品集与本地演示复用）---
_DEMO_USERNAME: str = (os.environ.get("PSMF_DEMO_USERNAME") or "psmf_coach").strip()
_DEMO_PASSWORD: str = (os.environ.get("PSMF_DEMO_PASSWORD") or "demo_password").strip()


def _web_user_id(login_name: str) -> str:
    """与 Telegram chat_id 区分的网页用户键。"""
    return f"web_{login_name.strip().lower().replace(' ', '_')}"


_COACH_SIDEBAR: str = """
**关于你的教练**

我是 PSMF Elite Coach：**专业、较真、但更懂坚持下来有多难**。  
我会用你的 **Category / 蛋白质 / 体征与打卡**，把复杂协议拆成你今天就能执行的一步——  
在安全红线上绝不妥协；其它时候，我们像战友一样把这件事做完。

💡 **提示**：`/summary` 或「今日复盘」等可触发完整长文；
日常打卡与问答在引擎侧会尽量 **极简**（与 Telegram 一致）。
"""


@st.cache_resource
def _get_shared_agent() -> AgentOrchestrator:
    """单例引擎：与 Bot 相同逻辑内核，共享默认 ``user_profiles.json`` 路径。"""
    ensure_all_local_indexes_ready()
    return AgentOrchestrator()


def _init_session_state() -> None:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if "login_name" not in st.session_state:
        st.session_state.login_name = ""
    if "pending_image_bytes" not in st.session_state:
        st.session_state.pending_image_bytes: Optional[bytes] = None


def _format_tp(tp: Any) -> str:
    if isinstance(tp, dict):
        lo = tp.get("min")
        hi = tp.get("max")
        if lo is not None and hi is not None:
            return f"{float(lo):.1f} – {float(hi):.1f} g/天"
        if lo is not None:
            return f"≥ {float(lo):.1f} g/天"
        if hi is not None:
            return f"≤ {float(hi):.1f} g/天"
    return "—"


def _render_sidebar_vitals(agent: AgentOrchestrator, uid: str) -> None:
    st.sidebar.markdown("### 当前档案（核心体征）")
    st.sidebar.caption(f"用户 ID：`{uid}`")
    prof: dict[str, Any] = agent.memory.get_user(uid)
    w = prof.get("weight")
    bf = prof.get("body_fat")
    cat = prof.get("category")
    lbm = prof.get("lbm")
    tp = prof.get("target_protein")
    stg = prof.get("onboarding_stage")

    st.sidebar.metric("体重 (kg)", f"{float(w):.1f}" if w is not None else "—")
    st.sidebar.metric("体脂 (%)", f"{float(bf):.1f}" if bf is not None else "—")
    st.sidebar.metric("Category", str(cat) if cat is not None else "—")
    if lbm is not None:
        st.sidebar.caption(f"估算 LBM：{float(lbm):.2f} kg")
    st.sidebar.caption(f"蛋白质目标：{_format_tp(tp)}")
    st.sidebar.caption(f"onboarding_stage：{stg if stg is not None else 0}")
    st.sidebar.divider()
    st.sidebar.markdown("### 今日微量元素补剂")
    try:
        sup = agent.memory.get_today_supplement_status(uid)
        st.sidebar.caption(format_supplement_status(sup))
        n_done = sum(1 for k in SUPPLEMENT_KEYS if sup.get(k))
        st.sidebar.progress(
            n_done / len(SUPPLEMENT_KEYS) if SUPPLEMENT_KEYS else 0.0,
            text=f"{n_done}/{len(SUPPLEMENT_KEYS)} 已打卡",
        )
        for key in SUPPLEMENT_KEYS:
            label = SUPPLEMENT_LABELS_ZH.get(key, key)
            st.sidebar.checkbox(label, value=bool(sup.get(key)), disabled=True)
    except Exception:
        st.sidebar.caption("补剂状态暂不可用")
    st.sidebar.divider()
    st.sidebar.markdown(_COACH_SIDEBAR)

    st.sidebar.divider()
    st.sidebar.markdown("##### 可选：营养表 / 补剂包装图片")
    st.sidebar.caption("选择图片后，在下方输入文字并发送；可识别食品营养表，也可录入补剂产品标签。")
    up = st.sidebar.file_uploader(
        "上传（PNG / JPG / JPEG / WebP）",
        type=["png", "jpg", "jpeg", "webp"],
        key="nutrition_uploader",
    )
    if up is not None:
        st.session_state.pending_image_bytes = up.getvalue()
        st.sidebar.success(f"已缓存图片（{len(st.session_state.pending_image_bytes)} bytes），发送下一条消息时附带。")
    if st.sidebar.button("清除已选图片"):
        st.session_state.pending_image_bytes = None
        st.rerun()

    if st.sidebar.button("退出登录"):
        st.session_state.auth_ok = False
        st.session_state.login_name = ""
        st.session_state.pending_image_bytes = None
        st.rerun()


def _load_chat_history(agent: AgentOrchestrator, uid: str) -> list[dict[str, Any]]:
    prof = agent.memory.get_user(uid)
    hist = prof.get("chat_history")
    if not isinstance(hist, list):
        return []
    return [x for x in hist if isinstance(x, dict)]


def _should_use_expander(assistant_text: str) -> bool:
    """渐进式展示：长文或明显「完整复盘」结构时折叠，避免挤满首屏。"""
    if len(assistant_text) < 2200:
        return False
    if re.search(r"(每日复盘|完整复盘|周报|##\s*【?生理趋势)", assistant_text):
        return True
    if assistant_text.count("\n##") >= 4:
        return True
    return len(assistant_text) >= 4500


def _render_assistant_markdown(text: str) -> None:
    if _should_use_expander(text):
        st.caption("回复较长，已折叠显示；日常极简回复通常不会看到此条。")
        with st.expander("展开完整回复", expanded=False):
            st.markdown(text)
    else:
        st.markdown(text)


def _render_login() -> None:
    st.title("PSMF Elite Coach")
    st.caption("网页演示入口 · 登录后开始对话")
    with st.form("login_form"):
        u = st.text_input("用户名")
        p = st.text_input("密码", type="password")
        sub = st.form_submit_button("登录")
    if sub:
        if (u or "").strip() == _DEMO_USERNAME and p == _DEMO_PASSWORD:
            st.session_state.auth_ok = True
            st.session_state.login_name = _DEMO_USERNAME
            st.session_state.pending_image_bytes = None
            st.success("登录成功")
            st.rerun()
        else:
            st.error("用户名或密码错误（演示账号见项目说明）")


def main() -> None:
    st.set_page_config(
        page_title="PSMF Elite Coach",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()

    if not st.session_state.auth_ok:
        _render_login()
        st.divider()
        st.markdown(
            "**演示账号**（硬编码）  \n"
            f"- 用户名：`{_DEMO_USERNAME}`  \n"
            f"- 密码：`{_DEMO_PASSWORD}`  \n\n"
            "与 Telegram Bot 使用 **同一** `user_profiles.json`；"
            "网页用户 ID 为 `web_psmf_coach`，与任意 `chat_id` 不冲突。"
        )
        return

    uid: str = _web_user_id(st.session_state.login_name)
    agent: AgentOrchestrator = _get_shared_agent()

    _render_sidebar_vitals(agent, uid)

    st.title("PSMF Elite Coach")
    st.caption("问啥答啥 · 需要完整复盘时请说「今日复盘」或发送 `/summary`")

    # 从磁盘同步展示 STM（与 memory_manager 一致）
    history: list[dict[str, Any]] = _load_chat_history(agent, uid)
    for turn in history:
        umsg = str(turn.get("user", ""))
        amsg = str(turn.get("assistant", ""))
        with st.chat_message("user"):
            st.markdown(umsg if umsg else "（空消息）")
        with st.chat_message("assistant"):
            _render_assistant_markdown(amsg)

    user_prompt: Optional[str] = st.chat_input("输入消息，或结合侧栏图片一起发送…")
    if user_prompt is None:
        return

    img_bytes: Optional[bytes] = st.session_state.pending_image_bytes
    st.session_state.pending_image_bytes = None

    with st.chat_message("user"):
        st.markdown(user_prompt)
        if img_bytes:
            st.caption("（本句附带图片：营养成分表 / 补剂包装标签）")

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考…（Planner → Tools → Observations → Final）"):
            try:
                reply: str = agent.process_message(
                    uid,
                    user_prompt.strip(),
                    img_bytes,
                )
            except Exception as e:
                st.error(f"处理失败：{e}")
                return
        if not (reply or "").strip():
            st.warning("本轮无可见回复，请补充体征或重试。")
        else:
            _render_assistant_markdown(reply)

    st.rerun()


if __name__ == "__main__":
    main()
