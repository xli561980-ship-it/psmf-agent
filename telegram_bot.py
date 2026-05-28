#!/usr/bin/env python3
"""
PSMF Agent — Telegram 线上入口（python-telegram-bot v20+ 异步）。

启动前：加载 .env、初始化本地 RAG 索引、校验 GEMINI_API_KEY 与 TELEGRAM_BOT_TOKEN。
支持文本与图片（营养成分表、补剂包装等）：图片经 ``process_message(..., image_bytes=…)`` 送入多模态提取。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

_PROJECT_ROOT: Path = Path(__file__).resolve().parent
_ENV_PATH: Path = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

from telegram import Update  # noqa: E402
from telegram.constants import ChatAction  # noqa: E402
from telegram.ext import (  # noqa: E402
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from psmf_engine import GeminiPSMFAgent  # noqa: E402
from rag_system import ensure_all_local_indexes_ready  # noqa: E402

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_MESSAGE_LEN: int = 4096


class _SecretRedactionFilter(logging.Filter):
    _TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        redacted = self._TOKEN_RE.sub("bot<redacted>", msg)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def _install_logging_hardening() -> None:
    redactor = _SecretRedactionFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(redactor)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _allowed_telegram_user_ids() -> set[int]:
    raw = (os.environ.get("ALLOWED_TELEGRAM_USER_IDS") or "").strip()
    if not raw:
        return set()
    out: set[int] = set()
    for part in re.split(r"[,\s]+", raw):
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            logger.warning("忽略无效 ALLOWED_TELEGRAM_USER_IDS 条目：%r", part)
    return out


def _is_allowed_update(update: Update) -> bool:
    allowed = _allowed_telegram_user_ids()
    if not allowed:
        return True
    user_id = update.effective_user.id if update.effective_user else None
    chat_id = update.effective_chat.id if update.effective_chat else None
    return user_id in allowed or chat_id in allowed


async def _reject_if_unauthorized(update: Update) -> bool:
    if _is_allowed_update(update):
        return False
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    logger.warning("拒绝未授权 Telegram 访问 user_id=%s chat_id=%s", uid, cid)
    if update.message:
        await update.message.reply_text("未授权访问。")
    return True


def _parse_hm(s: str) -> Optional[tuple[int, int]]:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", (s or "").strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return h, mi


def _merge_schedule_prefs_from_env(profile: dict) -> dict[str, str]:
    """用户对话建档优先；缺项时用 .env 兜底。"""
    prefs: dict[str, str] = dict(profile.get("schedule_prefs") or {})
    pairs: tuple[tuple[str, str], ...] = (
        ("MORNING_CHECKIN_TIME", "morning_time"),
        ("WORKOUT_REMINDER_TIME", "workout_time"),
        ("EVENING_REVIEW_TIME", "evening_time"),
        ("WEEKLY_REPORT_TIME", "weekly_time"),
    )
    for env_k, pk in pairs:
        if prefs.get(pk):
            continue
        raw = (os.environ.get(env_k) or "").strip()
        if raw:
            prefs[pk] = raw
    return prefs


_PROACTIVE_TEXTS: dict[str, str] = {
    "morning": (
        "【系统·晨间体重打卡】早安。请先回复今早空腹体重（kg），再补充睡眠、精神状态、身体不适和今日饮食计划。"
    ),
    "workout": "【系统·训练提醒】到了你设定的训练窗口，热身与容量请以恢复为先，必要时减量。",
    "evening_review": (
        "【系统·睡前复盘】下面是今日小结与明日提醒（含蛋白质是否达标与食材参考）。"
    ),
    "weekly": "【系统·周报】以下为最近 7 天的体重、力量与蛋白质汇总（滚动自然周）。",
}


async def send_proactive_message(application: Application, chat_id: int, kind: str) -> None:
    agent: GeminiPSMFAgent | None = application.bot_data.get("agent")
    if agent is None:
        return
    text: str = _PROACTIVE_TEXTS.get(kind, "【系统提醒】")
    try:
        reply: str = await asyncio.to_thread(
            lambda: agent.process_message(
                str(chat_id), text, None, proactive_trigger=kind
            )
        )
    except Exception:
        logger.exception("定时任务 process_message 失败 chat_id=%s kind=%s", chat_id, kind)
        return
    reply = _decorate_safety_alert_layout((reply or "").strip())
    if not reply:
        return
    bot = application.bot
    for part in _chunk_for_telegram(reply):
        try:
            await bot.send_message(chat_id=chat_id, text=part)
        except Exception:
            logger.exception("定时推送发送失败 chat_id=%s", chat_id)


def _profile_ready_for_scheduled_jobs(profile: dict) -> bool:
    """A user can receive scheduled jobs once core body metrics exist.

    schedule_prefs are optional because .env provides production defaults.
    """
    try:
        stage = int(profile.get("onboarding_stage") or 0)
    except (TypeError, ValueError):
        stage = 0
    if stage >= 2:
        return True
    if profile.get("target_protein"):
        return True
    metrics = profile.get("metrics_history")
    return isinstance(metrics, list) and bool(metrics)


def sync_user_jobs(application: Application, user_id: str) -> None:
    """
    按用户档案注册 / 刷新 Cron 任务。
    只要已有核心体征或蛋白目标，即使尚未完成日程建档，也使用 .env 默认时间兜底。
    """
    sched: AsyncIOScheduler | None = application.bot_data.get("scheduler")
    if sched is None:
        return
    agent: GeminiPSMFAgent | None = application.bot_data.get("agent")
    if agent is None:
        return
    uid: str = str(user_id).strip()
    chat_id: int = int(uid)

    for job in list(sched.get_jobs()):
        jid: str = getattr(job, "id", "") or ""
        if jid.startswith(f"u{uid}_"):
            sched.remove_job(jid)

    prof: dict = agent._memory.get_user(uid)
    try:
        stage: int = int(prof.get("onboarding_stage") or 0)
    except (TypeError, ValueError):
        stage = 0
    if not _profile_ready_for_scheduled_jobs(prof):
        logger.info("sync_user_jobs：用户 %s onboarding_stage=%s 且无核心体征，跳过调度", uid, stage)
        return

    prefs: dict[str, str] = _merge_schedule_prefs_from_env(prof)
    tz_name: str = (os.environ.get("SCHEDULER_TIMEZONE") or "Europe/Berlin").strip()

    sched_cfg: list[tuple[str, str, str]] = [
        ("morning_time", "morning", "morning"),
        ("workout_time", "workout", "workout"),
        ("evening_time", "evening_review", "evening"),
    ]
    for pref_key, kind, jid_suffix in sched_cfg:
        hm = prefs.get(pref_key)
        if not hm:
            continue
        parsed = _parse_hm(hm)
        if not parsed:
            logger.warning("无效时间 %s=%s user=%s", pref_key, hm, uid)
            continue
        h, mi = parsed
        jid: str = f"u{uid}_{jid_suffix}"
        sched.add_job(
            partial(send_proactive_message, application, chat_id, kind),
            CronTrigger(hour=h, minute=mi, timezone=tz_name),
            id=jid,
            replace_existing=True,
        )
        logger.info("已注册定时任务 id=%s %02d:%02d %s", jid, h, mi, kind)

    wt = prefs.get("weekly_time")
    if wt:
        wp = _parse_hm(wt)
        if wp:
            wh, wmi = wp
            dow: str = (os.environ.get("WEEKLY_REPORT_WEEKDAY") or "sun").strip().lower()
            jid_w: str = f"u{uid}_weekly"
            sched.add_job(
                partial(send_proactive_message, application, chat_id, "weekly"),
                CronTrigger(day_of_week=dow, hour=wh, minute=wmi, timezone=tz_name),
                id=jid_w,
                replace_existing=True,
            )
            logger.info("已注册周报任务 id=%s dow=%s %02d:%02d", jid_w, dow, wh, wmi)


async def _post_init(application: Application) -> None:
    tz_name: str = (os.environ.get("SCHEDULER_TIMEZONE") or "Europe/Berlin").strip()
    sched = AsyncIOScheduler(timezone=tz_name)
    application.bot_data["scheduler"] = sched
    agent: GeminiPSMFAgent = application.bot_data["agent"]
    application.bot_data["sync_user_jobs"] = lambda uid: sync_user_jobs(application, uid)
    print(
        f"[调度器] 正在根据档案注册定时任务（时区 {tz_name}）…",
        flush=True,
    )
    for uid in agent._memory.list_all_user_ids():
        sync_user_jobs(application, uid)
    sched.start()
    logger.info("APScheduler 已启动 timezone=%s", tz_name)
    print("[调度器] APScheduler 已启动。", flush=True)


async def _post_shutdown(application: Application) -> None:
    """优雅关闭：先停 APScheduler，减少 threading 在退出时报警或阻塞。"""
    sched: AsyncIOScheduler | None = application.bot_data.get("scheduler")
    if sched is not None:
        try:
            if sched.running:
                sched.shutdown(wait=False)
            print("[调度器] 已停止。", flush=True)
            logger.info("APScheduler 已关闭")
        except Exception:
            logger.exception("关闭 APScheduler 时出错")


_CHUNK_SAFE: int = 4000

# 命中任一即视为需要 🚨 排版强化（与引擎「安全警告」小节对齐）
_SAFETY_TRIGGER_SUBSTRINGS: tuple[str, ...] = (
    "【安全警告】",
    "## 安全警告",
    "### ⚠",
    "## ⚠️",
    "安全护栏",
    "高危拦截",
)

_START_MESSAGE: str = (
    "欢迎。这里是 PSMF（Protein-Sparing Modified Fast）专项教练通道。\n\n"
    "PSMF 不是「少吃就行」，而是在医疗监督语境下，用极高蛋白、极低脂碳水的方式尽量保住瘦体重；"
    "执行错误会导致低血糖、电解质紊乱、肌肉流失与心血管风险——我会按规则给你数字与红线，"
    "而不是鸡汤。\n\n"
    "「建档」必填：请用一条消息给出尽量完整的信息，便于计算 Category 与蛋白质区间：\n"
    "• 性别（男/女）\n"
    "• 身高（cm），用于上下文与后续调整参考\n"
    "• 体重（kg）\n"
    "• 体脂率（%）\n\n"
    "示例：男，身高181，体重85kg，体脂约24%，想了解目标和怎么吃。\n\n"
    "你也可以直接拍摄食品「营养成分表」或保健品「Supplement Facts/补剂包装」发给我；"
    "我会尝试识读宏量或补剂含量，并把补剂产品写入档案用于后续用量规划。\n\n"
    "若出现头晕、心悸、胸痛、呼吸困难、晕厥等，请立即中止节食并就医；"
    "聊天中出现此类关键词时我会给出明确安全段落。\n\n"
    "直接发送你的情况即可开始。"
)


_USER_FACING_ERROR: str = (
    "抱歉，系统在为你生成回复时遇到内部错误，请稍后再试。"
    "若问题持续，请联系管理员。此处已记录诊断信息。"
)


def _decorate_safety_alert_layout(text: str) -> str:
    """若包含安全警告 / 高危语义，用 🚨 做 Telegram 端视觉强化（纯文本）。"""
    if not any(s in text for s in _SAFETY_TRIGGER_SUBSTRINGS) and "安全警告" not in text:
        return text

    banner: str = "🚨 ━━━ 安全警告 · 请务必优先阅读 ━━━ 🚨\n\n"
    body: str = text
    body = body.replace("【安全警告】", "🚨 【安全警告】", 1)
    for head in ("## 安全警告", "## ⚠️", "## ⚠", "### ⚠️", "### 紧急"):
        if head in body and f"🚨 {head}" not in body:
            body = body.replace(head, f"🚨 {head}", 1)
            break
    return banner + body


def _chunk_for_telegram(text: str, max_len: int = _CHUNK_SAFE) -> list[str]:
    if len(text) <= max_len:
        return [text]
    out: list[str] = []
    rest: str = text
    while rest:
        out.append(rest[:max_len])
        rest = rest[max_len:]
    return out


async def _typing_keepalive(
    bot,
    chat_id: int,
    stop: asyncio.Event,
) -> None:
    """在 Agent 阻塞计算期间周期性发送「正在输入」，直至 stop 被置位。"""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            logger.debug("send_chat_action(TYPING) 失败", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.5)
        except asyncio.TimeoutError:
            continue


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if update.message:
        await update.message.reply_text(_START_MESSAGE)


async def _execute_agent_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    image_bytes: Optional[bytes],
    *,
    show_nutrition_status: bool,
) -> None:
    """统一调用 ``process_message`` 并分段回复；可选营养成分表状态提示与完成后删除。"""
    if not update.message:
        return

    chat_id: int = update.effective_chat.id
    agent: GeminiPSMFAgent | None = context.bot_data.get("agent")
    if agent is None:
        logger.error("bot_data['agent'] 未初始化")
        await update.message.reply_text(_USER_FACING_ERROR)
        return

    status_msg = None
    if show_nutrition_status:
        status_msg = await update.message.reply_text("正在分析图片标签…")

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(
        _typing_keepalive(context.bot, chat_id, stop_typing)
    )

    try:
        reply: str = await asyncio.to_thread(
            agent.process_message,
            str(chat_id),
            text,
            image_bytes,
        )
        sync_fn = context.application.bot_data.get("sync_user_jobs")
        if sync_fn:
            try:
                await asyncio.to_thread(sync_fn, str(chat_id))
            except Exception:
                logger.exception("sync_user_jobs 失败 chat_id=%s", chat_id)
        if not (reply or "").strip():
            reply = "（本轮未生成可见回复；请补充性别、体重、体脂率等信息或发送更清晰的营养成分表照片后重试。）"
        reply = _decorate_safety_alert_layout(reply)
        chunks: list[str] = _chunk_for_telegram(reply)
        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                logger.debug("删除状态消息失败", exc_info=True)
        for i, part in enumerate(chunks):
            await update.message.reply_text(part)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.35)
    except Exception:
        traceback.print_exc()
        logger.exception("GeminiPSMFAgent.process_message 失败 chat_id=%s", chat_id)
        try:
            await update.message.reply_text(_USER_FACING_ERROR)
        except Exception:
            logger.exception("发送错误提示给用户失败")
    finally:
        stop_typing.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def on_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if not update.message or update.message.text is None:
        return

    raw: str = update.message.text
    text: str = raw.strip()
    if not text:
        return

    await _execute_agent_and_reply(
        update,
        context,
        text,
        image_bytes=None,
        show_nutrition_status=False,
    )


async def on_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _reject_if_unauthorized(update):
        return
    if not update.message or not update.message.photo:
        return

    photos = update.message.photo
    best = photos[-1]
    tg_file = await context.bot.get_file(best.file_id)
    data = await tg_file.download_as_bytearray()
    image_bytes: bytes = bytes(data)

    caption: str = (update.message.caption or "").strip()

    await _execute_agent_and_reply(
        update,
        context,
        caption,
        image_bytes=image_bytes,
        show_nutrition_status=True,
    )


def _bootstrap_sync() -> GeminiPSMFAgent:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _install_logging_hardening()
    print("", flush=True)
    print("=" * 58, flush=True)
    print("  PSMF Agent — Telegram 启动流程", flush=True)
    print("=" * 58, flush=True)
    print("", flush=True)
    print("【1/3】正在加载向量库（Chroma / 嵌入，首次可能较慢）…", flush=True)
    logger.info("开始 ensure_all_local_indexes_ready")
    ensure_all_local_indexes_ready()
    print("【1/3】向量库就绪。", flush=True)
    print("", flush=True)
    print("【2/3】正在初始化本地存储（user_profiles.json）与 Gemini 引擎…", flush=True)
    logger.info("正在构造 GeminiPSMFAgent")
    agent = GeminiPSMFAgent()
    print("【2/3】存储与引擎初始化完成。", flush=True)
    print("", flush=True)
    print("【3/3】核心组件已就绪，即将连接 Telegram。", flush=True)
    return agent


def main() -> None:
    token: str = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        print(
            "错误：未配置 TELEGRAM_BOT_TOKEN。请在 .env 中设置："
            "TELEGRAM_BOT_TOKEN=你的 BotFather Token",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        agent: GeminiPSMFAgent = _bootstrap_sync()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"无法启动引擎：{exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    application: Application = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    application.bot_data["agent"] = agent

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_message)
    )
    application.add_handler(MessageHandler(filters.PHOTO, on_photo_message))

    print("", flush=True)
    print("-" * 58, flush=True)
    print("  Telegram Bot 已上线，请在手机端发送消息测试。", flush=True)
    print("  按 Ctrl+C 可停止服务（将尝试优雅关闭调度器）。", flush=True)
    print("-" * 58, flush=True)
    print("", flush=True)
    logger.info("开始 long polling…")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        print("\n[退出] 收到 KeyboardInterrupt（Ctrl+C）。", flush=True)
        logger.info("KeyboardInterrupt，用户主动停止")
    except SystemExit:
        raise
    except Exception:
        logger.exception("run_polling 异常退出")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[退出] 已中断。", flush=True)
