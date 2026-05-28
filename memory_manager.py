"""
用户长期记忆（LTM）与短期对话（STM）的本地持久化管理。

数据存储于 ``user_profiles.json``，主键 ``user_id``（可与 Telegram chat_id 对齐）。
体重/体脂/瘦体重以 ``metrics_history`` 时间序列为唯一真源；读时通过
``_hydrate_vitals_from_history`` 派生顶栏 ``weight`` 等供旧逻辑使用，写盘前会删除顶栏副本。
``user_memories`` 为软记忆（偏好/习惯/昵称等），不得替代或覆盖 ``metrics_history``。
``daily_logs`` / ``logged_food`` / ``logged_supplements`` 仍由规则 + 提取 JSON 写入，不经过软记忆判断。
``supplement_products`` 存放用户拍照录入的补剂产品标签，供剂量换算与服用时机规划使用。
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import secrets
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Final, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# metrics_history 最多保留条数（防止 JSON 无限增长）
_MAX_METRICS_HISTORY: Final[int] = 500

# 力量水平：与 ``psmf_engine`` 提取结果对齐（存于 metrics_history 每条记录）
_STRENGTH_LEVELS: Final[tuple[str, ...]] = ("stable", "drop", "crash")

# PSMF 日常追踪：脂肪 / 碳水「红线」（余量 = 限额 − 已摄入）
_FAT_LIMIT_G: Final[float] = 20.0
_CARB_LIMIT_G: Final[float] = 50.0

# 微量元素 / 必需补剂（每日打卡，布尔）
SUPPLEMENT_KEYS: Final[tuple[str, ...]] = (
    "zinc",
    "magnesium",
    "fish_oil",
    "b_complex",
    "k2_d3",
)
SUPPLEMENT_LABELS_ZH: Final[dict[str, str]] = {
    "zinc": "锌 (Zinc)",
    "magnesium": "镁 (Magnesium)",
    "fish_oil": "鱼油 (Omega-3)",
    "b_complex": "B族维生素",
    "k2_d3": "K2 + D3",
}

_DEFAULT_PROFILE: Final[dict[str, Any]] = {
    # 当前体征以 ``metrics_history`` 最近一条为准；读时由 ``_hydrate_vitals_from_history`` 灌入派生键
    "category": None,
    "target_protein": None,
    "gender": None,
    "chat_history": [],
    # 日期键 ISO ``YYYY-MM-DD`` → 当日日志；见 ``_new_day_log``
    "daily_logs": {},
    # 唯一时间序载体：weight / body_fat / lbm / strength_level
    "metrics_history": [],
    # 0=待体征 →1=待日程 →2=日常
    "onboarding_stage": 0,
    # morning_time / workout_time / evening_time / weekly_time，值为 "HH:MM"
    "schedule_prefs": {},
    # 软记忆：偏好/习惯/昵称等；体征真源仍为 metrics_history，不得替代
    "user_memories": [],
    # 用户已购买/拍照录入的补剂产品标签信息；由多模态提取模块写入
    "supplement_products": [],
}

# 短期记忆（STM）：最多 50 轮；总字符预算见 append_chat_turn
_MAX_CHAT_TURNS: Final[int] = 50
_MAX_STM_USER_CHARS: Final[int] = 16_000
_MAX_STM_ASSISTANT_CHARS: Final[int] = 32_000
_MAX_STM_TOTAL_CHARS: Final[int] = 120_000

# 软记忆上限（超出时删 confidence 最低且最久未确认者）
_MAX_USER_MEMORIES: Final[int] = 80

_VALID_MEMORY_TYPES: Final[tuple[str, ...]] = (
    "preference",
    "constraint",
    "habit",
    "goal",
    "equipment",
    "schedule",
    "nickname",
    "flirt_boundary",
    "other",
)

_MAX_SUPPLEMENT_PRODUCTS: Final[int] = 40
_DEFAULT_LOCAL_TIMEZONE: Final[str] = "Europe/Berlin"


def _local_tz() -> datetime.tzinfo:
    """本地业务时区：用于 daily_logs 分桶、周报窗口与自然周锚点。"""
    tz_name: str = (
        os.environ.get("PSMF_LOCAL_TIMEZONE")
        or os.environ.get("SCHEDULER_TIMEZONE")
        or _DEFAULT_LOCAL_TIMEZONE
    ).strip()
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            logger.warning("无效业务时区 %s，回退 %s", tz_name, _DEFAULT_LOCAL_TIMEZONE)
            return ZoneInfo(_DEFAULT_LOCAL_TIMEZONE)
    return datetime.timezone.utc


def _local_now() -> datetime.datetime:
    return datetime.datetime.now(_local_tz())


def _today_iso() -> str:
    return _local_now().date().isoformat()


def _local_date_from_timestamp(raw: Any) -> Optional[datetime.date]:
    dt = _parse_timestamp_iso(raw)
    if dt is None:
        return None
    return dt.astimezone(_local_tz()).date()


def _utc_now_iso_z() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _normalize_memory_text_key(text: str) -> str:
    """按 text 归一化去重：小写并去掉空白。"""
    return "".join(text.strip().lower().split())


def _normalize_user_memories(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text") or "").strip()
        if not text:
            continue
        mem_type = str(it.get("type") or "other").strip().lower()
        if mem_type not in _VALID_MEMORY_TYPES:
            mem_type = "other"
        try:
            conf = float(it.get("confidence", 0.85))
        except (TypeError, ValueError):
            conf = 0.85
        conf = max(0.0, min(1.0, conf))
        mid = str(it.get("id") or "").strip()
        if len(mid) != 8:
            mid = secrets.token_hex(4)
        created = str(it.get("created_at") or "").strip() or _utc_now_iso_z()
        confirmed = str(it.get("last_confirmed_at") or "").strip() or created
        out.append(
            {
                "id": mid,
                "type": mem_type,
                "text": text,
                "confidence": conf,
                "created_at": created,
                "last_confirmed_at": confirmed,
            }
        )
    return out


def _stm_turn_char_count(turn: Any) -> int:
    if not isinstance(turn, dict):
        return 0
    return len(str(turn.get("user") or "")) + len(str(turn.get("assistant") or ""))


def _trim_stm_history(hist: list[Any]) -> list[Any]:
    """维持轮数与总字符上限（从头部 pop）。"""
    while len(hist) > _MAX_CHAT_TURNS:
        hist.pop(0)
    while hist and sum(_stm_turn_char_count(t) for t in hist) > _MAX_STM_TOTAL_CHARS:
        hist.pop(0)
    return hist


def _default_supplements() -> dict[str, bool]:
    return {k: False for k in SUPPLEMENT_KEYS}


def _coerce_supplement_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "taken", "done", "已吃", "吃了", "打卡", "已补"):
            return True
        if s in ("false", "no", "0", "未吃", "没吃", "跳过"):
            return False
    return None


def _normalize_supplements(raw: Any) -> dict[str, bool]:
    out: dict[str, bool] = _default_supplements()
    if not isinstance(raw, dict):
        return out
    for k in SUPPLEMENT_KEYS:
        coerced = _coerce_supplement_bool(raw.get(k))
        if coerced is not None:
            out[k] = coerced
    return out


def format_supplement_status(supplements: dict[str, bool]) -> str:
    """单行中文：今日微量元素打卡完成情况。"""
    done: list[str] = [
        SUPPLEMENT_LABELS_ZH[k] for k in SUPPLEMENT_KEYS if supplements.get(k)
    ]
    pending: list[str] = [
        SUPPLEMENT_LABELS_ZH[k] for k in SUPPLEMENT_KEYS if not supplements.get(k)
    ]
    parts: list[str] = []
    if done:
        parts.append(f"已打卡：{'、'.join(done)}")
    if pending:
        parts.append(f"待补充：{'、'.join(pending)}")
    if not parts:
        return "微量元素补剂：今日尚未打卡。"
    return "微量元素补剂 — " + "；".join(parts) + "。"


def _new_day_log() -> dict[str, Any]:
    return {
        "consumed_protein": 0.0,
        "consumed_fat": 0.0,
        "consumed_carbs": 0.0,
        "food_items": [],
        "supplements": _default_supplements(),
        "supplement_log": [],
    }


def _normalize_day_log(raw: Any) -> dict[str, Any]:
    """保证单日结构完整、数值为 float。"""
    out: dict[str, Any] = _new_day_log()
    if not isinstance(raw, dict):
        return out
    try:
        out["consumed_protein"] = float(raw.get("consumed_protein") or 0)
        out["consumed_fat"] = float(raw.get("consumed_fat") or 0)
        out["consumed_carbs"] = float(raw.get("consumed_carbs") or 0)
    except (TypeError, ValueError):
        pass
    raw_items = raw.get("food_items")
    if isinstance(raw_items, list):
        cleaned: list[dict[str, Any]] = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            name = str(it.get("name", "")).strip() or "（未命名）"
            try:
                grams = float(it.get("grams") or 0)
            except (TypeError, ValueError):
                grams = 0.0
            cleaned_item: dict[str, Any] = {"name": name, "grams": grams}
            for src_key, dst_key in (
                ("protein", "protein"),
                ("fat", "fat"),
                ("net_carbs", "net_carbs"),
                ("carbs", "net_carbs"),
            ):
                if src_key not in it:
                    continue
                val = _coerce_opt_float(it.get(src_key))
                if val is not None:
                    cleaned_item[dst_key] = float(val)
            signature = str(it.get("signature") or "").strip()
            at = str(it.get("at") or "").strip()
            local_date = str(it.get("local_date") or "").strip()
            if signature:
                cleaned_item["signature"] = signature
            if at:
                cleaned_item["at"] = at
            if local_date:
                cleaned_item["local_date"] = local_date[:10]
            cleaned.append(cleaned_item)
        out["food_items"] = cleaned
    out["supplements"] = _normalize_supplements(raw.get("supplements"))
    raw_slog = raw.get("supplement_log")
    slog: list[dict[str, Any]] = []
    if isinstance(raw_slog, list):
        for entry in raw_slog:
            if not isinstance(entry, dict):
                continue
            note = str(entry.get("note", "")).strip()
            ts = str(entry.get("at", "")).strip()
            taken = _normalize_supplements(entry.get("supplements"))
            slog.append({"at": ts, "note": note, "supplements": taken})
    out["supplement_log"] = slog
    return out


def _normalize_daily_logs(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(k, str) and len(k) >= 8:
            out[k] = _normalize_day_log(v)
    return out


def _normalize_day_key(raw: Any) -> str:
    s = str(raw or "").strip()
    try:
        if s:
            return datetime.date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        pass
    return _today_iso()


def _normalize_amounts(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        key = str(k or "").strip()
        val = _coerce_opt_float(v)
        if key and val is not None:
            out[key] = float(val)
    return out


def _normalize_supplement_products(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        category = str(item.get("category") or "other").strip().lower() or "other"
        if not name:
            continue
        key = "".join(f"{category}:{name}".lower().split())
        if key in seen:
            continue
        seen.add(key)
        pid = str(item.get("id") or "").strip()
        if len(pid) != 8:
            pid = secrets.token_hex(4)
        created = str(item.get("created_at") or "").strip() or _utc_now_iso_z()
        updated = str(item.get("updated_at") or "").strip() or created
        out.append(
            {
                "id": pid,
                "name": name,
                "category": category,
                "form": str(item.get("form") or "").strip(),
                "serving_size": str(item.get("serving_size") or "").strip(),
                "units_per_serving": _coerce_opt_float(item.get("units_per_serving")),
                "amounts_per_serving": _normalize_amounts(
                    item.get("amounts_per_serving")
                ),
                "directions": str(item.get("directions") or "").strip(),
                "warnings": str(item.get("warnings") or "").strip(),
                "raw_label_notes": str(item.get("raw_label_notes") or "").strip(),
                "created_at": created,
                "updated_at": updated,
            }
        )
    return out[-_MAX_SUPPLEMENT_PRODUCTS:]


def _coerce_opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_strength_level(raw: Any) -> Optional[str]:
    """将提取值规范为 stable | drop | crash；未知则 None。"""
    if raw is None:
        return None
    s: str = str(raw).strip().lower()
    if not s or s in ("null", "none", "unknown", "未知"):
        return None
    if s in ("stable", "稳", "稳定", "保持", "持平"):
        return "stable"
    if s in ("drop", "下降", "下滑", "减退", "掉了"):
        return "drop"
    if s in ("crash", "暴跌", "崩", "塌", "崩盘"):
        return "crash"
    if s in _STRENGTH_LEVELS:
        return s
    return None


def _parse_timestamp_iso(s: Any) -> Optional[datetime.datetime]:
    if not isinstance(s, str) or not s.strip():
        return None
    t: str = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except ValueError:
        return None


def _normalize_metrics_history(raw: Any) -> list[dict[str, Any]]:
    """规范化体征历史列表并按时间排序。"""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        ts_raw = it.get("timestamp") or it.get("ts") or ""
        ts = str(ts_raw).strip()
        if not ts:
            continue
        entry: dict[str, Any] = {
            "timestamp": ts,
            "weight": _coerce_opt_float(it.get("weight")),
            "body_fat": _coerce_opt_float(it.get("body_fat")),
            "lbm": _coerce_opt_float(it.get("lbm")),
            "category": it.get("category"),
            "strength_level": _normalize_strength_level(it.get("strength_level")),
        }
        if entry["category"] is not None:
            try:
                entry["category"] = int(entry["category"])
            except (TypeError, ValueError):
                entry["category"] = None
        out.append(entry)

    def _key(e: dict[str, Any]) -> float:
        dt = _parse_timestamp_iso(e.get("timestamp"))
        return dt.timestamp() if dt else 0.0

    out.sort(key=_key)
    if len(out) > _MAX_METRICS_HISTORY:
        out = out[-_MAX_METRICS_HISTORY :]
    return out


def _migrate_legacy_top_level_metrics(base: dict[str, Any]) -> None:
    """旧版顶栏 ``weight`` / ``body_fat`` / ``lbm`` 迁入 ``metrics_history``（历史为空时一次性迁移）。"""
    hist: list[dict[str, Any]] = _normalize_metrics_history(base.get("metrics_history"))
    if hist:
        return
    w: Optional[float] = _coerce_opt_float(base.get("weight"))
    bf: Optional[float] = _coerce_opt_float(base.get("body_fat"))
    lb: Optional[float] = _coerce_opt_float(base.get("lbm"))
    if w is None and bf is None and lb is None:
        return
    ts_now: str = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    cat_val: Optional[int] = None
    cr = base.get("category")
    if cr is not None:
        try:
            cat_val = int(cr)
        except (TypeError, ValueError):
            cat_val = None
    hist.append(
        {
            "timestamp": ts_now,
            "weight": w,
            "body_fat": bf,
            "lbm": lb,
            "category": cat_val,
            "strength_level": None,
        }
    )
    base["metrics_history"] = _normalize_metrics_history(hist)


def _hydrate_vitals_from_history(base: dict[str, Any]) -> None:
    """从 ``metrics_history`` 最近一条有效快照填充派生键 ``weight`` / ``body_fat`` / ``lbm`` / ``strength_level``。"""
    hist: list[dict[str, Any]] = _normalize_metrics_history(base.get("metrics_history"))
    if not hist:
        # 保留磁盘上旧版顶栏字段（直至首次写入 metrics_history）
        return
    # 自最新往回找第一条含体重或体脂的记录；力量可取最新一条非空
    w: Optional[float] = None
    bf: Optional[float] = None
    lb: Optional[float] = None
    sl: Optional[str] = None
    for e in reversed(hist):
        if w is None:
            w = _coerce_opt_float(e.get("weight"))
        if bf is None:
            bf = _coerce_opt_float(e.get("body_fat"))
        if lb is None:
            lb = _coerce_opt_float(e.get("lbm"))
        if w is not None or bf is not None:
            break
    last_e: dict[str, Any] = hist[-1]
    sl = _normalize_strength_level(last_e.get("strength_level")) or sl
    if w is not None:
        base["weight"] = w
    else:
        base.pop("weight", None)
    if bf is not None:
        base["body_fat"] = bf
    else:
        base.pop("body_fat", None)
    if lb is not None:
        base["lbm"] = lb
    else:
        base.pop("lbm", None)
    if sl is not None:
        base["strength_level"] = sl
    else:
        base.pop("strength_level", None)


def _metrics_snapshot_changed(
    prev: Optional[dict[str, Any]],
    post_w: Optional[float],
    post_bf: Optional[float],
    post_lb: Optional[float],
    post_sl: Optional[str],
    post_cat: Optional[int],
) -> bool:
    """是否与上一快照相比需要追加新记录。"""
    if prev is None:
        return (
            post_w is not None
            or post_bf is not None
            or post_lb is not None
            or post_sl is not None
        )

    def _neq(
        a: Optional[float], b: Optional[float], eps: float = 1e-5
    ) -> bool:
        if a is None and b is None:
            return False
        if a is None or b is None:
            return True
        return abs(float(a) - float(b)) > eps

    pw = _coerce_opt_float(prev.get("weight"))
    pbf = _coerce_opt_float(prev.get("body_fat"))
    plb = _coerce_opt_float(prev.get("lbm"))
    psl = _normalize_strength_level(prev.get("strength_level"))
    pcl = prev.get("category")
    pci: Optional[int] = None
    if pcl is not None:
        try:
            pci = int(pcl)
        except (TypeError, ValueError):
            pci = None

    if _neq(pw, post_w) or _neq(pbf, post_bf) or _neq(plb, post_lb):
        return True
    if post_sl != psl and (post_sl is not None or psl is not None):
        return True
    if post_cat is not None and pci != post_cat:
        return True
    return False


def _food_log_signature(
    food_name: str,
    protein: float,
    fat: float,
    carbs: float,
    grams: float,
) -> str:
    name_key = "".join(str(food_name).strip().lower().split())
    return "|".join(
        (
            name_key,
            f"p={float(protein):.1f}",
            f"f={float(fat):.1f}",
            f"c={float(carbs):.1f}",
            f"g={float(grams):.1f}",
        )
    )


def _is_recent_duplicate_food_item(
    items: list[Any],
    signature: str,
    *,
    now: datetime.datetime,
    window_seconds: int = 20 * 60,
) -> bool:
    for raw in reversed(items):
        if not isinstance(raw, dict):
            continue
        if raw.get("signature") != signature:
            continue
        ts = _parse_timestamp_iso(raw.get("at"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        delta = abs((now - ts.astimezone(datetime.timezone.utc)).total_seconds())
        if delta <= window_seconds:
            return True
    return False


class UserStateManager:
    """
    读写 ``user_profiles.json``，提供 ``get_user`` / ``update_user`` / 对话追加。
    """

    def __init__(self, json_path: Optional[Path] = None) -> None:
        self._path: Path = (
            json_path
            if json_path is not None
            else Path(__file__).resolve().parent / "user_profiles.json"
        )
        self._lock: threading.RLock = threading.RLock()

    def _ensure_file(self) -> None:
        if not self._path.parent.is_dir():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text(
                json.dumps({"users": {}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_root(self) -> dict[str, Any]:
        self._ensure_file()
        try:
            raw: str = self._path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)
            if "users" not in data or not isinstance(data["users"], dict):
                data = {"users": {}}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.exception("读取 user_profiles 失败，将使用空存储：%s", exc)
            return {"users": {}}

    def _save_root(self, root: dict[str, Any]) -> None:
        self._ensure_file()
        tmp: Path = self._path.with_suffix(".json.tmp")
        text: str = json.dumps(root, ensure_ascii=False, indent=2)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self._path)

    def get_user(self, user_id: str) -> dict[str, Any]:
        """
        返回用户的完整状态字典；不存在则返回带默认字段的新用户结构（不写盘）。
        """
        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root["users"]
            if user_id not in users:
                fresh: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
                return fresh
            merged: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
            merged.update(users[user_id])
            if not isinstance(merged.get("chat_history"), list):
                merged["chat_history"] = []
            merged["daily_logs"] = _normalize_daily_logs(merged.get("daily_logs"))
            merged["metrics_history"] = _normalize_metrics_history(
                merged.get("metrics_history")
            )
            _hydrate_vitals_from_history(merged)
            try:
                merged["onboarding_stage"] = int(merged.get("onboarding_stage") or 0)
            except (TypeError, ValueError):
                merged["onboarding_stage"] = 0
            sp = merged.get("schedule_prefs")
            merged["schedule_prefs"] = sp if isinstance(sp, dict) else {}
            merged["user_memories"] = _normalize_user_memories(merged.get("user_memories"))
            merged["supplement_products"] = _normalize_supplement_products(
                merged.get("supplement_products")
            )
            return merged

    def update_user(self, user_id: str, updates: dict[str, Any]) -> None:
        """
        合并更新用户字段；``updates`` 中未出现的键保持不变。
        体征 ``weight`` / ``body_fat`` / ``lbm`` 仅存于 ``metrics_history`` 的时间序列；写盘前会去掉顶栏副本。
        当 ``updates`` 触及体征相关键且快照相对上一条有变化时，追加
        ``{"timestamp", "weight", "body_fat", "lbm", "category", "strength_level"}``。
        """
        _METRIC_TOUCH_KEYS: frozenset[str] = frozenset(
            {"weight", "body_fat", "lbm", "strength_level", "category"}
        )

        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root["users"]
            base: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
            if user_id in users:
                base.update(users[user_id])
            if not isinstance(base.get("chat_history"), list):
                base["chat_history"] = []

            _migrate_legacy_top_level_metrics(base)
            base["metrics_history"] = _normalize_metrics_history(base.get("metrics_history"))
            _hydrate_vitals_from_history(base)

            hist_before: list[dict[str, Any]] = _normalize_metrics_history(
                base.get("metrics_history")
            )
            prev_snap: Optional[dict[str, Any]] = (
                hist_before[-1] if hist_before else None
            )

            for key, val in updates.items():
                if key == "chat_history" and val is not None:
                    if not isinstance(val, list):
                        raise ValueError("chat_history 必须为列表")
                    base["chat_history"] = val
                elif key == "daily_logs" and val is not None:
                    if not isinstance(val, dict):
                        raise ValueError("daily_logs 必须为字典")
                    base["daily_logs"] = _normalize_daily_logs(val)
                elif key == "supplement_products" and val is not None:
                    if not isinstance(val, list):
                        raise ValueError("supplement_products 必须为列表")
                    base["supplement_products"] = _normalize_supplement_products(val)
                elif key == "metrics_history" and val is not None:
                    if not isinstance(val, list):
                        raise ValueError("metrics_history 必须为列表")
                    base["metrics_history"] = _normalize_metrics_history(val)
                else:
                    base[key] = val

            base["daily_logs"] = _normalize_daily_logs(base.get("daily_logs"))
            base["metrics_history"] = _normalize_metrics_history(base.get("metrics_history"))

            post_w: Optional[float] = _coerce_opt_float(base.get("weight"))
            post_bf: Optional[float] = _coerce_opt_float(base.get("body_fat"))
            post_lb: Optional[float] = _coerce_opt_float(base.get("lbm"))
            cat_raw = base.get("category")
            post_cat: Optional[int] = None
            if cat_raw is not None:
                try:
                    post_cat = int(cat_raw)
                except (TypeError, ValueError):
                    post_cat = None

            if "strength_level" in updates:
                post_sl = _normalize_strength_level(updates.get("strength_level"))
            elif prev_snap is not None:
                post_sl = _normalize_strength_level(prev_snap.get("strength_level"))
            else:
                post_sl = None

            if prev_snap is not None:
                if post_w is None:
                    post_w = _coerce_opt_float(prev_snap.get("weight"))
                if post_bf is None:
                    post_bf = _coerce_opt_float(prev_snap.get("body_fat"))
                if post_lb is None:
                    post_lb = _coerce_opt_float(prev_snap.get("lbm"))

            touches_metrics: bool = bool(_METRIC_TOUCH_KEYS.intersection(updates.keys()))

            if touches_metrics and _metrics_snapshot_changed(
                prev_snap, post_w, post_bf, post_lb, post_sl, post_cat
            ):
                hist_work: list[dict[str, Any]] = _normalize_metrics_history(
                    base.get("metrics_history")
                )
                ts_now: str = (
                    datetime.datetime.now(datetime.timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                hist_work.append(
                    {
                        "timestamp": ts_now,
                        "weight": post_w,
                        "body_fat": post_bf,
                        "lbm": post_lb,
                        "category": post_cat,
                        "strength_level": post_sl,
                    }
                )
                base["metrics_history"] = _normalize_metrics_history(hist_work)
            else:
                base["metrics_history"] = _normalize_metrics_history(
                    base.get("metrics_history")
                )

            base.pop("weight", None)
            base.pop("body_fat", None)
            base.pop("lbm", None)
            base.pop("strength_level", None)

            users[user_id] = base
            self._save_root(root)

    def get_today_supplement_status(self, user_id: str) -> dict[str, bool]:
        """返回今日微量元素补剂打卡状态（键见 ``SUPPLEMENT_KEYS``）。"""
        log: dict[str, Any] = self.get_today_log(user_id)
        return _normalize_supplements(log.get("supplements"))

    def get_today_log(self, user_id: str) -> dict[str, Any]:
        """
        获取或初始化「今天」（本地日期 ``YYYY-MM-DD``）的饮食日志；新建时会写盘。
        返回包含 ``consumed_protein`` / ``consumed_fat`` / ``consumed_carbs`` / ``food_items`` 的副本。
        """
        day_key: str = _today_iso()
        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root["users"]
            base: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
            if user_id in users:
                base.update(users[user_id])
            if not isinstance(base.get("chat_history"), list):
                base["chat_history"] = []
            logs: dict[str, Any] = _normalize_daily_logs(base.get("daily_logs"))
            if day_key not in logs:
                logs[day_key] = _new_day_log()
            else:
                logs[day_key] = _normalize_day_log(logs[day_key])
            base["daily_logs"] = logs
            base["metrics_history"] = _normalize_metrics_history(base.get("metrics_history"))
            users[user_id] = base
            self._save_root(root)
            return deepcopy(logs[day_key])

    def add_food_log(
        self,
        user_id: str,
        food_name: str,
        protein: float,
        fat: float,
        carbs: float,
        grams: float = 0.0,
        *,
        log_date: Optional[str] = None,
    ) -> None:
        """
        将单次进食的宏量养分累加到今日记录，并在 ``food_items`` 中追加名称与克数。
        ``grams`` 默认 0，便于仅已知宏量而未知克数的录入。
        """
        day_key: str = _normalize_day_key(log_date)
        name: str = str(food_name).strip() or "（未命名）"
        try:
            dp = float(protein)
            df = float(fat)
            dc = float(carbs)
            g = float(grams)
        except (TypeError, ValueError) as exc:
            raise ValueError("protein / fat / carbs / grams 须为数值") from exc

        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root["users"]
            base: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
            if user_id in users:
                base.update(users[user_id])
            if not isinstance(base.get("chat_history"), list):
                base["chat_history"] = []
            logs: dict[str, Any] = _normalize_daily_logs(base.get("daily_logs"))
            log: dict[str, Any] = _normalize_day_log(logs.get(day_key))
            items: list[Any] = list(log.get("food_items") or [])
            signature = _food_log_signature(name, dp, df, dc, g)
            now_iso = _utc_now_iso_z()
            now_dt = _parse_timestamp_iso(now_iso) or datetime.datetime.now(datetime.timezone.utc)
            if _is_recent_duplicate_food_item(items, signature, now=now_dt):
                logger.info(
                    "跳过短时间重复饮食写入 user_id=%s day=%s food=%s",
                    user_id,
                    day_key,
                    name,
                )
                return

            log["consumed_protein"] = float(log["consumed_protein"]) + dp
            log["consumed_fat"] = float(log["consumed_fat"]) + df
            log["consumed_carbs"] = float(log["consumed_carbs"]) + dc
            items.append(
                {
                    "name": name,
                    "grams": g,
                    "protein": dp,
                    "fat": df,
                    "net_carbs": dc,
                    "signature": signature,
                    "at": now_iso,
                    "local_date": day_key,
                }
            )
            log["food_items"] = items
            logs[day_key] = log
            base["daily_logs"] = logs
            base["metrics_history"] = _normalize_metrics_history(base.get("metrics_history"))
            users[user_id] = base
            self._save_root(root)

    def apply_supplement_updates(
        self,
        user_id: str,
        updates: dict[str, bool],
        *,
        note: str = "",
    ) -> None:
        """
        合并今日 ``supplements`` 打卡状态（仅处理 ``SUPPLEMENT_KEYS`` 中的键）。
        ``updates`` 中 True 表示已服用；False 可显式取消当日打卡。
        """
        if not updates:
            return
        day_key: str = _today_iso()
        cleaned: dict[str, bool] = {}
        for k, v in updates.items():
            if k not in SUPPLEMENT_KEYS:
                continue
            if isinstance(v, bool):
                cleaned[k] = v
        if not cleaned:
            return

        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root["users"]
            base: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
            if user_id in users:
                base.update(users[user_id])
            logs: dict[str, Any] = _normalize_daily_logs(base.get("daily_logs"))
            log: dict[str, Any] = _normalize_day_log(logs.get(day_key))
            sup: dict[str, bool] = _normalize_supplements(log.get("supplements"))
            sup.update(cleaned)
            log["supplements"] = sup
            if note.strip():
                slog: list[Any] = list(log.get("supplement_log") or [])
                slog.append(
                    {
                        "at": datetime.datetime.now(datetime.timezone.utc)
                        .replace(microsecond=0)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "note": note.strip(),
                        "supplements": dict(cleaned),
                    }
                )
                log["supplement_log"] = slog[-30:]
            logs[day_key] = log
            base["daily_logs"] = logs
            base["metrics_history"] = _normalize_metrics_history(
                base.get("metrics_history")
            )
            users[user_id] = base
            self._save_root(root)

    def upsert_supplement_products(
        self,
        user_id: str,
        products: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        合并用户拍照录入的补剂产品标签。
        同名 + 同类别视为同一产品，后续拍照会更新标签信息而不是重复追加。
        """
        cleaned: list[dict[str, Any]] = _normalize_supplement_products(products)
        if not cleaned:
            return []

        def _product_key(item: dict[str, Any]) -> str:
            return "".join(
                f"{item.get('category') or 'other'}:{item.get('name') or ''}"
                .lower()
                .split()
            )

        now = _utc_now_iso_z()
        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root["users"]
            base: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
            if user_id in users:
                base.update(users[user_id])
            existing: list[dict[str, Any]] = _normalize_supplement_products(
                base.get("supplement_products")
            )
            by_key: dict[str, dict[str, Any]] = {
                _product_key(item): item for item in existing
            }
            changed: list[dict[str, Any]] = []
            for item in cleaned:
                key = _product_key(item)
                prev = by_key.get(key)
                if prev is not None:
                    item["id"] = prev.get("id") or item.get("id") or secrets.token_hex(4)
                    item["created_at"] = prev.get("created_at") or item.get("created_at") or now
                item["updated_at"] = now
                by_key[key] = item
                changed.append(item)
            merged = _normalize_supplement_products(list(by_key.values()))
            base["supplement_products"] = merged
            base["daily_logs"] = _normalize_daily_logs(base.get("daily_logs"))
            base["metrics_history"] = _normalize_metrics_history(
                base.get("metrics_history")
            )
            users[user_id] = base
            self._save_root(root)
            return deepcopy(changed)

    def get_supplement_products(self, user_id: str) -> list[dict[str, Any]]:
        """返回用户已录入的补剂产品标签档案。"""
        return deepcopy(
            _normalize_supplement_products(
                self.get_user(user_id).get("supplement_products")
            )
        )

    def get_daily_summary(self, user_id: str) -> str:
        """
        返回今日已摄入宏量、相对 ``target_protein`` 区间的缺口/超出，
        以及相对 20 g 脂肪、50 g 碳水红线的余量（可为负表示已超标）。
        """
        log: dict[str, Any] = self.get_today_log(user_id)
        state: dict[str, Any] = self.get_user(user_id)
        day_key: str = _today_iso()

        cp: float = float(log.get("consumed_protein") or 0)
        cf: float = float(log.get("consumed_fat") or 0)
        cc: float = float(log.get("consumed_carbs") or 0)

        lines: list[str] = [
            f"【今日饮食摘要】{day_key}",
            f"已摄入 — 蛋白质：{cp:.1f} g，脂肪：{cf:.1f} g，碳水：{cc:.1f} g。",
        ]

        tp = state.get("target_protein")
        p_min: Optional[float] = None
        p_max: Optional[float] = None
        if isinstance(tp, dict):
            try:
                if tp.get("min") is not None:
                    p_min = float(tp["min"])
                if tp.get("max") is not None:
                    p_max = float(tp["max"])
            except (TypeError, ValueError):
                pass

        if p_min is None and p_max is None:
            lines.append(
                "蛋白质目标：档案中尚无 target_protein（请先完成体征与引擎计算）；无法估算蛋白缺口。"
            )
        else:
            if p_min is not None and p_max is not None:
                lo, hi = p_min, p_max
            elif p_min is not None:
                lo = hi = p_min
            else:
                lo = hi = p_max  # type: ignore[assignment]
            assert lo is not None and hi is not None
            if lo > hi:
                lo, hi = hi, lo
            if cp < lo:
                lines.append(
                    f"蛋白质目标区间：{lo:.1f}–{hi:.1f} g/天；当前低于下限，还差约 {lo - cp:.1f} g 蛋白可达下限。"
                )
            elif cp <= hi:
                lines.append(
                    f"蛋白质目标区间：{lo:.1f}–{hi:.1f} g/天；已在区间内，距离上限还可约 {hi - cp:.1f} g。"
                )
            else:
                lines.append(
                    f"蛋白质目标区间：{lo:.1f}–{hi:.1f} g/天；已超过上限约 {cp - hi:.1f} g。"
                )

        fat_margin: float = _FAT_LIMIT_G - cf
        carb_margin: float = _CARB_LIMIT_G - cc
        lines.append(
            f"脂肪红线：{_FAT_LIMIT_G:.0f} g（参考）；剩余余量约 {fat_margin:.1f} g"
            + ("（已超红线）。" if fat_margin < 0 else "。")
        )
        lines.append(
            f"碳水红线：{_CARB_LIMIT_G:.0f} g（参考）；剩余余量约 {carb_margin:.1f} g"
            + ("（已超红线）。" if carb_margin < 0 else "。")
        )

        n_items: int = len(log.get("food_items") or [])
        lines.append(f"今日已记录食物条目数：{n_items}。")

        sup: dict[str, bool] = _normalize_supplements(log.get("supplements"))
        n_done: int = sum(1 for k in SUPPLEMENT_KEYS if sup.get(k))
        lines.append(format_supplement_status(sup))
        lines.append(
            f"微量元素完成度：{n_done}/{len(SUPPLEMENT_KEYS)}（锌、镁、鱼油、B族、K2+D3）。"
        )

        return "\n".join(lines)

    def get_smoothed_weight(self, user_id: str) -> Optional[float]:
        """
        最近 **7 条**有效体重记录（按时间新老顺序取最新 7 次）的算术平均；
        用于平滑 PSMF 初期水分与糖原波动。
        """
        state: dict[str, Any] = self.get_user(user_id)
        hist: list[dict[str, Any]] = _normalize_metrics_history(state.get("metrics_history"))
        ws: list[float] = []
        for e in reversed(hist):
            w = _coerce_opt_float(e.get("weight"))
            if w is not None:
                ws.append(float(w))
            if len(ws) >= 7:
                break
        if not ws:
            return None
        return sum(ws) / float(len(ws))

    def get_two_window_smoothed_weights(
        self, user_id: str
    ) -> tuple[Optional[float], Optional[float]]:
        """
        返回 (当前窗口均值, 前一窗口均值)：当前窗口为最近 7 条体重，前一窗口为紧接着更早的 7 条。
        样本不足 7 条时返回 (None, None)；满 7 条但不足 14 条时返回 (当前均值, None)。
        """
        state: dict[str, Any] = self.get_user(user_id)
        hist: list[dict[str, Any]] = _normalize_metrics_history(state.get("metrics_history"))
        all_w: list[float] = []
        for e in reversed(hist):
            w = _coerce_opt_float(e.get("weight"))
            if w is not None:
                all_w.append(float(w))
        if len(all_w) < 7:
            return (None, None)
        cur_avg: float = sum(all_w[:7]) / 7.0
        if len(all_w) < 14:
            return (cur_avg, None)
        prev_avg: float = sum(all_w[7:14]) / 7.0
        return (cur_avg, prev_avg)

    def get_latest_strength_level(self, user_id: str) -> Optional[str]:
        """``metrics_history`` 最近一条快照中的 ``strength_level``。"""
        hist: list[dict[str, Any]] = _normalize_metrics_history(
            self.get_user(user_id).get("metrics_history")
        )
        if not hist:
            return None
        return _normalize_strength_level(hist[-1].get("strength_level"))

    def get_progress_report(self, user_id: str) -> str:
        """
        基于 ``metrics_history`` 生成进度文字：原始体重轨迹、7 次滑动均值（平滑体重）、
        以及力量水平（``strength_level``）。肌肉保留以力量反馈为主，不再作 LBM 数值警报。
        """
        state: dict[str, Any] = self.get_user(user_id)
        hist: list[dict[str, Any]] = _normalize_metrics_history(state.get("metrics_history"))

        lines: list[str] = ["【体能进度报告】"]

        if not hist:
            cw = _coerce_opt_float(state.get("weight"))
            extra: str = f"当前档案体重：{cw:.2f} kg。" if cw is not None else ""
            lines.append(
                "尚无体征历史记录（metrics_history 为空）。"
                + (extra if extra else " 请在更新体重/体脂后自动生成时间序列。")
            )
            return "\n".join(lines)

        sm: Optional[float] = self.get_smoothed_weight(user_id)
        cur_prev: tuple[Optional[float], Optional[float]] = (
            self.get_two_window_smoothed_weights(user_id)
        )
        cur_avg, prev_avg = cur_prev
        if sm is not None:
            lines.append(
                f"7 次滑动平均体重（平滑）：约 {sm:.2f} kg（最近至多 7 条有效体重记录之算术平均）。"
            )
        else:
            lines.append("7 次滑动平均体重：暂无足够体重记录。")

        if cur_avg is not None and prev_avg is not None:
            wk_delta: float = float(prev_avg) - float(cur_avg)
            lines.append(
                f"双窗口参考：当前窗口均值约 {cur_avg:.2f} kg，前一窗口均值约 {prev_avg:.2f} kg；"
                f"两窗口差约 {wk_delta:+.2f} kg（正数表示整体呈下降趋势，数值受打卡频率影响）。"
            )
        elif cur_avg is not None:
            lines.append(f"当前窗口（最近 7 条）平均体重约 {cur_avg:.2f} kg；尚不足以计算前一窗口。")

        # ----- 原始体重轨迹 -----
        w_series: list[tuple[str, float]] = []
        for e in hist:
            w = _coerce_opt_float(e.get("weight"))
            if w is not None:
                ts = str(e.get("timestamp") or "")
                w_series.append((ts, float(w)))

        cur_w: Optional[float] = _coerce_opt_float(state.get("weight"))
        if w_series:
            first_ts, first_w = w_series[0][0], w_series[0][1]
            last_ts, last_w = w_series[-1][0], w_series[-1][1]
            if cur_w is not None:
                last_w = float(cur_w)
            delta_kg: float = float(first_w) - float(last_w)
            dt_a = _parse_timestamp_iso(first_ts)
            dt_b = _parse_timestamp_iso(last_ts)
            weeks: float = 0.0
            if dt_a and dt_b and dt_b >= dt_a:
                weeks = max((dt_b - dt_a).total_seconds() / (86400.0 * 7.0), 1e-9)
            weekly: float = delta_kg / weeks if weeks > 0 else 0.0

            lines.append(
                f"体重轨迹（原始记录）：最早约 {first_w:.2f} kg（{first_ts[:10]}）→ "
                f"当前约 {last_w:.2f} kg。"
            )
            lines.append(
                f"期间总体重变化：{delta_kg:+.2f} kg（正数表示体重下降、负数表示上升）。"
            )
            if weeks >= 1 / 168:
                lines.append(
                    f"观测跨度约 {weeks:.2f} 周；估算平均每周体重变化约 {weekly:+.3f} kg/周 "
                    f"（正数表示平均每周减重）。"
                )
            else:
                lines.append("记录时间跨度较短，「每周平均」仅供参考。")
        else:
            lines.append("历史记录中尚无有效体重数值；请补充体重数据。")

        lines.append("")
        latest_sl: Optional[str] = _normalize_strength_level(hist[-1].get("strength_level"))
        if latest_sl is not None:
            lines.append(f"最近一条快照的力量水平 strength_level：{latest_sl}（stable=稳定，drop=减退，crash=暴跌）。")
        else:
            lines.append(
                "力量水平 strength_level：尚无记录；请在对话中反馈训练重量/主观发力（如深蹲是否掉重量）。"
            )

        sl_series: list[str] = []
        for e in hist:
            sl = _normalize_strength_level(e.get("strength_level"))
            if sl is not None:
                sl_series.append(sl)
        if len(sl_series) >= 2:
            lines.append(f"历史力量标签序列（节选）：{' → '.join(sl_series[-5:])}。")

        if latest_sl == "crash":
            lines.append(
                "【力量警报 · crash】最近快照标记为力量暴跌；必须在回复中按「高危肌肉流失」关切处理，"
                "优先复盘蛋白质摄入与恢复睡眠，并审慎减压训练强度。"
            )

        lines.append(f"体征快照条数（metrics_history）：{len(hist)}。")

        return "\n".join(lines)

    def get_weight_continuity_narrative(self, user_id: str) -> str:
        """
        生成一两句「体重 / LBM 连续性」评价文案，供 Agent 合成时引用。
        锚点优先取「上一自然周的周一」附近的历史体重；若无则退化为最早一条体重记录。
        """
        state: dict[str, Any] = self.get_user(user_id)
        hist: list[dict[str, Any]] = _normalize_metrics_history(state.get("metrics_history"))
        cur_w: Optional[float] = _coerce_opt_float(state.get("weight"))
        cur_lb: Optional[float] = _coerce_opt_float(state.get("lbm"))

        if cur_w is None:
            return "（暂无当前体重，无法生成连续性反馈。）"

        today: datetime.date = _local_now().date()
        this_monday: datetime.date = today - datetime.timedelta(days=today.weekday())
        anchor_monday: datetime.date = this_monday - datetime.timedelta(days=7)

        best_e: Optional[dict[str, Any]] = None
        best_dist: float = 1e18
        for e in hist:
            d = _local_date_from_timestamp(e.get("timestamp"))
            if d is None:
                continue
            w = _coerce_opt_float(e.get("weight"))
            if w is None:
                continue
            if d > today:
                continue
            dist: float = float(abs((d - anchor_monday).days))
            if dist < best_dist:
                best_dist = dist
                best_e = e

        if best_e is None:
            for e in hist:
                w = _coerce_opt_float(e.get("weight"))
                if w is not None:
                    best_e = e
                    break

        if best_e is None:
            return (
                f"（当前体重 {cur_w:.2f} kg；尚无历史锚点可对比「上周一」附近记录。）"
            )

        old_w = _coerce_opt_float(best_e.get("weight"))
        old_lb = _coerce_opt_float(best_e.get("lbm"))
        ts_short: str = str(best_e.get("timestamp") or "")[:16]
        if old_w is None:
            return f"（当前体重 {cur_w:.2f} kg；历史锚点不完整。）"

        delta_kg: float = float(old_w) - float(cur_w)
        parts: list[str] = [
            f"连续性参考：相较锚点记录（约 {ts_short}，体重约 {old_w:.2f} kg），"
            f"当前体重约 {cur_w:.2f} kg，变化约 {delta_kg:+.2f} kg（正数表示体重下降）。"
        ]

        if old_lb is not None and cur_lb is not None and old_lb > 1e-6:
            r: float = (float(cur_lb) - float(old_lb)) / float(old_lb)
            if abs(r) <= 0.02:
                parts.append(
                    "同期 LBM 波动约在 ±2% 内，可视为相对稳定；若体重稳步下降且 LBM 稳定，可鼓励继续保持。"
                )
            elif r < -0.02:
                parts.append(
                    "同期 LBM 降幅超过约 2%，需在回复中提醒复核蛋白质与训练负荷。"
                )
            else:
                parts.append("同期 LBM 有所上升，对保肌有利。")

        return " ".join(parts)

    def get_today_protein_gap_vs_min(
        self, user_id: str
    ) -> tuple[float, Optional[float], Optional[float]]:
        """
        返回 ``(今日已摄入蛋白质 g, 目标下限 g 或 None, 相对下限的缺口 g)``。
        缺口 > 0 表示未达下限；若无 target_protein 则后两项为 None。
        """
        log: dict[str, Any] = self.get_today_log(user_id)
        cp: float = float(log.get("consumed_protein") or 0)
        state: dict[str, Any] = self.get_user(user_id)
        tp = state.get("target_protein")
        p_min: Optional[float] = None
        if isinstance(tp, dict):
            try:
                if tp.get("min") is not None:
                    p_min = float(tp["min"])
            except (TypeError, ValueError):
                pass
        if p_min is None:
            return cp, None, None
        gap: float = max(0.0, float(p_min) - cp)
        return cp, p_min, gap

    def generate_weekly_analysis(self, user_id: str) -> str:
        """
        滚动最近 7 个自然日（含今天）：``metrics_history`` 体重/力量趋势 + ``daily_logs`` 蛋白质汇总。
        """
        state: dict[str, Any] = self.get_user(user_id)
        hist: list[dict[str, Any]] = _normalize_metrics_history(state.get("metrics_history"))
        today: datetime.date = _local_now().date()
        start: datetime.date = today - datetime.timedelta(days=6)

        entries: list[dict[str, Any]] = []
        for e in hist:
            d = _local_date_from_timestamp(e.get("timestamp"))
            if d is None:
                continue
            if start <= d <= today:
                entries.append(e)

        lines: list[str] = [
            "【周报数据 · 最近 7 天】",
            f"窗口：{start.isoformat()} — {today.isoformat()}",
        ]

        w_vals: list[float] = []
        for e in entries:
            w = _coerce_opt_float(e.get("weight"))
            if w is not None:
                w_vals.append(float(w))
        if len(w_vals) >= 2:
            delta = w_vals[0] - w_vals[-1]
            lines.append(
                f"体重登记点：窗口内最早样本约 {w_vals[0]:.2f} kg → 最近约 {w_vals[-1]:.2f} kg "
                f"（变化约 {delta:+.2f} kg，正数表示下降）。"
            )
        elif len(w_vals) == 1:
            lines.append(f"体重：窗口内仅 1 个有效点约 {w_vals[0]:.2f} kg。")
        else:
            lines.append("体重：窗口内无有效体重打点。")

        sl_seq: list[str] = []
        for e in entries:
            sl = _normalize_strength_level(e.get("strength_level"))
            if sl is not None:
                sl_seq.append(sl)
        if len(sl_seq) >= 2:
            first, last = sl_seq[0], sl_seq[-1]
            trend = "暂不明显"
            order = {"stable": 0, "drop": 1, "crash": 2}
            if order.get(last, 0) > order.get(first, 0):
                trend = "力量主观反馈呈走弱（stable→drop/crash 方向）"
            elif order.get(last, 0) < order.get(first, 0):
                trend = "力量主观反馈略有好转"
            lines.append(f"力量标签序列（节选）：{' → '.join(sl_seq[-5:])}；趋势判断：{trend}。")
        elif sl_seq:
            lines.append(f"力量标签：{sl_seq[-1]}（样本较少）。")
        else:
            lines.append("力量标签：窗口内无 strength_level 记录。")

        logs: dict[str, Any] = _normalize_daily_logs(state.get("daily_logs"))
        tp = state.get("target_protein")
        p_min: Optional[float] = None
        p_max: Optional[float] = None
        if isinstance(tp, dict):
            try:
                if tp.get("min") is not None:
                    p_min = float(tp["min"])
                if tp.get("max") is not None:
                    p_max = float(tp["max"])
            except (TypeError, ValueError):
                pass

        lines.append("")
        lines.append("【每日蛋白质摄入（来自 daily_logs）】")
        for i in range(7):
            dk: str = (start + datetime.timedelta(days=i)).isoformat()
            dl = logs.get(dk)
            if not isinstance(dl, dict):
                cp = 0.0
            else:
                cp = float(dl.get("consumed_protein") or 0)
            flag = ""
            if p_min is not None:
                if cp < p_min:
                    flag = f"（低于区间下限 {p_min:.1f} g）"
                elif p_max is not None and cp > p_max:
                    flag = f"（高于区间上限 {p_max:.1f} g）"
            lines.append(f"  {dk}：蛋白质累计约 {cp:.1f} g{flag}")

        lines.append("")
        lines.append("【每日微量元素补剂打卡（锌 / 镁 / 鱼油 / B族 / K2+D3）】")
        for i in range(7):
            dk2: str = (start + datetime.timedelta(days=i)).isoformat()
            dl2 = logs.get(dk2)
            if not isinstance(dl2, dict):
                sup2 = _default_supplements()
            else:
                sup2 = _normalize_supplements(dl2.get("supplements"))
            n_done2: int = sum(1 for k in SUPPLEMENT_KEYS if sup2.get(k))
            lines.append(f"  {dk2}：{n_done2}/{len(SUPPLEMENT_KEYS)} 项已打卡")

        return "\n".join(lines)

    def list_all_user_ids(self) -> list[str]:
        """返回 ``user_profiles.json`` 中所有 ``user_id``。"""
        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root.get("users")
            if not isinstance(users, dict):
                return []
            return [str(k) for k in users.keys()]

    def list_user_ids_by_onboarding_stage(self, stage: int) -> list[str]:
        """筛选 ``onboarding_stage`` 等于给定值的用户。"""
        out: list[str] = []
        for uid in self.list_all_user_ids():
            try:
                st = int(self.get_user(uid).get("onboarding_stage") or 0)
            except (TypeError, ValueError):
                st = 0
            if st == stage:
                out.append(uid)
        return out

    def append_chat_turn(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """追加一轮对话，维持最多 ``_MAX_CHAT_TURNS`` 条且总字符不超过 ``_MAX_STM_TOTAL_CHARS``。"""
        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root["users"]
            base: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
            if user_id in users:
                base.update(users[user_id])
            if not isinstance(base.get("chat_history"), list):
                base["chat_history"] = []
            base["daily_logs"] = _normalize_daily_logs(base.get("daily_logs"))
            hist: list[Any] = list(base["chat_history"])
            hist.append(
                {
                    "user": user_message.strip()[:_MAX_STM_USER_CHARS],
                    "assistant": assistant_message.strip()[:_MAX_STM_ASSISTANT_CHARS],
                }
            )
            hist = _trim_stm_history(hist)
            base["chat_history"] = hist
            base["metrics_history"] = _normalize_metrics_history(base.get("metrics_history"))
            users[user_id] = base
            self._save_root(root)

    def get_user_memories(self, user_id: str) -> list[dict[str, Any]]:
        """返回用户软记忆列表（副本）。"""
        return list(_normalize_user_memories(self.get_user(user_id).get("user_memories")))

    def merge_user_memories(self, user_id: str, new_items: list[dict[str, Any]]) -> None:
        """
        合并软记忆：按 type + 归一化 text 去重；命中则更新 ``last_confirmed_at``。
        最多 ``_MAX_USER_MEMORIES`` 条，超出删 confidence 最低且最久未确认者。
        不触碰 ``metrics_history`` / ``daily_logs``。
        """
        if not new_items:
            return
        with self._lock:
            root: dict[str, Any] = self._load_root()
            users: dict[str, Any] = root["users"]
            base: dict[str, Any] = deepcopy(_DEFAULT_PROFILE)
            if user_id in users:
                base.update(users[user_id])
            existing: list[dict[str, Any]] = _normalize_user_memories(
                base.get("user_memories")
            )
            now: str = _utc_now_iso_z()
            index: dict[tuple[str, str], int] = {}
            for i, mem in enumerate(existing):
                t = str(mem.get("type") or "other")
                tk = _normalize_memory_text_key(str(mem.get("text") or ""))
                index[(t, tk)] = i

            for item in new_items:
                if not isinstance(item, dict):
                    continue
                mem_type = str(item.get("type") or "other").strip().lower()
                if mem_type not in _VALID_MEMORY_TYPES:
                    mem_type = "other"
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                try:
                    conf = float(item.get("confidence", 0.85))
                except (TypeError, ValueError):
                    conf = 0.85
                conf = max(0.0, min(1.0, conf))
                key = (mem_type, _normalize_memory_text_key(text))
                if key in index:
                    idx = index[key]
                    existing[idx]["last_confirmed_at"] = now
                    existing[idx]["confidence"] = max(
                        float(existing[idx].get("confidence") or 0.0), conf
                    )
                else:
                    new_mem: dict[str, Any] = {
                        "id": secrets.token_hex(4),
                        "type": mem_type,
                        "text": text,
                        "confidence": conf,
                        "created_at": now,
                        "last_confirmed_at": now,
                    }
                    existing.append(new_mem)
                    index[key] = len(existing) - 1

            while len(existing) > _MAX_USER_MEMORIES:
                worst_i = 0
                for i in range(1, len(existing)):
                    wi = existing[worst_i]
                    wi_mem = existing[i]
                    wc = float(wi.get("confidence") or 0.0)
                    ic = float(wi_mem.get("confidence") or 0.0)
                    if ic < wc:
                        worst_i = i
                    elif ic == wc:
                        wt = _parse_timestamp_iso(wi.get("last_confirmed_at"))
                        it = _parse_timestamp_iso(wi_mem.get("last_confirmed_at"))
                        wt_ts = wt.timestamp() if wt else 0.0
                        it_ts = it.timestamp() if it else 0.0
                        if it_ts < wt_ts:
                            worst_i = i
                existing.pop(worst_i)

            base["user_memories"] = existing
            base["metrics_history"] = _normalize_metrics_history(base.get("metrics_history"))
            users[user_id] = base
            self._save_root(root)
