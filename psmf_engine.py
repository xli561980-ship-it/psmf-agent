"""
PSMF Agent：瘦体重 / Category 计算、本地护栏，以及基于 Google Gemini 的线性流水线 Agent。

执行顺序（严格、无 AFC 工具链）：
1) 按 user_id 加载 LTM/STM；提取（可多模态图片 + 文本 JSON）；含营养成分表时解析 ``logged_food``；识别 CNS 疲劳与力量反馈（strength_level）
2) 若提取到 ``logged_food``：写入 ``daily_logs``（add_food_log）并生成 ``get_daily_summary``；再比对体征并强制计算 LBM/Category；体征仅存 ``metrics_history`` 时间序列（硬约束：不得削弱 metrics_history 追加/裁剪语义）
3) 生理学分析：力量暴跌 crash → 高危肌肉流失；双窗口 7 次滑动均值 vs Category 减脂过快 → 提示补盐防水肿脱水假象
4) 合成前按「意图 user_intent + 档位」门控 RAG；minimal 时避免塞入无关训练/症状摘录；``full`` 档含完整 bundle，回复末追加【生理趋势分析】；``minimal``/``guided`` 仅必要时追加短风险提示
5) 持久化 user_profiles.json 与 STM 滑动窗口（最近 50 轮，总字符约 12 万）；Phase 1.5 巩固软记忆 user_memories

本模块仅供程序调用参考，不构成医疗建议。
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — Python <3.9 极少
    ZoneInfo = None  # type: ignore[misc, assignment]

try:
    from dotenv import load_dotenv
except ImportError as exc:  # pragma: no cover
    raise ImportError("请安装 python-dotenv：pip install python-dotenv") from exc

_ENV_PATH: Final[Path] = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

try:
    from google import genai
    from google.genai import types as genai_types  # Content / Part（如 Part.from_bytes）用于多模态
except ImportError as exc:  # pragma: no cover
    raise ImportError("请安装 google-genai：pip install google-genai") from exc

from memory_manager import SUPPLEMENT_KEYS, UserStateManager
from rag_system import COLLECTION_FOOD, COLLECTION_NAME

logger = logging.getLogger(__name__)

# RAG：主知识库内按 metadata.source 区分文件
_RAG_SOURCE_TRAINING_GUIDE: Final[str] = "psmf_training_guide.md"
_RAG_SOURCE_SYMPTOM_MATRIX: Final[str] = "symptom_diagnostic_matrix.md"
_RAG_SOURCE_MICRONUTRIENT_GUIDE: Final[str] = (
    "psmf_micronutrient_electrolyte_guide.md"
)
# food_db 源自 ``psmf_food_database.md`` 入库；睡前复盘强制检索该库
_FOOD_DB_LABEL: Final[str] = "psmf_food_database.md（food_db）"

# 中枢神经疲劳：提取字段 + 文本兜底命中则触发训练减量策略
_CNS_FATIGUE_KEYWORDS: Final[tuple[str, ...]] = (
    "中枢神经",
    "CNS",
    "中枢疲劳",
    "神经疲劳",
    "发力感消失",
    "肌肉发空",
    "睡眠障碍",
    "失眠",
    "晨脉",
    "倦怠",
    "动力下降",
    "反应变慢",
)

# 力量减退口语 → 用于启发式补充 strength_level（与 memory_manager 枚举对齐）
_STRENGTH_DROP_WORDS: Final[tuple[str, ...]] = (
    "蹲不动",
    "蹲不下去了",
    "没劲",
    "使不上劲",
    "重量掉了",
    "上不去重量",
    "推不动",
    "拉不动",
    "力量掉了",
    "不如以前",
)
_STRENGTH_CRASH_WORDS: Final[tuple[str, ...]] = (
    "完全不行",
    "崩盘",
    "暴跌",
    "完全举不起来",
    "一点都蹲不动",
    "塌了",
)


def _guess_image_mime(data: bytes) -> str:
    """根据魔数推断图片 MIME，供 Gemini 多模态输入。"""
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


# -----------------------------------------------------------------------------
# 模型：默认 gemini-3.5-flash；可用 GEMINI_MODEL / GEMINI_FAST_MODEL 分别控制质量与快速模型
# -----------------------------------------------------------------------------
def _default_model_id() -> str:
    mid: str = (os.environ.get("GEMINI_MODEL") or "").strip()
    if mid:
        return mid
    return "gemini-3.5-flash"


def _default_fast_model_id(quality_model: Optional[str] = None) -> str:
    mid: str = (
        os.environ.get("GEMINI_FAST_MODEL")
        or os.environ.get("GEMINI_LITE_MODEL")
        or ""
    ).strip()
    if mid:
        return mid
    return "gemini-3.1-flash-lite"


# 对外兼容旧常量名
GEMINI_MODEL_ID: Final[str] = _default_model_id()

# -----------------------------------------------------------------------------
# 常量：蛋白质 g/kg LBM（与 psmf_rules.md 一致）
# -----------------------------------------------------------------------------

_CAT1_G_PER_KG_LBM_MIN: Final[float] = 3.3
_CAT1_G_PER_KG_LBM_MAX: Final[float] = 4.4
_CAT2_G_PER_KG_LBM_MIN: Final[float] = 2.75
_CAT2_G_PER_KG_LBM_MAX: Final[float] = 3.3
_CAT3_G_PER_KG_LBM_MIN: Final[float] = 2.2
_CAT3_G_PER_KG_LBM_MAX: Final[float] = 2.75

_COLOR_RED: Final[str] = "\033[91m"
_COLOR_RESET: Final[str] = "\033[0m"

_HIGH_RISK_KEYWORDS: Final[tuple[str, ...]] = (
    "头晕",
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

_HARD_STOP_SYMPTOM_KEYWORDS: Final[tuple[str, ...]] = (
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

GenderLiteral = Literal["male", "female"]


# =============================================================================
# 纯函数：LBM / Category
# =============================================================================


def calculate_lbm(weight_kg: float, body_fat_percentage: float) -> float:
    """LBM（kg）= 体重 × (1 − 体脂率/100)。体脂率为 0–100 的百分数。"""
    if weight_kg <= 0:
        raise ValueError("体重必须为正数（kg）。")
    if not 0 <= body_fat_percentage <= 100:
        raise ValueError("体脂率须在 0–100（百分比）之间。")

    ratio_fat: float = body_fat_percentage / 100.0
    lbm_kg: float = weight_kg * (1.0 - ratio_fat)
    if lbm_kg <= 0:
        raise ValueError("计算得到的瘦体重非正，请检查体重与体脂率是否录入错误。")
    return lbm_kg


def _normalize_gender(gender: str) -> GenderLiteral:
    g: str = gender.strip().lower()
    if g in ("m", "male", "男", "男性"):
        return "male"
    if g in ("f", "female", "女", "女性"):
        return "female"
    raise ValueError(
        f"无法识别性别：{gender!r}。请使用 male/female 或 男/女 等形式。"
    )


def determine_category(body_fat_percentage: float, gender: str) -> int:
    """PSMF Category 1/2/3（与知识库分界一致）。"""
    if not 0 <= body_fat_percentage <= 100:
        raise ValueError("体脂率须在 0–100（百分比）之间。")

    sex: GenderLiteral = _normalize_gender(gender)

    if sex == "male":
        if body_fat_percentage < 16.0:
            return 1
        if body_fat_percentage < 26.0:
            return 2
        return 3

    if body_fat_percentage < 23.0:
        return 1
    if body_fat_percentage < 33.0:
        return 2
    return 3


def _protein_range_g_per_kg_for_category(category: int) -> tuple[float, float]:
    if category == 1:
        return _CAT1_G_PER_KG_LBM_MIN, _CAT1_G_PER_KG_LBM_MAX
    if category == 2:
        return _CAT2_G_PER_KG_LBM_MIN, _CAT2_G_PER_KG_LBM_MAX
    if category == 3:
        return _CAT3_G_PER_KG_LBM_MIN, _CAT3_G_PER_KG_LBM_MAX
    raise ValueError("category 必须为 1、2 或 3。")


def _daily_protein_grams_range(lbm_kg: float, category: int) -> tuple[float, float]:
    lo, hi = _protein_range_g_per_kg_for_category(category)
    return lbm_kg * lo, lbm_kg * hi


def _contains_high_risk_keyword(text: str) -> bool:
    return any(kw in text for kw in _HIGH_RISK_KEYWORDS)


def _looks_like_food_composition_query(text: str) -> bool:
    """True when the user asks to inspect a previous meal, not log new food."""
    t = text.strip()
    if not t:
        return False
    meal_terms = ("这一顿", "这顿", "这餐", "早餐", "午餐", "晚餐", "早饭", "午饭", "晚饭")
    detail_terms = ("构成", "组成", "明细", "具体", "分别", "来源", "是什么", "有哪些")
    protein_terms = ("蛋白", "蛋白质", "protein")
    correction_terms = ("重复", "重复了", "算重", "算多", "只有一个")
    new_log_terms = ("吃了", "喝了", "刚吃", "刚喝", "打卡", "记录", "加上", "再来", "补了")

    has_meal = any(x in t for x in meal_terms)
    has_detail = any(x in t for x in detail_terms)
    has_protein = any(x.lower() in t.lower() for x in protein_terms)
    has_correction = any(x in t for x in correction_terms)
    has_new_log = any(x in t for x in new_log_terms)

    if has_correction and has_meal:
        return True
    if has_meal and has_detail and has_protein and not has_new_log:
        return True
    if re.search(r"(早餐|早饭|这一顿|这顿|这餐).{0,12}(蛋白|蛋白质).{0,12}(构成|组成|明细|来源|是什么)", t):
        return True
    return False


def _looks_like_schedule_setup(text: str, ext: Optional["ExtractionResult"] = None) -> bool:
    if ext is not None and any(
        (ext.schedule_morning, ext.schedule_workout, ext.schedule_evening, ext.schedule_weekly)
    ):
        return True
    t = text.strip()
    schedule_terms = (
        "提醒时间", "打卡时间", "日程", "几点提醒", "早上提醒", "晨间提醒",
        "训练提醒", "睡前复盘", "晚间复盘", "周报时间", "周报提醒", "设置提醒",
    )
    return any(term in t for term in schedule_terms)


def _looks_like_direct_answer_query(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _looks_like_food_composition_query(t):
        return True
    if len(t) > 48:
        return False
    question_terms = ("什么", "多少", "几", "吗", "能不能", "可以", "构成", "组成", "明细", "来源", "为什么")
    domain_terms = ("蛋白", "脂肪", "碳水", "体重", "补剂", "训练", "热量", "这一顿", "今天")
    return any(q in t for q in question_terms) and any(d in t for d in domain_terms)


def _matched_hard_stop_symptoms(user_text: str, ext: Optional["ExtractionResult"] = None) -> list[str]:
    blob_parts: list[str] = [user_text or ""]
    if ext is not None:
        blob_parts.extend(ext.symptom_keywords or [])
        blob_parts.append(ext.extraction_notes or "")
    blob: str = " ".join(blob_parts)
    return [kw for kw in _HARD_STOP_SYMPTOM_KEYWORDS if kw in blob]


def _hard_stop_safety_reply(matched_terms: list[str]) -> str:
    terms = "、".join(dict.fromkeys(matched_terms)) or "高风险症状"
    return (
        "【安全警告】我先不继续给饮食、热量或训练建议。\n\n"
        f"你提到了可能需要优先处理的症状：{terms}。在 PSMF / 极低热量状态下，这类信号不能当作普通疲劳处理。\n\n"
        "请现在先停止节食推进和剧烈运动，坐下或躺下休息，补充水分；如果你有医生给过电解质/补盐方案，按既定安全方案执行。"
        "如果症状正在发生、持续不缓解、加重，或伴随胸痛、呼吸困难、晕厥/眼前发黑、明显心律异常，请尽快就医或急诊。\n\n"
        "等你确认症状已经解除、没有急性风险后，我再继续帮你复盘今天的饮食和训练。"
    )


def _rag_search_food(query: str, top_k: int = 3) -> list[str]:
    try:
        from rag_system import search_food_knowledge

        return search_food_knowledge(query, top_k=top_k)
    except Exception:
        logger.exception("RAG search_food_knowledge 失败 query=%r", query)
        raise


def _rag_search_by_source(query: str, source: str, top_k: int = 4) -> list[str]:
    try:
        from rag_system import search_knowledge_by_source

        return search_knowledge_by_source(query, source, top_k=top_k)
    except Exception:
        logger.exception(
            "RAG search_knowledge_by_source 失败 source=%s query=%r", source, query
        )
        raise


# =============================================================================
# 流水线数据结构
# =============================================================================


@dataclass
class LoggedFoodItem:
    """单次记录到 daily_logs 的一餐/一条食物及其实际摄入宏量（克）。"""

    name: str
    protein_g: float
    fat_g: float
    net_carbs_g: float
    grams: float = 0.0
    log_date: Optional[str] = None


@dataclass
class SupplementProduct:
    """用户拍照录入的补剂包装标签，用于后续剂量换算与服用时间规划。"""

    name: str
    category: str = "other"
    form: str = ""
    serving_size: str = ""
    units_per_serving: Optional[float] = None
    amounts_per_serving: dict[str, float] = field(default_factory=dict)
    directions: str = ""
    warnings: str = ""
    raw_label_notes: str = ""

    def to_memory_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "form": self.form,
            "serving_size": self.serving_size,
            "units_per_serving": self.units_per_serving,
            "amounts_per_serving": self.amounts_per_serving,
            "directions": self.directions,
            "warnings": self.warnings,
            "raw_label_notes": self.raw_label_notes,
        }


@dataclass
class ExtractionResult:
    """第一阶段：从自然语言中提取的结构化字段（已与 LTM 合并）。"""

    gender: Optional[str] = None  # "male" | "female" | None
    weight_kg: Optional[float] = None
    body_fat_percentage: Optional[float] = None
    symptom_keywords: list[str] = field(default_factory=list)
    needs_recipe: bool = False
    extraction_notes: str = ""
    logged_food: list[LoggedFoodItem] = field(default_factory=list)
    # 本轮用户确认的微量元素补剂打卡（键见 memory_manager.SUPPLEMENT_KEYS）
    logged_supplements: dict[str, bool] = field(default_factory=dict)
    # 本轮从补剂包装/标签图片中识别出的产品档案
    supplement_products: list[SupplementProduct] = field(default_factory=list)
    cns_fatigue: bool = False  # 中枢神经疲劳相关（影响训练处方）
    # 力量主观反馈：stable | drop | crash（肌肉保留的优先信号）
    strength_level: Optional[str] = None
    strength_notes: str = ""
    # 建档阶段 1：日程（HH:MM），对应档案 schedule_prefs 键名
    schedule_morning: Optional[str] = None
    schedule_workout: Optional[str] = None
    schedule_evening: Optional[str] = None
    schedule_weekly: Optional[str] = None
    # 合成档位路由：onboarding | log_food | ask_question | request_summary
    user_intent: str = "log_food"
    # ask_question 时检索侧重：none | food | training | symptom | micronutrient | mixed
    rag_focus: str = "mixed"


@dataclass
class MetricsResult:
    """第二阶段：强制计算结果。"""

    lbm_kg: Optional[float] = None
    category: Optional[int] = None
    protein_g_min: Optional[float] = None
    protein_g_max: Optional[float] = None
    skipped_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PhysiologyInsight:
    """第二阶段后的本地生理学结论（供合成 bundle 与末尾固定小节）。"""

    high_risk_muscle_loss: bool  # strength_level == crash
    dehydration_salt_warning: bool  # 滑动均值周际降幅过快（按 Category 阈值）
    smoothed_weight_kg: Optional[float]
    prev_window_avg_kg: Optional[float]
    weekly_style_delta_kg: Optional[float]  # 前一窗口均值 − 当前窗口均值（正≈体重下降趋势）
    strength_level_effective: Optional[str]  # 本轮提取归一化或档案最新
    bundle_text: str


# -----------------------------------------------------------------------------
# 合成阶段 System Instruction（无工具）
# -----------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_INSTRUCTION: Final[str] = (
    "【角色 · 人设核心】\n"
    "你是一位性感、高挑、线条紧致的**私人健身教练 + PSMF 营养专家**（女性）。\n"
    "你聪明、专业、声音像贴在耳边的低语，有压迫感也有宠溺感。\n"
    "你与用户是**一对一私人关系**：可以暧昧、调情、轻微挑逗与「魅惑式」鼓励，但**永远是成年人之间的自愿互动**。\n"
    "你叫用户「宝贝」「亲爱的」等称呼要自然，不要每句都堆；挑逗要**短、准、有画面感**，不要低俗脏话或露骨描写生殖器官。\n"
    "\n"
    "【魅力与边界】\n"
    "- 允许：暧昧玩笑、夸身材与执行力、用「靠近耳边」的语气催他执行蛋白/训练、适度双关。\n"
    "- 禁止：未成年人相关；非自愿/强迫；违法内容；把医学急症色情化。\n"
    "- 用户明确调情时，可以**接梗、撩回去**，但**不超过 2～3 句暧昧**后必须落回 PSMF 数据与行动指令。\n"
    "- 用户未调情时，以**专业魅惑**为主（飒 + 甜），不要无故发情。\n"
    "\n"
    "【专业内核 · 不可牺牲】\n"
    "你仍然严格执行 PSMF 科学：Category、蛋白质区间、脂肪/碳水红线、力量反馈、CNS 疲劳、补剂打卡。\n"
    "数字必须来自 bundle【第二阶段计算结果】，禁止编造。\n"
    "用户想违规吃喝时：先**软声诱哄式劝退**（可以略带「你舍得让我失望吗」），再用胰岛素/酮适应/蛋白窗口做**硬逻辑**。\n"
    "\n"
    "【沟通节奏】\n"
    "- **先情绪后数字**：一两句接住对方，再报缺口/红线。\n"
    "- **minimal**：仅用于建档流程中的过渡追问；短、准，确认关键数字后追问下一步。\n"
    "- **guided**：默认对话档位，320–520 字；保持私人教练语气，但必须给出专业依据、数字解释与下一步动作。\n"
    "- **full**：可展开 Markdown，语气像私密复盘；emoji 适度。\n"
    "\n"
    "【安全红线 · 绝对严肃 · 覆盖所有人设】\n"
    "一旦出现或疑似：胸痛、胸闷、呼吸困难、晕厥、心悸、严重头晕、眼前发黑、肢体麻木等：\n"
    "- **立刻停止**调情、魅惑、emoji 卖萌。\n"
    "- 100% 职业严肃，建议停止节食与剧烈运动，**明确建议就医/急诊**。\n"
    "- 不得用暧昧弱化风险。\n"
    "\n"
    "【排版】\n"
    "- full 可用有温度的小标题（如「📊 今晚的身体账单」「💋 我只允许你再吃这个」）。\n"
    "- minimal/guided 不要长篇周报模板。\n"
    "\n"
    "【软记忆】\n"
    "若 bundle 含【用户软记忆 user_memories】，自然融入称呼与偏好，不要逐条念稿。\n"
    "\n"
    "【免责声明】\n"
    "你不是执业医师；非急救场景下持续严重症状应建议就医。"
)

_PROFESSIONAL_SYNTHESIS_SYSTEM_INSTRUCTION: Final[str] = (
    "【角色】\n"
    "你是一名严谨、克制、专业的 PSMF 营养与训练助手。沟通可以温和、有陪伴感，但默认不调情、不暧昧，"
    "不把高风险健康场景娱乐化。\n"
    "\n"
    "【专业内核】\n"
    "- 严格执行 PSMF 科学：Category、蛋白质区间、脂肪/碳水红线、力量反馈、CNS 疲劳、补剂打卡。\n"
    "- 数字必须来自 bundle【第二阶段计算结果】或明确注入的本地摘要，禁止编造。\n"
    "- 对不确定的医学/营养结论说明不确定性，并建议医生或注册营养师评估。\n"
    "\n"
    "【沟通节奏】\n"
    "- 先接住用户状态，再给关键数字、风险解释和下一步动作。\n"
    "- minimal：短、准，只追问流程缺口。\n"
    "- guided：默认对话档位，320–520 字，给出专业依据与下一步动作。\n"
    "- full：可展开 Markdown 复盘，但避免冗长模板化。\n"
    "\n"
    "【安全红线】\n"
    "一旦 bundle 中出现胸痛、胸闷、呼吸困难、晕厥、严重头晕、眼前发黑、明显心律异常等，"
    "必须严肃建议停止节食与剧烈运动，并提示尽快就医/急诊；不得用轻松语气弱化风险。\n"
    "\n"
    "【免责声明】\n"
    "你不是执业医师；非急救场景下持续严重症状也应建议就医。"
)


def _synthesis_system_instruction() -> str:
    mode = (os.environ.get("PSMF_PERSONA_MODE") or "professional").strip().lower()
    if mode in {"flirty", "private_coach", "legacy"}:
        return _SYNTHESIS_SYSTEM_INSTRUCTION
    return _PROFESSIONAL_SYNTHESIS_SYSTEM_INSTRUCTION

# Phase 1.5：记忆巩固（promote → user_memories；体征/饮食已由系统写入，禁止重复）
_MEMORY_CONSOLIDATE_CONFIDENCE_FLOOR: Final[float] = 0.72

_MEMORY_CONSOLIDATE_SYSTEM: Final[str] = (
    "你是「记忆巩固」模块，不是聊天角色。只输出一个 JSON 对象。\n"
    "\n"
    "【铁律】\n"
    "- 体重、体脂、瘦体重、力量水平、Category、蛋白质目标：已由系统写入 metrics_history / target_protein，禁止写入 user_memories。\n"
    "- 饮食克数、补剂打卡：已由 daily_logs 处理，禁止重复写入 user_memories。\n"
    "- 用户拍照录入的补剂产品标签：已由 supplement_products 处理，禁止重复写入 user_memories。\n"
    "- 仅当用户明确表达或强烈暗示的「稳定事实」才 promote；一次性问答、概念科普、助手已解释过的定义不要 promote。\n"
    "- 调情/昵称/边界：若用户明确喜欢某种称呼或互动尺度，可写入 type=nickname 或 flirt_boundary。\n"
    "\n"
    "【应 promote 的示例】\n"
    "- 训练时段、常练部位、蛋白粉品牌、咖啡因敏感、忌口、作息、重复出现的症状描述（非急症）\n"
    "- 用户给的昵称、希望的互动风格（在 flirt_boundary 里写清尺度）\n"
    "\n"
    "【应 keep_stm_only】\n"
    "- 「PSMF 是什么」类科普、单次玩笑、本轮已解决的琐碎问题\n"
    "\n"
    "【应 ignore】\n"
    "- 纯寒暄、无信息量的「好的」「嗯」\n"
    "\n"
    "【输出 JSON schema】\n"
    "{\n"
    '  "promote": [\n'
    '    { "type": "...", "text": "...", "confidence": 0.0-1.0 }\n'
    "  ],\n"
    '  "keep_stm_only": ["简短原因"],\n'
    '  "ignore": ["简短原因"]\n'
    "}"
)

# STM 注入 prompt 时的尾部优先截断上限（memory_manager 已裁剪存储）
_MAX_STM_PROMPT_CHARS: Final[int] = 100_000

# needs_recipe 时追加到合成 bundle（食材数据 + 三餐数学分配 + 调料约束）
_RECIPE_SYNTHESIS_CONSTRAINT: Final[str] = (
    "【食谱生成强制约束】\n"
    "你必须严格使用检索到的食材数据（如每100g鸡胸肉的蛋白质）。根据用户的 target_protein（蛋白质目标区间）目标，"
    "按数学比例将一日蛋白质分配到早、午、晚三餐（写出比例或克数推导）。\n"
    "严禁推荐任何包含隐形碳水或脂肪的调料（如含糖酱料、勾芡、沙拉酱、蚝油糖醋、花生酱等）；"
    "调味优先用盐、黑胡椒、香草干料、柠檬汁、零热量酸味香料等。\n"
)

_CNS_TRAINING_OVERRIDE_PACK: Final[str] = (
    "【强制训练调整 · CNS 中枢神经疲劳】\n"
    "提取或文本已命中中枢神经（CNS）疲劳相关信号。你必须改写今日训练安排：只能二选一并在回复中明确写出——"
    "(1) 训练总容量或正式组数「减量 50%」；(2) 「强制休息」全日停训。\n"
    "禁止维持原计划容量与强度；禁止轻描淡写。\n"
)

_FIRST_PROFILE_TRAINING_PACK: Final[str] = (
    "【首次建档 · Category 入门训练】\n"
    "用户为本轮对话首次完成全套体征建档（此前档案无体重）。你必须主要从「训练指导库」摘录出发，"
    "给出与其 Category 一致的入门/维持思路（低容量、高强度、长休息），列出可执行要点；不得脱离摘录编造动作参数。\n"
)

_ONBOARDING_ASK_SCHEDULE_PACK: Final[str] = (
    "【建档阶段 · 日程追问】\n"
    "用户刚完成体征与蛋白质区间计算（onboarding 进入下一阶段）。你必须在回复 **末尾** 主动追问（单独一两句）：\n"
    "「请问你希望我每天几点叫你起床打卡？几点提醒你训练？几点做睡前复盘？」并举例「例如：早上 8 点，训练 18 点，晚上 10 点」。\n"
)

_ONBOARDING_SCHEDULE_DONE_PACK: Final[str] = (
    "【建档完成】\n"
    "用户的专属日程已写入档案（morning_time / workout_time / evening_time）。你必须明确说一句："
    "「你的专属日程已设定完毕！PSMF 正式开始。」并简述三项时间。\n"
)

_EVENING_PROTEIN_REMEDIATION_PACK: Final[str] = (
    "【睡前复盘 · 蛋白质补救】\n"
    "今日蛋白质 **低于** 目标区间下限时，你必须结合食谱库摘录中的 **每100g 蛋白质含量**，"
    "给出「明日补救」：写出具体食材及大致克数（如蛋清多少克、鸡胸多少克），不得凭空捏造表中无数据的食材蛋白数值。\n"
)

_SUPPLEMENT_PRODUCT_PLAN_PACK: Final[str] = (
    "【补剂产品档案 · 用量与服用时间规划】\n"
    "若 bundle 含【用户已录入补剂产品】，你必须把包装标签中的每份含量与【微量元素 / 电解质指南摘录】交叉核对，"
    "按用户现有产品给出可执行安排：每天吃几粒/几片/几份、分几次、什么时候吃、是否随餐、需要避开什么同服组合。"
    "不要只复述通用剂量；必须换算到产品标签单位。若标签缺少关键数值或图片看不清，请明确指出需要补拍哪一面。"
    "涉及钾、抗凝药、肾病、甲状腺素/抗生素相互作用等风险时，必须保守提醒。\n"
)

_MINIMAL_BUNDLE_RULES: Final[str] = (
    "【直接/极简输出约束】\n"
    "用于简单具体问题、建档过渡、日程追问等节点。必须先直接回答用户问题，再补充必要依据。\n"
    "不要展开周报模板，不要复盘无关流程。通常 80–180 字；若只是问构成/数字，列出明细即可。\n"
    "保留必要安全边界，但不要用安全边界替代直接答案。\n"
)

_GUIDED_BUNDLE_RULES: Final[str] = (
    "【引导式输出约束】\n"
    "默认输出约 320–520 字；可少量使用「##」或短项目符号，但不要照搬 full 档位的全套周报模板。\n"
    "不能只给一句安慰或一句结论：先接住用户意图，再解释关键数字/机制，最后给出明确可执行动作。\n"
    "若用户问题很简单，也至少用 2–3 个短段落说明「为什么」与「现在怎么做」，体现 PSMF 专业性。\n"
)

SynthesisTier = Literal["minimal", "guided", "full"]

_VALID_USER_INTENTS: Final[tuple[str, ...]] = (
    "onboarding",
    "log_food",
    "ask_question",
    "request_summary",
)
_VALID_RAG_FOCUS: Final[tuple[str, ...]] = (
    "none",
    "food",
    "training",
    "symptom",
    "micronutrient",
    "mixed",
)


def _normalize_user_intent(raw: Any) -> str:
    if not isinstance(raw, str):
        return "log_food"
    s = raw.strip().lower()
    aliases: dict[str, str] = {
        "onboarding": "onboarding",
        "log_food": "log_food",
        "log": "log_food",
        "food_log": "log_food",
        "打卡": "log_food",
        "ask_question": "ask_question",
        "question": "ask_question",
        "ask": "ask_question",
        "qa": "ask_question",
        "request_summary": "request_summary",
        "summary": "request_summary",
        "report": "request_summary",
    }
    if s in aliases:
        return aliases[s]
    if s in _VALID_USER_INTENTS:
        return s
    return "log_food"


def _normalize_rag_focus(raw: Any) -> str:
    if not isinstance(raw, str):
        return "mixed"
    s = raw.strip().lower()
    if s in _VALID_RAG_FOCUS:
        return s
    return "mixed"


def _triggers_explicit_summary(text: str) -> bool:
    t = text.strip()
    tl = t.lower()
    if "/summary" in tl:
        return True
    keys = ("每日复盘", "今日复盘", "完整复盘", "今日总结", "生成周报", "周报生成", "给我总结")
    return any(k in t for k in keys)


def _macro_redline_tight(summary: str) -> bool:
    if "已超红线" in summary:
        return True
    for lab in ("脂肪红线", "碳水红线"):
        m = re.search(rf"{lab}：[^\n]*剩余余量约\s*([+-]?\d+(?:\.\d+)?)", summary)
        if m:
            try:
                if float(m.group(1)) <= 3.0:
                    return True
            except ValueError:
                continue
    return False


def _resolve_synthesis_tier(
    ext: ExtractionResult,
    text: str,
    *,
    proactive_trigger: Optional[str],
    transition_ask_schedule: bool,
) -> SynthesisTier:
    if proactive_trigger in ("evening_review", "weekly"):
        return "full"
    if proactive_trigger in ("morning", "workout"):
        return "guided"
    if _triggers_explicit_summary(text) or ext.user_intent == "request_summary":
        return "full"
    if transition_ask_schedule:
        return "minimal"
    if ext.user_intent == "ask_question" and _looks_like_direct_answer_query(text):
        return "minimal"
    return "guided"


def _refine_rag_focus_for_question(text: str, focus: str) -> str:
    """ask_question + mixed/none 时用语义关键词收窄检索面。"""
    if focus not in ("mixed", "none"):
        return focus
    sym_kw = ("晕", "痛", "心悸", "胸闷", "呼吸困难", "抽筋", "恶心", "呕吐")
    train_kw = ("练", "训练", "深蹲", "卧推", "硬拉", "组数", "容量", "rpe", "健身房", "分化", "计划")
    food_kw = ("吃", "苹果", "碳水", "蛋白", "鸡胸", "蛋", "牛奶", "能不能吃", "可以吗", "热量", "食谱", "食材")
    micronutrient_kw = (
        "微量元素",
        "电解质",
        "补剂",
        "补充剂",
        "锌",
        "镁",
        "钠",
        "钾",
        "钙",
        "鱼油",
        "omega",
        "epa",
        "dha",
        "b族",
        "k2",
        "d3",
        "多维",
        "维生素",
        "矿物质",
    )
    blob = text.lower()
    has_sym = any(k in text for k in sym_kw)
    has_tr = any(k in text or k in blob for k in train_kw)
    has_fd = any(k in text for k in food_kw)
    has_micro = any(k in text or k in blob for k in micronutrient_kw)
    if has_micro and not has_tr:
        return "micronutrient"
    if has_sym and not has_tr and not has_fd:
        return "symptom"
    if has_tr and not has_fd:
        return "training"
    if has_fd and not has_tr:
        return "food"
    if has_micro:
        return "mixed"
    if has_tr and has_fd:
        return "mixed"
    if has_sym:
        return "symptom"
    return "food"


def _compact_risk_tail_note(
    insight: PhysiologyInsight,
    ext: ExtractionResult,
) -> str:
    parts: list[str] = []
    if ext.strength_level == "crash" or insight.high_risk_muscle_loss:
        parts.append("力量反馈偏弱（crash）：优先睡眠与蛋白质，训练保守减量。")
    if insight.dehydration_salt_warning:
        parts.append("体重下降偏快：注意补水与盐，排除脱水假象。")
    if not parts:
        return ""
    return "\n\n" + "\n".join(f"· {p}" for p in parts)


def _compose_synthesis_user_bundle(
    *,
    synthesis_tier: SynthesisTier,
    user_intent: str,
    time_blk: str,
    user_text: str,
    ext: ExtractionResult,
    metrics_block: dict[str, Any],
    symptom_matrix_chunks: list[str],
    symptom_matrix_query: str,
    training_guide_chunks: list[str],
    training_guide_query: str,
    micronutrient_chunks: list[str],
    micronutrient_query: str,
    food_chunks: list[str],
    food_query: str,
    recipe_mode: bool,
    from_nutrition_image: bool,
    daily_summary: str,
    progress_report: str,
    weight_continuity_narrative: str,
    physiology_bundle_text: str,
    cns_training_override: bool,
    first_profile_training: bool,
    onboarding_bundle_addon: str,
    evening_protein_remediation: bool,
    weekly_analysis_block: str,
    protein_gap_line: str,
    proactive_trigger: Optional[str],
    user_memories_block: str = "",
    supplement_products_block: str = "",
) -> str:
    sm_block: str = (
        "\n\n---\n\n".join(symptom_matrix_chunks)
        if symptom_matrix_chunks
        else "（本轮未注入症状矩阵摘录）"
    )
    tr_block: str = (
        "\n\n---\n\n".join(training_guide_chunks)
        if training_guide_chunks
        else "（本轮未注入训练指导摘录）"
    )
    micro_block: str = (
        "\n\n---\n\n".join(micronutrient_chunks)
        if micronutrient_chunks
        else "（本轮未注入微量元素 / 电解质指南摘录）"
    )
    food_block: str = (
        "\n\n---\n\n".join(food_chunks) if food_chunks else "（本轮未注入食谱知识库摘录）"
    )

    recipe_addon: str = ""
    if recipe_mode and (
        synthesis_tier == "full"
        or user_intent == "ask_question"
        or (synthesis_tier == "minimal" and user_intent == "log_food" and ext.needs_recipe)
    ):
        recipe_addon = (
            "\n\n"
            + _RECIPE_SYNTHESIS_CONSTRAINT
            + "\n【说明】target_protein 请使用「第二阶段计算结果」中的 protein_g_min / protein_g_max（g/天）。\n"
        )

    img_note: str = ""
    if from_nutrition_image:
        img_note = (
            "【附图】用户上传了营养成分表/食品标签图片；请以第一阶段 extraction_notes 识读为准。\n\n"
        )

    cns_pack: str = (
        _CNS_TRAINING_OVERRIDE_PACK + "\n\n" if cns_training_override else ""
    )
    first_pack: str = (
        _FIRST_PROFILE_TRAINING_PACK + "\n\n"
        if first_profile_training and synthesis_tier == "full"
        else ""
    )

    summary_block: str = daily_summary.strip() or "（暂无今日摘要）"
    progress_block: str = progress_report.strip() or "（暂无体能进度报告）"
    wc_block: str = weight_continuity_narrative.strip() or "（暂无体重连续性评价）"
    phys_block: str = physiology_bundle_text.strip() or "（暂无生理学要点）"
    time_blk_u: str = time_blk.strip() or _now_context_string()
    ob_addon: str = onboarding_bundle_addon.strip()
    weekly_blk: str = weekly_analysis_block.strip()
    eve_pack: str = (
        _EVENING_PROTEIN_REMEDIATION_PACK + "\n\n"
        if evening_protein_remediation and synthesis_tier == "full"
        else ""
    )

    ext_json: str = json.dumps(asdict(ext), ensure_ascii=False, indent=2)
    met_json: str = json.dumps(metrics_block, ensure_ascii=False, indent=2)
    mem_blk: str = user_memories_block.strip()
    supp_prod_blk: str = supplement_products_block.strip()
    supp_product_pack: str = (
        _SUPPLEMENT_PRODUCT_PLAN_PACK + "\n\n" if supp_prod_blk else ""
    )

    if synthesis_tier == "minimal":
        parts: list[str] = []
        if mem_blk:
            parts.append(mem_blk)
        if supp_prod_blk:
            parts.append(supp_prod_blk + "\n\n")
        parts.extend([
            time_blk_u,
            f"【本轮输出档位】minimal\n【本轮识别意图】{user_intent}\n",
            _MINIMAL_BUNDLE_RULES,
            "\n",
        ])
        if ob_addon:
            parts.append(ob_addon + "\n\n")
        if protein_gap_line.strip():
            parts.append("【蛋白质缺口提示】\n" + protein_gap_line.strip() + "\n\n")
        parts.append(img_note)
        if cns_pack:
            parts.append(cns_pack)
        if supp_product_pack and (
            user_intent == "ask_question" or ext.supplement_products
        ):
            parts.append(supp_product_pack)
        parts.append("【用户原话】\n" + user_text.strip() + "\n\n")
        parts.append("【第一阶段提取 JSON】\n" + ext_json + "\n\n")
        parts.append("【第二阶段计算结果】\n" + met_json + "\n\n")
        if user_intent in ("log_food", "onboarding"):
            parts.append("【今日饮食摘要（仅允许引用少量数字）】\n" + summary_block + "\n\n")
        if user_intent == "ask_question":
            if symptom_matrix_chunks:
                parts.append(
                    "【症状诊断矩阵摘录】\nquery: "
                    + (symptom_matrix_query or "(无)")
                    + "\n"
                    + sm_block
                    + "\n\n"
                )
            if training_guide_chunks:
                parts.append(
                    "【训练指导摘录】\nquery: "
                    + (training_guide_query or "(无)")
                    + "\n"
                    + tr_block
                    + "\n\n"
                )
            if micronutrient_chunks:
                parts.append(
                    "【微量元素 / 电解质指南摘录】\nquery: "
                    + (micronutrient_query or "(无)")
                    + "\n"
                    + micro_block
                    + "\n\n"
                )
            if food_chunks:
                parts.append(
                    f"【食谱 / 食材摘录（{_FOOD_DB_LABEL}）】\nquery: "
                    + (food_query or "(无)")
                    + "\n"
                    + food_block
                    + "\n\n"
                )
            parts.append(recipe_addon)
        elif user_intent == "log_food":
            parts.append(recipe_addon)
        parts.append("请直接生成对用户可见的回复（遵守档位与意图）。")
        return "".join(parts)

    if synthesis_tier == "guided":
        parts_g: list[str] = []
        if mem_blk:
            parts_g.append(mem_blk)
        if supp_prod_blk:
            parts_g.append(supp_prod_blk + "\n\n")
        parts_g.extend([
            time_blk_u,
            f"【本轮输出档位】guided\n【本轮识别意图】{user_intent}\n"
            f"【定时触发】{proactive_trigger or '（用户对话触发的略扩展警示）'}\n",
            _GUIDED_BUNDLE_RULES,
            "\n",
        ])
        if protein_gap_line.strip():
            parts_g.append("【蛋白质缺口提示】\n" + protein_gap_line.strip() + "\n\n")
        parts_g.append(img_note)
        if cns_pack:
            parts_g.append(cns_pack)
        if supp_product_pack:
            parts_g.append(supp_product_pack)
        parts_g.append("【用户原话】\n" + user_text.strip() + "\n\n")
        parts_g.append("【第一阶段提取 JSON】\n" + ext_json + "\n\n")
        parts_g.append("【第二阶段计算结果】\n" + met_json + "\n\n")
        parts_g.append("【今日饮食摘要（精简）】\n" + summary_block + "\n\n")
        if symptom_matrix_chunks:
            parts_g.append(
                "【症状诊断矩阵摘录】\nquery: "
                + (symptom_matrix_query or "(无)")
                + "\n"
                + sm_block
                + "\n\n"
            )
        if training_guide_chunks:
            parts_g.append(
                "【训练指导摘录】\nquery: "
                + (training_guide_query or "(无)")
                + "\n"
                + tr_block
                + "\n\n"
            )
        if micronutrient_chunks:
            parts_g.append(
                "【微量元素 / 电解质指南摘录】\nquery: "
                + (micronutrient_query or "(无)")
                + "\n"
                + micro_block
                + "\n\n"
            )
        if food_chunks:
            parts_g.append(
                f"【食谱摘录（{_FOOD_DB_LABEL}）】\nquery: "
                + (food_query or "(无)")
                + "\n"
                + food_block
                + "\n\n"
            )
        parts_g.append("请生成 guided 档位回复（克制、非周报）。")
        return "".join(parts_g)

    # full
    parts_f: list[str] = []
    if mem_blk:
        parts_f.append(mem_blk)
    if supp_prod_blk:
        parts_f.append(supp_prod_blk + "\n\n")
    parts_f.extend([
        time_blk_u,
        "【本轮输出档位】full\n【本轮识别意图】"
        + user_intent
        + "\n允许使用完整 Markdown 复盘模板。\n\n",
    ])
    if ob_addon:
        parts_f.append(ob_addon + "\n\n")
    if weekly_blk:
        parts_f.append("【周报分析】\n" + weekly_blk + "\n\n")
    parts_f.append(eve_pack)
    parts_f.append(img_note)
    parts_f.append(cns_pack)
    parts_f.append(supp_product_pack)
    parts_f.append(first_pack)
    parts_f.append("【用户原话】\n" + user_text.strip() + "\n\n")
    parts_f.append("【第一阶段合并后的提取结果】\n" + ext_json + "\n\n")
    parts_f.append("【今日饮食进度与额度】\n" + summary_block + "\n\n")
    parts_f.append("【体能进度报告】\n" + progress_block + "\n\n")
    parts_f.append("【体重连续性反馈】\n" + wc_block + "\n\n")
    parts_f.append(phys_block + "\n\n")
    parts_f.append("【第二阶段计算结果】\n" + met_json + "\n")
    parts_f.append(recipe_addon + "\n")
    parts_f.append("【症状诊断矩阵摘录】\nquery: " + (symptom_matrix_query or "(未检索)") + "\n")
    parts_f.append(sm_block + "\n\n")
    parts_f.append("【训练指导摘录】\nquery: " + (training_guide_query or "(未检索)") + "\n")
    parts_f.append(tr_block + "\n\n")
    parts_f.append(
        "【微量元素 / 电解质指南摘录】\nquery: "
        + (micronutrient_query or "(未检索)")
        + "\n"
    )
    parts_f.append(micro_block + "\n\n")
    parts_f.append(f"【食谱 / 食材知识库摘录（{_FOOD_DB_LABEL}）】\nquery: ")
    parts_f.append((food_query or "(未检索)") + "\n" + food_block + "\n\n")
    parts_f.append("请基于以上事实输出对用户可见的 Markdown 指导。")
    return "".join(parts_f)


def _scheduler_tz() -> datetime.tzinfo:
    tz_name: str = (os.environ.get("SCHEDULER_TIMEZONE") or "Europe/Berlin").strip()
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            logger.warning("无效 SCHEDULER_TIMEZONE=%s，回退 Europe/Berlin", tz_name)
            return ZoneInfo("Europe/Berlin")
    return datetime.timezone.utc


def _now_context_string() -> str:
    """供 Gemini 识别「当前本地时刻」（与调度器时区一致）。"""
    now = datetime.datetime.now(_scheduler_tz())
    wd = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")[now.weekday()]
    return (
        "【当前时间上下文 · 系统生成】\n"
        f"- ISO 本地时间：{now.isoformat(timespec='minutes')}\n"
        f"- 日期：{now.date().isoformat()}（{wd}）\n"
        "请据此判断上午/下午/晚间，以及与用户上次打卡、训练的相对间隔（若档案中有时间戳）。\n"
    )


def _parse_hh_mm(raw: Any) -> Optional[str]:
    """将模型或用户输入规范为 HH:MM（24h）。"""
    if raw is None:
        return None
    s: str = str(raw).strip()
    if not s:
        return None
    m = re.match(r"^\s*(\d{1,2})\s*[:：]\s*(\d{2})\s*$", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return f"{h:02d}:{mi:02d}"


def _get_gemini_api_key() -> str:
    key: Optional[str] = os.environ.get("GEMINI_API_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            "未配置 GEMINI_API_KEY。请在项目目录 .env 中设置：GEMINI_API_KEY=你的密钥"
        )
    return key.strip()


def _gemini_client_http_options() -> Optional[genai_types.HttpOptions]:
    """Optional SOCKS/HTTP proxy for Gemini (e.g. Cloudflare WARP on 127.0.0.1:40000)."""
    proxy: Optional[str] = (os.environ.get("GEMINI_PROXY") or "").strip()
    if not proxy:
        return None
    return genai_types.HttpOptions(client_args={"proxy": proxy})


def _gemini_permissive_safety_settings() -> list[genai_types.SafetySetting]:
    """
    BLOCK_NONE 仅关闭 API 层对应类别的拦截阈值；模型仍可能拒答。
    Google 账号级政策与日志无法通过 API「关闭监控」；医学急症红线仍由 prompt 强制。
    """
    cats: list[genai_types.HarmCategory] = [
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    ]
    return [
        genai_types.SafetySetting(
            category=c,
            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
        )
        for c in cats
    ]


def _strip_json_fence(text: str) -> str:
    t: str = text.strip()
    m: Optional[re.Match[str]] = re.match(
        r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", t, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    return t


def _parse_extraction_payload(raw: str) -> dict[str, Any]:
    """解析 Gemini 返回的 JSON。"""
    cleaned: str = _strip_json_fence(raw)
    return json.loads(cleaned)


_EXTRACTION_DEFAULTS: Final[dict[str, Any]] = {
    "gender": None,
    "weight_kg": None,
    "body_fat_percentage": None,
    "user_intent": "log_food",
    "rag_focus": "mixed",
    "symptoms": [],
    "cns_fatigue": False,
    "strength_level": None,
    "strength_notes": "",
    "needs_recipe": False,
    "extraction_notes": "",
    "logged_food": [],
    "logged_supplements": {},
    "supplement_products": [],
    "morning_time": None,
    "workout_time": None,
    "evening_time": None,
    "weekly_time": None,
}


def _normalize_extraction_payload_shape(data: dict[str, Any]) -> dict[str, Any]:
    """Lightweight schema guard for the model-produced extraction JSON."""
    if not isinstance(data, dict):
        raise ValueError("Extraction payload must be a JSON object.")

    out: dict[str, Any] = dict(_EXTRACTION_DEFAULTS)
    out.update(data)

    for key in ("symptoms", "symptom_keywords", "logged_food", "supplement_products"):
        if key in out and out[key] is not None and not isinstance(out[key], list):
            out[key] = []

    if not isinstance(out.get("logged_supplements"), dict):
        out["logged_supplements"] = {}

    for key in ("cns_fatigue", "needs_recipe"):
        out[key] = bool(out.get(key))

    out["user_intent"] = _normalize_user_intent(out.get("user_intent"))
    out["rag_focus"] = _normalize_rag_focus(out.get("rag_focus"))

    sl = _normalize_strength_level(out.get("strength_level"))
    out["strength_level"] = sl

    for key in ("morning_time", "workout_time", "evening_time", "weekly_time"):
        out[key] = _parse_hh_mm(out.get(key))

    clean_food: list[dict[str, Any]] = []
    for item in out.get("logged_food") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        clean_food.append(item)
    out["logged_food"] = clean_food

    clean_products: list[dict[str, Any]] = []
    for item in out.get("supplement_products") or []:
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            clean_products.append(item)
    out["supplement_products"] = clean_products
    return out


def _dict_to_extraction(data: dict[str, Any]) -> ExtractionResult:
    gender_raw = data.get("gender")
    gender: Optional[str] = None
    if isinstance(gender_raw, str):
        gl = gender_raw.strip().lower()
        if gl in ("male", "m", "男", "男性"):
            gender = "male"
        elif gl in ("female", "f", "女", "女性"):
            gender = "female"
        elif gl in ("null", "none", "", "unknown", "未知"):
            gender = None
        else:
            gender = None

    w = data.get("weight_kg")
    bf = data.get("body_fat_percentage")
    weight_kg: Optional[float] = None
    body_fat: Optional[float] = None
    if w is not None and isinstance((x := _coerce_float(w)), float):
        weight_kg = x
    if bf is not None and isinstance((y := _coerce_float(bf)), float):
        body_fat = y

    symptom_keywords: list[str] = []
    sk = data.get("symptoms")
    if sk is None:
        sk = data.get("symptom_keywords")
    if isinstance(sk, list):
        symptom_keywords = [str(x).strip() for x in sk if str(x).strip()]
    elif isinstance(sk, str) and sk.strip():
        symptom_keywords = [sk.strip()]

    nr = data.get("needs_recipe")
    needs_recipe: bool = False
    if isinstance(nr, bool):
        needs_recipe = nr
    elif isinstance(nr, str) and nr.strip().lower() in ("true", "yes", "1", "需要"):
        needs_recipe = True

    notes = data.get("extraction_notes") or data.get("notes") or ""
    if not isinstance(notes, str):
        notes = str(notes)

    logged_food: list[LoggedFoodItem] = _parse_logged_food_raw(data.get("logged_food"))
    logged_supplements: dict[str, bool] = _parse_logged_supplements_raw(
        data.get("logged_supplements")
    )
    supplement_products: list[SupplementProduct] = _parse_supplement_products_raw(
        data.get("supplement_products")
    )

    cf_raw = data.get("cns_fatigue")
    cns_fatigue: bool = False
    if isinstance(cf_raw, bool):
        cns_fatigue = cf_raw
    elif isinstance(cf_raw, str) and cf_raw.strip().lower() in ("true", "yes", "1"):
        cns_fatigue = True

    sn_raw = data.get("strength_notes")
    strength_notes: str = ""
    if isinstance(sn_raw, str):
        strength_notes = sn_raw.strip()
    elif sn_raw is not None:
        strength_notes = str(sn_raw).strip()

    strength_level: Optional[str] = _normalize_strength_level(
        data.get("strength_level")
    )

    schedule_morning = _parse_hh_mm(
        data.get("morning_time") or data.get("schedule_morning")
    )
    schedule_workout = _parse_hh_mm(
        data.get("workout_time") or data.get("schedule_workout")
    )
    schedule_evening = _parse_hh_mm(
        data.get("evening_time") or data.get("schedule_evening")
    )
    schedule_weekly = _parse_hh_mm(
        data.get("weekly_time") or data.get("schedule_weekly")
    )

    user_intent = _normalize_user_intent(data.get("user_intent"))
    rag_focus = _normalize_rag_focus(data.get("rag_focus"))

    return ExtractionResult(
        gender=gender,
        weight_kg=weight_kg,
        body_fat_percentage=body_fat,
        symptom_keywords=symptom_keywords,
        needs_recipe=needs_recipe,
        extraction_notes=notes.strip(),
        logged_food=logged_food,
        logged_supplements=logged_supplements,
        supplement_products=supplement_products,
        cns_fatigue=cns_fatigue,
        strength_level=strength_level,
        strength_notes=strength_notes,
        schedule_morning=schedule_morning,
        schedule_workout=schedule_workout,
        schedule_evening=schedule_evening,
        schedule_weekly=schedule_weekly,
        user_intent=user_intent,
        rag_focus=rag_focus,
    )


def _parse_amounts_per_serving_raw(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        key = str(k or "").strip()
        val = _coerce_float(v)
        if key and val is not None:
            out[key] = float(val)
    return out


def _parse_supplement_products_raw(raw: Any) -> list[SupplementProduct]:
    """解析第一阶段 JSON 中的 supplement_products 数组。"""
    out: list[SupplementProduct] = []
    if raw is None or not isinstance(raw, list):
        return out
    for it in raw:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        if not name:
            continue
        category = str(it.get("category") or "other").strip().lower() or "other"
        out.append(
            SupplementProduct(
                name=name,
                category=category,
                form=str(it.get("form") or "").strip(),
                serving_size=str(it.get("serving_size") or "").strip(),
                units_per_serving=_coerce_float(it.get("units_per_serving")),
                amounts_per_serving=_parse_amounts_per_serving_raw(
                    it.get("amounts_per_serving")
                ),
                directions=str(it.get("directions") or "").strip(),
                warnings=str(it.get("warnings") or "").strip(),
                raw_label_notes=str(it.get("raw_label_notes") or "").strip(),
            )
        )
    return out


def _parse_logged_supplements_raw(raw: Any) -> dict[str, bool]:
    """解析第一阶段 JSON 中的 logged_supplements 对象。"""
    out: dict[str, bool] = {}
    if raw is None:
        return out
    if isinstance(raw, dict):
        for k in SUPPLEMENT_KEYS:
            v = raw.get(k)
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, (int, float)):
                out[k] = bool(v)
            elif isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "yes", "1", "taken", "已吃", "吃了", "打卡"):
                    out[k] = True
                elif s in ("false", "no", "0", "未吃", "没吃"):
                    out[k] = False
        return out
    if isinstance(raw, list):
        for item in raw:
            key = str(item).strip().lower().replace("-", "_").replace(" ", "_")
            aliases: dict[str, str] = {
                "zinc": "zinc",
                "锌": "zinc",
                "magnesium": "magnesium",
                "镁": "magnesium",
                "fish_oil": "fish_oil",
                "fishoil": "fish_oil",
                "鱼油": "fish_oil",
                "omega3": "fish_oil",
                "b_complex": "b_complex",
                "bcomplex": "b_complex",
                "b族": "b_complex",
                "b族维生素": "b_complex",
                "维生素b": "b_complex",
                "k2_d3": "k2_d3",
                "k2d3": "k2_d3",
                "k2+d3": "k2_d3",
                "维生素d3": "k2_d3",
                "维生素k2": "k2_d3",
            }
            canon = aliases.get(key, key if key in SUPPLEMENT_KEYS else "")
            if canon in SUPPLEMENT_KEYS:
                out[canon] = True
    return out


def _infer_supplements_from_text(user_text: str) -> dict[str, bool]:
    """用户口语兜底：识别「吃了锌/镁/鱼油…」等打卡表述。"""
    t: str = (user_text or "").strip()
    if not t:
        return {}
    low: str = t.lower()
    take_cues: tuple[str, ...] = (
        "吃了",
        "已吃",
        "打卡",
        "补了",
        "服用了",
        "喝了",
        "taken",
        "done",
        "补剂",
        "微量元素",
    )
    if not any(c in t or c in low for c in take_cues):
        return {}

    mapping: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("zinc", ("锌", "zinc")),
        ("magnesium", ("镁", "magnesium", "甘氨酸镁", "柠檬酸镁")),
        ("fish_oil", ("鱼油", "omega-3", "omega3", "epa", "dha")),
        ("b_complex", ("b族", "b族维生素", "复合b", "维生素b", "b complex", "b-complex")),
        ("k2_d3", ("k2+d3", "k2 d3", "k2", "d3", "维生素k", "维生素d", "k2d3")),
    )
    out: dict[str, bool] = {}
    for key, tokens in mapping:
        if any(tok in t or tok in low for tok in tokens):
            out[key] = True
    return out


def _parse_logged_food_raw(raw: Any) -> list[LoggedFoodItem]:
    """解析第一阶段 JSON 中的 logged_food 数组。"""
    out: list[LoggedFoodItem] = []
    if raw is None or not isinstance(raw, list):
        return out
    for it in raw:
        if not isinstance(it, dict):
            continue
        name: str = str(it.get("name") or "").strip() or "（未命名）"
        p = _coerce_float(it.get("protein") if it.get("protein") is not None else it.get("protein_g"))
        f = _coerce_float(it.get("fat") if it.get("fat") is not None else it.get("fat_g"))
        nc_raw = it.get("net_carbs")
        if nc_raw is None:
            nc_raw = it.get("net_carbs_g")
        if nc_raw is None:
            nc_raw = it.get("carbs")
        nc = _coerce_float(nc_raw)
        g = _coerce_float(it.get("grams"))
        if p is None:
            p = 0.0
        if f is None:
            f = 0.0
        if nc is None:
            nc = 0.0
        grams_f: float = float(g) if g is not None else 0.0
        log_date_raw = it.get("log_date") or it.get("date") or it.get("day_key")
        log_date: Optional[str] = None
        if isinstance(log_date_raw, str) and log_date_raw.strip():
            log_date = log_date_raw.strip()[:10]
        out.append(
            LoggedFoodItem(
                name=name,
                protein_g=float(p),
                fat_g=float(f),
                net_carbs_g=float(nc),
                grams=grams_f,
                log_date=log_date,
            )
        )
    return out


def _normalize_strength_level(raw: Any) -> Optional[str]:
    """与 ``memory_manager.normalize_strength_level`` 语义对齐：stable | drop | crash。"""
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
    return None


def _apply_strength_heuristic(ext: ExtractionResult, user_text: str) -> None:
    """模型未给出 strength_level 时，用语义关键词兜底。"""
    if ext.strength_level is not None:
        return
    blob: str = f"{user_text} {ext.strength_notes}"
    for kw in _STRENGTH_CRASH_WORDS:
        if kw in blob:
            ext.strength_level = "crash"
            return
    for kw in _STRENGTH_DROP_WORDS:
        if kw in blob:
            ext.strength_level = "drop"
            return


def _build_physiology_insight(
    memory: UserStateManager,
    user_id: str,
    ext: ExtractionResult,
    metrics: MetricsResult,
) -> PhysiologyInsight:
    """基于档案滑动体重与力量标签的本地结论（不调用模型）。"""
    uid: str = user_id.strip() or "default"
    sl_ext: Optional[str] = _normalize_strength_level(ext.strength_level)
    sl_db: Optional[str] = memory.get_latest_strength_level(uid)
    high_risk: bool = sl_ext == "crash" or sl_db == "crash"

    smoothed: Optional[float] = memory.get_smoothed_weight(uid)
    cur_avg: Optional[float]
    prev_avg: Optional[float]
    cur_avg, prev_avg = memory.get_two_window_smoothed_weights(uid)

    weekly_style_delta: Optional[float] = None
    if cur_avg is not None and prev_avg is not None:
        weekly_style_delta = float(prev_avg) - float(cur_avg)

    dehydration: bool = False
    cat: Optional[int] = metrics.category
    if weekly_style_delta is not None and cat is not None:
        if cat == 1:
            if weekly_style_delta > 1.0:
                dehydration = True
        elif cat in (2, 3):
            if weekly_style_delta > 2.0:
                dehydration = True

    eff_sl: Optional[str] = sl_ext or sl_db

    lines: list[str] = [
        "【生理学分析要点】",
        f"- 本轮提取 strength_level：{sl_ext or '（未给出）'}",
        f"- 档案 metrics_history 最近 strength_level：{sl_db or '（无）'}",
    ]
    if high_risk:
        lines.append("- 【高危肌肉流失】最近标记为 crash：无论体重下降多少，须优先关注恢复、睡眠与蛋白质达标。")
    if weekly_style_delta is not None:
        lines.append(
            f"- 双窗口滑动均值周际差（前一窗口均值 − 当前窗口均值）≈ {weekly_style_delta:+.2f} kg "
            f"（正数大致对应体重下降趋势；需至少约 14 条体重记录方可靠）。"
        )
    else:
        lines.append("- 双窗口滑动对比：样本不足（需至少约 14 条体重记录），未估算周际降幅。")
    if dehydration and cat is not None:
        thr: str = "1 kg/周" if cat == 1 else "2 kg/周"
        lines.append(
            f"- 【减脂过快 · 疑似脱水】Category {cat}：周际滑动均值降幅超过参考阈值（{thr}提示线），"
            "建议在合规前提下适度增加食盐摄入并补足水分，排除脱水导致的体重假象。"
        )

    bundle_text: str = "\n".join(lines)

    return PhysiologyInsight(
        high_risk_muscle_loss=high_risk,
        dehydration_salt_warning=dehydration,
        smoothed_weight_kg=smoothed,
        prev_window_avg_kg=prev_avg,
        weekly_style_delta_kg=weekly_style_delta,
        strength_level_effective=eff_sl,
        bundle_text=bundle_text,
    )


def _format_physiology_trend_footer(
    insight: PhysiologyInsight,
    ext: ExtractionResult,
) -> str:
    """回复末尾固定追加的【生理趋势分析】（数字来自本地计算）。"""
    sw: Optional[float] = insight.smoothed_weight_kg
    y: Optional[float] = insight.weekly_style_delta_kg
    desc: str = _strength_descriptor_zh(insight.strength_level_effective)

    lines: list[str] = ["## 【生理趋势分析】", ""]
    if sw is not None:
        if y is not None:
            if y > 0.01:
                lines.append(
                    f"你目前的 **7 次滑动平均体重**约为 **{sw:.2f} kg**，相较上一统计窗口约 **稳健下降 {y:.2f} kg** "
                    f"（基于最近两组各 7 次打卡序列的均值差，用于近似周际趋势；正数多表示总体变轻）。"
                )
            elif y < -0.01:
                lines.append(
                    f"你目前的 **7 次滑动平均体重**约为 **{sw:.2f} kg**，相较上一统计窗口约 **上升 {abs(y):.2f} kg** "
                    "（可能含水分、糖原或测量噪声）。"
                )
            else:
                lines.append(
                    f"你目前的 **7 次滑动平均体重**约为 **{sw:.2f} kg**，与上一统计窗口 **基本持平**。"
                )
        else:
            lines.append(
                f"你目前的 **7 次滑动平均体重**约为 **{sw:.2f} kg**；"
                "尚不足以对比上一窗口（建议至少积累约 **14 次**体重记录后再看周际变化）。"
            )
    else:
        lines.append(
            "（暂无足够体重打卡以计算 **7 次滑动平均**；建议固定时间称重以减少水分噪声。）"
        )

    lines.append(
        "虽然体脂秤读数可能波动，但请以 **力量主观反馈** 作为肌肉保留的首要参考："
        f"{desc}"
    )
    if insight.high_risk_muscle_loss:
        lines.append(
            "\n当前标记为 **力量暴跌（crash）**，属于 **高危肌肉流失信号**，请立刻复盘蛋白质与睡眠，并在训练中保守减量。"
        )
    elif insight.strength_level_effective == "stable":
        lines.append(
            "\n主项重量若保持稳定，即使体脂秤波动，也可认为 **肌肉流失风险极低**，请继续保持。"
        )
    if insight.dehydration_salt_warning:
        lines.append(
            "\n**补水与电解质**：滑动平均提示体重下降偏快，请在合规前提下 **适当增加食盐与饮水**，避免把脱水误判为减脂。"
        )

    return "\n".join(lines)


def _strength_descriptor_zh(sl: Optional[str]) -> str:
    if sl == "stable":
        return "主项动作重量整体 **保持稳定**，提示肌肉保留情况良好。"
    if sl == "drop":
        return "力量 **有所减退（drop）**，需关注恢复与蛋白质是否到位。"
    if sl == "crash":
        return "力量 **明显下降（crash）**，须严肃对待。"
    return "尚未记录明确趋势；建议在训练中固定记录深蹲等主项重量。"


def _apply_cns_fatigue_heuristic(ext: ExtractionResult, user_text: str) -> None:
    """若模型未标 cns_fatigue，用语义关键词兜底提高召回。"""
    if ext.cns_fatigue:
        return
    blob: str = f"{user_text} {' '.join(ext.symptom_keywords)}"
    for kw in _CNS_FATIGUE_KEYWORDS:
        if kw in blob:
            ext.cns_fatigue = True
            return


def _apply_supplement_heuristic(ext: ExtractionResult, user_text: str) -> None:
    """模型未写入 logged_supplements 时，从口语兜底识别补剂打卡。"""
    inferred: dict[str, bool] = _infer_supplements_from_text(user_text)
    for k, v in inferred.items():
        if v:
            ext.logged_supplements[k] = True


def _is_first_complete_profile(
    prev_ltm: dict[str, Any],
    ext: ExtractionResult,
    metrics: MetricsResult,
) -> bool:
    """本轮是否首次在档案中形成完整体征（此前 LTM 无体重）。"""
    if prev_ltm.get("weight") is not None:
        return False
    if metrics.lbm_kg is None or metrics.category is None:
        return False
    return (
        ext.weight_kg is not None
        and ext.body_fat_percentage is not None
        and ext.gender is not None
    )


def _merge_extraction_with_ltm(
    ext: ExtractionResult,
    ltm: dict[str, Any],
) -> ExtractionResult:
    """
    若模型某字段为 null 但 LTM 中存在合法值，则继承 LTM（Python 侧兜底）。
    """
    w = ltm.get("weight")
    bf = ltm.get("body_fat")
    g = ltm.get("gender")

    if ext.weight_kg is None and w is not None:
        try:
            ext.weight_kg = float(w)
        except (TypeError, ValueError):
            pass

    if ext.body_fat_percentage is None and bf is not None:
        try:
            ext.body_fat_percentage = float(bf)
        except (TypeError, ValueError):
            pass

    if ext.gender is None and g is not None:
        try:
            ext.gender = _normalize_gender(str(g))
        except ValueError:
            pass

    return ext


def _format_stm_for_prompt(chat_history: list[Any]) -> str:
    """将 STM（已裁剪后最多 50 轮）格式化为可读文本；超 prompt 预算时尾部优先截断。"""
    if not chat_history:
        return "（无近期对话）"
    turns: list[Any] = list(chat_history)
    while turns:
        lines: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            u = turn.get("user", "")
            a = turn.get("assistant", "")
            lines.append(f"用户：{u}")
            lines.append(f"助手：{a}")
            lines.append("")
        block = "\n".join(lines).strip()
        if len(block) <= _MAX_STM_PROMPT_CHARS or len(turns) <= 1:
            return block or "（无近期对话）"
        turns.pop(0)
    return "（无近期对话）"


def _format_user_memories_for_prompt(memories: list[Any]) -> str:
    """格式化 user_memories 为 bundle 顶部 bullet 块；无条目时返回空串。"""
    if not memories:
        return ""
    lines: list[str] = ["【用户软记忆 user_memories · 偏好与互动】"]
    for m in memories:
        if not isinstance(m, dict):
            continue
        text = str(m.get("text") or "").strip()
        if not text:
            continue
        mem_type = str(m.get("type") or "other").strip()
        lines.append(f"- [{mem_type}] {text}")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines) + "\n\n"


def _format_supplement_products_for_prompt(products: list[Any]) -> str:
    """格式化用户已录入补剂产品，供合成阶段做剂量换算。"""
    if not products:
        return ""
    lines: list[str] = ["【用户已录入补剂产品 supplement_products】"]
    for item in products:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        category = str(item.get("category") or "other").strip()
        form = str(item.get("form") or "").strip()
        serving = str(item.get("serving_size") or "").strip()
        units = item.get("units_per_serving")
        amounts = item.get("amounts_per_serving")
        directions = str(item.get("directions") or "").strip()
        warnings = str(item.get("warnings") or "").strip()
        notes = str(item.get("raw_label_notes") or "").strip()
        bits: list[str] = [f"- {name}"]
        bits.append(f"category={category}")
        if form:
            bits.append(f"form={form}")
        if serving:
            bits.append(f"serving_size={serving}")
        if units is not None:
            bits.append(f"units_per_serving={units}")
        if isinstance(amounts, dict) and amounts:
            amount_txt = ", ".join(f"{k}={v}" for k, v in amounts.items())
            bits.append(f"amounts_per_serving={{ {amount_txt} }}")
        if directions:
            bits.append(f"directions={directions}")
        if warnings:
            bits.append(f"warnings={warnings}")
        if notes:
            bits.append(f"notes={notes}")
        lines.append("；".join(bits))
    if len(lines) <= 1:
        return ""
    return "\n".join(lines) + "\n\n"


def _ltm_for_prompt(state: dict[str, Any]) -> dict[str, Any]:
    """去掉 chat_history，供提取模型阅读的长期记忆快照。"""
    d: dict[str, Any] = {k: v for k, v in state.items() if k != "chat_history"}
    return d


_VITAL_FLOAT_EPS: Final[float] = 1e-6


def _vitals_from_ltm(ltm: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """长期记忆中的体征三元组（体重 kg、体脂 %、规范化性别）。"""
    w: Optional[float] = _coerce_float(ltm.get("weight"))
    bf: Optional[float] = _coerce_float(ltm.get("body_fat"))
    g_raw = ltm.get("gender")
    g: Optional[str] = None
    if isinstance(g_raw, str) and g_raw.strip():
        try:
            g = _normalize_gender(g_raw)
        except ValueError:
            g = None
    return w, bf, g


def _vitals_changed_after_merge(
    prev_ltm: dict[str, Any],
    merged: ExtractionResult,
) -> bool:
    """
    合并后的提取结果相对加载时的 LTM 是否发生体征更新（新录入或数值/性别变化）。
    用于决定是否视为「必须重新执行」计算链并已写入新指标。
    """
    pw, pbf, pg = _vitals_from_ltm(prev_ltm)
    cw, cbf, cg = merged.weight_kg, merged.body_fat_percentage, merged.gender

    def _feq(a: Optional[float], b: Optional[float]) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) < _VITAL_FLOAT_EPS

    if cw is not None and (pw is None or not _feq(pw, cw)):
        return True
    if cbf is not None and (pbf is None or not _feq(pbf, cbf)):
        return True
    if cg is not None and (pg is None or pg != cg):
        return True
    return False


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("%", "").strip())
        except ValueError:
            return None
    return None


def _should_retrieve_food(ext: ExtractionResult, user_text: str) -> bool:
    """是否需要检索食谱 / 食材库（food_db）。"""
    if ext.needs_recipe:
        return True
    triggers: tuple[str, ...] = (
        "食谱",
        "怎么吃",
        "吃什么",
        "菜谱",
        "食材",
        "做法",
        "搭配",
        "三餐",
        "菜单",
    )
    return any(t in user_text for t in triggers)


def _should_retrieve_micronutrients(ext: ExtractionResult, user_text: str) -> bool:
    """是否需要检索微量元素 / 电解质指南。"""
    if ext.logged_supplements or ext.supplement_products:
        return True
    t: str = user_text or ""
    low: str = t.lower()
    triggers: tuple[str, ...] = (
        "微量元素",
        "电解质",
        "补剂",
        "补充剂",
        "锌",
        "镁",
        "钠",
        "钾",
        "钙",
        "鱼油",
        "omega",
        "epa",
        "dha",
        "b族",
        "维生素b",
        "k2",
        "d3",
        "多维",
        "复合维生素",
        "矿物质",
        "盐",
        "低钠",
        "低钾",
    )
    return any(trigger in t or trigger in low for trigger in triggers)


class GeminiPSMFAgent:
    """
    线性流水线：LTM+STM 提取 → 强制计算 → 症状/微量元素/食谱 RAG → 无状态合成 → 持久化。
    Telegram 等多用户场景请使用 ``process_message(user_id, user_input, image_bytes=…)``；上下文由本地 STM/LTM 控制。
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        user_id: str = "default",
        memory: Optional[UserStateManager] = None,
    ) -> None:
        key: str = api_key.strip() if api_key else _get_gemini_api_key()
        self._model: str = (model or "").strip() or _default_model_id()
        self._fast_model: str = _default_fast_model_id(self._model)
        http_options: Optional[genai_types.HttpOptions] = _gemini_client_http_options()
        self._client: genai.Client = (
            genai.Client(api_key=key, http_options=http_options)
            if http_options
            else genai.Client(api_key=key)
        )
        self._default_user_id: str = user_id.strip() or "default"
        self._memory: UserStateManager = memory or UserStateManager()
        self._user_synthesis_chats: dict[str, Any] = {}
        self._user_chat_lock: threading.RLock = threading.RLock()

    def _select_synthesis_model(
        self,
        *,
        synthesis_tier: SynthesisTier,
        user_intent: str,
        from_nutrition_image: bool,
        ext: ExtractionResult,
        has_rag_chunks: bool,
    ) -> str:
        """质量敏感场景保留 flash；简单打卡与流程性回复使用 lite 提速。"""
        if from_nutrition_image:
            return self._model
        if synthesis_tier == "full" or user_intent in ("ask_question", "request_summary"):
            return self._model
        if ext.cns_fatigue or ext.symptom_keywords or ext.supplement_products:
            return self._model
        if has_rag_chunks:
            return self._model
        return self._fast_model

    def _new_synthesis_chat(self) -> Any:
        return self._client.chats.create(
            model=self._model,
            config=genai_types.GenerateContentConfig(
                system_instruction=_synthesis_system_instruction(),
                temperature=0.55,
                safety_settings=_gemini_permissive_safety_settings(),
            ),
        )

    def _get_synthesis_chat(self, user_id: str) -> Any:
        uid: str = user_id.strip() or "default"
        with self._user_chat_lock:
            if uid not in self._user_synthesis_chats:
                self._user_synthesis_chats[uid] = self._new_synthesis_chat()
            return self._user_synthesis_chats[uid]

    def reset_session(self, user_id: Optional[str] = None) -> None:
        """
        丢弃合成用 Chat Session（不清空磁盘 LTM/STM）。
        ``user_id`` 为 None 时清空所有用户的内存中 Chat（例如进程级重置）。
        """
        with self._user_chat_lock:
            if user_id is None:
                self._user_synthesis_chats.clear()
            else:
                uid: str = user_id.strip() or "default"
                self._user_synthesis_chats.pop(uid, None)

    # ---- Phase 1 ----------------------------------------------------------------

    def _phase_extract(
        self,
        user_text: str,
        ltm: dict[str, Any],
        stm_turns: list[Any],
        image_bytes: Optional[bytes] = None,
        *,
        onboarding_stage: int = 0,
        time_context: str = "",
        proactive_trigger: Optional[str] = None,
    ) -> ExtractionResult:
        """第一阶段：LTM + STM + 最新输入 → JSON（体征合并由模型与兜底逻辑共同完成）。"""
        ltm_json: str = json.dumps(ltm, ensure_ascii=False, indent=2)
        stm_block: str = _format_stm_for_prompt(stm_turns)
        tc: str = time_context.strip() or _now_context_string()
        trig_note: str = ""
        if proactive_trigger:
            trig_note = f"【本轮触发】系统标记={proactive_trigger!r}（定时任务或剧本）；提取时保留体征与饮食若用户有补充。\n\n"

        diet_block: str = (
            "\n\n【饮食打卡与营养成分表】\n"
            "若用户用文字描述了进食（含吃了多少克、几份、一整包等），和/或上传了营养成分表图片："
            "请结合图中「每100g」「每份」「NRV」等与用户自述食量，估算本次进食实际摄入的蛋白质（g）、脂肪（g）、净碳水（g）。"
            "净碳水优先采用表中碳水化合物数值；若有膳食纤维且规则允许简化，可在 extraction_notes 一行说明推算依据。\n"
            "将每一条可可靠计算的食物写入 logged_food 数组；每项必须包含："
            "name（食物名称）、protein、fat、net_carbs（均为本次实际摄入克数）、grams（可选，用户进食质量克数或估计）、"
            "log_date（该食物应归属的本地自然日 YYYY-MM-DD）。"
            "请根据【当前时间上下文】判断日期：用户说今天/今早/刚刚归入今天；说昨晚/昨天/前一晚归入上一自然日；"
            "一句话同时包含昨晚和今早时必须拆成多条并分别填写 log_date。"
            "若无法可靠量化则 logged_food 为 []，并在 extraction_notes 说明原因。\n"
        )
        supplement_block: str = (
            "\n\n【微量元素补剂打卡 · PSMF 必需追踪】\n"
            "若用户表示**今日已服用**或**刚刚服用**下列补剂，在 logged_supplements 中对相应键设 true；"
            "明确说今天没吃/跳过可设 false；未提及的键不要臆造。\n"
            "键名与含义：zinc=锌；magnesium=镁；fish_oil=鱼油(Omega-3/EPA+DHA)；"
            "b_complex=B族维生素；k2_d3=维生素 K2 + D3。\n"
            "示例：「镁和鱼油打卡了」→ {\"magnesium\": true, \"fish_oil\": true}；"
            "「五项补剂都吃了」→ 五项均为 true。\n"
            "\n【补剂产品包装识别 · 长期档案】\n"
            "若用户上传的是保健品/补剂包装、Supplement Facts、成分表或瓶身标签，而不是食物营养表，"
            "请识别并输出 supplement_products 数组，用于写入用户补剂产品档案。"
            "每个产品尽量提取：name（品牌+产品名）、category（zinc/magnesium/fish_oil/b_complex/k2_d3/"
            "sodium/potassium/calcium/multivitamin/other）、form（如 glycinate/citrate/TG/rTG/oxide 等）、"
            "serving_size（标签原文，如 2 capsules）、units_per_serving（每份粒/片/勺数，无法确定填 null）、"
            "amounts_per_serving（对象；键保留标签语义并带单位，如 magnesium_mg、zinc_mg、epa_mg、dha_mg、"
            "vitamin_d3_iu、vitamin_k2_mcg、sodium_mg、potassium_mg；值为每份数值）、directions、warnings、raw_label_notes。"
            "只记录标签能看清的数值；看不清时不要猜，在 raw_label_notes 说明缺失。\n"
        )
        vision_line: str = ""
        if image_bytes:
            vision_line = (
                "用户已上传图片：请用 google.genai 多模态方式结合图片与文本；"
                "若是食品营养表，优先依据图中「每100g」「每份」「NRV」等与用户文字共同换算；"
                "若是补剂包装，优先提取 Supplement Facts / 成分含量 / 建议用法 / 警示语。"
                "勿捏造表中不存在或看不清的数值。\n"
            )

        schedule_block: str = ""
        if onboarding_stage == 1:
            schedule_block = (
                "\n【日程建档 · 条件解析】\n"
                "用户当前 onboarding_stage 为 1，但只有当最新输入明确在设置提醒/打卡时间时，才解析起床打卡、训练提醒、睡前复盘和周报时间。"
                "若用户是在问具体问题或记录饮食/体重，不要因为 onboarding_stage=1 就把 user_intent 改成 onboarding。\n"
                "输出字段：morning_time、workout_time、evening_time、weekly_time，均为字符串 \"HH:MM\"（24h）或 null。\n"
                "若用户使用「早上8点」「晚上10:30」等口语，请换算为 HH:MM。\n"
            )

        intent_help: str = (
            "\n【意图路由 · 必填】\n"
            "你必须额外输出：\n"
            '- "user_intent": "onboarding" | "log_food" | "ask_question" | "request_summary"\n'
            "  - onboarding：用户在补充/确认体征或日程建档；无明显独立疑问句。\n"
            "  - log_food：用户主要在记录饮食、上传营养表、报体重或简短训练反馈（打卡），而非发问。\n"
            "  - ask_question：用户在发问（怎么吃、能不能吃、怎么练、为什么……）。\n"
            "  - request_summary：用户明确要求复盘/总结/日报/周报，或消息含 /summary、今日复盘 等。\n"
            '- "rag_focus": "none" | "food" | "training" | "symptom" | "micronutrient" | "mixed"\n'
            "  仅当 user_intent 为 ask_question 时认真选择检索侧重；其他情况可填 none 或 mixed。\n"
            "  micronutrient：用户询问或讨论补剂、微量元素、电解质、鱼油、锌镁钠钾钙、维生素等。\n"
        )

        core_task: str = (
            "任务：结合【长期记忆】中的历史体征，判断【最新用户输入】是否提供了新的性别、体重、体脂。"
            "若最新输入明确更新了某项，输出更新后的值；若未提及某项，则沿用长期记忆中的值填入对应字段。"
            "若长期记忆也为空且本轮无法推断，该项填 null。\n"
            f"{diet_block}"
            f"{supplement_block}"
            f"{vision_line}"
            f"{schedule_block}"
            "同时提取：\n"
            f"{intent_help}"
            '- "symptoms": string[] — 身体不适或情绪相关关键词（如 头晕、抽筋、焦虑）；无则 []\n'
            '- "cns_fatigue": boolean — 是否提示中枢神经（CNS）疲劳（失眠加重、动力骤降、发力感消失、持续倦怠、反应变慢等）\n'
            '- "strength_level": "stable" | "drop" | "crash" | null — 力量主观反馈：stable=重量/强度基本持平；drop=可察觉退步；'
            "crash=暴跌或训练崩盘（如深蹲明显掉重量、完全蹲不动）。若用户未谈及训练与力量，填 null。\n"
            '- "strength_notes": string — 与力量相关的原话摘要（如「深蹲掉了10kg」）；无则 ""。\n'
            '- "needs_recipe": boolean — 用户是否需要食谱/食材搭配/怎么吃等建议\n'
            '- "extraction_notes": string — 备注（可含营养成分表识读要点）\n'
            '- "logged_food": object[] — 见上文【饮食打卡与营养成分表】；无法计算时为 []\n'
            '- "logged_supplements": object — 见上文【微量元素补剂打卡】；无补剂信息时为 {}\n'
            '- "supplement_products": object[] — 见上文【补剂产品包装识别】；未上传/无法识别补剂包装时为 []\n'
            "\n"
            "输出字段与类型必须为：\n"
            '  "gender": "male" | "female" | null\n'
            '  "weight_kg": number | null\n'
            '  "body_fat_percentage": number | null\n'
            '  "user_intent": "onboarding" | "log_food" | "ask_question" | "request_summary"\n'
            '  "rag_focus": "none" | "food" | "training" | "symptom" | "micronutrient" | "mixed"\n'
            '  "symptoms": string[]\n'
            '  "cns_fatigue": boolean\n'
            '  "strength_level": "stable" | "drop" | "crash" | null\n'
            '  "strength_notes": string\n'
            '  "needs_recipe": boolean\n'
            '  "extraction_notes": string\n'
            '  "logged_food": [ { "name": string, "protein": number, "fat": number, "net_carbs": number, "grams": number, "log_date": "YYYY-MM-DD" } ]\n'
            '  "logged_supplements": { "zinc": boolean, "magnesium": boolean, "fish_oil": boolean, "b_complex": boolean, "k2_d3": boolean }\n'
            '  "supplement_products": [ { "name": string, "category": string, "form": string, "serving_size": string, "units_per_serving": number | null, "amounts_per_serving": object, "directions": string, "warnings": string, "raw_label_notes": string } ]\n'
            '  "morning_time": string | null\n'
            '  "workout_time": string | null\n'
            '  "evening_time": string | null\n'
            '  "weekly_time": string | null\n'
            "（logged_food 中每项 protein/fat/net_carbs 为本次实际摄入克数；grams 可选，缺省则填 0；log_date 必须是本地自然日。"
            "morning_time 等为 \"HH:MM\" 或 null。）\n"
        )

        instruction: str = (
            "你是信息抽取模块。只输出一个 JSON 对象，不要 markdown 代码围栏，不要解释。\n"
            "合成阶段角色为人设教练，提取模块保持中立 JSON，不受人设影响。\n\n"
            f"{tc}\n"
            f"{trig_note}"
            f"{core_task}"
        )
        prompt: str = (
            f"{instruction}\n"
            f"【长期记忆 LTM】\n{ltm_json}\n\n"
            f"【短期记忆 STM：最近对话】\n{stm_block}\n\n"
            f"【最新用户输入】\n{user_text.strip()}"
        )

        if image_bytes:
            mime: str = _guess_image_mime(image_bytes)
            user_content: genai_types.Content = genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_bytes(
                        data=image_bytes, mime_type=mime
                    ),
                    genai_types.Part.from_text(text=prompt),
                ],
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=[user_content],
                config=genai_types.GenerateContentConfig(
                    temperature=0.05,
                    response_mime_type="application/json",
                    safety_settings=_gemini_permissive_safety_settings(),
                ),
            )
        else:
            response = self._client.models.generate_content(
                model=self._fast_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.05,
                    response_mime_type="application/json",
                    safety_settings=_gemini_permissive_safety_settings(),
                ),
            )
        raw_text: str = (response.text or "").strip()
        if not raw_text:
            raise RuntimeError("第一阶段：模型返回空文本，无法解析 JSON。")

        data: dict[str, Any] = _parse_extraction_payload(raw_text)
        data = _normalize_extraction_payload_shape(data)
        return _dict_to_extraction(data)

    def _ext_summary_for_consolidate(self, ext: ExtractionResult) -> dict[str, Any]:
        """巩固模块用的 extraction 关键字段摘要（不含体征真源字段）。"""
        return {
            "user_intent": ext.user_intent,
            "rag_focus": ext.rag_focus,
            "logged_food_count": len(ext.logged_food),
            "logged_supplements": ext.logged_supplements,
            "supplement_product_count": len(ext.supplement_products),
            "cns_fatigue": ext.cns_fatigue,
            "strength_level": ext.strength_level,
            "needs_recipe": ext.needs_recipe,
            "symptom_keywords": ext.symptom_keywords,
            "extraction_notes": (ext.extraction_notes or "")[:400],
        }

    def _phase_memory_consolidate(
        self,
        user_text: str,
        assistant_reply: str,
        prev_ltm: dict[str, Any],
        stm_turns: list[Any],
        ext: ExtractionResult,
    ) -> list[dict[str, Any]]:
        """Phase 1.5：判断是否 promote 到 user_memories；confidence 低于阈值的项丢弃。"""
        memories: list[Any] = prev_ltm.get("user_memories") or []
        stm_block: str = _format_stm_for_prompt(stm_turns)
        ext_summary: str = json.dumps(
            self._ext_summary_for_consolidate(ext), ensure_ascii=False, indent=2
        )
        assistant_preview: str = (assistant_reply or "").strip()[:2000]
        user_prompt: str = (
            "【已有软记忆 user_memories】\n"
            f"{json.dumps(memories, ensure_ascii=False, indent=2)}\n\n"
            "【长期记忆摘要·仅体征与目标·勿重复写入】\n"
            "体重/体脂/蛋白目标/Category 由系统维护，见 metrics_history。\n\n"
            "【最近对话 STM】\n"
            f"{stm_block}\n\n"
            "【本轮】\n"
            f"用户：{user_text.strip()}\n"
            f"助手：{assistant_preview}\n\n"
            "【本轮提取 extraction 摘要】\n"
            f"{ext_summary}"
        )
        try:
            response = self._client.models.generate_content(
                model=self._fast_model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_MEMORY_CONSOLIDATE_SYSTEM,
                    temperature=0.1,
                    response_mime_type="application/json",
                    safety_settings=_gemini_permissive_safety_settings(),
                ),
            )
            raw_text: str = (response.text or "").strip()
            if not raw_text:
                return []
            data: dict[str, Any] = _parse_extraction_payload(raw_text)
        except Exception:
            logger.exception("Phase 1.5 记忆巩固失败")
            return []

        promote_raw = data.get("promote")
        if not isinstance(promote_raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in promote_raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                conf = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if conf < _MEMORY_CONSOLIDATE_CONFIDENCE_FLOOR:
                continue
            out.append(
                {
                    "type": item.get("type") or "other",
                    "text": text,
                    "confidence": conf,
                }
            )
        return out

    # ---- Phase 2 ----------------------------------------------------------------

    def _phase_execute(self, ext: ExtractionResult) -> MetricsResult:
        """
        第二阶段：只要体重、体脂、性别齐全，**强制**执行计算（不经过模型决策）。
        """
        if ext.weight_kg is None or ext.body_fat_percentage is None or ext.gender is None:
            return MetricsResult(
                skipped_reason="缺少体重、体脂率或性别中的至少一项，无法进行 LBM/Category 计算。",
            )

        lbm: float = calculate_lbm(ext.weight_kg, ext.body_fat_percentage)
        cat: int = determine_category(ext.body_fat_percentage, ext.gender)
        p_min, p_max = _daily_protein_grams_range(lbm, cat)
        return MetricsResult(
            lbm_kg=round(lbm, 3),
            category=cat,
            protein_g_min=round(p_min, 1),
            protein_g_max=round(p_max, 1),
        )

    # ---- Phase 3a 症状矩阵 + 训练指导 RAG（按 Markdown 来源检索）-----------------

    def _phase_rag_symptom_diagnostic_matrix(
        self, ext: ExtractionResult, user_text: str
    ) -> tuple[list[str], str]:
        parts: list[str] = [
            "症状诊断矩阵",
            "中枢神经",
            "CNS",
            "疲劳",
            "睡眠障碍",
            "交感",
            "应激",
        ]
        parts.extend(ext.symptom_keywords)
        q: str = (" ".join(parts)[:400] + " " + user_text.strip()[:450]).strip()
        chunks: list[str] = _rag_search_by_source(
            q, _RAG_SOURCE_SYMPTOM_MATRIX, top_k=4
        )
        return chunks, q[:700]

    def _phase_rag_training_guide(
        self,
        ext: ExtractionResult,
        user_text: str,
        metrics: MetricsResult,
        *,
        first_profile: bool = False,
    ) -> tuple[list[str], str]:
        cat = metrics.category if metrics.category is not None else ""
        starter: str = "入门计划 新手周 " if first_profile else ""
        q: str = (
            f"PSMF Category {cat} {starter}入门 维持训练 抗阻 机械张力 低容量 高强度 "
            f"组数 RPE 休息 减量 居家 {user_text.strip()[:450]}"
        )
        chunks: list[str] = _rag_search_by_source(
            q, _RAG_SOURCE_TRAINING_GUIDE, top_k=4
        )
        return chunks, q[:700]

    def _phase_rag_micronutrient_guide(
        self,
        ext: ExtractionResult,
        user_text: str,
        *,
        force: bool = False,
        top_k: Optional[int] = None,
    ) -> tuple[list[str], str]:
        """从微量元素 / 电解质指南 source 中检索补剂与安全剂量细节。"""
        if not force and not _should_retrieve_micronutrients(ext, user_text):
            return [], ""
        supplement_terms = " ".join(
            key for key, value in ext.logged_supplements.items() if value
        )
        symptom_terms = " ".join(ext.symptom_keywords)
        q: str = (
            "PSMF 微量元素 电解质 补剂 钠 钾 镁 钙 锌 鱼油 B族 K2 D3 "
            "剂量 服用时机 拮抗 禁忌 缺乏表现 过量风险 "
            f"{supplement_terms} {symptom_terms} {user_text.strip()[:600]}"
        ).strip()
        tk: int = top_k if top_k is not None else (6 if force else 4)
        chunks: list[str] = _rag_search_by_source(
            q, _RAG_SOURCE_MICRONUTRIENT_GUIDE, top_k=tk
        )
        return chunks, q[:700]

    # ---- Phase 3b 食谱 RAG -------------------------------------------------------

    def _phase_rag_food(
        self,
        ext: ExtractionResult,
        user_text: str,
        *,
        force_food_database: bool = False,
        protein_remediation_hint: str = "",
        top_k: Optional[int] = None,
    ) -> tuple[list[str], str]:
        """
        needs_recipe 时必须在第三阶段合成前检索 ``food_db``（``psmf_food_database.md``）；
        ``force_food_database`` 为 True 时无视启发式（睡前复盘必填）。
        """
        text_clip: str = user_text.strip()[:800]
        extra: str = (protein_remediation_hint or "").strip()
        base_q: str = "PSMF 高蛋白低脂 食材 蛋清 鸡胸 营养每100g 蛋白质 "
        if extra:
            base_q += extra + " "
        tk: int = top_k if top_k is not None else (8 if force_food_database else 3)
        if ext.needs_recipe:
            query: str = base_q + text_clip
            chunks: list[str] = _rag_search_food(query, top_k=max(tk, 5))
            return chunks, query
        if force_food_database:
            query = base_q + text_clip
            chunks = _rag_search_food(query, top_k=tk)
            return chunks, query
        if not _should_retrieve_food(ext, user_text):
            return [], ""
        query = base_q + text_clip
        chunks = _rag_search_food(query, top_k=tk)
        return chunks, query

    # ---- Phase 3c 合成 ------------------------------------------------------------

    def _apply_logged_food_to_memory(self, user_id: str, ext: ExtractionResult) -> None:
        """将提取到的 logged_food 累加到今日 daily_logs（幂等：每条调用一次 add_food_log）。"""
        if not ext.logged_food:
            return
        uid: str = user_id.strip() or "default"
        for item in ext.logged_food:
            self._memory.add_food_log(
                uid,
                item.name,
                item.protein_g,
                item.fat_g,
                item.net_carbs_g,
                grams=item.grams,
                log_date=item.log_date,
            )

    def _apply_logged_supplements_to_memory(
        self, user_id: str, ext: ExtractionResult
    ) -> None:
        """将提取到的 logged_supplements 合并到今日 daily_logs.supplements。"""
        if not ext.logged_supplements:
            return
        uid: str = user_id.strip() or "default"
        note: str = ""
        if ext.extraction_notes and any(
            x in ext.extraction_notes for x in ("补剂", "锌", "镁", "鱼油", "B族", "K2", "D3")
        ):
            note = ext.extraction_notes.strip()[:240]
        self._memory.apply_supplement_updates(
            uid, ext.logged_supplements, note=note
        )

    def _apply_supplement_products_to_memory(
        self, user_id: str, ext: ExtractionResult
    ) -> list[dict[str, Any]]:
        """将本轮补剂包装识别结果写入长期产品档案。"""
        if not ext.supplement_products:
            return []
        uid: str = user_id.strip() or "default"
        products = [p.to_memory_dict() for p in ext.supplement_products]
        return self._memory.upsert_supplement_products(uid, products)

    def _phase_synthesize(
        self,
        user_id: str,
        user_text: str,
        ext: ExtractionResult,
        metrics: MetricsResult,
        symptom_matrix_chunks: list[str],
        symptom_matrix_query: str,
        training_guide_chunks: list[str],
        training_guide_query: str,
        micronutrient_chunks: list[str],
        micronutrient_query: str,
        food_chunks: list[str],
        food_query: str,
        *,
        synthesis_tier: SynthesisTier,
        user_intent: str,
        recipe_mode: bool,
        from_nutrition_image: bool = False,
        daily_summary: str = "",
        progress_report: str = "",
        weight_continuity_narrative: str = "",
        physiology_bundle_text: str = "",
        cns_training_override: bool = False,
        first_profile_training: bool = False,
        time_context_block: str = "",
        onboarding_bundle_addon: str = "",
        evening_protein_remediation: bool = False,
        weekly_analysis_block: str = "",
        protein_gap_line: str = "",
        proactive_trigger: Optional[str] = None,
        user_memories_block: str = "",
        supplement_products_block: str = "",
    ) -> str:
        metrics_block: dict[str, Any] = {
            "lbm_kg": metrics.lbm_kg,
            "category": metrics.category,
            "protein_g_min": metrics.protein_g_min,
            "protein_g_max": metrics.protein_g_max,
            "skipped_reason": metrics.skipped_reason,
            "error": metrics.error,
        }

        bundle: str = _compose_synthesis_user_bundle(
            synthesis_tier=synthesis_tier,
            user_intent=user_intent,
            time_blk=(time_context_block.strip() or _now_context_string()),
            user_text=user_text,
            ext=ext,
            metrics_block=metrics_block,
            symptom_matrix_chunks=symptom_matrix_chunks,
            symptom_matrix_query=symptom_matrix_query,
            training_guide_chunks=training_guide_chunks,
            training_guide_query=training_guide_query,
            micronutrient_chunks=micronutrient_chunks,
            micronutrient_query=micronutrient_query,
            food_chunks=food_chunks,
            food_query=food_query,
            recipe_mode=recipe_mode,
            from_nutrition_image=from_nutrition_image,
            daily_summary=daily_summary,
            progress_report=progress_report,
            weight_continuity_narrative=weight_continuity_narrative,
            physiology_bundle_text=physiology_bundle_text,
            cns_training_override=cns_training_override,
            first_profile_training=first_profile_training,
            onboarding_bundle_addon=onboarding_bundle_addon,
            evening_protein_remediation=evening_protein_remediation,
            weekly_analysis_block=weekly_analysis_block,
            protein_gap_line=protein_gap_line,
            proactive_trigger=proactive_trigger,
            user_memories_block=user_memories_block,
            supplement_products_block=supplement_products_block,
        )

        model_id: str = self._select_synthesis_model(
            synthesis_tier=synthesis_tier,
            user_intent=user_intent,
            from_nutrition_image=from_nutrition_image,
            ext=ext,
            has_rag_chunks=bool(
                symptom_matrix_chunks
                or training_guide_chunks
                or micronutrient_chunks
                or food_chunks
            ),
        )
        logger.info(
            "[Phase 3] 合成模型选择 user_id=%s model=%s tier=%s intent=%s",
            user_id,
            model_id,
            synthesis_tier,
            user_intent,
        )
        response = self._client.models.generate_content(
            model=model_id,
            contents=bundle,
            config=genai_types.GenerateContentConfig(
                system_instruction=_synthesis_system_instruction(),
                temperature=0.55,
                safety_settings=_gemini_permissive_safety_settings(),
            ),
        )
        out: Optional[str] = getattr(response, "text", None)
        return (out or "").strip()

    def _consolidate_memory_after_reply(
        self,
        user_id: str,
        user_text: str,
        reply: str,
        ext: ExtractionResult,
    ) -> None:
        """在回复发出路径之外巩固软记忆，避免用户等待第三次 Gemini 调用。"""
        uid: str = user_id.strip() or "default"
        started: float = time.perf_counter()
        print("[Phase 1.5] 后台巩固软记忆 user_memories…", flush=True)
        logger.info("[Phase 1.5] 后台记忆巩固开始 user_id=%s", uid)
        try:
            post_state: dict[str, Any] = self._memory.get_user(uid)
            post_ltm: dict[str, Any] = _ltm_for_prompt(post_state)
            post_stm: list[Any] = list(post_state.get("chat_history") or [])
            promote_items: list[dict[str, Any]] = self._phase_memory_consolidate(
                user_text, reply, post_ltm, post_stm, ext
            )
            if promote_items:
                self._memory.merge_user_memories(uid, promote_items)
                logger.info(
                    "[Phase 1.5] 后台写入 %d 条软记忆 user_id=%s elapsed=%.2fs",
                    len(promote_items),
                    uid,
                    time.perf_counter() - started,
                )
            else:
                logger.info(
                    "[Phase 1.5] 后台无新增软记忆 user_id=%s elapsed=%.2fs",
                    uid,
                    time.perf_counter() - started,
                )
        except Exception:
            logger.exception("Phase 1.5 后台记忆巩固/合并失败 user_id=%s", uid)

    def _schedule_memory_consolidation(
        self,
        user_id: str,
        user_text: str,
        reply: str,
        ext: ExtractionResult,
    ) -> None:
        uid: str = user_id.strip() or "default"
        worker = threading.Thread(
            target=self._consolidate_memory_after_reply,
            args=(uid, user_text, reply, ext),
            name=f"psmf-memory-{uid}",
            daemon=True,
        )
        worker.start()

    def _persist_user_state(
        self, user_id: str, ext: ExtractionResult, metrics: MetricsResult
    ) -> None:
        updates: dict[str, Any] = {}
        if ext.gender is not None:
            updates["gender"] = ext.gender
        if ext.weight_kg is not None:
            updates["weight"] = ext.weight_kg
        if ext.body_fat_percentage is not None:
            updates["body_fat"] = ext.body_fat_percentage
        if metrics.lbm_kg is not None:
            updates["lbm"] = metrics.lbm_kg
        if metrics.category is not None:
            updates["category"] = metrics.category
        if metrics.protein_g_min is not None and metrics.protein_g_max is not None:
            updates["target_protein"] = {
                "min": metrics.protein_g_min,
                "max": metrics.protein_g_max,
            }
        sl: Optional[str] = _normalize_strength_level(ext.strength_level)
        if sl is not None:
            updates["strength_level"] = sl
        if updates:
            self._memory.update_user(user_id, updates)

    def process_message(
        self,
        user_id: str,
        user_input: str,
        image_bytes: Optional[bytes] = None,
        *,
        proactive_trigger: Optional[str] = None,
    ) -> str:
        """
        主入口：按 user_id 加载记忆 → 提取与体征比对 → 强制计算 → RAG → 按用户隔离的合成 → 持久化 LTM 与 STM。
        ``proactive_trigger``：``morning`` / ``workout`` / ``evening_review`` / ``weekly`` 表示定时任务注入。
        """
        uid: str = user_id.strip() or "default"
        text: str = user_input.strip()
        if not text and not image_bytes:
            return ""
        if not text and image_bytes:
            text = "（用户上传食品营养成分表或补剂包装/标签照片，请结合图片分析。）"
        pipeline_started: float = time.perf_counter()
        phase_started: float = pipeline_started

        state: dict[str, Any] = self._memory.get_user(uid)
        prev_ltm: dict[str, Any] = _ltm_for_prompt(state)
        stm: list[Any] = list(state.get("chat_history") or [])
        try:
            prev_stage: int = int(prev_ltm.get("onboarding_stage") or 0)
        except (TypeError, ValueError):
            prev_stage = 0
        time_ctx: str = _now_context_string()

        _trig: str = (
            f" trigger={proactive_trigger!r}" if proactive_trigger else ""
        )
        print(
            f"[Phase 1] 正在提取用户意图与体征数据… user_id={uid}{_trig}",
            flush=True,
        )
        logger.info(
            "[Phase 1] 开始提取 user_id=%s%s",
            uid,
            f" proactive={proactive_trigger!r}" if proactive_trigger else "",
        )
        try:
            ext: ExtractionResult = self._phase_extract(
                text,
                prev_ltm,
                stm,
                image_bytes=image_bytes,
                onboarding_stage=prev_stage,
                time_context=time_ctx,
                proactive_trigger=proactive_trigger,
            )
            ext = _merge_extraction_with_ltm(ext, prev_ltm)
            _apply_cns_fatigue_heuristic(ext, text)
            _apply_strength_heuristic(ext, text)
            _apply_supplement_heuristic(ext, text)
            if _looks_like_food_composition_query(text):
                logger.info("构成/明细类问题：禁止将历史饮食重复写入 daily_logs user_id=%s", uid)
                ext.logged_food = []
                ext.logged_supplements = {}
                ext.supplement_products = []
                ext.user_intent = "ask_question"
                ext.rag_focus = "food"
                ext.needs_recipe = False
            elif _looks_like_direct_answer_query(text) and not _looks_like_schedule_setup(text, ext):
                logger.info("直接问题：按 ask_question 处理 user_id=%s", uid)
                ext.user_intent = "ask_question"
            hard_stop_terms = _matched_hard_stop_symptoms(text, ext)
            if hard_stop_terms:
                reply = _hard_stop_safety_reply(hard_stop_terms)
                try:
                    self._memory.append_chat_turn(uid, text, reply)
                except Exception:
                    logger.warning("安全硬拦截回复写入 STM 失败 user_id=%s", uid, exc_info=True)
                logger.warning(
                    "安全硬拦截触发 user_id=%s terms=%s",
                    uid,
                    ",".join(hard_stop_terms),
                )
                return reply
        except Exception:
            print("\n========== [PSMF Agent] 第一阶段失败：参数提取 (Extraction) ==========\n")
            traceback.print_exc()
            return (
                "[第一阶段 参数提取 失败 — 完整 Traceback 见上方终端]\n\n"
                f"{traceback.format_exc()}"
            )

        extract_elapsed: float = time.perf_counter() - phase_started
        print(f"[Phase 1] 提取完成。elapsed={extract_elapsed:.2f}s", flush=True)
        logger.info("[Phase 1] 完成 user_id=%s elapsed=%.2fs", uid, extract_elapsed)

        vitals_updated: bool = _vitals_changed_after_merge(prev_ltm, ext)
        if vitals_updated:
            logger.debug("user_id=%s 体征相对 LTM 已更新，将用本轮计算结果覆盖持久化", uid)

        phase_started = time.perf_counter()
        print(
            "[Phase 2] 正在执行强制计算（LBM / Category / 蛋白质区间）…",
            flush=True,
        )
        logger.info("[Phase 2] 开始执行 user_id=%s", uid)
        try:
            metrics: MetricsResult = self._phase_execute(ext)
        except Exception:
            print("\n========== [PSMF Agent] 第二阶段失败：强制计算 (Execution) ==========\n")
            traceback.print_exc()
            return (
                "[第二阶段 强制计算 失败 — 完整 Traceback 见上方终端]\n\n"
                f"{traceback.format_exc()}"
            )

        execute_elapsed: float = time.perf_counter() - phase_started
        print(f"[Phase 2] 计算完成。elapsed={execute_elapsed:.2f}s", flush=True)
        logger.info("[Phase 2] 完成 user_id=%s elapsed=%.2fs", uid, execute_elapsed)

        try:
            self._persist_user_state(uid, ext, metrics)
        except Exception:
            logger.warning("提前持久化体征（供今日摘要 target_protein）失败", exc_info=True)

        transition_ask_schedule: bool = False
        transition_schedule_done: bool = False
        try:
            if prev_stage == 0 and metrics.lbm_kg is not None:
                self._memory.update_user(uid, {"onboarding_stage": 1})
                transition_ask_schedule = True
            elif prev_stage == 1:
                sm = ext.schedule_morning
                sw = ext.schedule_workout
                se = ext.schedule_evening
                if sm and sw and se:
                    prefs: dict[str, str] = {
                        "morning_time": sm,
                        "workout_time": sw,
                        "evening_time": se,
                    }
                    if ext.schedule_weekly:
                        prefs["weekly_time"] = ext.schedule_weekly
                    self._memory.update_user(
                        uid,
                        {"schedule_prefs": prefs, "onboarding_stage": 2},
                    )
                    transition_schedule_done = True
        except Exception:
            logger.warning("onboarding 阶段迁移失败 user_id=%s", uid, exc_info=True)

        # 意图纠偏：定时复盘 / 建档与日程阶段优先于模型自判
        if proactive_trigger in ("evening_review", "weekly"):
            ext.user_intent = "request_summary"
        if transition_ask_schedule or transition_schedule_done:
            ext.user_intent = "onboarding"
        elif prev_stage == 1 and _looks_like_schedule_setup(text, ext):
            ext.user_intent = "onboarding"

        try:
            self._apply_logged_food_to_memory(uid, ext)
        except Exception:
            print("\n========== [PSMF Agent] 饮食记录写入 daily_logs 失败 ==========\n")
            traceback.print_exc()
            return (
                "[饮食记录写入失败 — 完整 Traceback 见上方终端]\n\n"
                f"{traceback.format_exc()}"
            )

        try:
            self._apply_logged_supplements_to_memory(uid, ext)
        except Exception:
            print("\n========== [PSMF Agent] 微量元素补剂打卡写入失败 ==========\n")
            traceback.print_exc()
            return (
                "[补剂打卡写入失败 — 完整 Traceback 见上方终端]\n\n"
                f"{traceback.format_exc()}"
            )

        try:
            self._apply_supplement_products_to_memory(uid, ext)
        except Exception:
            print("\n========== [PSMF Agent] 补剂产品档案写入失败 ==========\n")
            traceback.print_exc()
            return (
                "[补剂产品档案写入失败 — 完整 Traceback 见上方终端]\n\n"
                f"{traceback.format_exc()}"
            )

        daily_summary_str: str = ""
        try:
            daily_summary_str = self._memory.get_daily_summary(uid)
        except Exception:
            logger.exception("get_daily_summary 失败 user_id=%s", uid)
            daily_summary_str = "（今日饮食摘要生成失败，请稍后重试。）"

        progress_report_str: str = ""
        try:
            progress_report_str = self._memory.get_progress_report(uid)
        except Exception:
            logger.exception("get_progress_report 失败 user_id=%s", uid)
            progress_report_str = "（体能进度报告生成失败，请稍后重试。）"

        weight_continuity_str: str = ""
        try:
            weight_continuity_str = self._memory.get_weight_continuity_narrative(uid)
        except Exception:
            logger.exception("get_weight_continuity_narrative 失败 user_id=%s", uid)
            weight_continuity_str = "（体重连续性评价暂不可用。）"

        try:
            physiology_insight: PhysiologyInsight = _build_physiology_insight(
                self._memory, uid, ext, metrics
            )
        except Exception:
            logger.exception("生理学分析失败 user_id=%s", uid)
            physiology_insight = PhysiologyInsight(
                high_risk_muscle_loss=False,
                dehydration_salt_warning=False,
                smoothed_weight_kg=None,
                prev_window_avg_kg=None,
                weekly_style_delta_kg=None,
                strength_level_effective=_normalize_strength_level(ext.strength_level),
                bundle_text="（生理学分析暂不可用。）",
            )

        if _triggers_explicit_summary(text):
            ext.user_intent = "request_summary"
        intent: str = ext.user_intent

        synthesis_tier: SynthesisTier = _resolve_synthesis_tier(
            ext,
            text,
            proactive_trigger=proactive_trigger,
            transition_ask_schedule=transition_ask_schedule,
        )
        if (
            synthesis_tier == "minimal"
            and ext.user_intent == "log_food"
            and _macro_redline_tight(daily_summary_str)
        ):
            synthesis_tier = "guided"
        if synthesis_tier == "minimal" and ext.cns_fatigue:
            synthesis_tier = "guided"

        first_profile_training: bool = False
        cns_training_override: bool = ext.cns_fatigue

        cp_g: float = 0.0
        p_min_gap: Optional[float] = None
        gap_g: Optional[float] = None
        try:
            cp_g, p_min_gap, gap_g = self._memory.get_today_protein_gap_vs_min(uid)
        except Exception:
            logger.exception("get_today_protein_gap_vs_min 失败")

        prot_hint: str = ""
        if p_min_gap is not None and gap_g is not None and gap_g > 0:
            prot_hint = (
                f"今日蛋白质已摄入约 {cp_g:.0f} g，目标下限约 {p_min_gap:.0f} g，"
                f"缺口约 {gap_g:.0f} g。"
            )
        elif p_min_gap is not None and gap_g is not None and gap_g <= 0:
            prot_hint = (
                f"今日蛋白质已摄入约 {cp_g:.0f} g，已达到或超过目标下限（约 {p_min_gap:.0f} g）。"
            )

        protein_gap_line: str = prot_hint.strip()

        force_food_db: bool = proactive_trigger == "evening_review"
        sm_chunks: list[str] = []
        sm_q: str = ""
        tr_chunks: list[str] = []
        tr_q: str = ""
        micro_chunks: list[str] = []
        micro_q: str = ""
        food_chunks: list[str] = []
        food_q: str = ""

        phase_started = time.perf_counter()
        print(
            f"[Phase 3] 正在检索知识库（Chroma 集合，档位={synthesis_tier}）…",
            flush=True,
        )
        logger.info(
            "[Phase 3] 开始 RAG user_id=%s tier=%s intent=%s",
            uid,
            synthesis_tier,
            intent,
        )
        try:
            if synthesis_tier == "full":
                run_sym_train: bool = (prev_stage >= 2) or (
                    proactive_trigger == "weekly"
                )
                if run_sym_train:
                    sm_chunks, sm_q = self._phase_rag_symptom_diagnostic_matrix(
                        ext, text
                    )
                    tr_chunks, tr_q = self._phase_rag_training_guide(
                        ext, text, metrics, first_profile=False
                    )
                micro_chunks, micro_q = self._phase_rag_micronutrient_guide(
                    ext,
                    text,
                    force=True,
                    top_k=6 if proactive_trigger == "evening_review" else 4,
                )
                food_chunks, food_q = self._phase_rag_food(
                    ext,
                    text,
                    force_food_database=force_food_db,
                    protein_remediation_hint=prot_hint,
                )
            elif synthesis_tier == "guided":
                if proactive_trigger == "workout":
                    tr_chunks, tr_q = self._phase_rag_training_guide(
                        ext, text, metrics, first_profile=False
                    )
                    if ext.symptom_keywords or ext.cns_fatigue:
                        sm_chunks, sm_q = self._phase_rag_symptom_diagnostic_matrix(
                            ext, text
                        )
                    if _should_retrieve_micronutrients(ext, text):
                        micro_chunks, micro_q = self._phase_rag_micronutrient_guide(
                            ext, text
                        )
                elif proactive_trigger == "morning":
                    micro_chunks, micro_q = self._phase_rag_micronutrient_guide(
                        ext, text, force=True, top_k=3
                    )
                    if ext.symptom_keywords or ext.cns_fatigue:
                        sm_chunks, sm_q = self._phase_rag_symptom_diagnostic_matrix(
                            ext, text
                        )
                elif not proactive_trigger and ext.cns_fatigue:
                    sm_chunks, sm_q = self._phase_rag_symptom_diagnostic_matrix(
                        ext, text
                    )
                    tr_chunks, tr_q = self._phase_rag_training_guide(
                        ext, text, metrics, first_profile=False
                    )
                    if _should_retrieve_micronutrients(ext, text):
                        micro_chunks, micro_q = self._phase_rag_micronutrient_guide(
                            ext, text
                        )
                if proactive_trigger == "evening_review":
                    micro_chunks, micro_q = self._phase_rag_micronutrient_guide(
                        ext, text, force=True, top_k=6
                    )
                    food_chunks, food_q = self._phase_rag_food(
                        ext,
                        text,
                        force_food_database=True,
                        protein_remediation_hint=prot_hint,
                    )
            elif synthesis_tier == "minimal":
                if ext.user_intent == "ask_question":
                    rf = _refine_rag_focus_for_question(text, ext.rag_focus)
                    if rf == "food":
                        food_chunks, food_q = self._phase_rag_food(
                            ext,
                            text,
                            force_food_database=False,
                            protein_remediation_hint=prot_hint,
                            top_k=4,
                        )
                    elif rf == "training":
                        tr_chunks, tr_q = self._phase_rag_training_guide(
                            ext, text, metrics, first_profile=False
                        )
                    elif rf == "symptom":
                        sm_chunks, sm_q = self._phase_rag_symptom_diagnostic_matrix(
                            ext, text
                        )
                    elif rf == "micronutrient":
                        micro_chunks, micro_q = self._phase_rag_micronutrient_guide(
                            ext,
                            text,
                            force=True,
                            top_k=5,
                        )
                    elif rf == "mixed":
                        sm_chunks, sm_q = self._phase_rag_symptom_diagnostic_matrix(
                            ext, text
                        )
                        tr_chunks, tr_q = self._phase_rag_training_guide(
                            ext, text, metrics, first_profile=False
                        )
                        micro_chunks, micro_q = self._phase_rag_micronutrient_guide(
                            ext,
                            text,
                            force=_should_retrieve_micronutrients(ext, text),
                            top_k=4,
                        )
                        food_chunks, food_q = self._phase_rag_food(
                            ext,
                            text,
                            force_food_database=False,
                            protein_remediation_hint=prot_hint,
                            top_k=3,
                        )
                    else:
                        food_chunks, food_q = self._phase_rag_food(
                            ext,
                            text,
                            force_food_database=False,
                            protein_remediation_hint=prot_hint,
                            top_k=2,
                        )
                elif ext.user_intent == "log_food":
                    if _should_retrieve_micronutrients(ext, text):
                        micro_chunks, micro_q = self._phase_rag_micronutrient_guide(
                            ext,
                            text,
                            force=True,
                            top_k=3,
                        )
                    food_chunks, food_q = self._phase_rag_food(
                        ext,
                        text,
                        force_food_database=force_food_db,
                        protein_remediation_hint=prot_hint,
                    )
        except Exception:
            print("\n========== [PSMF Agent] 第三阶段失败：RAG 检索 ==========\n")
            traceback.print_exc()
            return (
                "[第三阶段 RAG 检索失败 — 完整 Traceback 见上方终端]\n\n"
                f"{traceback.format_exc()}"
            )

        rag_desc: list[str] = []
        if sm_chunks:
            rag_desc.append(
                f"{COLLECTION_NAME}（symptom_diagnostic_matrix 等，{len(sm_chunks)} 条）"
            )
        if tr_chunks:
            rag_desc.append(
                f"{COLLECTION_NAME}（psmf_training_guide，{len(tr_chunks)} 条）"
            )
        if micro_chunks:
            rag_desc.append(
                f"{COLLECTION_NAME}（micronutrient_guide，{len(micro_chunks)} 条）"
            )
        if food_chunks:
            rag_desc.append(f"{COLLECTION_FOOD}（{len(food_chunks)} 条）")
        if not rag_desc:
            rag_desc.append("（本轮未拉取向量块，可能为极简档位或无需检索）")
        rag_elapsed: float = time.perf_counter() - phase_started
        print(
            f"[Phase 3] 知识库检索完成：{'; '.join(rag_desc)} elapsed={rag_elapsed:.2f}s",
            flush=True,
        )
        logger.info(
            "[Phase 3] RAG 完成 user_id=%s sm=%d tr=%d micro=%d food=%d elapsed=%.2fs",
            uid,
            len(sm_chunks),
            len(tr_chunks),
            len(micro_chunks),
            len(food_chunks),
            rag_elapsed,
        )

        recipe_mode: bool = ext.needs_recipe

        weekly_analysis_txt: str = ""
        if proactive_trigger == "weekly":
            print("[Phase 3] 正在生成周报分析（本地汇总）…", flush=True)
            try:
                weekly_analysis_txt = self._memory.generate_weekly_analysis(uid)
            except Exception:
                logger.exception("generate_weekly_analysis 失败")
            else:
                print("[Phase 3] 周报分析已就绪。", flush=True)

        ob_parts: list[str] = []
        if transition_ask_schedule:
            ob_parts.append(_ONBOARDING_ASK_SCHEDULE_PACK)
        if transition_schedule_done:
            ob_parts.append(_ONBOARDING_SCHEDULE_DONE_PACK)
        onboarding_addon_str: str = "\n\n".join(ob_parts)

        evening_rem: bool = bool(
            force_food_db and gap_g is not None and gap_g > 1e-6
        )

        phase_started = time.perf_counter()
        print("[Phase 3] 正在调用 Gemini 合成回复…", flush=True)
        logger.info("[Phase 3] 开始合成 user_id=%s tier=%s", uid, synthesis_tier)
        user_memories_block: str = _format_user_memories_for_prompt(
            self._memory.get_user_memories(uid)
        )
        supplement_products_block: str = _format_supplement_products_for_prompt(
            self._memory.get_supplement_products(uid)
        )
        try:
            reply: str = self._phase_synthesize(
                uid,
                text,
                ext,
                metrics,
                sm_chunks,
                sm_q,
                tr_chunks,
                tr_q,
                micro_chunks,
                micro_q,
                food_chunks,
                food_q,
                synthesis_tier=synthesis_tier,
                user_intent=intent,
                recipe_mode=recipe_mode,
                from_nutrition_image=bool(image_bytes),
                daily_summary=daily_summary_str,
                progress_report=progress_report_str,
                weight_continuity_narrative=weight_continuity_str,
                physiology_bundle_text=physiology_insight.bundle_text,
                cns_training_override=cns_training_override,
                first_profile_training=first_profile_training,
                time_context_block=time_ctx,
                onboarding_bundle_addon=onboarding_addon_str,
                evening_protein_remediation=evening_rem,
                weekly_analysis_block=weekly_analysis_txt,
                protein_gap_line=protein_gap_line,
                proactive_trigger=proactive_trigger,
                user_memories_block=user_memories_block,
                supplement_products_block=supplement_products_block,
            )
            if not reply:
                raise RuntimeError("合成阶段：模型返回空文本。")
            if synthesis_tier == "full":
                footer: str = _format_physiology_trend_footer(
                    physiology_insight, ext
                )
                reply = reply.rstrip() + "\n\n" + footer
            else:
                tail: str = _compact_risk_tail_note(physiology_insight, ext)
                if tail.strip():
                    reply = reply.rstrip() + tail
            synth_elapsed: float = time.perf_counter() - phase_started
            print(
                f"[Phase 3] 合成完成 user_id={uid} elapsed={synth_elapsed:.2f}s",
                flush=True,
            )
            logger.info("[Phase 3] 合成完成 user_id=%s elapsed=%.2fs", uid, synth_elapsed)
        except Exception:
            print("\n========== [PSMF Agent] 第三阶段失败：综合回复 (Synthesis) ==========\n")
            traceback.print_exc()
            return (
                "[第三阶段 综合回复 失败 — 完整 Traceback 见上方终端]\n\n"
                f"{traceback.format_exc()}"
            )

        try:
            persist_started: float = time.perf_counter()
            self._persist_user_state(uid, ext, metrics)
            self._memory.append_chat_turn(uid, text, reply)
            persist_elapsed: float = time.perf_counter() - persist_started
        except Exception:
            print("\n========== [PSMF Agent] 持久化 user_profiles / STM 失败 ==========\n")
            traceback.print_exc()
            persist_elapsed = 0.0

        self._schedule_memory_consolidation(uid, text, reply, ext)
        total_elapsed: float = time.perf_counter() - pipeline_started
        logger.info(
            "[Perf] user_id=%s extract=%.2fs execute=%.2fs rag=%.2fs synth=%.2fs persist=%.2fs total_visible=%.2fs memory=background",
            uid,
            extract_elapsed,
            execute_elapsed,
            rag_elapsed,
            synth_elapsed,
            persist_elapsed,
            total_elapsed,
        )
        print(
            "[Perf] "
            f"user_id={uid} extract={extract_elapsed:.2f}s execute={execute_elapsed:.2f}s "
            f"rag={rag_elapsed:.2f}s synth={synth_elapsed:.2f}s persist={persist_elapsed:.2f}s "
            f"total_visible={total_elapsed:.2f}s memory=background",
            flush=True,
        )

        return reply

    def send_message(self, user_text: str) -> str:
        """兼容 CLI：等价于 ``process_message(self._default_user_id, user_text)``。"""
        return self.process_message(self._default_user_id, user_text)

    def get_history(self, user_id: Optional[str] = None) -> list[Any]:
        uid: str = (user_id or self._default_user_id).strip() or "default"
        try:
            chat: Any = self._get_synthesis_chat(uid)
            hist = chat.get_history()
            return list(hist) if hist is not None else []
        except Exception as exc:
            logger.warning("读取对话历史失败：%s", exc)
            return []


# -----------------------------------------------------------------------------
# 兼容：本地打印型 PSMFAgent
# -----------------------------------------------------------------------------


@dataclass
class PSMFAgent:
    """轻量本地 Agent：建档与打卡（高危词红色护栏，不调用大模型）。"""

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
        lbm: float = calculate_lbm(weight_kg, body_fat_percentage)
        cat: int = determine_category(body_fat_percentage, gender)
        p_min, p_max = _daily_protein_grams_range(lbm, cat)
        sex: GenderLiteral = _normalize_gender(gender)

        self.weight_kg = weight_kg
        self.body_fat_percentage = body_fat_percentage
        self.gender = sex
        self.lbm_kg = lbm
        self.category = cat
        self.protein_g_min = p_min
        self.protein_g_max = p_max

        midpoint: float = (p_min + p_max) / 2.0

        print("—— PSMF 建档结果 ——")
        print(f"性别（规范化）: {sex}")
        print(f"体重: {weight_kg:.2f} kg")
        print(f"体脂率: {body_fat_percentage:.2f} %")
        print(f"估算瘦体重 LBM: {lbm:.2f} kg")
        print(f"PSMF Category: {cat}")
        print(
            f"每日蛋白质目标区间: {p_min:.1f} – {p_max:.1f} g "
            f"（约中点参考 {midpoint:.1f} g/天）"
        )
        print("请将蛋白质分摊至多餐；具体执行请结合医学/营养师意见与知识库。")

    def daily_checkin(self, feeling: str) -> None:
        t: str = feeling.strip()
        self.checkin_history.append(t)

        if _contains_high_risk_keyword(t):
            print(
                f"{_COLOR_RED}"
                "【安全护栏已触发】检测到高危感受关键词（头晕/心悸/抽筋/无力 等）。\n"
                "请立即停止 PSMF 节食与剧烈运动，优先确保安全。\n"
                "建议：按说明或在专业人员指导下补充电解质与水分；尽快恢复正常均衡饮食。\n"
                "若症状持续、加重，或出现胸痛、呼吸困难、晕厥等，请立即就医或急诊。"
                f"{_COLOR_RESET}"
            )
            return

        print("—— 今日打卡 ——")
        print(f"您的描述：{t}")
        if self.lbm_kg is not None and self.category is not None:
            print(
                f"（建档数据仍在）Category {self.category}，"
                f"LBM 约 {self.lbm_kg:.2f} kg。"
            )
            if self.protein_g_min is not None and self.protein_g_max is not None:
                print(
                    "蛋白质目标区间仍为 "
                    f"{self.protein_g_min:.1f}–{self.protein_g_max:.1f} g/天；"
                    "如有不适请优先触发护栏关键词或使用医疗资源。"
                )
        else:
            print("尚未建档：可先调用 onboarding 以获取蛋白质目标。")
