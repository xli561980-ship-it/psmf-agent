"""PSMF calculation tools."""

from __future__ import annotations

from typing import Any, Final, Literal

from agent.schemas import AgentContext, ToolResult
from agent.tool_registry import ToolSpec

GenderLiteral = Literal["male", "female"]

CAT1_G_PER_KG_LBM_MIN: Final[float] = 3.3
CAT1_G_PER_KG_LBM_MAX: Final[float] = 4.4
CAT2_G_PER_KG_LBM_MIN: Final[float] = 2.75
CAT2_G_PER_KG_LBM_MAX: Final[float] = 3.3
CAT3_G_PER_KG_LBM_MIN: Final[float] = 2.2
CAT3_G_PER_KG_LBM_MAX: Final[float] = 2.75


def normalize_gender(gender: str) -> GenderLiteral:
    g = gender.strip().lower()
    if g in ("m", "male", "男", "男性"):
        return "male"
    if g in ("f", "female", "女", "女性"):
        return "female"
    raise ValueError(f"无法识别性别：{gender!r}。请使用 male/female 或 男/女。")


def calculate_lbm(weight_kg: float, body_fat_percentage: float) -> float:
    """LBM (kg) = body weight * (1 - body fat percentage)."""
    if weight_kg <= 0:
        raise ValueError("体重必须为正数（kg）。")
    if not 0 <= body_fat_percentage <= 100:
        raise ValueError("体脂率须在 0–100（百分比）之间。")
    lbm_kg = weight_kg * (1.0 - body_fat_percentage / 100.0)
    if lbm_kg <= 0:
        raise ValueError("计算得到的瘦体重非正，请检查体重与体脂率。")
    return lbm_kg


def determine_category(body_fat_percentage: float, gender: str) -> int:
    """Determine PSMF category using the public project rule thresholds."""
    if not 0 <= body_fat_percentage <= 100:
        raise ValueError("体脂率须在 0–100（百分比）之间。")

    sex = normalize_gender(gender)
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


def protein_range_g_per_kg_for_category(category: int) -> tuple[float, float]:
    if category == 1:
        return CAT1_G_PER_KG_LBM_MIN, CAT1_G_PER_KG_LBM_MAX
    if category == 2:
        return CAT2_G_PER_KG_LBM_MIN, CAT2_G_PER_KG_LBM_MAX
    if category == 3:
        return CAT3_G_PER_KG_LBM_MIN, CAT3_G_PER_KG_LBM_MAX
    raise ValueError("category 必须为 1、2 或 3。")


def daily_protein_grams_range(lbm_kg: float, category: int) -> tuple[float, float]:
    lo, hi = protein_range_g_per_kg_for_category(category)
    return lbm_kg * lo, lbm_kg * hi


def calculate_psmf_targets(
    *,
    gender: str,
    weight_kg: float,
    body_fat_percentage: float,
) -> dict[str, Any]:
    sex = normalize_gender(gender)
    lbm = calculate_lbm(weight_kg, body_fat_percentage)
    category = determine_category(body_fat_percentage, sex)
    p_min, p_max = daily_protein_grams_range(lbm, category)
    return {
        "gender": sex,
        "weight_kg": float(weight_kg),
        "body_fat_percentage": float(body_fat_percentage),
        "lbm_kg": round(lbm, 3),
        "category": category,
        "protein_g_min": round(p_min, 1),
        "protein_g_max": round(p_max, 1),
    }


def calculate_psmf_targets_tool(
    arguments: dict[str, Any],
    context: AgentContext,
) -> ToolResult:
    try:
        result = calculate_psmf_targets(
            gender=str(arguments.get("gender") or ""),
            weight_kg=float(arguments["weight_kg"]),
            body_fat_percentage=float(arguments["body_fat_percentage"]),
        )
    except KeyError as exc:
        return ToolResult(
            name="calculate_psmf_targets",
            ok=False,
            content="缺少 PSMF 计算所需字段。",
            error=f"missing_argument:{exc}",
        )
    except Exception as exc:
        return ToolResult(
            name="calculate_psmf_targets",
            ok=False,
            content="PSMF 指标计算失败。",
            error=str(exc),
        )

    return ToolResult(
        name="calculate_psmf_targets",
        ok=True,
        content=(
            f"LBM={result['lbm_kg']:.3f}kg, Category={result['category']}, "
            f"protein={result['protein_g_min']:.1f}-{result['protein_g_max']:.1f}g/day"
        ),
        data=result,
    )


def psmf_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="calculate_psmf_targets",
            description=(
                "Calculates LBM, PSMF Category, and daily protein target range "
                "from gender, body weight, and body-fat percentage."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "gender": {"type": "string"},
                    "weight_kg": {"type": "number"},
                    "body_fat_percentage": {"type": "number"},
                },
                "required": ["gender", "weight_kg", "body_fat_percentage"],
            },
            handler=calculate_psmf_targets_tool,
        )
    ]
