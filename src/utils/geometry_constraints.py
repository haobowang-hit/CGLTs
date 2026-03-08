"""
CGLT 多约束几何构造与约束验证模块
====================================
实现web宽度计算、截面面积、质量削减、硬约束验证等功能

作者: 王浩博
单位: 哈尔滨工业大学
"""

import math
from typing import Dict, Tuple, Optional
import numpy as np


# 归一化基准面积 (mm^2)
A0 = 1000.0 * 1200.0

# 设计空间参数 (mm)
DESIGN_SPACE = 120.0

# 硬约束阈值
W_MIN = 3.0        # 最小web宽度 (mm) - 根据数据分布调整，获得33.9%有效设计
WR_MIN = 7.5       # 最小web半径 (mm) - 腹板曲率半径约束

# 梁长 (mm)，用于质量削减计算
L_BEAM = 1000.0

# 展平应变约束 (可选)
EPSILON_FLAT_MAX = 0.01  # 1% 最大展平应变


def compute_wr(H: float, r1: float, a_deg: float) -> float:
    """
    计算腹板半径 wr (web radius)

    公式: wr = (sr - r1 + r1·cos(α)) / (1 - cos(α))
         其中 sr = H/2 (截面半径)

    参数:
        H: 截面高度 (mm)
        r1: 主曲率半径 (mm)
        a_deg: 角度 (度)

    返回:
        wr: 腹板半径 (mm)，失败返回-1
    """
    sr = H / 2.0
    a_rad = math.radians(a_deg)
    cos_a = math.cos(a_rad)

    try:
        denominator = 1.0 - cos_a
        if abs(denominator) < 1e-8:
            return -1.0
        wr = (sr - r1 + r1 * cos_a) / denominator
        return wr
    except (ZeroDivisionError, ValueError):
        return -1.0


def compute_w(L: float, r1: float, wr: float, a_deg: float) -> float:
    """
    计算web有效宽度 w

    公式: w = 120·0.5 - α·r1·π/180 - α·wr·π/90 - L/2

    参数:
        L: Lumbus宽度参数 (mm)
        r1: 主曲率半径 (mm)
        wr: 腹板半径 (mm)
        a_deg: 角度 (度)

    返回:
        w: web有效宽度 (mm)
    """
    a = a_deg
    return DESIGN_SPACE * 0.5 - a * r1 * math.pi / 180.0 - a * wr * math.pi / 90.0 - 0.5 * L


def section_area(H: float, L: float, r1: float, wr: float, a_deg: float) -> float:
    """
    计算单截面面积

    公式: area = H·L + 2·H·r1·sin(α)
                - 4·(r1²·sin(α) - π·r1²·α/360 - r1²·sin(α)·cos(α)/2)
                + 4·(wr²·sin(α) - π·wr²·α/360 - wr²·sin(α)·cos(α)/2)

    参数:
        H: 截面高度 (mm)
        L: Lumbus宽度参数 (mm)
        r1: 主曲率半径 (mm)
        wr: 腹板半径 (mm)
        a_deg: 角度 (度)

    返回:
        area: 截面面积 (mm²)
    """
    a_rad = math.radians(a_deg)
    sin_a = math.sin(a_rad)
    cos_a = math.cos(a_rad)

    r1_sq = r1 * r1
    wr_sq = wr * wr

    # 主体矩形面积
    term1 = H * L + 2 * H * r1 * sin_a

    # r1贡献（减去）
    term2 = -4.0 * (r1_sq * sin_a - math.pi * r1_sq * (a_deg / 360.0) - r1_sq * sin_a * cos_a / 2.0)

    # wr贡献（加上）
    term3 = 4.0 * (wr_sq * sin_a - math.pi * wr_sq * (a_deg / 360.0) - wr_sq * sin_a * cos_a / 2.0)

    return term1 + term2 + term3


def check_hard_constraints(wr: float, w: float) -> Tuple[bool, bool]:
    """
    检查硬约束

    参数:
        wr: 腹板半径 (mm)
        w: web宽度 (mm)

    返回:
        ok_wr: wr > 7.5mm
        ok_w: w ≥ 3.0mm
    """
    ok_wr = (wr > WR_MIN)
    ok_w = (w >= W_MIN)
    return ok_wr, ok_w


def check_flattening_strain(t: float, r: float) -> bool:
    """
    检查展平应变约束（可选）

    公式: ε_flat = t / (2r) ≤ 1%

    参数:
        t: 厚度 (mm)
        r: 曲率半径 (mm)

    返回:
        是否满足应变约束
    """
    if r <= 0:
        return False
    epsilon = t / (2.0 * r)
    return epsilon <= EPSILON_FLAT_MAX


def compute_mass_trimming(w1: float, w2: float) -> float:
    """
    计算归一化质量削减

    公式: M_trim = ((Δw1 + Δw2) × L_beam) / A0
         其中 Δw_i = max(w_i - 10, 0)

    参数:
        w1: 端1的web宽度 (mm)
        w2: 端2的web宽度 (mm)

    返回:
        M_trim: 归一化质量削减
    """
    delta_w1 = max(w1 - W_MIN, 0.0)
    delta_w2 = max(w2 - W_MIN, 0.0)
    M_trim = ((delta_w1 + delta_w2) * L_BEAM) / A0
    return M_trim


def normalize_section_mass(area: float) -> float:
    """
    计算归一化端部质量

    公式: M_sec = area / A0

    参数:
        area: 截面面积 (mm²)

    返回:
        M_sec: 归一化端部质量
    """
    return area / A0


def end_section_metrics(H: float, L: float, r1: float, r2_param: float, a_deg: float) -> Dict[str, float]:
    """
    计算端部几何指标

    注意：r2_param 是设计参数（不参与计算），wr 是计算得到的腹板半径

    参数:
        H: 截面高度 (mm)
        L: Lumbus宽度 (mm)
        r1: 主曲率半径 (mm)
        r2_param: 设计参数 r2（不用于约束检查）
        a_deg: 角度 (度)

    返回:
        字典包含: w, wr, area, M_sec, ok_wr, ok_w
    """
    # 计算腹板半径 wr
    wr = compute_wr(H, r1, a_deg)

    # 计算web宽度
    w = compute_w(L, r1, wr, a_deg)

    # 计算截面面积
    area = section_area(H, L, r1, wr, a_deg)

    # 归一化质量
    M_sec = normalize_section_mass(area)

    # 硬约束检查（检查 wr，不是 r2_param）
    ok_wr, ok_w = check_hard_constraints(wr, w)

    return {
        "w": w,
        "wr": wr,
        "area": area,
        "M_sec": M_sec,
        "ok_wr": ok_wr,
        "ok_w": ok_w
    }


def assemble_row(
    H1: float, L1: float, r1: float, a1: float,
    H2: float, L2: float, r2: float, a2: float,
    sampled_min_wr: Optional[float] = None,
    sampled_min_w: Optional[float] = None
) -> Dict[str, float]:
    """
    生成一行完整的几何记录

    参数:
        H1, L1, r1, a1: 端1参数 (Height, Lumbus, Radius, Angle)
        H2, L2, r2, a2: 端2参数 (Height, Lumbus, Radius, Angle)
        sampled_min_wr: 全长采样最小 wr 值（可选）
        sampled_min_w: 全长采样最小 w 值（可选）

    返回:
        包含主字段和所有派生字段的字典
    """
    # 计算端1指标（使用 r1）
    m1 = end_section_metrics(H1, L1, r1, None, a1)

    # 计算端2指标（使用 r2）
    m2 = end_section_metrics(H2, L2, r2, None, a2)

    # 计算质量削减
    M_trim = compute_mass_trimming(m1["w"], m2["w"])

    # 两端约束验证
    valid_ends = int(m1["ok_wr"] and m1["ok_w"] and m2["ok_wr"] and m2["ok_w"])

    # 全长约束验证（可选）
    valid_full = ""
    if (sampled_min_wr is not None) and (sampled_min_w is not None):
        valid_full = int((sampled_min_wr > WR_MIN) and (sampled_min_w >= W_MIN))

    return {

        "H1": H1, "L1": L1, "r1": r1, "a1": a1,
        "H2": H2, "L2": L2, "r2": r2, "a2": a2,

        "w1": m1["w"], "w2": m2["w"],
        "wr1": m1["wr"], "wr2": m2["wr"],

        "ok_wr_1": int(m1["ok_wr"]), "ok_w1_10": int(m1["ok_w"]),
        "ok_wr_2": int(m2["ok_wr"]), "ok_w2_10": int(m2["ok_w"]),
        "valid_ends": valid_ends,
        "valid_full": valid_full,

        "M_trim": M_trim,
        "M_sec1": m1["M_sec"], "M_sec2": m2["M_sec"],
        "M_sec_sum": m1["M_sec"] + m2["M_sec"],

        "area1": m1["area"], "area2": m2["area"],
    }


def validate_design_parameters(
    H: float, L: float, Angle: float, Radius: float,
    check_r2_w: bool = True,
    strict_integer: bool = False
) -> Tuple[bool, str]:
    """
    验证设计参数是否满足所有约束

    参数:
        H: 截面高度 (mm)
        L: Lumbus宽度 (mm)
        Angle: 角度 (度)
        Radius: 主曲率半径 (mm)
        check_r2_w: 是否检查r2和w约束
        strict_integer: 是否严格要求整数/偶数（False则允许小数）

    返回:
        (是否有效, 失败原因)
    """
    # 基本范围约束
    if not (24 <= H <= 36):
        return False, f"H={H} 不在[24,36]范围内"

    # 如果严格模式，检查H是否为偶数
    if strict_integer and H % 2 != 0:
        return False, f"H={H} 必须为偶数"

    if not (30 <= Angle <= 80):
        return False, f"Angle={Angle} 不在[30,80]范围内"

    if not (10 <= Radius <= H / 2):
        return False, f"Radius={Radius} 不在[10,{H/2}]范围内"

    if not (1 <= L <= 10):
        return False, f"L={L} 不在[1,10]范围内"

    # 几何约束检查
    if check_r2_w:
        r2 = compute_r2(H, Radius, Angle)
        if r2 <= 0:
            return False, f"r2={r2:.2f} 计算失败"

        w = compute_w(L, Radius, r2, Angle)

        if r2 <= R2_MIN:
            return False, f"r2={r2:.2f} ≤ {R2_MIN}"

        if w < W_MIN:
            return False, f"w={w:.2f} < {W_MIN}"

    return True, "valid"


def validate_design_row(row: Dict[str, float]) -> Tuple[bool, str]:
    """
    验证一行设计参数

    参数:
        row: 包含H1,L1,r1,a1,H2,L2,r2_1,a2,r2_2的字典

    返回:
        (是否有效, 失败原因)
    """
    # 验证端1
    valid1, msg1 = validate_design_parameters(
        row["H1"], row["L1"], row["a1"], row["r1"], check_r2_w=False
    )
    if not valid1:
        return False, f"端1: {msg1}"

    # 验证端2
    valid2, msg2 = validate_design_parameters(
        row["H2"], row["L2"], row["a2"], row["r1"], check_r2_w=False
    )
    if not valid2:
        return False, f"端2: {msg2}"

    # 检查几何约束
    if row.get("valid_ends", 0) != 1:
        return False, "几何约束不满足"

    return True, "valid"


def print_constraint_summary():
    """打印约束摘要"""
    print("=" * 60)
    print("CGLT 多约束几何设计规范")
    print("=" * 60)
    print(f"硬约束:")
    print(f"  - 腹板半径: wr > {WR_MIN} mm (通过 H, r1, a 计算得到)")
    print(f"  - Web宽度: w ≥ {W_MIN} mm")
    print(f"\n参数范围 (支持小数，精度0.01):")
    print(f"  - H (截面高度): [24.00, 36.00] mm")
    print(f"  - L (Lumbus): [1.00, 10.00] mm")
    print(f"  - Angle (角度): [30.00, 80.00] 度")
    print(f"  - Radius (r1): [10.00, H/2] mm")
    print(f"  - r2 (设计参数): 无约束")
    print(f"\n质量计算:")
    print(f"  - 归一化基准: A0 = {A0:.0f} mm²")
    print(f"  - 梁长: L_beam = {L_BEAM:.0f} mm")
    print(f"  - 质量削减: M_trim = ((Δw1+Δw2)×L_beam)/A0")
    print("=" * 60)


if __name__ == "__main__":
    # 测试示例
    print_constraint_summary()

    print("\n测试案例:")
    H1, L1, r1, a1 = 30, 5, 12, 45
    H2, L2, a2 = 32, 6, 50

    # 计算r2
    r2_1 = compute_r2(H1, r1, a1)
    r2_2 = compute_r2(H2, r1, a2)

    print(f"\n端1: H={H1}, L={L1}, r1={r1}, a={a1}°")
    print(f"  → r2_1 = {r2_1:.2f} mm")

    print(f"\n端2: H={H2}, L={L2}, r1={r1}, a={a2}°")
    print(f"  → r2_2 = {r2_2:.2f} mm")

    # 组装完整记录
    row = assemble_row(H1, L1, r1, a1, H2, L2, r2_1, a2, r2_2)

    print(f"\n完整记录:")
    print(f"  w1={row['w1']:.2f}, w2={row['w2']:.2f}")
    print(f"  valid_ends={row['valid_ends']}")
    print(f"  M_trim={row['M_trim']:.6f}")
    print(f"  M_sec_sum={row['M_sec_sum']:.6f}")
