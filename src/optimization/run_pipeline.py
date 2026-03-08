"""
CGLT 多约束几何构造与优化Pipeline
==================================
一键运行批处理：几何验证 + 约束检查 + 质量计算 + 代理预测 + 结果可视化

作者: 王浩博
单位: 哈尔滨工业大学
"""

import argparse
import csv
import os
import sys
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from utils.geometry_constraints import (
    assemble_row, compute_wr, A0, L_BEAM, W_MIN, WR_MIN,
    print_constraint_summary
)


def read_seeds_csv(path: str) -> List[Dict[str, float]]:
    """
    读取种子参数CSV

    参数:
        path: CSV文件路径

    返回:
        参数字典列表
    """
    rows = []
    # Seeds需要的8个设计参数
    required_cols = ['H1', 'L1', 'r1', 'a1', 'H2', 'L2', 'r2', 'a2']

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 只读取必需的列，跳过空值列
            row = {}
            for col in required_cols:
                if col in r and r[col] and r[col].strip():
                    row[col] = float(r[col])
                elif col in r:
                    # 如果列存在但为空，跳过此行
                    continue

            # 只有包含所有必需列的行才添加
            if len(row) == len(required_cols):
                rows.append(row)

    return rows


def write_csv(path: str, rows: List[Dict[str, object]], field_order: List[str]):
    """
    写入CSV文件

    参数:
        path: 输出路径
        rows: 数据行列表
        field_order: 字段顺序
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in field_order})



def predict_curve_from_geometry(row: Dict, predictor=None) -> np.ndarray:
    """
    从几何参数预测曲线

    参数:
        row: 参数字典
        predictor: 代理模型预测器（可选）

    返回:
        [N, 2] 曲线
    """
    if predictor is None:
        # 占位：返回空数组
        return np.array([])

    # 构造参数向量
    params = [
        row['H1'], row['L1'], row['a1'], row['r1'],
        row['H2'], row['L2'], row['a2'], row['r1']
    ]

    return predictor.predict_curve(params)


def extract_indicators(curve: np.ndarray) -> Dict[str, float]:
    """
    从曲线提取性能指标

    参数:
        curve: [N, 2] 曲线

    返回:
        {PBS, NCL, NCA}
    """
    if len(curve) == 0:
        return {"PBS": float("nan"), "NCL": float("nan"), "NCA": float("nan")}

    from optimization.surrogate import extract_all_indicators
    return extract_all_indicators(curve)


def compute_dtw(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算DTW距离

    参数:
        pred, target: [N, 2] 曲线

    返回:
        DTW距离
    """
    if len(pred) == 0 or len(target) == 0:
        return float("nan")

    from optimization.surrogate import compute_dtw_distance
    return compute_dtw_distance(pred, target)



def make_result_plots(rows: List[Dict[str, object]], outdir: str, enable_surrogate: bool = False):
    """
    生成结果图表

    参数:
        rows: 结果数据
        outdir: 输出目录
        enable_surrogate: 是否启用代理模型预测
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
    except Exception as e:
        print(f"[WARN] 无法绘图（未安装 matplotlib）：{e}")
        return

    # 过滤有效样本
    valid_rows = [r for r in rows if r["valid_ends"] == 1]

    if not valid_rows:
        print("[WARN] 无满足硬约束的样本，跳过绘图。")
        return

    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    data = [v["M_trim"] for v in valid_rows]
    plt.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel("M_trim (归一化质量削减)", fontsize=12)
    plt.ylabel("频数", fontsize=12)
    plt.title("质量削减分布 (满足硬约束)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig6_b_mtrim_hist.png"), dpi=240)
    plt.close()
    print(f"[OK] 生成: {outdir}/fig6_b_mtrim_hist.png")

    plt.figure(figsize=(8, 6))
    x = [v["M_trim"] for v in valid_rows]

    if enable_surrogate and "DTW_after" in valid_rows[0]:
        # 使用真实性能指标
        y = [v["DTW_after"] for v in valid_rows]
        ylabel = "DTW距离 (越小越好)"
    else:
        # 占位：使用M_sec_sum
        y = [v["M_sec_sum"] for v in valid_rows]
        ylabel = "M_sec_sum (归一化端部质量) [占位]"

    plt.scatter(x, y, s=20, alpha=0.6, c='coral', edgecolors='black', linewidths=0.5)
    plt.xlabel("M_trim (归一化质量削减)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title("质量-性能帕累托 (Pareto Front)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig6_d_pareto.png"), dpi=240)
    plt.close()
    print(f"[OK] 生成: {outdir}/fig6_d_pareto.png")

    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)
    print(f"总样本数: {len(rows)}")
    print(f"满足约束样本数: {len(valid_rows)}")
    print(f"约束通过率: {len(valid_rows) / len(rows) * 100:.2f}%")
    print(f"\nM_trim统计:")
    print(f"  平均值: {np.mean(data):.6f}")
    print(f"  标准差: {np.std(data):.6f}")
    print(f"  最小值: {np.min(data):.6f}")
    print(f"  最大值: {np.max(data):.6f}")
    print("=" * 60)



def main():
    ap = argparse.ArgumentParser(description="CGLT 多约束几何构造与结果导出")
    ap.add_argument("--seeds", required=True, help="输入 seeds.csv，列：H1,L1,r1,a1,H2,L2,r2,a2 (8个设计参数)")
    ap.add_argument("--out", default="candidates.csv", help="输出 candidates.csv")
    ap.add_argument("--make-fig6", action="store_true", help="生成结果型图（Figure 6 的关键面板）")
    ap.add_argument("--checkpoint-dir", type=str, default=None,
                    help="模型检查点目录（启用代理预测）")
    ap.add_argument("--target-curve", type=str, default=None,
                    help="目标曲线CSV路径（用于DTW计算）")
    ap.add_argument("--device", type=str, default='cuda', help="计算设备")
    args = ap.parse_args()

    # 打印约束摘要
    print_constraint_summary()

    # 读取种子参数
    print(f"\n[Data] 读取种子参数: {args.seeds}")
    seeds = read_seeds_csv(args.seeds)
    print(f"[Data] 加载 {len(seeds)} 个种子样本")

    # 初始化代理模型（如果提供）
    predictor = None
    enable_surrogate = False
    if args.checkpoint_dir is not None:
        print(f"\n[Model] 加载代理模型: {args.checkpoint_dir}")
        try:
            from optimization.surrogate import SurrogatePredictor
            predictor = SurrogatePredictor(
                checkpoint_dir=args.checkpoint_dir,
                device=args.device
            )
            enable_surrogate = True
            print("[Model] 代理模型加载成功!")
        except Exception as e:
            print(f"[WARN] 代理模型加载失败: {e}")
            print("[WARN] 将跳过代理预测")

    # 加载目标曲线（如果提供）
    target_curve = None
    if args.target_curve is not None and os.path.exists(args.target_curve):
        print(f"\n[Target] 加载目标曲线: {args.target_curve}")
        target_curve = np.loadtxt(args.target_curve, delimiter=',')
        print(f"[Target] 目标曲线形状: {target_curve.shape}")

    # 处理每个样本
    rows: List[Dict[str, object]] = []

    print(f"\n[Processing] 处理种子样本...")
    for r in tqdm(seeds, desc="处理中"):
        # 组装几何记录（8个设计参数）
        row = assemble_row(
            r["H1"], r["L1"], r["r1"], r["a1"],
            r["H2"], r["L2"], r["r2"], r["a2"]
        )

        # 如果启用代理模型，进行预测
        if enable_surrogate and predictor is not None:
            try:
                # 预测曲线
                pred_curve = predict_curve_from_geometry(row, predictor)

                # 提取指标
                indicators = extract_indicators(pred_curve)
                row.update({f"pred_{k}": v for k, v in indicators.items()})

                # 计算DTW（如果有目标曲线）
                if target_curve is not None:
                    dtw_dist = compute_dtw(pred_curve, target_curve)
                    row["DTW_after"] = dtw_dist
                else:
                    row["DTW_after"] = float("nan")

            except Exception as e:
                print(f"\n[WARN] 样本预测失败: {e}")
                row["pred_PBS"] = float("nan")
                row["pred_NCL"] = float("nan")
                row["pred_NCA"] = float("nan")
                row["DTW_after"] = float("nan")

        rows.append(row)

    # 定义输出字段顺序
    order = [
        # 8个设计参数
        "H1", "L1", "r1", "a1", "H2", "L2", "r2", "a2",
        # 计算得到的参数
        "w1", "w2", "wr1", "wr2",
        # 约束检查
        "ok_wr_1", "ok_w1_10", "ok_wr_2", "ok_w2_10",
        "valid_ends", "valid_full",
        # 性能指标
        "M_trim", "M_sec1", "M_sec2", "M_sec_sum", "area1", "area2"
    ]

    # 如果启用代理预测，添加预测字段
    if enable_surrogate:
        order.extend(["DTW_after", "pred_PBS", "pred_NCL", "pred_NCA"])

    # 写出CSV
    write_csv(args.out, rows, order)
    print(f"\n[OK] 写出 {args.out}，样本数={len(rows)}")
    print(f"    约束：wr > {WR_MIN} mm, w≥ {W_MIN} mm；归一化：A0={A0:.0f}, L_beam={L_BEAM:.0f} mm")

    # 生成图表
    if args.make_fig6:
        print(f"\n[Plotting] 生成结果图表...")
        make_result_plots(rows, outdir="fig_out", enable_surrogate=enable_surrogate)

    print("\n[完成] Pipeline执行完毕!")


if __name__ == "__main__":
    main()
