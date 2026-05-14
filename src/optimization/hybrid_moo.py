

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import NSGAIISampler
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple, Optional

# 确保优先从当前目录导入模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# 导入代理预测器
from optimization.surrogate import SurrogatePredictor, extract_all_indicators

# 强制从repo目录导入geometry_constraints
import importlib.util
from utils.geometry_constraints import (
    compute_wr, compute_w, section_area, compute_mass_trimming, assemble_row
)


plt.rcParams.update({
    'font.family': ['Times New Roman'],
    'font.serif': ['Times New Roman'],
    'font.size': 48,           # 增大：38 -> 48
    'axes.titlesize': 56,      # 增大：44 -> 56
    'axes.labelsize': 52,      # 增大：42 -> 52
    'xtick.labelsize': 46,     # 增大：36 -> 46
    'ytick.labelsize': 46,     # 增大：36 -> 46
    'legend.fontsize': 42,     # 增大：34 -> 42
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 2.0,     # 增大：1.5 -> 2.0
    'lines.linewidth': 3.5,    # 增大：2.5 -> 3.5
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold'
})

def style_axis(ax):
    """统一坐标轴样式"""
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)



def is_pareto_efficient_3d(costs):
    """
    三维帕累托前沿提取（快速算法）

    参数:
        costs: numpy array, shape (n_points, 3)
               对于要最大化的目标，传入负值

    返回:
        pareto_mask: boolean array
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient



def load_existing_data(results_csv: str = 'results.csv') -> pd.DataFrame:
    """
    加载现有的结果数据

    参数:
        results_csv: 结果CSV路径

    返回:
        DataFrame
    """
    print(f"\n{'='*70}")
    print("加载现有数据")
    print(f"{'='*70}")

    if not os.path.exists(results_csv):
        print(f"[警告] 未找到 {results_csv}，将仅使用模型生成数据")
        return None

    df = pd.read_csv(results_csv)
    print(f"总设计数: {len(df)}")

    # 过滤有效设计
    df = df[df['valid_ends'] == 1].copy()
    print(f"有效设计: {len(df)}")

    # 统一列名
    if 'pred_PBS' in df.columns:
        df.rename(columns={
            'pred_PBS': 'PBS',
            'pred_NCL': 'NCL',
            'pred_NCA': 'NCA'
        }, inplace=True)

    # 检查必需列
    required_cols = ['PBS', 'NCL', 'NCA', 'M_trim']
    if all(col in df.columns for col in required_cols):
        print(f"[✓] 数据完整，包含性能指标")
        print(f"    PBS 范围: [{df['PBS'].min():.4f}, {df['PBS'].max():.4f}]")
        print(f"    NCL 范围: [{df['NCL'].min():.4f}, {df['NCL'].max():.4f}]")
        print(f"    NCA 范围: [{df['NCA'].min():.4f}, {df['NCA'].max():.4f}]")
        print(f"    M_trim 范围: [{df['M_trim'].min():.6f}, {df['M_trim'].max():.6f}]")
        return df
    else:
        print(f"[警告] 缺少性能指标列，数据不完整")
        return None


def generate_new_designs_bayesian(
    predictor: SurrogatePredictor,
    n_trials: int = 500,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    使用贝叶斯优化生成新设计

    参数:
        predictor: 代理模型预测器
        n_trials: 试验次数
        device: 计算设备

    返回:
        新设计的DataFrame
    """
    print(f"\n{'='*70}")
    print(f"贝叶斯优化生成新设计")
    print(f"{'='*70}")
    print(f"试验次数: {n_trials}")
    print(f"设备: {device}")

    valid_designs = []

    def objective(trial):
        # 采样参数
        params = {
            'H1': trial.suggest_int('H1', 24, 36, step=2),
            'L1': trial.suggest_float('L1', 1, 10),
            'a1': trial.suggest_float('a1', 30, 80),
            'r1': trial.suggest_float('r1', 10, 18),
            'H2': trial.suggest_int('H2', 24, 36, step=2),
            'L2': trial.suggest_float('L2', 1, 10),
            'a2': trial.suggest_float('a2', 30, 80),
            'r2': trial.suggest_float('r2', 10, 18),
        }

        # 验证约束
        row = assemble_row(
            params['H1'], params['L1'], params['r1'], params['a1'],
            params['H2'], params['L2'], params['r2'], params['a2']
        )

        if row['valid_ends'] != 1:
            return (-1e6, -1e6, -1e6, 1e6)

        # 预测性能
        try:
            param_vector = [
                params['H1'], params['L1'], params['a1'], params['r1'],
                params['H2'], params['L2'], params['a2'], params['r1']
            ]
            curve = predictor.predict_curve(param_vector)
            performance = extract_all_indicators(curve)

            # 记录有效设计
            design = {
                **params,
                'PBS': performance['PBS'],
                'NCL': performance['NCL'],
                'NCA': performance['NCA'],
                'M_trim': row['M_trim'],
                'w1': row['w1'],
                'w2': row['w2'],
                'wr1': row['wr1'],
                'wr2': row['wr2'],
                'source': 'bayesian'
            }
            valid_designs.append(design)

            omega = 1.0 - row['M_trim']

            # 返回目标值（最大化PBS,NCL,NCA，最小化omega）
            return (-performance['PBS'], -performance['NCL'], -performance['NCA'], omega)

        except Exception as e:
            # 异常情况：返回极差的目标值
            return (-1e6, -1e6, -1e6, 1e6)

    # 创建study
    sampler = NSGAIISampler(population_size=50)
    study = optuna.create_study(
        directions=['minimize', 'minimize', 'minimize', 'minimize'],
        sampler=sampler
    )

    # 执行优化
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n优化完成:")
    print(f"  有效设计: {len(valid_designs)}")

    return pd.DataFrame(valid_designs)



def densify_pareto_region(
    pareto_df: pd.DataFrame,
    predictor: SurrogatePredictor,
    n_samples_per_design: int = 5
) -> pd.DataFrame:
    """
    在帕累托前沿附近密集采样

    参数:
        pareto_df: 帕累托前沿设计
        predictor: 代理模型
        n_samples_per_design: 每个设计的扰动样本数

    返回:
        密集化后的DataFrame
    """
    print(f"\n{'='*70}")
    print("帕累托区域密集化")
    print(f"{'='*70}")

    dense_designs = []

    for idx, row in tqdm(pareto_df.iterrows(), total=len(pareto_df), desc="密集采样"):
        # 原始设计
        base_params = {
            'H1': row['H1'], 'L1': row['L1'], 'a1': row['a1'], 'r1': row['r1'],
            'H2': row['H2'], 'L2': row['L2'], 'a2': row['a2'], 'r2': row['r2']
        }

        # 生成扰动样本
        for _ in range(n_samples_per_design):
            perturbed = base_params.copy()

            # 添加小扰动
            perturbed['H1'] = int(np.clip(base_params['H1'] + np.random.choice([-2, 0, 2]), 24, 36))
            perturbed['H2'] = int(np.clip(base_params['H2'] + np.random.choice([-2, 0, 2]), 24, 36))
            perturbed['L1'] = np.clip(base_params['L1'] + np.random.uniform(-0.5, 0.5), 1, 10)
            perturbed['L2'] = np.clip(base_params['L2'] + np.random.uniform(-0.5, 0.5), 1, 10)
            perturbed['a1'] = np.clip(base_params['a1'] + np.random.uniform(-5, 5), 30, 80)
            perturbed['a2'] = np.clip(base_params['a2'] + np.random.uniform(-5, 5), 30, 80)
            perturbed['r1'] = np.clip(base_params['r1'] + np.random.uniform(-1, 1), 10, 18)
            perturbed['r2'] = np.clip(base_params['r2'] + np.random.uniform(-1, 1), 10, 18)

            # 验证约束
            geo_row = assemble_row(
                perturbed['H1'], perturbed['L1'], perturbed['r1'], perturbed['a1'],
                perturbed['H2'], perturbed['L2'], perturbed['r2'], perturbed['a2']
            )

            if geo_row['valid_ends'] != 1:
                continue

            # 预测性能
            try:
                param_vector = [
                    perturbed['H1'], perturbed['L1'], perturbed['a1'], perturbed['r1'],
                    perturbed['H2'], perturbed['L2'], perturbed['a2'], perturbed['r1']
                ]
                curve = predictor.predict_curve(param_vector)
                performance = extract_all_indicators(curve)

                design = {
                    **perturbed,
                    'PBS': performance['PBS'],
                    'NCL': performance['NCL'],
                    'NCA': performance['NCA'],
                    'M_trim': geo_row['M_trim'],
                    'w1': geo_row['w1'],
                    'w2': geo_row['w2'],
                    'source': 'densified'
                }
                dense_designs.append(design)

            except:
                continue

    print(f"密集化生成设计: {len(dense_designs)}")

    return pd.DataFrame(dense_designs)



def plot_pareto_with_paths(
    all_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    output_dir: str = './hybrid_moo_results'
):
    """
    绘制带多条优化路径的帕累托前沿

    参数:
        all_df: 所有有效设计
        pareto_df: 帕累托前沿
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("生成多路径可视化")
    print(f"{'='*70}")

    print("生成 3D 帕累托前沿 + 优化路径...")

    # 使用合理比例的画布（ω在Z轴，参考analyze_3d_pareto.py）
    fig = plt.figure(figsize=(24, 18))
    ax = fig.add_subplot(111, projection='3d')

    scatter_all = ax.scatter(
        all_df['PBS'], all_df['NCL'], all_df['omega'],  # PBS为X轴，NCL为Y轴，ω为Z轴
        c=all_df['NCA'], cmap='plasma', s=150, alpha=0.25,
        edgecolors='none', label='All Valid Designs'
    )

    # 2. 绘制帕累托前沿
    scatter_pareto = ax.scatter(
        pareto_df['PBS'], pareto_df['NCL'], pareto_df['omega'],
        c=pareto_df['NCA'], cmap='plasma', s=450, alpha=0.95,
        edgecolors='black', linewidths=3.0, label='Pareto Front', marker='D'
    )

    # 3. 绘制多条优化路径
    pareto_sorted = pareto_df.reset_index(drop=True)

    # 路径1: ω优先（omega从小到大，轻量化优先）
    path_omega = pareto_sorted.sort_values('omega').reset_index(drop=True)
    ax.plot(path_omega['PBS'], path_omega['NCL'], path_omega['omega'],
            'r-', linewidth=4.5, alpha=0.9, zorder=5, label=r'Path: $\omega$ Priority')

    # 路径2: PBS优先（PBS从大到小）
    path_pbs = pareto_sorted.sort_values('PBS', ascending=False).reset_index(drop=True)
    ax.plot(path_pbs['PBS'], path_pbs['NCL'], path_pbs['omega'],
            'b-', linewidth=4.5, alpha=0.9, zorder=5, label='Path: PBS Priority')

    # 路径3: NCL优先（NCL从大到小）
    path_ncl = pareto_sorted.sort_values('NCL', ascending=False).reset_index(drop=True)
    ax.plot(path_ncl['PBS'], path_ncl['NCL'], path_ncl['omega'],
            'g-', linewidth=4.5, alpha=0.9, zorder=5, label='Path: NCL Priority')

    cbar = plt.colorbar(scatter_pareto, ax=ax, pad=0.06, shrink=0.75)
    cbar.set_label('NCA [rad]', fontweight='bold', rotation=270, labelpad=50, fontsize=52)
    cbar.ax.tick_params(labelsize=34)

    ax.set_xlabel('PBS [Nm/rad]', fontweight='bold', labelpad=40, fontsize=52)
    ax.set_ylabel('NCL [Nm]', fontweight='bold', labelpad=40, fontsize=52)
    ax.set_zlabel(r'$\omega$', fontweight='bold', labelpad=35, fontsize=52)

    ax.set_title(r'PBS-NCL-$\omega$-NCA Pareto Front', fontweight='bold', pad=40, fontsize=56)

    # 图例放在右上角，稍微下移
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.92),
              frameon=True, fancybox=False, edgecolor='black',
              framealpha=0.7, borderaxespad=0)

    ax.view_init(elev=25, azim=135)

    # 调整所有坐标轴刻度字体大小
    ax.tick_params(axis='x', labelsize=44)
    ax.tick_params(axis='y', labelsize=44)
    ax.tick_params(axis='z', labelsize=44, pad=20)

    # 设置Z轴刻度间隔
    z_min = pareto_df['omega'].min()
    z_max = pareto_df['omega'].max()
    z_min_tick = np.floor(z_min*100) / 100
    z_max_tick = min(1.0, np.ceil(z_max*100) / 100)
    z_ticks = np.arange(z_min_tick, z_max_tick + 0.001, 0.01)
    ax.set_zticks(z_ticks)

    # 开启网格，使坐标更清晰
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=1)

    # 设置背景为白色，轴线更清晰
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_linewidth(2)
    ax.yaxis.pane.set_linewidth(2)
    ax.zaxis.pane.set_linewidth(2)

    # 设置坐标轴刻度线
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    # 使用tight_layout自动调整布局
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'pareto_3d_with_paths.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[✓] {fig_path}")

    print("生成 2D 投影 + 优化路径（4张独立图）...")

    # 准备路径数据
    path_nca = pareto_sorted.sort_values('NCA', ascending=False).reset_index(drop=True)

    fig1 = plt.figure(figsize=(16, 14))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(all_df['NCL'], all_df['PBS'], c=all_df['NCA'], cmap='plasma',
               s=150, alpha=0.25, edgecolors='none')
    scatter1 = ax1.scatter(pareto_df['NCL'], pareto_df['PBS'], c=pareto_df['NCA'],
                        cmap='plasma', s=450, alpha=0.95, edgecolors='black', linewidths=3.0, marker='D')
    ax1.plot(path_omega['NCL'], path_omega['PBS'], 'r-', linewidth=4, alpha=0.8, label=r'$\omega$ Priority')
    ax1.plot(path_pbs['NCL'], path_pbs['PBS'], 'b-', linewidth=4, alpha=0.8, label='PBS Priority')
    ax1.plot(path_ncl['NCL'], path_ncl['PBS'], 'g-', linewidth=4, alpha=0.8, label='NCL Priority')
    ax1.set_xlabel('NCL [Nm]', fontweight='bold')
    ax1.set_ylabel('PBS [Nm/rad]', fontweight='bold')
    ax1.set_title('NCL vs PBS', fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('NCA [rad]', fontweight='bold', rotation=270, labelpad=50)
    cbar1.ax.tick_params(labelsize=40)
    ax1.legend(loc='lower right', framealpha=0.7)
    style_axis(ax1)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'pareto_2d_ncl_pbs.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] {fig1_path}")

    fig2 = plt.figure(figsize=(16, 14))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(all_df['omega'], all_df['PBS'], c=all_df['NCA'], cmap='plasma',
               s=150, alpha=0.25, edgecolors='none')
    scatter2 = ax2.scatter(pareto_df['omega'], pareto_df['PBS'], c=pareto_df['NCA'],
                        cmap='plasma', s=450, alpha=0.95, edgecolors='black', linewidths=3.0, marker='D')
    ax2.plot(path_omega['omega'], path_omega['PBS'], 'r-', linewidth=4, alpha=0.8, label=r'$\omega$ Priority')
    ax2.plot(path_pbs['omega'], path_pbs['PBS'], 'b-', linewidth=4, alpha=0.8, label='PBS Priority')
    ax2.plot(path_ncl['omega'], path_ncl['PBS'], 'g-', linewidth=4, alpha=0.8, label='NCL Priority')
    ax2.set_xlabel(r'$\omega$', fontweight='bold')
    ax2.set_ylabel('PBS [Nm/rad]', fontweight='bold')
    ax2.set_title(r'$\omega$ vs PBS', fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('NCA [rad]', fontweight='bold', rotation=270, labelpad=50)
    cbar2.ax.tick_params(labelsize=40)
    ax2.legend(loc='lower right', framealpha=0.7)
    style_axis(ax2)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'pareto_2d_omega_pbs.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] {fig2_path}")

    fig3 = plt.figure(figsize=(16, 14))
    ax3 = fig3.add_subplot(111)
    ax3.scatter(all_df['omega'], all_df['NCL'], c=all_df['NCA'], cmap='plasma',
               s=150, alpha=0.25, edgecolors='none')
    scatter3 = ax3.scatter(pareto_df['omega'], pareto_df['NCL'], c=pareto_df['NCA'],
                        cmap='plasma', s=450, alpha=0.95, edgecolors='black', linewidths=3.0, marker='D')
    ax3.plot(path_omega['omega'], path_omega['NCL'], 'r-', linewidth=4, alpha=0.8, label=r'$\omega$ Priority')
    ax3.plot(path_pbs['omega'], path_pbs['NCL'], 'b-', linewidth=4, alpha=0.8, label='PBS Priority')
    ax3.plot(path_ncl['omega'], path_ncl['NCL'], 'g-', linewidth=4, alpha=0.8, label='NCL Priority')
    ax3.set_xlabel(r'$\omega$', fontweight='bold')
    ax3.set_ylabel('NCL [Nm]', fontweight='bold')
    ax3.set_title(r'$\omega$ vs NCL', fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('NCA [rad]', fontweight='bold', rotation=270, labelpad=50)
    cbar3.ax.tick_params(labelsize=40)
    ax3.legend(loc='lower right', framealpha=0.7)
    style_axis(ax3)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'pareto_2d_omega_ncl.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] {fig3_path}")

    fig4 = plt.figure(figsize=(16, 14))
    ax4 = fig4.add_subplot(111)
    ax4.scatter(all_df['NCA'], all_df['PBS'], c=all_df['omega'], cmap='plasma',
               s=150, alpha=0.25, edgecolors='none')
    scatter4 = ax4.scatter(pareto_df['NCA'], pareto_df['PBS'], c=pareto_df['omega'],
                        cmap='plasma', s=450, alpha=0.95, edgecolors='black', linewidths=3.0, marker='D')
    ax4.plot(path_nca['NCA'], path_nca['PBS'], 'purple', linewidth=4, alpha=0.8, label='NCA Priority')
    ax4.plot(path_pbs['NCA'], path_pbs['PBS'], 'b-', linewidth=4, alpha=0.8, label='PBS Priority')
    ax4.set_xlabel('NCA [rad]', fontweight='bold')
    ax4.set_ylabel('PBS [Nm/rad]', fontweight='bold')
    ax4.set_title('NCA vs PBS', fontweight='bold')
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label(r'$\omega$', fontweight='bold', rotation=270, labelpad=50)
    cbar4.ax.tick_params(labelsize=40)
    ax4.legend(loc='lower right', framealpha=0.7)
    style_axis(ax4)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, 'pareto_2d_nca_pbs.png')
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] {fig4_path}")


def extract_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 PBS↑, NCL↑, omega↓ 提取三目标帕累托前沿。
    """
    if df.empty:
        return df.copy()

    costs = np.column_stack([
        -df['PBS'].values,
        -df['NCL'].values,
        df['omega'].values,
    ])
    mask = is_pareto_efficient_3d(costs)
    return df[mask].copy().reset_index(drop=True)


def compute_recommendations(pareto_df: pd.DataFrame) -> pd.DataFrame:
    """
    从帕累托前沿中按多种策略筛选推荐设计。
    """
    if pareto_df.empty:
        return pd.DataFrame()

    eps = 1e-8
    pbs_norm = (pareto_df['PBS'] - pareto_df['PBS'].min()) / (pareto_df['PBS'].max() - pareto_df['PBS'].min() + eps)
    ncl_norm = (pareto_df['NCL'] - pareto_df['NCL'].min()) / (pareto_df['NCL'].max() - pareto_df['NCL'].min() + eps)
    nca_norm = (pareto_df['NCA'] - pareto_df['NCA'].min()) / (pareto_df['NCA'].max() - pareto_df['NCA'].min() + eps)
    mtrim_norm = (pareto_df['M_trim'] - pareto_df['M_trim'].min()) / (pareto_df['M_trim'].max() - pareto_df['M_trim'].min() + eps)

    recs = []

    ideal_dist = np.sqrt((1.0 - pbs_norm) ** 2 + (1.0 - ncl_norm) ** 2 + (1.0 - nca_norm) ** 2 + mtrim_norm ** 2)
    idx = ideal_dist.idxmin()
    row = pareto_df.loc[idx].to_dict()
    row['strategy'] = 'Ideal Point'
    row['score'] = float(1.0 / (ideal_dist.loc[idx] + eps))
    recs.append(row)

    weighted_balanced = 0.3 * pbs_norm + 0.3 * ncl_norm + 0.2 * nca_norm + 0.2 * (1.0 - mtrim_norm)
    idx = weighted_balanced.idxmax()
    row = pareto_df.loc[idx].to_dict()
    row['strategy'] = 'Weighted Sum-Balanced'
    row['score'] = float(weighted_balanced.loc[idx])
    recs.append(row)

    weighted_strength = 0.5 * pbs_norm + 0.3 * ncl_norm + 0.1 * nca_norm + 0.1 * (1.0 - mtrim_norm)
    idx = weighted_strength.idxmax()
    row = pareto_df.loc[idx].to_dict()
    row['strategy'] = 'Weighted Sum-Strength'
    row['score'] = float(weighted_strength.loc[idx])
    recs.append(row)

    weighted_light = 0.2 * pbs_norm + 0.2 * ncl_norm + 0.1 * nca_norm + 0.5 * (1.0 - mtrim_norm)
    idx = weighted_light.idxmax()
    row = pareto_df.loc[idx].to_dict()
    row['strategy'] = 'Weighted Sum-Lightweight'
    row['score'] = float(weighted_light.loc[idx])
    recs.append(row)

    return pd.DataFrame(recs)


def run_mode_bayesian(args, predictor):
    """
    仅基于贝叶斯优化生成设计，不合并历史数据。
    """
    print(f"\n{'='*70}")
    print("运行模式: bayesian")
    print(f"{'='*70}")

    all_df = generate_new_designs_bayesian(predictor, args.n_trials, args.device)
    if all_df.empty:
        print("[警告] 未生成有效设计，流程结束")
        return

    all_df['omega'] = 1.0 - all_df['M_trim']
    pareto_df = extract_pareto_front(all_df)

    os.makedirs(args.output, exist_ok=True)
    all_path = os.path.join(args.output, 'all_valid_designs.csv')
    pareto_path = os.path.join(args.output, 'pareto_front_designs.csv')
    rec_path = os.path.join(args.output, 'recommended_designs.csv')

    all_df.to_csv(all_path, index=False)
    pareto_df.to_csv(pareto_path, index=False)
    compute_recommendations(pareto_df).to_csv(rec_path, index=False)

    print(f"[✓] 所有有效设计: {all_path}")
    print(f"[✓] 帕累托前沿: {pareto_path}")
    print(f"[✓] 推荐设计: {rec_path}")

    plot_pareto_with_paths(all_df, pareto_df, args.output)


def run_mode_hybrid(args, predictor):
    """
    混合模式：历史结果 + 贝叶斯优化 + （可选）帕累托区域密集化。
    """
    print(f"\n{'='*70}")
    print("运行模式: hybrid")
    print(f"{'='*70}")

    # 1. 加载现有数据
    existing_df = load_existing_data(args.results_csv)

    # 2. 生成新设计（贝叶斯优化）
    new_df = generate_new_designs_bayesian(predictor, args.n_trials, args.device)

    # 3. 合并数据
    print(f"\n{'='*70}")
    print("合并数据集")
    print(f"{'='*70}")

    if existing_df is not None:
        existing_df['source'] = 'historical'
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"历史数据: {len(existing_df)}")
        print(f"新生成: {len(new_df)}")
    else:
        combined_df = new_df
        print(f"仅新数据: {len(new_df)}")

    print(f"合并后总数: {len(combined_df)}")
    combined_df['omega'] = 1.0 - combined_df['M_trim']

    # 4. 提取帕累托前沿
    print(f"\n{'='*70}")
    print("提取帕累托前沿")
    print(f"{'='*70}")

    pareto_df = extract_pareto_front(combined_df)
    print(f"帕累托前沿设计: {len(pareto_df)} ({len(pareto_df) / len(combined_df) * 100:.2f}%)")

    # 5. 密集化帕累托区域（可选）
    if args.n_densify > 0:
        dense_df = densify_pareto_region(pareto_df, predictor, args.n_densify)
        if len(dense_df) > 0:
            dense_df['omega'] = 1.0 - dense_df['M_trim']
            combined_df = pd.concat([combined_df, dense_df], ignore_index=True)
            pareto_df = extract_pareto_front(combined_df)
            print(f"密集化后帕累托前沿: {len(pareto_df)}")
            print(f"最终数据集大小: {len(combined_df)}")

    # 6. 保存结果
    print(f"\n{'='*70}")
    print("保存结果")
    print(f"{'='*70}")

    os.makedirs(args.output, exist_ok=True)
    combined_path = os.path.join(args.output, 'all_designs_hybrid.csv')
    pareto_path = os.path.join(args.output, 'pareto_front_hybrid.csv')
    combined_df.to_csv(combined_path, index=False)
    pareto_df.to_csv(pareto_path, index=False)

    print(f"[✓] 所有设计: {combined_path}")
    print(f"[✓] 帕累托前沿: {pareto_path}")

    # 7. 可视化
    plot_pareto_with_paths(combined_df, pareto_df, args.output)


def main():
    parser = argparse.ArgumentParser(
        description='CGLT 多目标优化统一脚本（bayesian / hybrid）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', type=str, default='hybrid',
                       choices=['hybrid', 'bayesian'],
                       help='运行模式：hybrid(默认) 或 bayesian')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='模型检查点目录')
    parser.add_argument('--results-csv', type=str, default='results.csv',
                       help='现有结果CSV（默认: results.csv）')
    parser.add_argument('--n-trials', type=int, default=500,
                       help='贝叶斯优化试验次数（默认: 500）')
    parser.add_argument('--n-densify', type=int, default=5,
                       help='每个帕累托设计的密集化样本数（默认: 5）')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--output', type=str, default='./moo_results',
                       help='输出目录')
    parser.add_argument('--timeout', type=int, default=None,
                       help='优化超时时间(秒)，当前版本中保留兼容参数')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("CGLT 多目标优化统一脚本")
    print(f"{'='*70}")
    print(f"运行模式: {args.mode}")
    print(f"模型路径: {args.checkpoint_dir}")
    print(f"新增试验: {args.n_trials}")
    if args.mode == 'hybrid':
        print(f"现有数据: {args.results_csv}")
        print(f"密集化因子: {args.n_densify}")
    print(f"输出目录: {args.output}")
    print(f"{'='*70}")

    # 1. 加载代理模型
    print(f"\n{'='*70}")
    print("加载深度学习模型")
    print(f"{'='*70}")
    predictor = SurrogatePredictor(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    print("[✓] 模型加载成功!")

    if args.mode == 'bayesian':
        run_mode_bayesian(args, predictor)
    else:
        run_mode_hybrid(args, predictor)


if __name__ == "__main__":
    main()

