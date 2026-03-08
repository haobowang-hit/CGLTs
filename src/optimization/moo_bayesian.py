"""
多目标贝叶斯优化主动搜索 - CGLT设计优化
================================================
整合深度学习模型、贝叶斯优化和多目标帕累托分析

优化目标:
  1. 最大化 PBS (Pre-Buckling Stiffness) - 屈曲前刚度
  2. 最大化 NCL (Normalized Critical Load) - 归一化临界载荷
  3. 最大化 NCA (Normalized Critical Area) - 能量吸收
  4. 最小化 M_trim - 质量削减率（轻量化）

使用:
  python multi_objective_bayesian_search.py --checkpoint-dir ../src/checkpoints --n-trials 200 --device cuda

作者: 王浩博
单位: 哈尔滨工业大学
"""

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

# 确保优先从当前目录（repo）导入模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 将当前目录放在最前面，优先级最高
sys.path.insert(0, SCRIPT_DIR)

# 导入代理预测器（会自动添加src到路径）
from optimization.surrogate import SurrogatePredictor, extract_all_indicators

# 强制重新导入geometry_constraints，确保从repo目录导入
import importlib.util
from utils.geometry_constraints import (
    compute_wr, compute_w, section_area, compute_mass_trimming, assemble_row
)


plt.rcParams.update({
    'font.family': ['Times New Roman'],
    'font.serif': ['Times New Roman'],
    'font.size': 38,
    'axes.titlesize': 44,
    'axes.labelsize': 42,
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'legend.fontsize': 34,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold'
})

def style_axis(ax):
    """统一坐标轴样式 - 无网格"""
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)



class MultiObjectiveBayesianOptimizer:
    """
    多目标贝叶斯优化器

    使用深度学习代理模型进行性能预测
    使用NSGA-II进行多目标优化
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = 'cuda',
        objectives: List[str] = ['PBS', 'NCL', 'NCA', 'M_trim'],
        directions: List[str] = ['maximize', 'maximize', 'maximize', 'minimize']
    ):
        """
        初始化优化器

        参数:
            checkpoint_dir: 模型检查点目录
            device: 计算设备
            objectives: 优化目标列表
            directions: 优化方向 ('maximize' 或 'minimize')
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.objectives = objectives
        self.directions = directions

        # 加载代理模型
        print(f"\n{'='*70}")
        print("加载深度学习代理模型")
        print(f"{'='*70}")
        self.predictor = SurrogatePredictor(
            checkpoint_dir=checkpoint_dir,
            device=device
        )
        print("[✓] 模型加载成功!\n")

        # 定义设计空间（基于CGLT约束）
        self.param_ranges = {
            'H1': (24, 36),      # 截面高度必须是偶数
            'L1': (1, 10),       # Lumbus宽度
            'a1': (30, 80),      # 角度
            'r1': (10, 18),      # 曲率半径
            'H2': (24, 36),
            'L2': (1, 10),
            'a2': (30, 80),
            'r2': (10, 18),      # r2独立于r1
        }

        # 优化历史
        self.history: List[Dict] = []
        self.valid_designs: List[Dict] = []
        self.pareto_front: List[Dict] = []

    def _validate_design(self, params: Dict[str, float]) -> Tuple[bool, Dict]:
        """
        验证设计约束

        参数:
            params: 设计参数字典

        返回:
            (is_valid, geo_info)
        """
        # 使用assemble_row进行完整验证
        row = assemble_row(
            params['H1'], params['L1'], params['r1'], params['a1'],
            params['H2'], params['L2'], params['r2'], params['a2']
        )

        is_valid = (row['valid_ends'] == 1)

        geo_info = {
            'w1': row['w1'],
            'w2': row['w2'],
            'wr1': row['wr1'],
            'wr2': row['wr2'],
            'area1': row['area1'],
            'area2': row['area2'],
            'M_trim': row['M_trim'],
            'valid_ends': row['valid_ends']
        }

        return is_valid, geo_info

    def _predict_performance(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        预测性能指标

        参数:
            params: 设计参数

        返回:
            性能指标字典
        """
        # 构造参数向量
        param_vector = [
            params['H1'], params['L1'], params['a1'], params['r1'],
            params['H2'], params['L2'], params['a2'], params['r1']  # 注意：模型输入使用r1
        ]

        # 预测曲线
        curve = self.predictor.predict_curve(param_vector)

        # 提取性能指标
        indicators = extract_all_indicators(curve)

        return indicators

    def _objective_function(self, trial: optuna.Trial) -> Tuple[float, ...]:
        """
        Optuna目标函数

        参数:
            trial: Optuna试验对象

        返回:
            目标值元组
        """
        # 采样设计参数
        params = {}
        params['H1'] = trial.suggest_int('H1', 24, 36, step=2)  # 偶数
        params['L1'] = trial.suggest_float('L1', 1, 10)
        params['a1'] = trial.suggest_float('a1', 30, 80)
        params['r1'] = trial.suggest_float('r1', 10, 18)

        params['H2'] = trial.suggest_int('H2', 24, 36, step=2)
        params['L2'] = trial.suggest_float('L2', 1, 10)
        params['a2'] = trial.suggest_float('a2', 30, 80)
        params['r2'] = trial.suggest_float('r2', 10, 18)

        # 验证约束
        is_valid, geo_info = self._validate_design(params)

        if not is_valid:
            # 约束违反，返回惩罚值
            trial.set_user_attr('valid', False)
            if 'maximize' in self.directions:
                penalty = tuple([-1e6 if d == 'maximize' else 1e6 for d in self.directions])
            else:
                penalty = tuple([1e6] * len(self.objectives))
            return penalty

        # 预测性能
        try:
            performance = self._predict_performance(params)
        except Exception as e:
            print(f"[警告] 预测失败: {e}")
            trial.set_user_attr('valid', False)
            penalty = tuple([-1e6 if d == 'maximize' else 1e6 for d in self.directions])
            return penalty

        # 记录结果
        record = {
            'trial': trial.number,
            'params': params.copy(),
            'geo_info': geo_info,
            'performance': performance,
            'valid': True
        }
        self.history.append(record)
        self.valid_designs.append(record)

        # 存储用户属性
        trial.set_user_attr('valid', True)
        trial.set_user_attr('PBS', performance['PBS'])
        trial.set_user_attr('NCL', performance['NCL'])
        trial.set_user_attr('NCA', performance['NCA'])
        trial.set_user_attr('M_trim', geo_info['M_trim'])

        # 构造目标值（注意符号）
        objectives = []
        for obj, direction in zip(self.objectives, self.directions):
            if obj == 'M_trim':
                value = geo_info[obj]
            else:
                value = performance[obj]

            # Optuna多目标优化默认最小化，所以最大化目标需要取负
            if direction == 'maximize':
                objectives.append(-value)  # 取负变为最小化
            else:
                objectives.append(value)

        return tuple(objectives)

    def optimize(
        self,
        n_trials: int = 200,
        n_startup_trials: int = 50,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> optuna.Study:
        """
        执行多目标优化

        参数:
            n_trials: 试验次数
            n_startup_trials: 启动试验次数（随机采样）
            timeout: 超时时间（秒）
            show_progress: 显示进度条

        返回:
            Optuna Study对象
        """
        print(f"{'='*70}")
        print("开始多目标贝叶斯优化")
        print(f"{'='*70}")
        print(f"优化目标: {', '.join(self.objectives)}")
        print(f"优化方向: {', '.join(self.directions)}")
        print(f"试验次数: {n_trials}")
        print(f"设备: {self.device}")
        print(f"{'='*70}\n")

        # 创建Optuna study（多目标优化）
        sampler = NSGAIISampler(
            population_size=50,
            mutation_prob=0.1,
            crossover_prob=0.9
        )

        study = optuna.create_study(
            directions=self.directions,
            sampler=sampler
        )

        # 执行优化
        if show_progress:
            study.optimize(
                self._objective_function,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
        else:
            study.optimize(
                self._objective_function,
                n_trials=n_trials,
                timeout=timeout
            )

        print(f"\n{'='*70}")
        print("优化完成")
        print(f"{'='*70}")
        print(f"总试验次数: {len(study.trials)}")
        print(f"有效设计: {len(self.valid_designs)} ({len(self.valid_designs)/len(study.trials)*100:.1f}%)")
        print(f"帕累托前沿设计: {len(study.best_trials)}")

        return study

    def extract_pareto_front(self, study: optuna.Study) -> pd.DataFrame:
        """
        提取帕累托前沿

        参数:
            study: Optuna Study对象

        返回:
            帕累托前沿DataFrame
        """
        print(f"\n{'='*70}")
        print("提取帕累托前沿")
        print(f"{'='*70}")

        pareto_trials = study.best_trials
        pareto_designs = []

        for trial in pareto_trials:
            if not trial.user_attrs.get('valid', False):
                continue

            design = {
                'trial': trial.number,
                **trial.params,
                'PBS': trial.user_attrs['PBS'],
                'NCL': trial.user_attrs['NCL'],
                'NCA': trial.user_attrs['NCA'],
                'M_trim': trial.user_attrs['M_trim']
            }
            pareto_designs.append(design)

        df = pd.DataFrame(pareto_designs)

        print(f"帕累托前沿设计数: {len(df)}")
        if len(df) > 0:
            print(f"\nPBS 范围: [{df['PBS'].min():.4f}, {df['PBS'].max():.4f}]")
            print(f"NCL 范围: [{df['NCL'].min():.4f}, {df['NCL'].max():.4f}]")
            print(f"NCA 范围: [{df['NCA'].min():.4f}, {df['NCA'].max():.4f}]")
            print(f"M_trim 范围: [{df['M_trim'].min():.6f}, {df['M_trim'].max():.6f}]")

        self.pareto_front = pareto_designs

        return df

    def save_results(self, study: optuna.Study, output_dir: str = './moo_results'):
        """
        保存优化结果

        参数:
            study: Optuna Study对象
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("保存结果")
        print(f"{'='*70}")

        # 1. 保存所有有效设计
        if len(self.valid_designs) > 0:
            valid_data = []
            for record in self.valid_designs:
                row = {
                    'trial': record['trial'],
                    **record['params'],
                    'PBS': record['performance']['PBS'],
                    'NCL': record['performance']['NCL'],
                    'NCA': record['performance']['NCA'],
                    'M_trim': record['geo_info']['M_trim'],
                    'w1': record['geo_info']['w1'],
                    'w2': record['geo_info']['w2'],
                    'wr1': record['geo_info']['wr1'],
                    'wr2': record['geo_info']['wr2'],
                    'area1': record['geo_info']['area1'],
                    'area2': record['geo_info']['area2']
                }
                valid_data.append(row)

            valid_df = pd.DataFrame(valid_data)
            valid_path = os.path.join(output_dir, 'all_valid_designs.csv')
            valid_df.to_csv(valid_path, index=False)
            print(f"[✓] 所有有效设计: {valid_path}")

        # 2. 保存帕累托前沿
        pareto_df = self.extract_pareto_front(study)
        if len(pareto_df) > 0:
            pareto_path = os.path.join(output_dir, 'pareto_front_designs.csv')
            pareto_df.to_csv(pareto_path, index=False)
            print(f"[✓] 帕累托前沿: {pareto_path}")

        # 3. 保存推荐设计（使用多种策略）
        if len(pareto_df) > 0:
            recommendations = self._compute_recommendations(pareto_df)
            rec_path = os.path.join(output_dir, 'recommended_designs.csv')
            rec_df = pd.DataFrame(recommendations)
            rec_df.to_csv(rec_path, index=False)
            print(f"[✓] 推荐设计: {rec_path}")

        print(f"\n结果已保存到: {output_dir}")

    def _compute_recommendations(self, pareto_df: pd.DataFrame) -> List[Dict]:
        """
        计算推荐设计（多种策略）

        参数:
            pareto_df: 帕累托前沿DataFrame

        返回:
            推荐设计列表
        """
        recommendations = []

        # 归一化
        pbs_norm = (pareto_df['PBS'] - pareto_df['PBS'].min()) / (pareto_df['PBS'].max() - pareto_df['PBS'].min() + 1e-8)
        ncl_norm = (pareto_df['NCL'] - pareto_df['NCL'].min()) / (pareto_df['NCL'].max() - pareto_df['NCL'].min() + 1e-8)
        nca_norm = (pareto_df['NCA'] - pareto_df['NCA'].min()) / (pareto_df['NCA'].max() - pareto_df['NCA'].min() + 1e-8)
        mtrim_norm = (pareto_df['M_trim'] - pareto_df['M_trim'].min()) / (pareto_df['M_trim'].max() - pareto_df['M_trim'].min() + 1e-8)

        # 策略1: 理想点法
        ideal_point = np.array([1.0, 1.0, 1.0, 0.0])  # PBS↑, NCL↑, NCA↑, M_trim↓
        distances = np.sqrt(
            (pbs_norm - ideal_point[0])**2 +
            (ncl_norm - ideal_point[1])**2 +
            (nca_norm - ideal_point[2])**2 +
            (mtrim_norm - ideal_point[3])**2
        )
        idx_ideal = distances.idxmin()
        rec = pareto_df.iloc[idx_ideal].to_dict()
        rec['strategy'] = '理想点法 (Ideal Point)'
        rec['score'] = float(1.0 / (distances.iloc[idx_ideal] + 1e-8))
        recommendations.append(rec)

        weighted_sum = 0.3 * pbs_norm + 0.3 * ncl_norm + 0.2 * nca_norm + 0.2 * (1 - mtrim_norm)
        idx_weighted = weighted_sum.idxmax()
        rec = pareto_df.iloc[idx_weighted].to_dict()
        rec['strategy'] = '加权和法-均衡 (Weighted Sum-Balanced)'
        rec['score'] = float(weighted_sum.iloc[idx_weighted])
        recommendations.append(rec)

        weighted_sum_strength = 0.5 * pbs_norm + 0.3 * ncl_norm + 0.1 * nca_norm + 0.1 * (1 - mtrim_norm)
        idx_strength = weighted_sum_strength.idxmax()
        rec = pareto_df.iloc[idx_strength].to_dict()
        rec['strategy'] = '加权和法-强度优先 (Weighted Sum-Strength)'
        rec['score'] = float(weighted_sum_strength.iloc[idx_strength])
        recommendations.append(rec)

        weighted_sum_light = 0.2 * pbs_norm + 0.2 * ncl_norm + 0.1 * nca_norm + 0.5 * (1 - mtrim_norm)
        idx_light = weighted_sum_light.idxmax()
        rec = pareto_df.iloc[idx_light].to_dict()
        rec['strategy'] = '加权和法-轻量化优先 (Weighted Sum-Lightweight)'
        rec['score'] = float(weighted_sum_light.iloc[idx_light])
        recommendations.append(rec)

        # 策略5: 膝点法
        origin = np.array([0.0, 0.0, 0.0, 1.0])
        distances_origin = np.sqrt(
            pbs_norm**2 + ncl_norm**2 + nca_norm**2 + (1 - mtrim_norm)**2
        )
        idx_knee = distances_origin.idxmax()
        rec = pareto_df.iloc[idx_knee].to_dict()
        rec['strategy'] = '膝点法 (Knee Point)'
        rec['score'] = float(distances_origin.iloc[idx_knee])
        recommendations.append(rec)

        return recommendations



def visualize_results(
    pareto_df: pd.DataFrame,
    all_valid_df: pd.DataFrame,
    output_dir: str = './moo_results'
):
    """
    生成可视化结果

    参数:
        pareto_df: 帕累托前沿DataFrame
        all_valid_df: 所有有效设计DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("生成可视化")
    print(f"{'='*70}")

    # 图1: 3D帕累托前沿 (PBS, NCL, M_trim)
    print("生成 3D 帕累托前沿...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制所有有效设计
    scatter_all = ax.scatter(
        all_valid_df['PBS'],
        all_valid_df['NCL'],
        all_valid_df['M_trim'],
        c=all_valid_df['NCA'],
        cmap='plasma',
        s=100,
        alpha=0.3,
        edgecolors='none',
        label='All Valid Designs'
    )

    # 绘制帕累托前沿
    scatter_pareto = ax.scatter(
        pareto_df['PBS'],
        pareto_df['NCL'],
        pareto_df['M_trim'],
        c=pareto_df['NCA'],
        cmap='plasma',
        s=300,
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        label='Pareto Front',
        marker='D'
    )

    # 添加色标
    cbar = plt.colorbar(scatter_pareto, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('NCA', rotation=270, labelpad=50)
    cbar.ax.tick_params(labelsize=32)

    # 坐标轴标签
    ax.set_xlabel('PBS', fontweight='bold', labelpad=25)
    ax.set_ylabel('NCL', fontweight='bold', labelpad=25)
    ax.set_zlabel(r'$\omega$', fontweight='bold', labelpad=35)
    ax.set_title('3D Pareto Front: PBS-NCL-$\omega$-NCA', fontweight='bold', pad=35)

    # 图例
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1.0), framealpha=0.7,
              fancybox=False, edgecolor='black')

    # 调整视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'pareto_3d_front.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] 3D图: {fig_path}")

    # 图2: 2D投影 (2x2布局)
    print("生成 2D 投影...")
    fig, axes = plt.subplots(2, 2, figsize=(28, 24))

    # (1) PBS vs NCL
    ax = axes[0, 0]
    ax.scatter(all_valid_df['PBS'], all_valid_df['NCL'],
               c=all_valid_df['NCA'], cmap='plasma', s=100, alpha=0.3, edgecolors='none')
    scatter = ax.scatter(pareto_df['PBS'], pareto_df['NCL'],
                        c=pareto_df['NCA'], cmap='plasma', s=300, alpha=0.9,
                        edgecolors='black', linewidths=2, marker='D')
    ax.set_xlabel('PBS', fontweight='bold')
    ax.set_ylabel('NCL', fontweight='bold')
    ax.set_title('PBS vs NCL', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('NCA', rotation=270, labelpad=40)
    cbar.ax.tick_params(labelsize=28)
    ax.legend(['All Designs', 'Pareto Front'], loc='lower right', framealpha=0.7)
    style_axis(ax)

    # (2) M_trim vs PBS
    ax = axes[0, 1]
    ax.scatter(all_valid_df['M_trim'], all_valid_df['PBS'],
               c=all_valid_df['NCA'], cmap='plasma', s=100, alpha=0.3, edgecolors='none')
    scatter = ax.scatter(pareto_df['M_trim'], pareto_df['PBS'],
                        c=pareto_df['NCA'], cmap='plasma', s=300, alpha=0.9,
                        edgecolors='black', linewidths=2, marker='D')
    ax.set_xlabel(r'$\omega$', fontweight='bold')
    ax.set_ylabel('PBS', fontweight='bold')
    ax.set_title(r'$\omega$ vs PBS', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('NCA', rotation=270, labelpad=40)
    cbar.ax.tick_params(labelsize=28)
    ax.legend(['All Designs', 'Pareto Front'], loc='lower right', framealpha=0.7)
    style_axis(ax)

    # (3) M_trim vs NCL
    ax = axes[1, 0]
    ax.scatter(all_valid_df['M_trim'], all_valid_df['NCL'],
               c=all_valid_df['NCA'], cmap='plasma', s=100, alpha=0.3, edgecolors='none')
    scatter = ax.scatter(pareto_df['M_trim'], pareto_df['NCL'],
                        c=pareto_df['NCA'], cmap='plasma', s=300, alpha=0.9,
                        edgecolors='black', linewidths=2, marker='D')
    ax.set_xlabel(r'$\omega$', fontweight='bold')
    ax.set_ylabel('NCL', fontweight='bold')
    ax.set_title(r'$\omega$ vs NCL', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('NCA', rotation=270, labelpad=40)
    cbar.ax.tick_params(labelsize=28)
    ax.legend(['All Designs', 'Pareto Front'], loc='lower right', framealpha=0.7)
    style_axis(ax)

    # (4) NCA vs PBS
    ax = axes[1, 1]
    ax.scatter(all_valid_df['NCA'], all_valid_df['PBS'],
               c=all_valid_df['M_trim'], cmap='plasma', s=100, alpha=0.3, edgecolors='none')
    scatter = ax.scatter(pareto_df['NCA'], pareto_df['PBS'],
                        c=pareto_df['M_trim'], cmap='plasma', s=300, alpha=0.9,
                        edgecolors='black', linewidths=2, marker='D')
    ax.set_xlabel('NCA', fontweight='bold')
    ax.set_ylabel('PBS', fontweight='bold')
    ax.set_title('NCA vs PBS', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\omega$', rotation=270, labelpad=40)
    cbar.ax.tick_params(labelsize=28)
    ax.legend(['All Designs', 'Pareto Front'], loc='lower right', framealpha=0.7)
    style_axis(ax)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'pareto_2d_projections.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] 2D投影: {fig_path}")

    # 图3: 优化历史
    print("生成优化历史...")
    fig, axes = plt.subplots(2, 2, figsize=(28, 20))

    trials = all_valid_df['trial'].values

    ax = axes[0, 0]
    ax.plot(trials, all_valid_df['PBS'].values, 'o-', alpha=0.6, linewidth=2, markersize=8)
    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel('PBS', fontweight='bold')
    ax.set_title('PBS Optimization History', fontweight='bold')
    ax.axhline(y=pareto_df['PBS'].max(), color='r', linestyle='--', linewidth=2, label='Pareto Max')
    ax.legend()
    style_axis(ax)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(trials, all_valid_df['NCL'].values, 'o-', alpha=0.6, linewidth=2, markersize=8)
    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel('NCL', fontweight='bold')
    ax.set_title('NCL Optimization History', fontweight='bold')
    ax.axhline(y=pareto_df['NCL'].max(), color='r', linestyle='--', linewidth=2, label='Pareto Max')
    ax.legend()
    style_axis(ax)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(trials, all_valid_df['NCA'].values, 'o-', alpha=0.6, linewidth=2, markersize=8)
    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel('NCA', fontweight='bold')
    ax.set_title('NCA Optimization History', fontweight='bold')
    ax.axhline(y=pareto_df['NCA'].max(), color='r', linestyle='--', linewidth=2, label='Pareto Max')
    ax.legend()
    style_axis(ax)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(trials, all_valid_df['M_trim'].values, 'o-', alpha=0.6, linewidth=2, markersize=8)
    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel(r'$\omega$', fontweight='bold')
    ax.set_title(r'$\omega$ Optimization History', fontweight='bold')
    ax.axhline(y=pareto_df['M_trim'].min(), color='r', linestyle='--', linewidth=2, label='Pareto Min')
    ax.legend()
    style_axis(ax)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'optimization_history.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] 优化历史: {fig_path}")

    print(f"\n可视化完成! 所有图片已保存到: {output_dir}")



def main():
    parser = argparse.ArgumentParser(
        description='CGLT多目标贝叶斯优化主动搜索',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础运行
  python multi_objective_bayesian_search.py --checkpoint-dir ../src/checkpoints --n-trials 200

  # GPU加速
  python multi_objective_bayesian_search.py --checkpoint-dir ../src/checkpoints --n-trials 500 --device cuda

  # 长时间优化
  python multi_objective_bayesian_search.py --checkpoint-dir ../src/checkpoints --n-trials 1000 --device cuda --output ./moo_results_v2
        """
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='模型检查点目录 (例: ../src/checkpoints)'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=200,
        help='优化试验次数 (默认: 200)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备 (默认: cuda)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./moo_results',
        help='输出目录 (默认: ./moo_results)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='优化超时时间(秒), None表示无限制'
    )

    args = parser.parse_args()

    # 打印配置
    print(f"\n{'='*70}")
    print("CGLT 多目标贝叶斯优化主动搜索")
    print(f"{'='*70}")
    print(f"模型路径: {args.checkpoint_dir}")
    print(f"试验次数: {args.n_trials}")
    print(f"计算设备: {args.device}")
    print(f"输出目录: {args.output}")
    print(f"{'='*70}\n")

    # 检查模型路径
    if not os.path.exists(args.checkpoint_dir):
        print(f"[错误] 模型检查点目录不存在: {args.checkpoint_dir}")
        sys.exit(1)

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[警告] CUDA不可用，切换到CPU")
        args.device = 'cpu'

    # 初始化优化器
    optimizer = MultiObjectiveBayesianOptimizer(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        objectives=['PBS', 'NCL', 'NCA', 'M_trim'],
        directions=['maximize', 'maximize', 'maximize', 'minimize']
    )

    # 执行优化
    study = optimizer.optimize(
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress=True
    )

    # 保存结果
    optimizer.save_results(study, output_dir=args.output)

    # 提取数据用于可视化
    pareto_df = optimizer.extract_pareto_front(study)

    if len(optimizer.valid_designs) > 0:
        valid_data = []
        for record in optimizer.valid_designs:
            row = {
                'trial': record['trial'],
                **record['params'],
                'PBS': record['performance']['PBS'],
                'NCL': record['performance']['NCL'],
                'NCA': record['performance']['NCA'],
                'M_trim': record['geo_info']['M_trim']
            }
            valid_data.append(row)
        all_valid_df = pd.DataFrame(valid_data)

        # 生成可视化
        visualize_results(pareto_df, all_valid_df, output_dir=args.output)

    # 打印最终摘要
    print(f"\n{'='*70}")
    print("优化完成!")
    print(f"{'='*70}")
    print(f"\n生成的文件:")
    print(f"  - {args.output}/all_valid_designs.csv")
    print(f"  - {args.output}/pareto_front_designs.csv")
    print(f"  - {args.output}/recommended_designs.csv")
    print(f"  - {args.output}/pareto_3d_front.png")
    print(f"  - {args.output}/pareto_2d_projections.png")
    print(f"  - {args.output}/optimization_history.png")
    print(f"\n推荐查看 recommended_designs.csv 以选择最适合您需求的设计!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

