

import torch
import torch.nn as nn
import numpy as np
import optuna
from typing import Dict, Tuple, Callable, Optional
import warnings

from utils.geometry_constraints import (
    compute_wr, compute_w, compute_mass_trimming,
    validate_design_parameters, assemble_row,
    W_MIN, WR_MIN
)

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False
    print("[INFO] fastdtw not available, using built-in DTW implementation")



def extract_pbs(curve: np.ndarray) -> float:
   
    if len(curve) == 0:
        return 0.0
    return float(np.max(curve[:, 1]))


def extract_ncl(curve: np.ndarray) -> float:
    
    if len(curve) < 3:
        return 0.0

    forces = curve[:, 1]

   
    for i in range(1, len(forces) - 1):
        if forces[i] > forces[i-1] and forces[i] > forces[i+1]:
            return float(forces[i])

  
    return float(np.max(forces))


def extract_nca(curve: np.ndarray) -> float:
    
    if len(curve) < 2:
        return 0.0


    displacement = curve[:, 0]
    force = curve[:, 1]
    area = float(np.trapz(force, displacement))

    return area


def extract_all_indicators(curve: np.ndarray) -> Dict[str, float]:
   
    return {
        'PBS': extract_pbs(curve),
        'NCL': extract_ncl(curve),
        'NCA': extract_nca(curve)
    }



def _dtw_builtin(curve1: np.ndarray, curve2: np.ndarray) -> float:
    
    n, m = len(curve1), len(curve2)


    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(curve1[i - 1] - curve2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    return float(dtw_matrix[n, m])


def compute_dtw_distance(curve1: np.ndarray, curve2: np.ndarray) -> float:
    
    if FASTDTW_AVAILABLE:
        # 使用fastdtw（更快）
        try:
            distance, _ = fastdtw(curve1, curve2, dist=euclidean)
            return float(distance)
        except:
            # fastdtw失败，降级到内置实现
            return _dtw_builtin(curve1, curve2)
    else:
        # 使用内置DTW实现
        return _dtw_builtin(curve1, curve2)


def compute_smooth_l1_loss(curve1: np.ndarray, curve2: np.ndarray, beta: float = 1.0) -> float:
   
    if len(curve1) != len(curve2):
        return float('inf')

    diff = np.abs(curve1 - curve2)
    loss = np.where(diff < beta,
                    0.5 * diff ** 2 / beta,
                    diff - 0.5 * beta)

    return float(np.mean(loss))


def compute_indicator_consistency_loss(
    pred_indicators: Dict[str, float],
    target_indicators: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
   
    if weights is None:
        weights = {'PBS': 1.0, 'NCL': 1.0, 'NCA': 1.0}

    total_loss = 0.0
    for key in ['PBS', 'NCL', 'NCA']:
        if key in pred_indicators and key in target_indicators:
            pred_val = pred_indicators[key]
            target_val = target_indicators[key]

      
            if abs(target_val) > 1e-6:
                relative_error = abs(pred_val - target_val) / abs(target_val)
            else:
                relative_error = abs(pred_val - target_val)

            total_loss += weights[key] * relative_error

    return total_loss



class ConstrainedBayesianOptimizer:
   

    def __init__(
        self,
        surrogate_predictor: Callable,
        target_curve: np.ndarray,
        target_indicators: Optional[Dict[str, float]] = None,
        device: str = 'cuda'
    ):
      
        self.surrogate_predictor = surrogate_predictor
        self.target_curve = target_curve
        self.device = device

        # 如果未提供目标指标，从目标曲线提取
        if target_indicators is None:
            self.target_indicators = extract_all_indicators(target_curve)
        else:
            self.target_indicators = target_indicators

        # 参数范围 (基于CGLT设计空间)
        self.param_ranges = {
            'H1': (24, 36, 2),      # (min, max, step) - H必须是偶数
            'L1': (1, 10, 1),
            'a1': (30, 80, 5),
            'r1': (10, 18, 1),
            'H2': (24, 36, 2),
            'L2': (1, 10, 1),
            'a2': (30, 80, 5),
            # r1在两端共享
        }

        # 损失权重
        self.loss_weights = {
            'dtw': 1.0,
            'smooth_l1': 0.5,
            'indicator': 0.3,
            'mass_penalty': 0.1  # 质量惩罚（鼓励轻量化）
        }

        # 优化历史
        self.optimization_history = []

    def _validate_constraints(self, params: Dict[str, float]) -> Tuple[bool, str, Dict]:
        """
        验证所有约束

        返回:
            (是否有效, 失败原因, 几何信息)
        """
        H1, L1, a1, r1 = params['H1'], params['L1'], params['a1'], params['r1']
        H2, L2, a2 = params['H2'], params['L2'], params['a2']

        # 计算r2值
        r2_1 = compute_wr(H1, r1, a1)
        r2_2 = compute_wr(H2, r1, a2)

        if r2_1 <= 0 or r2_2 <= 0:
            return False, "r2计算失败", {}

        # 计算web宽度
        w1 = compute_w(L1, r1, r2_1, a1)
        w2 = compute_w(L2, r1, r2_2, a2)

        # 硬约束检查
        if r2_1 <= WR_MIN:
            return False, f"r2_1={r2_1:.2f} ≤ {WR_MIN}", {}

        if r2_2 <= WR_MIN:
            return False, f"r2_2={r2_2:.2f} ≤ {WR_MIN}", {}

        if w1 < W_MIN:
            return False, f"w1={w1:.2f} < {W_MIN}", {}

        if w2 < W_MIN:
            return False, f"w2={w2:.2f} < {W_MIN}", {}

        # 组装几何信息
        geo_info = assemble_row(H1, L1, r1, a1, H2, L2, r2_1, a2, r2_2)

        return True, "valid", geo_info

    def _compute_total_loss(
        self,
        pred_curve: np.ndarray,
        geo_info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算总损失

        返回:
            (总损失, 损失分解字典)
        """
        # 1. DTW损失
        dtw_loss = compute_dtw_distance(pred_curve, self.target_curve)

        # 2. Smooth L1损失
        smooth_l1_loss = compute_smooth_l1_loss(pred_curve, self.target_curve)

        # 3. 指标一致性损失
        pred_indicators = extract_all_indicators(pred_curve)
        indicator_loss = compute_indicator_consistency_loss(
            pred_indicators, self.target_indicators
        )

        # 4. 质量惩罚（鼓励减重）
        # 注意: M_trim越大越好（削减越多），所以用负值作为惩罚
        mass_penalty = -geo_info['M_trim']  # 负号表示鼓励增大M_trim

        # 总损失
        total_loss = (
            self.loss_weights['dtw'] * dtw_loss +
            self.loss_weights['smooth_l1'] * smooth_l1_loss +
            self.loss_weights['indicator'] * indicator_loss +
            self.loss_weights['mass_penalty'] * mass_penalty
        )

        loss_breakdown = {
            'total': total_loss,
            'dtw': dtw_loss,
            'smooth_l1': smooth_l1_loss,
            'indicator': indicator_loss,
            'mass_penalty': mass_penalty,
            'M_trim': geo_info['M_trim']
        }

        return total_loss, loss_breakdown

    def _objective_function(self, trial: optuna.Trial) -> float:
        """
        Optuna目标函数

        参数:
            trial: Optuna试验对象

        返回:
            损失值（越小越好）
        """
        # 采样参数（支持浮点数，精度0.01）
        params = {}
        params['H1'] = round(trial.suggest_float('H1', 24.0, 36.0), 2)
        params['L1'] = round(trial.suggest_float('L1', 1.0, 10.0), 2)
        params['a1'] = round(trial.suggest_float('a1', 30.0, 80.0), 2)
        params['r1'] = round(trial.suggest_float('r1', 10.0, 18.0), 2)

        params['H2'] = round(trial.suggest_float('H2', 24.0, 36.0), 2)
        params['L2'] = round(trial.suggest_float('L2', 1.0, 10.0), 2)
        params['a2'] = round(trial.suggest_float('a2', 30.0, 80.0), 2)

        # 验证约束
        valid, reason, geo_info = self._validate_constraints(params)

        if not valid:
            # 返回大损失值以惩罚违反约束的解
            return 1e6

        # 构造参数向量 (格式: [H1, L1, a1, r1, H2, L2, a2, r1])
        param_vector = [
            params['H1'], params['L1'], params['a1'], params['r1'],
            params['H2'], params['L2'], params['a2'], params['r1']
        ]

        try:
            # 使用代理模型预测
            pred_curve = self.surrogate_predictor(param_vector)

            # 计算总损失
            total_loss, loss_breakdown = self._compute_total_loss(pred_curve, geo_info)

            # 记录历史
            record = {
                'trial': trial.number,
                'params': params.copy(),
                'geo_info': geo_info,
                'loss_breakdown': loss_breakdown,
                'pred_indicators': extract_all_indicators(pred_curve)
            }
            self.optimization_history.append(record)

            # 设置用户属性以便后续分析
            trial.set_user_attr('dtw_loss', loss_breakdown['dtw'])
            trial.set_user_attr('M_trim', geo_info['M_trim'])
            trial.set_user_attr('valid', True)

            return total_loss

        except Exception as e:
            print(f"预测失败: {e}")
            return 1e6

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        执行贝叶斯优化

        参数:
            n_trials: 试验次数
            timeout: 超时时间（秒）
            show_progress: 是否显示进度

        返回:
            最优结果字典
        """
        # 创建Optuna研究
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # 执行优化
        study.optimize(
            self._objective_function,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )

        # 获取最优结果
        best_trial = study.best_trial
        best_params = best_trial.params

        # 重新验证最优解
        valid, reason, geo_info = self._validate_constraints(best_params)

        if not valid:
            raise ValueError(f"最优解验证失败: {reason}")

        # 重新预测最优曲线
        param_vector = [
            best_params['H1'], best_params['L1'], best_params['a1'], best_params['r1'],
            best_params['H2'], best_params['L2'], best_params['a2'], best_params['r1']
        ]

        best_curve = self.surrogate_predictor(param_vector)
        best_indicators = extract_all_indicators(best_curve)

        _, loss_breakdown = self._compute_total_loss(best_curve, geo_info)

        # 构造结果
        result = {
            'best_params': best_params,
            'best_curve': best_curve,
            'geo_info': geo_info,
            'indicators': best_indicators,
            'loss_breakdown': loss_breakdown,
            'n_trials': len(study.trials),
            'study': study,
            'optimization_history': self.optimization_history
        }

        return result

    def set_loss_weights(self, weights: Dict[str, float]):
        """设置损失权重"""
        self.loss_weights.update(weights)

    def get_pareto_front(self, top_k: int = 50) -> list:
        """
        获取帕累托前沿（质量-性能权衡）

        参数:
            top_k: 返回前k个解

        返回:
            帕累托解列表
        """
        # 按总损失排序
        sorted_history = sorted(
            self.optimization_history,
            key=lambda x: x['loss_breakdown']['total']
        )

        # 返回前k个
        return sorted_history[:top_k]



def create_optimizer_from_checkpoint(
    checkpoint_dir: str,
    target_curve: np.ndarray,
    device: str = 'cuda'
) -> ConstrainedBayesianOptimizer:
    """
    从检查点创建优化器

    参数:
        checkpoint_dir: 检查点目录
        target_curve: 目标曲线
        device: 设备

    返回:
        优化器实例
    """
    import sys
    import os

    # 添加src到路径
    sys.path.append(os.path.join(os.path.dirname(checkpoint_dir), '..', 'src'))

    from models.vae import EnhancedConditionalEncoder, EnhancedConditionalDecoder, EnhancedANNMapper
    from utils.dataloader import load_normalization_params
    from utils.utils import load_model

    # 加载模型
    encoder = EnhancedConditionalEncoder(z_dim=16, feature_dim=8).to(device)
    decoder = EnhancedConditionalDecoder(z_dim=16, feature_dim=8).to(device)
    mapper = EnhancedANNMapper(feature_dim=8, z_dim=16).to(device)

    load_model(encoder, os.path.join(checkpoint_dir, 'best_encoder.pt'), device)
    load_model(decoder, os.path.join(checkpoint_dir, 'best_decoder.pt'), device)
    load_model(mapper, os.path.join(checkpoint_dir, 'best_mapper.pt'), device)

    encoder.eval()
    decoder.eval()
    mapper.eval()

    # 加载归一化参数
    norm_params = load_normalization_params(checkpoint_dir)

    # 定义预测函数
    def surrogate_predictor(params_raw):
        """params_raw: [H1, L1, a1, r1, H2, L2, a2, r1]"""
        # 归一化
        scaler = norm_params[0]
        params_norm = scaler.transform([params_raw])[0]
        params_tensor = torch.tensor(params_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            z = mapper(params_tensor)
            curve = decoder(z, params_tensor)

        return curve[0].cpu().numpy()

    # 创建优化器
    optimizer = ConstrainedBayesianOptimizer(
        surrogate_predictor=surrogate_predictor,
        target_curve=target_curve,
        device=device
    )

    return optimizer


if __name__ == "__main__":
    print("多约束贝叶斯优化器模块")
    print("=" * 60)
    print("功能:")
    print("  - 硬约束验证 (r2 > 7.5mm, w ≥ 10mm)")
    print("  - 质量削减优化 (M_trim)")
    print("  - 轨迹级损失 (DTW + SmoothL1)")
    print("  - 性能指标一致性 (PBS/NCL/NCA)")
    print("=" * 60)
