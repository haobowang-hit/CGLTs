"""
CGLT 代理模型预测接口
=====================
封装深度学习模型的预测功能，提供统一的API

功能:
- 从设计参数预测力-位移曲线
- 提取性能指标 (PBS/NCL/NCA)
- 计算DTW距离
- 批量预测

作者: 王浩博
单位: 哈尔滨工业大学
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pickle

# 添加src路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')
sys.path.insert(0, SRC_DIR)

try:
    from models.vae import (
        EnhancedConditionalEncoder,
        EnhancedConditionalDecoder,
        EnhancedANNMapper
    )
    from utils.utils import load_model
except ImportError as e:
    print(f"Warning: Could not import from src: {e}")
    print("Please make sure CGLT/src is in the Python path")



def extract_pbs(curve: np.ndarray) -> float:
    """
    提取PBS (Peak Buckling Strength) - 峰值屈曲强度

    参数:
        curve: [N, 2] 曲线 (位移, 力)

    返回:
        PBS值
    """
    if len(curve) == 0:
        return 0.0
    return float(np.max(curve[:, 1]))


def extract_ncl(curve: np.ndarray) -> float:
    """
    提取NCL (Normalized Critical Load) - 归一化临界载荷

    参数:
        curve: [N, 2] 曲线

    返回:
        NCL值（第一个局部最大值）
    """
    if len(curve) < 3:
        return 0.0

    forces = curve[:, 1]

    # 寻找第一个局部最大值
    for i in range(1, len(forces) - 1):
        if forces[i] > forces[i-1] and forces[i] > forces[i+1]:
            return float(forces[i])

    # 如果没找到局部最大值，返回最大值
    return float(np.max(forces))


def extract_nca(curve: np.ndarray) -> float:
    """
    提取NCA (Normalized Critical Area) - 归一化临界面积
    计算曲线下面积（能量吸收）

    参数:
        curve: [N, 2] 曲线

    返回:
        NCA值（曲线下面积）
    """
    if len(curve) < 2:
        return 0.0

    displacement = curve[:, 0]
    force = curve[:, 1]
    area = float(np.trapz(force, displacement))

    return area


def extract_all_indicators(curve: np.ndarray) -> Dict[str, float]:
    """
    提取所有性能指标

    参数:
        curve: [N, 2] 曲线

    返回:
        字典 {PBS, NCL, NCA}
    """
    return {
        'PBS': extract_pbs(curve),
        'NCL': extract_ncl(curve),
        'NCA': extract_nca(curve)
    }



def compute_dtw_distance(curve1: np.ndarray, curve2: np.ndarray, use_fastdtw: bool = True) -> float:
    """
    计算两条曲线之间的DTW距离

    参数:
        curve1, curve2: [N, 2] 曲线
        use_fastdtw: 是否尝试使用fastdtw（如果已安装）

    返回:
        DTW距离
    """
    # 尝试使用fastdtw（更快）
    if use_fastdtw:
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            distance, _ = fastdtw(curve1, curve2, dist=euclidean)
            return float(distance)
        except ImportError:
            pass  # 降级到内置实现
        except Exception:
            pass  # 其他错误也降级

    # 内置DTW实现
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



class SurrogatePredictor:
    """
    代理模型预测器

    封装CGLT深度学习模型，提供简洁的预测API
    """

    def __init__(
        self,
        checkpoint_dir: str,
        z_dim: int = 16,
        feature_dim: int = 8,
        device: str = 'cuda'
    ):
        """
        初始化预测器

        参数:
            checkpoint_dir: 模型检查点目录
            z_dim: 潜在空间维度
            feature_dim: 特征维度
            device: 计算设备
        """
        self.checkpoint_dir = checkpoint_dir
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 加载模型
        self._load_models()

        # 加载归一化参数
        self._load_normalization_params()

    def _load_models(self):
        """加载训练好的模型"""
        print(f"Loading models from {self.checkpoint_dir}...")

        # 初始化模型
        self.encoder = EnhancedConditionalEncoder(
            z_dim=self.z_dim,
            feature_dim=self.feature_dim
        ).to(self.device)

        self.decoder = EnhancedConditionalDecoder(
            z_dim=self.z_dim,
            feature_dim=self.feature_dim
        ).to(self.device)

        # 加载权重路径
        encoder_path = os.path.join(self.checkpoint_dir, 'best_encoder.pt')
        decoder_path = os.path.join(self.checkpoint_dir, 'best_decoder.pt')
        mapper_path = os.path.join(self.checkpoint_dir, 'best_mapper.pt')

        # 如果best模型不存在，尝试加载普通模型
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join(self.checkpoint_dir, 'encoder.pt')
        if not os.path.exists(decoder_path):
            decoder_path = os.path.join(self.checkpoint_dir, 'decoder.pt')
        if not os.path.exists(mapper_path):
            mapper_path = os.path.join(self.checkpoint_dir, 'mapper.pt')

        # 加载encoder和decoder
        load_model(self.encoder, encoder_path, self.device)
        load_model(self.decoder, decoder_path, self.device)

        try:
            self.mapper = EnhancedANNMapper(
                feature_dim=self.feature_dim,
                z_dim=self.z_dim
            ).to(self.device)
            load_model(self.mapper, mapper_path, self.device)
            print("Loaded current version mapper")
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key" in str(e):
                print("Current mapper version mismatch, trying legacy version...")
                try:
                    from legacy_model import LegacyANNMapper
                    self.mapper = LegacyANNMapper(
                        feature_dim=self.feature_dim,
                        z_dim=self.z_dim
                    ).to(self.device)
                    load_model(self.mapper, mapper_path, self.device)
                    print("Loaded legacy version mapper successfully!")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load mapper with both current and legacy versions: {e2}")
            else:
                raise

        # 设置为评估模式
        self.encoder.eval()
        self.decoder.eval()
        self.mapper.eval()

        print("Models loaded successfully!")

    def _load_normalization_params(self):
        """加载归一化参数"""
        norm_path = os.path.join(self.checkpoint_dir, 'normalization_params.pkl')

        if os.path.exists(norm_path):
            with open(norm_path, 'rb') as f:
                self.norm_params = pickle.load(f)
            print("Normalization parameters loaded!")
        else:
            print("Warning: Normalization parameters not found, using default ranges")
            self.norm_params = None

    def normalize_parameters(self, params_raw: Union[List, np.ndarray]) -> np.ndarray:
        """
        归一化设计参数

        参数:
            params_raw: [H1, L1, a1, r1, H2, L2, a2, r1] or [H1, L1, a1, r1, H2, L2, a2, r2]

        返回:
            归一化后的参数
        """
        params_raw = np.array(params_raw, dtype=np.float32)

        if self.norm_params is not None:
            # 使用保存的scaler
            scaler = self.norm_params[0]
            params_norm = scaler.transform([params_raw])[0]
        else:
            # 使用默认范围归一化
            # 参数顺序: [H1, L1, a1, r1, H2, L2, a2, r2]
            ranges = [
                (24, 36), (1, 10), (30, 80), (10, 18),  # Section 1
                (24, 36), (1, 10), (30, 80), (10, 18)   # Section 2
            ]

            params_norm = []
            for param, (min_val, max_val) in zip(params_raw, ranges):
                normalized = (param - min_val) / (max_val - min_val)
                params_norm.append(normalized)

            params_norm = np.array(params_norm, dtype=np.float32)

        return params_norm

    def denormalize_curve(self, curve_norm: np.ndarray) -> np.ndarray:
        """
        反归一化曲线

        参数:
            curve_norm: [N, 2] 归一化曲线

        返回:
            原始尺度的曲线
        """
        if self.norm_params is not None and len(self.norm_params) >= 3:
            _, curve_min, curve_max = self.norm_params
            return curve_norm * (curve_max - curve_min) + curve_min
        else:
            # 如果没有归一化参数，返回原值
            return curve_norm

    def predict_curve(
        self,
        params_raw: Union[List, np.ndarray],
        denormalize: bool = False
    ) -> np.ndarray:
        """
        从设计参数预测力-位移曲线

        参数:
            params_raw: [H1, L1, a1, r1, H2, L2, a2, r1/r2]
            denormalize: 是否反归一化曲线

        返回:
            [N, 2] 预测曲线
        """
        # 归一化参数
        params_norm = self.normalize_parameters(params_raw)
        params_tensor = torch.tensor(params_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.mapper(params_tensor)

            curve_pred = self.decoder(z, params_tensor)

        # 转换为numpy
        curve = curve_pred[0].cpu().numpy()

        # 可选反归一化
        if denormalize:
            curve = self.denormalize_curve(curve)

        return curve

    def predict_with_indicators(
        self,
        params_raw: Union[List, np.ndarray],
        denormalize: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        预测曲线并提取性能指标

        参数:
            params_raw: 设计参数
            denormalize: 是否反归一化

        返回:
            (曲线, 性能指标字典)
        """
        curve = self.predict_curve(params_raw, denormalize)
        indicators = extract_all_indicators(curve)

        return curve, indicators

    def predict_batch(
        self,
        params_batch: List[Union[List, np.ndarray]],
        denormalize: bool = False
    ) -> List[np.ndarray]:
        """
        批量预测

        参数:
            params_batch: 参数列表
            denormalize: 是否反归一化

        返回:
            曲线列表
        """
        curves = []
        for params in params_batch:
            curve = self.predict_curve(params, denormalize)
            curves.append(curve)

        return curves

    def compute_dtw_to_target(
        self,
        params_raw: Union[List, np.ndarray],
        target_curve: np.ndarray,
        use_fastdtw: bool = True
    ) -> float:
        """
        计算预测曲线与目标曲线的DTW距离

        参数:
            params_raw: 设计参数
            target_curve: 目标曲线 [N, 2]
            use_fastdtw: 是否使用fastdtw

        返回:
            DTW距离
        """
        pred_curve = self.predict_curve(params_raw, denormalize=False)
        return compute_dtw_distance(pred_curve, target_curve, use_fastdtw)

    def evaluate_design(
        self,
        params_raw: Union[List, np.ndarray],
        target_curve: Optional[np.ndarray] = None,
        target_indicators: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        全面评估设计

        参数:
            params_raw: 设计参数
            target_curve: 目标曲线（可选）
            target_indicators: 目标指标（可选）

        返回:
            评估结果字典
        """
        # 预测曲线和指标
        pred_curve, pred_indicators = self.predict_with_indicators(params_raw)

        result = {
            'params': params_raw,
            'curve': pred_curve,
            'indicators': pred_indicators
        }

        # 如果有目标曲线，计算DTW
        if target_curve is not None:
            dtw_dist = compute_dtw_distance(pred_curve, target_curve)
            result['dtw_distance'] = dtw_dist

        # 如果有目标指标，计算相对误差
        if target_indicators is not None:
            indicator_errors = {}
            for key in ['PBS', 'NCL', 'NCA']:
                if key in pred_indicators and key in target_indicators:
                    pred_val = pred_indicators[key]
                    target_val = target_indicators[key]

                    if abs(target_val) > 1e-6:
                        rel_error = abs(pred_val - target_val) / abs(target_val)
                    else:
                        rel_error = abs(pred_val - target_val)

                    indicator_errors[key] = rel_error

            result['indicator_errors'] = indicator_errors

        return result



def create_predictor(checkpoint_dir: str, device: str = 'cuda') -> SurrogatePredictor:
    """
    创建预测器的便捷函数

    参数:
        checkpoint_dir: 检查点目录
        device: 设备

    返回:
        SurrogatePredictor实例
    """
    return SurrogatePredictor(checkpoint_dir=checkpoint_dir, device=device)



if __name__ == "__main__":
    print("=" * 60)
    print("CGLT 代理模型预测接口")
    print("=" * 60)

    # 示例用法
    print("\n使用示例:")
    print("```python")
    print("from optimization.surrogate import SurrogatePredictor")
    print()
    print("# 创建预测器")
    print("predictor = SurrogatePredictor(")
    print("    checkpoint_dir='../src/checkpoints',")
    print("    device='cuda'")
    print(")")
    print()
    print("# 预测曲线")
    print("params = [30, 5, 45, 12, 32, 6, 50, 12]  # H1,L1,a1,r1,H2,L2,a2,r1")
    print("curve, indicators = predictor.predict_with_indicators(params)")
    print()
    print("print(f'PBS={indicators[\"PBS\"]:.4f}')")
    print("print(f'NCL={indicators[\"NCL\"]:.4f}')")
    print("print(f'NCA={indicators[\"NCA\"]:.4f}')")
    print("```")
    print("=" * 60)
