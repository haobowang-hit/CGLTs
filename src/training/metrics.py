from typing import Callable, Dict, List, Tuple

import numpy as np
import torch



def extract_pbs(curve: np.ndarray) -> float:
    return float(np.max(curve[:, 1])) if len(curve) else 0.0



def extract_ncl(curve: np.ndarray) -> float:
    if len(curve) < 3:
        return 0.0
    y = curve[:, 1]
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            return float(y[i])
    return float(np.max(y))



def extract_nca(curve: np.ndarray) -> float:
    if len(curve) < 2:
        return 0.0
    return float(np.trapz(curve[:, 1], curve[:, 0]))



def compute_dtw_distance(curve1: np.ndarray, curve2: np.ndarray) -> float:
    n, m = len(curve1), len(curve2)
    mat = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    mat[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(curve1[i - 1] - curve2[j - 1])
            mat[i, j] = cost + min(mat[i - 1, j], mat[i, j - 1], mat[i - 1, j - 1])
    return float(mat[n, m])



def nrmse_curve(curve_pred: np.ndarray, curve_true: np.ndarray) -> float:
    diff = curve_pred - curve_true
    rmse = np.sqrt(np.mean(diff * diff))
    denom = np.max(curve_true) - np.min(curve_true) + 1e-8
    return float(rmse / denom)



def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)



def evaluate_curve_model(
    loader,
    device: str,
    predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Dict[str, float]:
    dtw_vals: List[float] = []
    nrmse_vals: List[float] = []
    pbs_true, pbs_pred = [], []
    ncl_true, ncl_pred = [], []
    nca_true, nca_pred = [], []

    with torch.no_grad():
        for features, curves in loader:
            features = features.to(device)
            curves = curves.to(device)
            pred = predict_fn(features, curves)

            pred_np = pred.detach().cpu().numpy()
            true_np = curves.detach().cpu().numpy()

            for curve_p, curve_t in zip(pred_np, true_np):
                dtw_vals.append(compute_dtw_distance(curve_p, curve_t))
                nrmse_vals.append(nrmse_curve(curve_p, curve_t))

                pbs_true.append(extract_pbs(curve_t))
                pbs_pred.append(extract_pbs(curve_p))
                ncl_true.append(extract_ncl(curve_t))
                ncl_pred.append(extract_ncl(curve_p))
                nca_true.append(extract_nca(curve_t))
                nca_pred.append(extract_nca(curve_p))

    pbs_true = np.asarray(pbs_true)
    pbs_pred = np.asarray(pbs_pred)
    ncl_true = np.asarray(ncl_true)
    ncl_pred = np.asarray(ncl_pred)
    nca_true = np.asarray(nca_true)
    nca_pred = np.asarray(nca_pred)

    return {
        "DTW": float(np.mean(dtw_vals)) if dtw_vals else float("nan"),
        "NRMSE": float(np.mean(nrmse_vals)) if nrmse_vals else float("nan"),
        "R2_PBS": _r2_score_np(pbs_true, pbs_pred),
        "R2_NCL": _r2_score_np(ncl_true, ncl_pred),
        "R2_NCA": _r2_score_np(nca_true, nca_pred),
        "N": int(len(dtw_vals)),
    }
