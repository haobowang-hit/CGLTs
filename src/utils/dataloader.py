import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

CHECKPOINT_DIR = "./checkpoints"


class CGLTDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        curves: np.ndarray,
        features_raw: Optional[np.ndarray] = None,
        jobnums: Optional[np.ndarray] = None,
        scaler: Optional[StandardScaler] = None,
        curve_min: Optional[np.ndarray] = None,
        curve_max: Optional[np.ndarray] = None,
    ):
        self.features = features.astype(np.float32)
        self.curves = curves.astype(np.float32)
        self.features_raw = features_raw.astype(np.float32) if features_raw is not None else self.features
        self.jobnums = jobnums.astype(int) if jobnums is not None else np.arange(len(self.features))
        self.scaler = scaler
        self.curve_min = curve_min
        self.curve_max = curve_max

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.curves[idx], dtype=torch.float32)

    def get_normalization_params(self):
        return {
            "scaler": self.scaler,
            "curve_min": self.curve_min,
            "curve_max": self.curve_max,
        }

    def get_raw_features(self, idx):
        return self.features_raw[idx]

    def denormalize_curve(self, curve):
        if self.curve_min is None or self.curve_max is None:
            return curve
        if isinstance(curve, torch.Tensor):
            curve = curve.detach().cpu().numpy()
        return curve * (self.curve_max - self.curve_min) + self.curve_min



def _resolve_column(columns: List[str], candidates: List[str], field_name: str) -> str:
    for col in candidates:
        if col in columns:
            return col
    raise KeyError(
        f"Missing required field '{field_name}'. Tried columns: {candidates}. "
        f"Available columns: {list(columns)}"
    )



def _load_raw_data(feature_csv_path: str, curve_folder_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_data = pd.read_csv(feature_csv_path)

    cols = feature_data.columns
    h1_col = _resolve_column(cols, ["H1", "h1", "Sectionalradius1", "sectionalradius1"], "H1")
    l1_col = _resolve_column(cols, ["Lumbus1", "lumbus1", "L1", "l1"], "Lumbus1")
    a1_col = _resolve_column(cols, ["Angle1", "angle1", "a1"], "Angle1")
    r1_col = _resolve_column(cols, ["Radius1", "radius1", "r1"], "Radius1")
    h2_col = _resolve_column(cols, ["H2", "h2", "Sectionalradius2", "sectionalradius2"], "H2")
    l2_col = _resolve_column(cols, ["Lumbus2", "lumbus2", "L2", "l2"], "Lumbus2")
    a2_col = _resolve_column(cols, ["Angle2", "angle2", "a2"], "Angle2")
    r2_col = _resolve_column(cols, ["Radius2", "radius2", "r2"], "Radius2")
    job_col = _resolve_column(cols, ["Jobnum", "jobnum", "JobNum"], "Jobnum")

    h1 = feature_data[h1_col].values.astype(np.float32)
    h2 = feature_data[h2_col].values.astype(np.float32)

    # Legacy datasets may store Sectionalradius (H/2). Convert to H to match code constraints.
    if "sectionalradius" in h1_col.lower():
        h1 = h1 * 2.0
    if "sectionalradius" in h2_col.lower():
        h2 = h2 * 2.0

    features_raw = np.stack(
        [
            h1,
            feature_data[l1_col].values.astype(np.float32),
            feature_data[a1_col].values.astype(np.float32),
            feature_data[r1_col].values.astype(np.float32),
            h2,
            feature_data[l2_col].values.astype(np.float32),
            feature_data[a2_col].values.astype(np.float32),
            feature_data[r2_col].values.astype(np.float32),
        ],
        axis=1,
    )

    jobnums = feature_data[job_col].values.astype(int)
    curves = []
    for jobnum in jobnums:
        fname = f"cgltNonLinear_no{jobnum}_resampled.csv"
        fpath = os.path.join(curve_folder_path, fname)
        curve = pd.read_csv(fpath, header=None).values.astype(np.float32)
        curves.append(curve)

    curves_raw = np.stack(curves, axis=0)  # [N, T, 2]
    return features_raw, curves_raw, jobnums



def save_normalization_params(params: Dict, save_dir: str = CHECKPOINT_DIR):
    os.makedirs(save_dir, exist_ok=True)

    scaler = params["scaler"]
    curve_min = params["curve_min"]
    curve_max = params["curve_max"]

    with open(os.path.join(save_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    np.savez(os.path.join(save_dir, "curve_norm.npz"), curve_min=curve_min, curve_max=curve_max)

    # Legacy compatibility (optimization/surrogate scripts).
    with open(os.path.join(save_dir, "normalization_params.pkl"), "wb") as f:
        pickle.dump((scaler, curve_min, curve_max), f)



def load_normalization_params(load_dir: str = CHECKPOINT_DIR):
    with open(os.path.join(load_dir, "feature_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    curve_params = np.load(os.path.join(load_dir, "curve_norm.npz"))
    curve_min = curve_params["curve_min"]
    curve_max = curve_params["curve_max"]
    return scaler, curve_min, curve_max



def _build_splits(total_len: int, val_split: float, test_split: float, seed: int):
    if val_split < 0 or test_split < 0 or (val_split + test_split) >= 1.0:
        raise ValueError("Require 0 <= val_split, test_split and val_split + test_split < 1")

    indices = np.arange(total_len)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_len = int(round(total_len * test_split))
    val_len = int(round(total_len * val_split))
    train_len = total_len - val_len - test_len

    train_idx = indices[:train_len]
    val_idx = indices[train_len : train_len + val_len]
    test_idx = indices[train_len + val_len :]
    return train_idx, val_idx, test_idx



def get_dataloader(
    feature_csv_path="./data/input/selected_pairs.csv",
    curve_folder_path="./data/output",
    batch_size=128,
    shuffle=True,
    use_saved_norm=False,
    val_split=0.2,
    seed=42,
    checkpoint_dir=CHECKPOINT_DIR,
    test_split=0.0,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
):
    scaler_path = os.path.join(checkpoint_dir, "feature_scaler.pkl")
    curve_norm_path = os.path.join(checkpoint_dir, "curve_norm.npz")
    has_saved_norm = os.path.exists(scaler_path) and os.path.exists(curve_norm_path)

    features_raw, curves_raw, jobnums = _load_raw_data(feature_csv_path, curve_folder_path)
    n_samples = len(features_raw)

    train_idx, val_idx, test_idx = _build_splits(n_samples, val_split=val_split, test_split=test_split, seed=seed)

    if use_saved_norm and has_saved_norm:
        scaler, curve_min, curve_max = load_normalization_params(checkpoint_dir)
    else:
        # Fit normalization only on train split to avoid leakage.
        scaler = StandardScaler().fit(features_raw[train_idx])
        train_curves = curves_raw[train_idx]
        curve_min = train_curves.reshape(-1, train_curves.shape[-1]).min(axis=0)
        curve_max = train_curves.reshape(-1, train_curves.shape[-1]).max(axis=0)

    features_norm = scaler.transform(features_raw)
    curves_norm = (curves_raw - curve_min) / (curve_max - curve_min + 1e-8)

    full_dataset = CGLTDataset(
        features=features_norm,
        curves=curves_norm,
        features_raw=features_raw,
        jobnums=jobnums,
        scaler=scaler,
        curve_min=curve_min,
        curve_max=curve_max,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)

    train_loader = DataLoader(Subset(full_dataset, train_idx.tolist()), shuffle=shuffle, **loader_kwargs)
    val_loader = None
    test_loader = None

    if len(val_idx) > 0:
        val_loader = DataLoader(Subset(full_dataset, val_idx.tolist()), shuffle=False, **loader_kwargs)
    if len(test_idx) > 0:
        test_loader = DataLoader(Subset(full_dataset, test_idx.tolist()), shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, full_dataset
