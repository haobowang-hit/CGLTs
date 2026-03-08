import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

CHECKPOINT_DIR = "./checkpoints"

class CGLTDataset(Dataset):
    def __init__(self, feature_csv_path, curve_folder_path,
                 scaler: StandardScaler = None,
                 curve_min: np.ndarray = None,
                 curve_max: np.ndarray = None):
        self.feature_data = pd.read_csv(feature_csv_path)
        self.curve_folder_path = curve_folder_path

        self.features_raw = self.feature_data[['H1', 'Lumbus1', 'Angle1', 'Radius1',
                                               'H2', 'Lumbus2', 'Angle2', 'Radius2']].values.astype(np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features_raw)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features_raw)

        self.jobnums = self.feature_data['Jobnum'].values.astype(int)

        if curve_min is None or curve_max is None:
            all_curves = []
            for jobnum in self.jobnums:
                fname = f'cgltNonLinear_no{jobnum}_resampled.csv'
                fpath = os.path.join(self.curve_folder_path, fname)
                curve = pd.read_csv(fpath, header=None).values.astype(np.float32)
                all_curves.append(curve)
            all_data = np.concatenate(all_curves, axis=0)
            self.curve_min = all_data.min(axis=0)
            self.curve_max = all_data.max(axis=0)
        else:
            self.curve_min = curve_min
            self.curve_max = curve_max

    def __len__(self):
        return len(self.jobnums)

    def __getitem__(self, idx):
        feature = self.features[idx]
        jobnum = self.jobnums[idx]
        curve_filename = f'cgltNonLinear_no{jobnum}_resampled.csv'
        curve_path = os.path.join(self.curve_folder_path, curve_filename)

        curve = pd.read_csv(curve_path, header=None).values.astype(np.float32)
        curve = (curve - self.curve_min) / (self.curve_max - self.curve_min + 1e-8)

        return torch.tensor(feature), torch.tensor(curve)

    def get_normalization_params(self):
        return {
            "scaler": self.scaler,
            "curve_min": self.curve_min,
            "curve_max": self.curve_max
        }
    
    def get_raw_features(self, idx):
        return self.features_raw[idx]
    
    def denormalize_curve(self, curve):
        """将归一化曲线转换回原始值"""
        if isinstance(curve, torch.Tensor):
            curve = curve.detach().cpu().numpy()
        return curve * (self.curve_max - self.curve_min) + self.curve_min

def save_normalization_params(params: dict, save_dir: str = CHECKPOINT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(params["scaler"], f)
    np.savez(os.path.join(save_dir, "curve_norm.npz"),
             curve_min=params["curve_min"],
             curve_max=params["curve_max"])

def load_normalization_params(load_dir: str = CHECKPOINT_DIR):
    with open(os.path.join(load_dir, "feature_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    curve_params = np.load(os.path.join(load_dir, "curve_norm.npz"))
    curve_min = curve_params["curve_min"]
    curve_max = curve_params["curve_max"]
    return scaler, curve_min, curve_max

def get_dataloader(feature_csv_path, curve_folder_path, batch_size, shuffle=True,
                   use_saved_norm=False, val_split=0.2, seed=42):
    if use_saved_norm and os.path.exists(CHECKPOINT_DIR):
        scaler, curve_min, curve_max = load_normalization_params(CHECKPOINT_DIR)
    else:
        scaler = curve_min = curve_max = None

    dataset = CGLTDataset(feature_csv_path, curve_folder_path,
                          scaler=scaler,
                          curve_min=curve_min,
                          curve_max=curve_max)
    
    # 创建训练集和验证集
    if val_split > 0:
        train_len = int(len(dataset) * (1 - val_split))
        val_len = len(dataset) - train_len
        
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, dataset
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), None, dataset