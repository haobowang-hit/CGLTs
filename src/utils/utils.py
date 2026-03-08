import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dtaidistance import dtw


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def plot_curve_comparison(original, reconstructed, title="Curve Comparison", save_path=None):
   
    original = to_numpy(original)
    reconstructed = to_numpy(reconstructed)

    plt.figure(figsize=(8, 4))
    plt.plot(original[:, 0], original[:, 1], label='Original', marker='o')
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], label='Predicted', marker='x')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y / 1000')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
        plt.close()
    else:
        plt.show()


class SoftDTWLoss(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, x, y):
       
        batch_size = x.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            
            D = torch.cdist(x[i], y[i], p=2)  # [seq_len, seq_len]
        
            R = torch.zeros_like(D)
            R[0, 0] = D[0, 0]
            
        
            n, m = D.shape
            for j in range(1, n):
                R[j, 0] = R[j-1, 0] + D[j, 0]
            
            for j in range(1, m):
                R[0, j] = R[0, j-1] + D[0, j]
            
            for j in range(1, n):
                for k in range(1, m):
                    R[j, k] = D[j, k] + torch.min(torch.stack([
                        R[j-1, k],
                        R[j, k-1],
                        R[j-1, k-1]
                    ]))
            
         
            total_loss += R[-1, -1] / (n + m)  
        
        return total_loss / batch_size

def dtw_2d_loss(recon_x, x):
   
    recon_np = recon_x.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    
    total_loss = 0.0
    for pred_curve, true_curve in zip(recon_np, x_np):
      
        dtw_x = dtw.distance(pred_curve[:, 0], true_curve[:, 0])
        dtw_y = dtw.distance(pred_curve[:, 1], true_curve[:, 1])
        total_loss += (dtw_x + dtw_y) / 2.0

    avg_loss = total_loss / len(recon_np)
    return torch.tensor(avg_loss, dtype=torch.float32, device=recon_x.device)

def enhanced_loss_function(recon_x, x, mu, logvar, gamma=1.0):
 
    recon_loss = F.smooth_l1_loss(recon_x, x, reduction='mean')
    
 
    dtw_loss = dtw_2d_loss(recon_x, x)
 
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
 
    latent_reg = compute_latent_regularization(mu)
    

    total_loss = recon_loss + 0.1 * dtw_loss + gamma * kl_div + 0.01 * latent_reg
    
    return total_loss, recon_loss.item(), kl_div.item(), dtw_loss.item()


def compute_covariance_z(z):
    
    batch_size = z.size(0)
    z_centered = z - torch.mean(z, dim=0)
    return (1 / (batch_size - 1)) * torch.matmul(z_centered.T, z_centered)

def compute_latent_regularization(mu):
  
    sparsity_loss = torch.mean(torch.abs(mu))
    
    z_cov = compute_covariance_z(mu)
    diag_mask = torch.eye(z_cov.shape[0], device=z_cov.device)
    off_diag_mask = 1.0 - diag_mask
    off_diag_cov = z_cov * off_diag_mask
    
    orthogonal_loss = torch.sum(off_diag_cov.pow(2))
    
    return sparsity_loss + 0.1 * orthogonal_loss


def augment_curve_data(curves, features, augmentation_strength=0.1):
  
    batch_size = curves.size(0)
    device = curves.device
    
  
    augmented_curves = curves.clone()
    augmented_features = features.clone()
 
    noise_level = augmentation_strength * torch.rand(batch_size, 1, 1, device=device)
    augmented_curves += noise_level * torch.randn_like(augmented_curves)
    
    feature_noise = augmentation_strength * torch.randn_like(augmented_features)
    augmented_features += feature_noise
    
  
    time_scale = 1.0 + 0.05 * torch.randn(batch_size, 1, 1, device=device)
    time_indices = torch.linspace(0, 1, curves.size(1), device=device).view(1, -1, 1)
    time_indices = time_indices * time_scale
   
    augmented_curves = torch.clamp(augmented_curves, 0.0, 1.0)
    
    return augmented_curves, augmented_features

def visualize_reconstructions(epoch, encoder, decoder, test_loader, device, save_dir="./checkpoints/visualizations"):
  
    os.makedirs(save_dir, exist_ok=True)
    
   
    features, curves = next(iter(test_loader))
    features = features.to(device)
    curves = curves.to(device)
   
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        mu, logvar = encoder(curves, features)
        z = mu  
        recon = decoder(z, features)
    
    # 可视化前8个样本
    plt.figure(figsize=(20, 15))
    for i in range(min(8, curves.size(0))):
        plt.subplot(2, 4, i+1)
        orig_curve = to_numpy(curves[i])
        recon_curve = to_numpy(recon[i])
        
        plt.plot(orig_curve[:, 0], orig_curve[:, 1], 'b-', label='Original')
        plt.plot(recon_curve[:, 0], recon_curve[:, 1], 'r--', label='Reconstructed')
        
        plt.title(f'Sample {i+1}')
        plt.legend()
        plt.grid(True)
    
    plt.suptitle(f'Epoch {epoch} Reconstructions')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/recon_epoch_{epoch}.png", dpi=300)
    plt.close()
    
    # 恢复训练模式
    encoder.train()
    decoder.train()

# 重参数化函数
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=10))
    eps = torch.randn_like(std)
    return mu + eps * std