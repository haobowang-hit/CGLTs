import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PerceptualMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query comes from sequence features; key/value come from 8 conditioning parameters as tokens.
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(1, embed_dim)
        self.value_proj = nn.Linear(1, embed_dim)
        self.param_pos_embed = nn.Parameter(torch.randn(1, 8, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, features):
        batch_size = x.shape[0]

        x_norm = self.norm1(x)

        q = self.query_proj(x_norm)  # [B, T, E]
        param_tokens = features.view(batch_size, -1, 1)  # [B, 8, 1]
        k = self.key_proj(param_tokens) + self.param_pos_embed[:, :param_tokens.size(1), :]
        v = self.value_proj(param_tokens) + self.param_pos_embed[:, :param_tokens.size(1), :]
        attn_output, _ = self.cross_attn(q, k, v, need_weights=False)  # [B, T, E]
        attn_output = self.output_proj(attn_output)

        x = x + attn_output
        x = self.norm2(x)

        return x



class FeatureExtractor(nn.Module):
    
    def __init__(self, input_channels=2, output_dim=16):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 4, kernel_size=3, padding=1),
            nn.GroupNorm(1, 4),
            nn.SiLU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.GroupNorm(1, 8),
            nn.SiLU(),
            nn.Conv1d(8, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, output_dim),
            nn.SiLU(),
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
       
        features = self.conv_layers(x)  # [B, output_dim, T]
        
        #  [B, T, output_dim]
        features = features.transpose(1, 2)
        
        return features



class ResidualBlock(nn.Module):
 
    def __init__(self, dim, expansion_factor=1.5):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
            
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        x = self.layers(x)
        x = residual + x
        x = self.norm(x)
        return x



class LatentRegularizer(nn.Module):
   
    def __init__(self, z_dim, regularization_type='tanh'):
        super().__init__()
        self.regularization_type = regularization_type
        
        if regularization_type == 'tanh':
         
            self.regularize = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.LayerNorm(z_dim),
                nn.Tanh()
            )
        elif regularization_type == 'vMF':
            self.regularize = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.LayerNorm(z_dim)
                
            )
        else:
            
            self.regularize = nn.Identity()
    
    def forward(self, z):
        if self.regularization_type == 'vMF':
            
            z = self.regularize(z)
            z = F.normalize(z, p=2, dim=1)
        else:
            z = self.regularize(z)
        return z



class EnhancedConditionalEncoder(nn.Module):
   
    def __init__(self, z_dim, feature_dim=8, latent_regularization='tanh'):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        
        
        self.curve_net = FeatureExtractor(input_channels=2, output_dim=16)
        
        self.attention = PerceptualMultiHeadAttention(embed_dim=16, num_heads=2)
        
      
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.joint_net = nn.Sequential(
            nn.Linear(16 * 30 + 32, 64),  
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Identity(),
        )
        
      
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)
        
     
        self.latent_regularizer = LatentRegularizer(z_dim, regularization_type=latent_regularization)

    def forward(self, curve, features):
       
        f_processed = self.feature_net(features)  # [B, 32]
        
        c_processed = self.curve_net(curve)  # [B, 30, 16]
        
        attended_features = self.attention(c_processed, features)  # [B, 30, 16]
        
     
        c_flat = attended_features.reshape(attended_features.size(0), -1)  # [B, 30*16]
        
     
        x = torch.cat([c_flat, f_processed], dim=1)
        x = self.joint_net(x)
        
      
        x = self.res_block1(x)
        x = self.res_block2(x)
        
    
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
       
        mu = self.latent_regularizer(mu)
        
        return mu, logvar



class EnhancedConditionalDecoder(nn.Module):
   
    def __init__(self, z_dim, feature_dim=8):
        super().__init__()
        
       
        self.fusion_net = nn.Sequential(
            nn.Linear(z_dim + feature_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Identity(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        
       
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        self.spatial_net = nn.Sequential(
            nn.Linear(64, 30 * 16),  
            nn.SiLU(),
        )
        
        self.attention = PerceptualMultiHeadAttention(embed_dim=16, num_heads=2)
        
       
        self.output_layer = nn.Sequential(
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.SiLU(),
            nn.Linear(8, 2)
        )
        
    def forward(self, z, features):
       
        x = torch.cat([z, features], dim=1)  # [B, z_dim + feature_dim]
        x = self.fusion_net(x)  # [B, 64]
        
      
        x = self.res_block1(x)
        x = self.res_block2(x)
        
       
        x = self.spatial_net(x)  # [B, 30*16]
        x = x.view(-1, 30, 16)  # [B, 30, 16]
        
        x = self.attention(x, features)  # [B, 30, 16]
        
      
        x = self.output_layer(x)  # [B, 30, 2]
        
        return x



class EnhancedANNMapper(nn.Module):
    
    def __init__(self, feature_dim, z_dim):
        super().__init__()
        
        self.input_block = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Identity(),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(32) for _ in range(2)
        ])
        
       
        self.output_layer = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, z_dim)
        )
        
       
        self.latent_regularizer = LatentRegularizer(z_dim, regularization_type='tanh')
        
    def forward(self, x):
       
        x = self.input_block(x)
        
  
        for res_block in self.res_blocks:
            x = res_block(x)
        
 
        z = self.output_layer(x)
        

        z = self.latent_regularizer(z)
        
        return z



def reparameterize(mu, logvar):
   
    std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=10))
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_shape_consistency(pred, target):
    
 
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    
 
    shape_loss = F.mse_loss(pred_diff, target_diff)
    return shape_loss

def get_adaptive_weights(curve, emphasis_factor=2.0):
    
    diffs = torch.diff(curve, dim=1)
    curvature = torch.sum(diffs**2, dim=-1)
    
    
    curvature = F.pad(curvature, (0, 1), "constant", 1.0)  
    weights = 1.0 + (curvature / torch.mean(curvature)) * (emphasis_factor - 1.0)
    
    return weights

def weighted_smooth_l1_loss(pred, target, weights=None, beta=1.0):
   
    if weights is None:
        weights = torch.ones_like(pred[:, :, 0])
        
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    loss = torch.sum(loss, dim=-1)  
    

    weighted_loss = loss * weights
    return torch.mean(weighted_loss)

def dtw_2d_loss(pred, target, gamma=0.1, sample_size=None, window=None):
  
    batch_size = pred.size(0)
    if batch_size == 0:
        return pred.new_tensor(0.0)

    # Subsample for speed when batch is large.
    if sample_size is not None and sample_size > 0 and sample_size < batch_size:
        idx = torch.randperm(batch_size, device=pred.device)[:sample_size]
        pred = pred[idx]
        target = target[idx]
        batch_size = pred.size(0)

    loss = pred.new_tensor(0.0)
    
    for i in range(batch_size):
      
        D = torch.cdist(pred[i], target[i], p=2)
        
      
        seq_len = D.size(0)
        R = torch.zeros_like(D)
        R[0, 0] = D[0, 0]
        
     
        for j in range(1, seq_len):
            R[j, 0] = R[j-1, 0] + D[j, 0]
            R[0, j] = R[0, j-1] + D[0, j]
     
        for j in range(1, seq_len):
            if window is None or window <= 0:
                k_start, k_end = 1, seq_len
            else:
                k_start = max(1, j - window)
                k_end = min(seq_len, j + window + 1)
            for k in range(k_start, k_end):
                R[j, k] = D[j, k] + torch.min(torch.stack([R[j-1, k], R[j, k-1], R[j-1, k-1]]))
        
        loss += R[-1, -1] / (seq_len * 2)  
    
    return loss / batch_size

def enhanced_loss_function(
    recon_x,
    x,
    mu,
    logvar,
    gamma=1.0,
    dtw_weight=0.1,
    shape_weight=0.1,
    dtw_sample_size=None,
    dtw_window=None
):
   
    point_weights = get_adaptive_weights(x)
   
    recon_loss = weighted_smooth_l1_loss(recon_x, x, weights=point_weights)
    
    dtw_loss = dtw_2d_loss(recon_x, x, sample_size=dtw_sample_size, window=dtw_window)
    
    shape_loss = compute_shape_consistency(recon_x, x)
    
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
 
    total_loss = recon_loss + dtw_weight * dtw_loss + shape_weight * shape_loss + gamma * kl_div
    
    return total_loss, recon_loss.item(), kl_div.item(), dtw_loss.item()

def advanced_curve_augmentation(curves, features, aug_strength=0.15):
  
    batch_size = curves.size(0)
    device = curves.device
    
  
    aug_curves = curves.clone()
    aug_features = features.clone()
    
    noise_level = aug_strength * torch.rand(batch_size, 1, 1, device=device)
    aug_curves += noise_level * torch.randn_like(aug_curves)
    

    feature_noise = aug_strength * torch.randn_like(aug_features)
    aug_features += feature_noise
    
  
    for i in range(batch_size):
        
        if torch.rand(1).item() > 0.5:
           
            intensity = 0.1 * torch.rand(1, device=device).item()
            
       
            t = torch.linspace(0, 1, aug_curves.size(1), device=device)
            distortion = torch.sin(t * (2 * math.pi)) * intensity
            
           
            aug_curves[i, :, 0] += distortion
            aug_curves[i, :, 1] += distortion * torch.rand(1, device=device).item()
    
    peak_indices = torch.rand(batch_size, device=device) < 0.3
    if peak_indices.any():
       
        for i in range(batch_size):
            if peak_indices[i]:
               
                peak_idx = torch.argmax(aug_curves[i, :, 1])
                
                
                window = 3  
                start_idx = max(0, peak_idx - window)
                end_idx = min(aug_curves.size(1), peak_idx + window + 1)
                
              
                local_noise = 0.15 * torch.randn(end_idx - start_idx, 2, device=device)
                aug_curves[i, start_idx:end_idx] += local_noise
    
  
    aug_curves = torch.clamp(aug_curves, 0.0, 1.0)
    
    return aug_curves, aug_features
