import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PerceptualMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Projection of queries, keys, values
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(8, embed_dim)  # Feature dimension is 8
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # normalization layer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, features):
        """forward propagation"""
        batch_size, seq_len, embed_dim = x.shape
        
        # Application layer normalization
        x_norm = self.norm1(x)
        
        # Projection queries, keys, values
        q = self.query_proj(x_norm)  # [B, T, E]
        k = self.key_proj(features).unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, E]
        v = self.value_proj(x_norm)  # [B, T, E]
        
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(embed_dim)  # [B, T, T]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights
        attn_output = torch.bmm(attn_weights, v)  # [B, T, E]
        attn_output = self.output_proj(attn_output)
        
        # Residual connections and layer normalization
        x = x + attn_output
        x = self.norm2(x)
        
        return x



class FeatureExtractor(nn.Module):
    """Simplified 1D convolutional feature extractor"""
    def __init__(self, input_channels=2, output_dim=16):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.SiLU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.SiLU(),
            nn.Conv1d(8, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(),
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        # Apply convolutional layer
        features = self.conv_layers(x)  # [B, output_dim, T]
        
        # Convert back to [B, T, output_dim]
        features = features.transpose(1, 2)
        
        return features



class ResidualBlock(nn.Module):
    """residual block"""
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
    """Latent space regularization module"""
    def __init__(self, z_dim, regularization_type='tanh'):
        super().__init__()
        self.regularization_type = regularization_type
        
        if regularization_type == 'tanh':
            # Boundedization using Tanh
            self.regularize = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.LayerNorm(z_dim),
                nn.Tanh()
            )
        elif regularization_type == 'vMF':
            self.regularize = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.LayerNorm(z_dim)
                # Then use the normalization constraint in forward to the unit hypersphere
            )
        else:
            # no regularization
            self.regularize = nn.Identity()
    
    def forward(self, z):
        if self.regularization_type == 'vMF':
            # constrained to unit hypersphere
            z = self.regularize(z)
            z = F.normalize(z, p=2, dim=1)
        else:
            z = self.regularize(z)
        return z



class EnhancedConditionalEncoder(nn.Module):
    """Conditional VAE encoder"""
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
        
        # Using 1D convolutional feature extractor instead of BiLSTM
        self.curve_net = FeatureExtractor(input_channels=2, output_dim=16)
        
        self.attention = PerceptualMultiHeadAttention(embed_dim=16, num_heads=2)
        
        # Global feature pooling layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # joint processing network
        self.joint_net = nn.Sequential(
            nn.Linear(16 * 30 + 32, 64),  # 30 time step features + design features
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        
        # residual block
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        # latent space mapping
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)
        
        # latent space regularization
        self.latent_regularizer = LatentRegularizer(z_dim, regularization_type=latent_regularization)

    def forward(self, curve, features):
        # Feature processing
        f_processed = self.feature_net(features)  # [B, 32]
        
        c_processed = self.curve_net(curve)  # [B, 30, 16]
        
        # Apply attention mechanism to fuse feature and curve information
        attended_features = self.attention(c_processed, features)  # [B, 30, 16]
        
        # Flat features
        c_flat = attended_features.reshape(attended_features.size(0), -1)  # [B, 30*16]
        
        # joint processing
        x = torch.cat([c_flat, f_processed], dim=1)
        x = self.joint_net(x)
        
        # Apply residual block
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # map to latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Apply latent space regularization
        mu = self.latent_regularizer(mu)
        
        return mu, logvar



class EnhancedConditionalDecoder(nn.Module):
    """Conditional VAE decoder"""
    def __init__(self, z_dim, feature_dim=8):
        super().__init__()
        
        # Latent vector and feature fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(z_dim + feature_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        
        # residual block
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        # spatial upsampling network
        self.spatial_net = nn.Sequential(
            nn.Linear(64, 30 * 16),  # Generate 30 time steps
            nn.SiLU(),
        )
        
        self.attention = PerceptualMultiHeadAttention(embed_dim=16, num_heads=2)
        
        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.SiLU(),
            nn.Linear(8, 2)
        )
        
    def forward(self, z, features):
        # Fusion of latent vectors and features
        x = torch.cat([z, features], dim=1)  # [B, z_dim + feature_dim]
        x = self.fusion_net(x)  # [B, 64]
        
        # Apply residual block
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # space expansion
        x = self.spatial_net(x)  # [B, 30*16]
        x = x.view(-1, 30, 16)  # [B, 30, 16]
        
        x = self.attention(x, features)  # [B, 30, 16]
        
        # generate final output
        x = self.output_layer(x)  # [B, 30, 2]
        
        return x



class EnhancedANNMapper(nn.Module):
    """Mapping networks - from design parameters to latent space"""
    def __init__(self, feature_dim, z_dim):
        super().__init__()
        
        # Input preprocessing
        self.input_block = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(32) for _ in range(2)
        ])
        
        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, z_dim)
        )
        
        # latent space regularization
        self.latent_regularizer = LatentRegularizer(z_dim, regularization_type='tanh')
        
    def forward(self, x):
        # Input processing
        x = self.input_block(x)
        
        # Apply residual block
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Generate latent representation
        z = self.output_layer(x)
        
        # Apply latent space regularization
        z = self.latent_regularizer(z)
        
        return z



def reparameterize(mu, logvar):
    """VAE reparameterization techniques"""
    std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=10))
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_shape_consistency(pred, target):
    """Compute shape consistency loss - ensure similarity of derivatives"""
    # Calculate the difference between adjacent points (approximate derivative)
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    
    # Compute similarity of differences
    shape_loss = F.mse_loss(pred_diff, target_diff)
    return shape_loss

def get_adaptive_weights(curve, emphasis_factor=2.0):
    """Calculate adaptive weights for different parts of the curve - focusing on twists and turns"""
    # Calculate the curvature of a curve (simplified version)
    diffs = torch.diff(curve, dim=1)
    curvature = torch.sum(diffs**2, dim=-1)
    
    # Normalize curvature and add base weights
    curvature = F.pad(curvature, (0, 1), "constant", 1.0)  # Complete the last point
    weights = 1.0 + (curvature / torch.mean(curvature)) * (emphasis_factor - 1.0)
    
    return weights

def weighted_smooth_l1_loss(pred, target, weights=None, beta=1.0):
    """Smooth L1 loss with weights"""
    if weights is None:
        weights = torch.ones_like(pred[:, :, 0])
        
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    loss = torch.sum(loss, dim=-1)  # Sum along feature dimensions
    
    # Apply weights
    weighted_loss = loss * weights
    return torch.mean(weighted_loss)

def dtw_2d_loss(pred, target, gamma=0.1):
    """Simplified version of differentiable DTW loss"""
    # Practical implementations often require GPU acceleration or the use of specialized libraries
    loss = 0.0
    batch_size = pred.size(0)
    
    for i in range(batch_size):
        # Calculate the L2 distance matrix for each dimension
        D = torch.cdist(pred[i], target[i], p=2)
        
        # cumulative distance matrix
        seq_len = D.size(0)
        R = torch.zeros_like(D)
        R[0, 0] = D[0, 0]
        
        # Fill the first row and column
        for j in range(1, seq_len):
            R[j, 0] = R[j-1, 0] + D[j, 0]
            R[0, j] = R[0, j-1] + D[0, j]
        
        # dynamic programming
        for j in range(1, seq_len):
            for k in range(1, seq_len):
                R[j, k] = D[j, k] + torch.min(torch.stack([
                    R[j-1, k], R[j, k-1], R[j-1, k-1]
                ]))
        
        loss += R[-1, -1] / (seq_len * 2)  # normalization
    
    return loss / batch_size

def enhanced_loss_function(recon_x, x, mu, logvar, gamma=1.0, dtw_weight=0.1, shape_weight=0.1):
    """Enhanced VAE loss function"""
    # adaptive weights
    point_weights = get_adaptive_weights(x)
    
    # Weighted reconstruction loss
    recon_loss = weighted_smooth_l1_loss(recon_x, x, weights=point_weights)
    
    dtw_loss = dtw_2d_loss(recon_x, x)
    
    shape_loss = compute_shape_consistency(recon_x, x)
    
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # total loss
    total_loss = recon_loss + dtw_weight * dtw_loss + shape_weight * shape_loss + gamma * kl_div
    
    return total_loss, recon_loss.item(), kl_div.item(), dtw_loss.item()

def advanced_curve_augmentation(curves, features, aug_strength=0.15):
    """Advanced curve data enhancement"""
    batch_size = curves.size(0)
    device = curves.device
    
    # Create copies of enhanced curves and features
    aug_curves = curves.clone()
    aug_features = features.clone()
    
    # 1. Gaussian noise (random intensity)
    noise_level = aug_strength * torch.rand(batch_size, 1, 1, device=device)
    aug_curves += noise_level * torch.randn_like(aug_curves)
    
    # 2. Characteristic random disturbance
    feature_noise = aug_strength * torch.randn_like(aug_features)
    aug_features += feature_noise
    
    # 3. Curve shape preservation perturbation
    for i in range(batch_size):
        # Randomly decide whether to apply
        if torch.rand(1).item() > 0.5:
            # Define adjustment strength
            intensity = 0.1 * torch.rand(1, device=device).item()
            
            # Spline-based smooth perturbation
            t = torch.linspace(0, 1, aug_curves.size(1), device=device)
            distortion = torch.sin(t * (2 * math.pi)) * intensity
            
            # Apply to X and Y respectively
            aug_curves[i, :, 0] += distortion
            aug_curves[i, :, 1] += distortion * torch.rand(1, device=device).item()
    
    peak_indices = torch.rand(batch_size, device=device) < 0.3
    if peak_indices.any():
        # Find the approximate peak position of each curve
        for i in range(batch_size):
            if peak_indices[i]:
                # Assume that the point near the maximum Y value is the peak value
                peak_idx = torch.argmax(aug_curves[i, :, 1])
                
                # Only perturb points near the peak
                window = 3  # Number of points before and after the peak
                start_idx = max(0, peak_idx - window)
                end_idx = min(aug_curves.size(1), peak_idx + window + 1)
                
                # Apply local perturbation
                local_noise = 0.15 * torch.randn(end_idx - start_idx, 2, device=device)
                aug_curves[i, start_idx:end_idx] += local_noise
    
    # Make sure the enhanced data is within reasonable limits
    aug_curves = torch.clamp(aug_curves, 0.0, 1.0)
    
    return aug_curves, aug_features