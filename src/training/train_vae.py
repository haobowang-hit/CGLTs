import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

# 导入增强版模型和工具函数
from models.vae import (
    EnhancedConditionalEncoder, 
    EnhancedConditionalDecoder,
    reparameterize, 
    enhanced_loss_function,
    advanced_curve_augmentation
)
from utils.utils import visualize_reconstructions, save_model

class WarmupExponentialLR:
    """带预热的指数衰减学习率调度器"""
    def __init__(self, optimizer, warmup_epochs=5, total_epochs=300, min_lr_ratio=0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        
        # 计算指数衰减的gamma
        decay_epochs = total_epochs - warmup_epochs
        self.gamma = (min_lr_ratio) ** (1/decay_epochs) if decay_epochs > 0 else 1.0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段: 线性增加
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
        else:
            # 指数衰减阶段
            decay_steps = self.current_epoch - self.warmup_epochs
            lr_scale = (self.gamma ** decay_steps)
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def train_vae(train_loader, val_loader, args):
    """增强版VAE训练函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建模型
    encoder = EnhancedConditionalEncoder(
        z_dim=args.z_dim, 
        feature_dim=8,
        latent_regularization=args.latent_reg
    ).to(args.device)
    
    decoder = EnhancedConditionalDecoder(
        z_dim=args.z_dim, 
        feature_dim=8
    ).to(args.device)

    # 优化器
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    from torch.optim.lr_scheduler import ExponentialLR
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    step_scheduler_each_batch = False
    
    # 保存数据集划分信息
    train_indices = train_loader.dataset.indices
    val_indices = val_loader.dataset.indices
    
    split_info = pd.DataFrame({
        "Index": list(train_indices) + list(val_indices),
        "Set": ["train"] * len(train_indices) + ["val"] * len(val_indices)
    })
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    split_info.to_csv(f"{args.checkpoint_dir}/dataset_split_info_vae.csv", index=False)
    
    # 训练跟踪变量
    loss_history = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = args.early_stop_patience
    
    os.makedirs(f"{args.checkpoint_dir}/visualizations", exist_ok=True)
    
    # 训练循环
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_dtw_loss = 0.0
        
        # 训练阶段
        for features, curves in tqdm(train_loader, desc=f"[Train CVAE] Epoch {epoch+1}/{args.epochs}"):
            features = features.to(args.device)
            curves = curves.to(args.device)
            
            # 应用数据增强
            if args.use_augmentation:
                curves_aug, features_aug = advanced_curve_augmentation(curves, features, args.aug_strength)
            else:
                curves_aug, features_aug = curves, features
            
            # 前向传播
            mu, logvar = encoder(curves_aug, features_aug)
            z = reparameterize(mu, logvar)
            recon = decoder(z, features_aug)
            
            # 计算损失
            loss, recon_loss, kl_loss, dtw_loss = enhanced_loss_function(
                recon, curves_aug, mu, logvar, 
                gamma=args.kl_weight,
                dtw_weight=args.dtw_weight,
                shape_weight=args.shape_weight
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 
                max_norm=args.grad_clip
            )
            
            optimizer.step()
            
            # 更新学习率
            if step_scheduler_each_batch:
                scheduler.step()
            
            # 累计损失
            train_loss += loss.item()
            train_recon_loss += recon_loss
            train_kl_loss += kl_loss
            train_dtw_loss += dtw_loss
        
        # 计算平均损失
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        train_dtw_loss /= len(train_loader)
        
        # 验证阶段
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_dtw_loss = 0.0
        
        with torch.no_grad():
            for features, curves in val_loader:
                features = features.to(args.device)
                curves = curves.to(args.device)
                
                mu, logvar = encoder(curves, features)
                z = mu  # 验证时不添加噪声
                recon = decoder(z, features)
                
                loss, recon_loss, kl_loss, dtw_loss = enhanced_loss_function(
                    recon, curves, mu, logvar, 
                    gamma=args.kl_weight,
                    dtw_weight=args.dtw_weight,
                    shape_weight=args.shape_weight
                )
                
                val_loss += loss.item()
                val_recon_loss += recon_loss
                val_kl_loss += kl_loss
                val_dtw_loss += dtw_loss
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)
        val_dtw_loss /= len(val_loader)
        
        # 更新学习率（如果是余弦退火）
        if not step_scheduler_each_batch:
            scheduler.step()
        
        # 记录损失
        loss_history.append((
            epoch + 1, 
            train_loss, val_loss,
            train_recon_loss, val_recon_loss,
            train_kl_loss, val_kl_loss,
            train_dtw_loss, val_dtw_loss
        ))
        
        # 输出当前训练状态
        print(f"CVAE Epoch {epoch+1}/{args.epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Train Recon: {train_recon_loss:.4f}, "
              f"Val Recon: {val_recon_loss:.4f}")
        
        # 每隔指定epoch可视化结果
        if (epoch + 1) % args.vis_interval == 0:
            visualize_reconstructions(epoch, encoder, decoder, val_loader, args.device, 
                                     f"{args.checkpoint_dir}/visualizations")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # 保存最佳模型
            save_model(encoder, f"{args.checkpoint_dir}/best_encoder.pt")
            save_model(decoder, f"{args.checkpoint_dir}/best_decoder.pt")
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            early_stop_counter += 1
            
        # 提前停止检查
        if early_stop_counter >= early_stop_patience and args.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # 保存当前模型
        if (epoch + 1) % args.save_interval == 0:
            save_model(encoder, f"{args.checkpoint_dir}/encoder_epoch_{epoch+1}.pt")
            save_model(decoder, f"{args.checkpoint_dir}/decoder_epoch_{epoch+1}.pt")
    
    # 保存最终模型
    save_model(encoder, f"{args.checkpoint_dir}/final_encoder.pt")
    save_model(decoder, f"{args.checkpoint_dir}/final_decoder.pt")
    
    # 保存损失历史
    loss_df = pd.DataFrame(loss_history, columns=[
        "Epoch", "Train_Loss", "Val_Loss",
        "Train_Recon", "Val_Recon",
        "Train_KL", "Val_KL",
        "Train_DTW", "Val_DTW"
    ])
    loss_df.to_csv(f"{args.checkpoint_dir}/vae_loss_history.csv", index=False)
    
    # 绘制损失曲线
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(loss_df["Epoch"], loss_df["Train_Loss"], label="Train Loss", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("CVAE Training Loss Curve")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(loss_df["Epoch"], loss_df["Train_Recon"], label="Train Recon", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Recon"], label="Val Recon", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(loss_df["Epoch"], loss_df["Train_KL"], label="Train KL", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_KL"], label="Val KL", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(loss_df["Epoch"], loss_df["Train_DTW"], label="Train DTW", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_DTW"], label="Val DTW", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("DTW Loss")
    plt.title("DTW Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.checkpoint_dir}/vae_loss_curves.png", dpi=300)
    plt.close()
    
    # 可视化学习率变化
    if hasattr(scheduler, 'get_last_lr'):
        plt.figure(figsize=(10, 6))
        plt.plot(scheduler.get_last_lr())
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(f"{args.checkpoint_dir}/learning_rate_schedule.png", dpi=300)
        plt.close()
    
    # 加载最佳模型并返回
    encoder.load_state_dict(torch.load(f"{args.checkpoint_dir}/best_encoder.pt"))
    decoder.load_state_dict(torch.load(f"{args.checkpoint_dir}/best_decoder.pt"))
    
    return encoder, decoder