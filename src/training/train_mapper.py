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
    EnhancedANNMapper,
    dtw_2d_loss,
    advanced_curve_augmentation
)
from utils.utils import save_model, visualize_reconstructions

def get_step_schedule(optimizer, step_size=25, gamma=0.85):
    """创建有序下降的学习率调度器"""
    from torch.optim.lr_scheduler import StepLR
    return StepLR(optimizer, step_size=step_size, gamma=gamma)

def train_mapper(train_loader, val_loader, encoder, decoder, args):
    """增强版映射网络训练函数"""
    # 设置为评估模式
    encoder.eval()
    decoder.eval()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建增强型映射网络
    mapper = EnhancedANNMapper(
        feature_dim=8, 
        z_dim=args.z_dim
    ).to(args.device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        mapper.parameters(), 
        lr=args.mapper_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = get_step_schedule(optimizer, step_size=25, gamma=0.85)
    step_scheduler_each_batch = False
    
    # 保存数据集划分信息
    train_indices = train_loader.dataset.indices
    val_indices = val_loader.dataset.indices
    
    split_info = pd.DataFrame({
        "Index": list(train_indices) + list(val_indices),
        "Set": ["train"] * len(train_indices) + ["val"] * len(val_indices)
    })
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    split_info.to_csv(f"{args.checkpoint_dir}/dataset_split_info_mapper.csv", index=False)
    
    # 训练跟踪变量
    loss_history = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = args.early_stop_patience
    
    # 创建可视化目录
    vis_dir = f"{args.checkpoint_dir}/mapper_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 训练循环
    for epoch in range(args.epochs):
        mapper.train()
        train_loss = 0.0
        train_curve_loss = 0.0
        train_dtw_loss = 0.0
        train_z_loss = 0.0
        
        # 训练阶段
        for features, curves in tqdm(train_loader, desc=f"[Mapper Train] Epoch {epoch+1}/{args.epochs}"):
            features = features.to(args.device)
            curves = curves.to(args.device)
            
            # 应用数据增强
            if args.use_augmentation:
                # 对特征添加轻微噪声，增强鲁棒性
                features_noisy = features + args.aug_strength * 0.5 * torch.randn_like(features)
            else:
                features_noisy = features
            
            # 获取目标潜在编码
            with torch.no_grad():
                mu, _ = encoder(curves, features)
            
            # 从特征预测潜在编码
            pred_z = mapper(features_noisy)
            
            # 从预测的潜在编码重建曲线
            with torch.no_grad():
                recon = decoder(pred_z, features_noisy)
            
            # 计算损失
            
            curve_loss = F.smooth_l1_loss(recon, curves)
            
            dtw_penalty = dtw_2d_loss(recon, curves)
            
            z_loss = F.mse_loss(pred_z, mu)
            
            # 根据训练阶段调整损失权重
            progress = epoch / args.epochs
            # 早期更注重潜在空间对齐，后期更注重曲线重建
            z_weight = args.z_weight * (1.0 - 0.5 * progress)
            curve_weight = 1.0 + 0.5 * progress
            
            # 总损失
            total_loss = (
                curve_weight * curve_loss + 
                args.dtw_weight * dtw_penalty + 
                z_weight * z_loss
            )
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=args.grad_clip)
            
            optimizer.step()
            
            # 更新学习率
            if step_scheduler_each_batch:
                scheduler.step()
            
            # 累计损失
            train_loss += total_loss.item()
            train_curve_loss += curve_loss.item()
            train_dtw_loss += dtw_penalty.item()
            train_z_loss += z_loss.item()
        
        # 计算平均损失
        train_loss /= len(train_loader)
        train_curve_loss /= len(train_loader)
        train_dtw_loss /= len(train_loader)
        train_z_loss /= len(train_loader)
        
        # 验证阶段
        mapper.eval()
        val_loss = 0.0
        val_curve_loss = 0.0
        val_dtw_loss = 0.0
        val_z_loss = 0.0
        
        with torch.no_grad():
            for features, curves in val_loader:
                features = features.to(args.device)
                curves = curves.to(args.device)
                
                # 获取目标潜在编码
                mu, _ = encoder(curves, features)
                
                # 从特征预测潜在编码
                pred_z = mapper(features)
                
                # 从预测的潜在编码重建曲线
                recon = decoder(pred_z, features)
                
                # 计算损失
                curve_loss = F.smooth_l1_loss(recon, curves)
                dtw_penalty = dtw_2d_loss(recon, curves)
                z_loss = F.mse_loss(pred_z, mu)
                
                # 总损失
                total_loss = curve_loss + args.dtw_weight * dtw_penalty + args.z_weight * z_loss
                
                # 累计损失
                val_loss += total_loss.item()
                val_curve_loss += curve_loss.item()
                val_dtw_loss += dtw_penalty.item()
                val_z_loss += z_loss.item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_curve_loss /= len(val_loader)
        val_dtw_loss /= len(val_loader)
        val_z_loss /= len(val_loader)
        
        # 更新学习率（如果是余弦退火）
        if not step_scheduler_each_batch:
            scheduler.step()
        
        # 记录损失
        loss_history.append((
            epoch + 1, 
            train_loss, val_loss,
            train_curve_loss, val_curve_loss,
            train_dtw_loss, val_dtw_loss,
            train_z_loss, val_z_loss
        ))
        
        # 输出当前训练状态
        print(f"Mapper Epoch {epoch+1}/{args.epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Train Curve: {train_curve_loss:.4f}, "
              f"Val Curve: {val_curve_loss:.4f}")
        
        # 可视化生成结果
        if (epoch + 1) % args.vis_interval == 0:
            plt.figure(figsize=(15, 10))
            for i, (features, curves) in enumerate(val_loader):
                if i >= 4:  # 只可视化前4个批次
                    break
                    
                features = features.to(args.device)
                curves = curves.to(args.device)
                
                # 通过映射器预测潜在向量
                pred_z = mapper(features)
                # 通过解码器重建曲线
                pred_curves = decoder(pred_z, features)
                
                # 可视化8个样本
                for j in range(min(4, features.size(0))):
                    plt.subplot(4, 4, i*4+j+1)
                    
                    # 获取原始曲线和预测曲线
                    orig = curves[j].detach().cpu().numpy()
                    pred = pred_curves[j].detach().cpu().numpy()
                    
                    plt.plot(orig[:, 0], orig[:, 1], 'b-', label='Original')
                    plt.plot(pred[:, 0], pred[:, 1], 'r--', label='Predicted')
                    
                    if j == 0 and i == 0:
                        plt.legend()
                    
                    plt.title(f'Batch {i+1}, Sample {j+1}')
                    plt.grid(True)
            
            plt.suptitle(f'Epoch {epoch+1} Mapper Predictions')
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/mapper_epoch_{epoch+1}.png", dpi=300)
            plt.close()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # 保存最佳模型
            save_model(mapper, f"{args.checkpoint_dir}/best_mapper.pt")
            print(f"Saved best mapper with val_loss: {val_loss:.4f}")
        else:
            early_stop_counter += 1
            
        # 提前停止检查
        if early_stop_counter >= early_stop_patience and args.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_model(mapper, f"{args.checkpoint_dir}/mapper_epoch_{epoch+1}.pt")
    
    # 保存最终模型
    save_model(mapper, f"{args.checkpoint_dir}/final_mapper.pt")
    
    # 保存损失历史
    loss_df = pd.DataFrame(loss_history, columns=[
        "Epoch", "Train_Loss", "Val_Loss",
        "Train_Curve", "Val_Curve",
        "Train_DTW", "Val_DTW",
        "Train_Z", "Val_Z"
    ])
    loss_df.to_csv(f"{args.checkpoint_dir}/mapper_loss_history.csv", index=False)
    
    # 绘制损失曲线
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(loss_df["Epoch"], loss_df["Train_Loss"], label="Train Loss", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Mapper Training Loss Curve")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(loss_df["Epoch"], loss_df["Train_Curve"], label="Train Curve", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Curve"], label="Val Curve", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Curve Loss")
    plt.title("Curve Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(loss_df["Epoch"], loss_df["Train_DTW"], label="Train DTW", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_DTW"], label="Val DTW", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("DTW Loss")
    plt.title("DTW Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(loss_df["Epoch"], loss_df["Train_Z"], label="Train Z", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Z"], label="Val Z", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Z Loss")
    plt.title("Latent Space Alignment")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.checkpoint_dir}/mapper_loss_curves.png", dpi=300)
    plt.close()
    
    # 可视化学习率变化
    if hasattr(scheduler, 'get_last_lr'):
        plt.figure(figsize=(10, 6))
        plt.plot(scheduler.get_last_lr())
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(f"{args.checkpoint_dir}/mapper_lr_schedule.png", dpi=300)
        plt.close()
    
    # 加载最佳模型并返回
    mapper.load_state_dict(torch.load(f"{args.checkpoint_dir}/best_mapper.pt"))
    
    return mapper