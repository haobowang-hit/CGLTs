import argparse
import os
import random
import sys
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.vae import (
    EnhancedANNMapper,
    EnhancedConditionalDecoder,
    EnhancedConditionalEncoder,
    dtw_2d_loss,
)
from training.metrics import evaluate_curve_model
from utils.dataloader import get_dataloader
from utils.utils import save_model


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, fallback to CPU.")
        return "cpu"
    return device_arg


def set_seed(seed: int, deterministic: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _save_split_info(train_loader, val_loader, test_loader, checkpoint_dir: str):
    rows = []
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        subset = loader.dataset
        if hasattr(subset, "indices"):
            for idx in subset.indices:
                rows.append({"Index": int(idx), "Set": name})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(checkpoint_dir, "dataset_split_info_mapper.csv"), index=False)


def train_mapper(train_loader, val_loader, test_loader, encoder, decoder, args):
    set_seed(args.seed, deterministic=args.deterministic)

    # Stage-1: freeze encoder/decoder, train mapper.
    encoder.eval()
    decoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    mapper = EnhancedANNMapper(feature_dim=8, z_dim=args.z_dim).to(args.device)

    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=args.mapper_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    use_amp = bool(args.amp and args.device.startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_ctx = (lambda: torch.autocast(device_type="cuda", enabled=use_amp)) if args.device.startswith("cuda") else nullcontext

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    vis_dir = os.path.join(args.checkpoint_dir, "mapper_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    _save_split_info(train_loader, val_loader, test_loader, args.checkpoint_dir)

    loss_history = []
    best_val_loss = float("inf")
    early_stop_counter = 0
    lr_history = []

    for epoch in range(args.epochs):
        mapper.train()
        train_loss = 0.0
        train_curve = 0.0
        train_dtw = 0.0
        train_z = 0.0

        for features, curves in tqdm(train_loader, desc=f"[Mapper Train] Epoch {epoch + 1}/{args.epochs}"):
            features = features.to(args.device, non_blocking=True)
            curves = curves.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                mu, _ = encoder(curves, features)

            with amp_ctx():
                pred_z = mapper(features)
                recon = decoder(pred_z, features)
                curve_loss = F.smooth_l1_loss(recon, curves)
                dtw_penalty = dtw_2d_loss(recon, curves, sample_size=args.dtw_sample_size, window=args.dtw_window)
                z_loss = F.mse_loss(pred_z, mu)
                total_loss = curve_loss + args.dtw_weight * dtw_penalty + args.z_weight * z_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(total_loss.item())
            train_curve += float(curve_loss.item())
            train_dtw += float(dtw_penalty.item())
            train_z += float(z_loss.item())

        train_loss /= len(train_loader)
        train_curve /= len(train_loader)
        train_dtw /= len(train_loader)
        train_z /= len(train_loader)

        mapper.eval()
        val_loss = 0.0
        val_curve = 0.0
        val_dtw = 0.0
        val_z = 0.0

        with torch.no_grad():
            for features, curves in val_loader:
                features = features.to(args.device, non_blocking=True)
                curves = curves.to(args.device, non_blocking=True)

                mu, _ = encoder(curves, features)
                with amp_ctx():
                    pred_z = mapper(features)
                    recon = decoder(pred_z, features)
                    curve_loss = F.smooth_l1_loss(recon, curves)
                    dtw_penalty = dtw_2d_loss(recon, curves, sample_size=args.dtw_sample_size, window=args.dtw_window)
                    z_loss = F.mse_loss(pred_z, mu)
                    total_loss = curve_loss + args.dtw_weight * dtw_penalty + args.z_weight * z_loss

                val_loss += float(total_loss.item())
                val_curve += float(curve_loss.item())
                val_dtw += float(dtw_penalty.item())
                val_z += float(z_loss.item())

        val_loss /= len(val_loader)
        val_curve /= len(val_loader)
        val_dtw /= len(val_loader)
        val_z /= len(val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)

        loss_history.append(
            (
                epoch + 1,
                train_loss,
                val_loss,
                train_curve,
                val_curve,
                train_dtw,
                val_dtw,
                train_z,
                val_z,
                current_lr,
            )
        )

        print(
            f"Mapper Epoch {epoch + 1}/{args.epochs}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Curve: {train_curve:.4f}, Val Curve: {val_curve:.4f}, "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_model(mapper, os.path.join(args.checkpoint_dir, "best_mapper.pt"))
            print(f"Saved best mapper with val_loss: {val_loss:.4f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop_patience and args.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % args.save_interval == 0:
            save_model(mapper, os.path.join(args.checkpoint_dir, f"mapper_epoch_{epoch + 1}.pt"))

        if (epoch + 1) % args.vis_interval == 0:
            mapper.eval()
            with torch.no_grad():
                features, curves = next(iter(val_loader))
                features = features.to(args.device)
                curves = curves.to(args.device)
                pred_z = mapper(features)
                pred_curves = decoder(pred_z, features)

            plt.figure(figsize=(12, 8))
            k = min(6, curves.size(0))
            for i in range(k):
                ax = plt.subplot(2, 3, i + 1)
                orig = curves[i].detach().cpu().numpy()
                pred = pred_curves[i].detach().cpu().numpy()
                ax.plot(orig[:, 0], orig[:, 1], "b-", label="Original")
                ax.plot(pred[:, 0], pred[:, 1], "r--", label="Predicted")
                if i == 0:
                    ax.legend()
                ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"mapper_epoch_{epoch + 1}.png"), dpi=300)
            plt.close()

    save_model(mapper, os.path.join(args.checkpoint_dir, "final_mapper.pt"))

    # Stage-2: optional short joint fine-tune of mapper + decoder.
    if args.joint_finetune_epochs > 0:
        print(f"\n[Joint Fine-tune] epochs={args.joint_finetune_epochs}")
        for p in decoder.parameters():
            p.requires_grad_(True)
        decoder.train()
        mapper.train()

        joint_opt = torch.optim.AdamW(
            list(mapper.parameters()) + list(decoder.parameters()),
            lr=args.joint_lr,
            weight_decay=args.weight_decay,
        )
        joint_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for j_epoch in range(args.joint_finetune_epochs):
            epoch_loss = 0.0
            for features, curves in tqdm(train_loader, desc=f"[Joint] {j_epoch + 1}/{args.joint_finetune_epochs}"):
                features = features.to(args.device, non_blocking=True)
                curves = curves.to(args.device, non_blocking=True)

                with torch.no_grad():
                    mu, _ = encoder(curves, features)

                joint_opt.zero_grad(set_to_none=True)
                with amp_ctx():
                    pred_z = mapper(features)
                    recon = decoder(pred_z, features)
                    curve_loss = F.smooth_l1_loss(recon, curves)
                    dtw_penalty = dtw_2d_loss(recon, curves, sample_size=args.dtw_sample_size, window=args.dtw_window)
                    z_loss = F.mse_loss(pred_z, mu)
                    loss = curve_loss + args.dtw_weight * dtw_penalty + args.z_weight * z_loss

                joint_scaler.scale(loss).backward()
                joint_scaler.unscale_(joint_opt)
                torch.nn.utils.clip_grad_norm_(list(mapper.parameters()) + list(decoder.parameters()), max_norm=args.grad_clip)
                joint_scaler.step(joint_opt)
                joint_scaler.update()
                epoch_loss += float(loss.item())

            print(f"Joint Epoch {j_epoch + 1}/{args.joint_finetune_epochs}, Loss={epoch_loss / len(train_loader):.4f}")

        save_model(decoder, os.path.join(args.checkpoint_dir, "best_decoder_for_mapper.pt"))
        save_model(mapper, os.path.join(args.checkpoint_dir, "best_mapper.pt"))

    loss_df = pd.DataFrame(
        loss_history,
        columns=[
            "Epoch",
            "Train_Loss",
            "Val_Loss",
            "Train_Curve",
            "Val_Curve",
            "Train_DTW",
            "Val_DTW",
            "Train_Z",
            "Val_Z",
            "LR",
        ],
    )
    loss_df.to_csv(os.path.join(args.checkpoint_dir, "mapper_loss_history.csv"), index=False)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(loss_df["Epoch"], loss_df["Train_Loss"], label="Train Loss", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Loss"], label="Val Loss", linewidth=2)
    plt.title("Mapper Total Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(loss_df["Epoch"], loss_df["Train_Curve"], label="Train Curve", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Curve"], label="Val Curve", linewidth=2)
    plt.title("Curve Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(loss_df["Epoch"], loss_df["Train_DTW"], label="Train DTW", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_DTW"], label="Val DTW", linewidth=2)
    plt.title("DTW Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(loss_df["Epoch"], loss_df["Train_Z"], label="Train Z", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Z"], label="Val Z", linewidth=2)
    plt.title("Latent Alignment")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, "mapper_loss_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(lr_history) + 1), lr_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Mapper Learning Rate")
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, "mapper_lr_schedule.png"), dpi=300)
    plt.close()

    best_mapper_path = os.path.join(args.checkpoint_dir, "best_mapper.pt")
    if not os.path.exists(best_mapper_path):
        best_mapper_path = os.path.join(args.checkpoint_dir, "final_mapper.pt")
    mapper.load_state_dict(torch.load(best_mapper_path, map_location=args.device))
    mapper.eval()

    # If joint fine-tuned decoder exists, evaluate with it; otherwise use input decoder.
    tuned_decoder_path = os.path.join(args.checkpoint_dir, "best_decoder_for_mapper.pt")
    if os.path.exists(tuned_decoder_path):
        decoder.load_state_dict(torch.load(tuned_decoder_path, map_location=args.device))
    decoder.eval()

    def _predict_fn(features, curves):
        pred_z = mapper(features)
        return decoder(pred_z, features)

    summary_rows = []
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        m = evaluate_curve_model(loader, args.device, _predict_fn)
        summary_rows.append({"Split": split_name, **m})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.checkpoint_dir, "mapper_eval_metrics.csv"), index=False)
    with open(os.path.join(args.checkpoint_dir, "mapper_eval_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(summary_df.to_string(index=False))
        f.write("\n")

    print("\n[Mapper Evaluation]\n" + summary_df.to_string(index=False))
    return mapper


def parse_args():
    parser = argparse.ArgumentParser(description="Train enhanced mapper network")

    parser.add_argument("--feature_csv", type=str, default="./data/input/selected_pairs.csv")
    parser.add_argument("--curve_dir", type=str, default="./data/output")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--encoder_path", type=str, default="")
    parser.add_argument("--decoder_path", type=str, default="")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--mapper_lr", type=float, default=8e-4)
    parser.add_argument("--lr_step_size", type=int, default=25)
    parser.add_argument("--lr_gamma", type=float, default=0.85)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dtw_weight", type=float, default=0.1)
    parser.add_argument("--z_weight", type=float, default=0.6)
    parser.add_argument("--dtw_sample_size", type=int, default=32)
    parser.add_argument("--dtw_window", type=int, default=6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--joint_finetune_epochs", type=int, default=5)
    parser.add_argument("--joint_lr", type=float, default=3e-4)

    parser.add_argument("--early_stop", dest="early_stop", action="store_true")
    parser.add_argument("--no_early_stop", dest="early_stop", action="store_false")
    parser.set_defaults(early_stop=True)
    parser.add_argument("--early_stop_patience", type=int, default=30)

    parser.add_argument("--vis_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=50)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)

    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=False)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.device)

    train_loader, val_loader, test_loader, _ = get_dataloader(
        feature_csv_path=args.feature_csv,
        curve_folder_path=args.curve_dir,
        batch_size=args.batch_size,
        shuffle=True,
        use_saved_norm=True,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    if val_loader is None:
        raise ValueError("val_split must be > 0 for mapper training.")

    encoder_path = args.encoder_path or os.path.join(args.checkpoint_dir, "best_encoder.pt")
    decoder_path = args.decoder_path or os.path.join(args.checkpoint_dir, "best_decoder.pt")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_path}")

    encoder = EnhancedConditionalEncoder(z_dim=args.z_dim, feature_dim=8).to(args.device)
    decoder = EnhancedConditionalDecoder(z_dim=args.z_dim, feature_dim=8).to(args.device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=args.device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=args.device))

    train_mapper(train_loader, val_loader, test_loader, encoder, decoder, args)


if __name__ == "__main__":
    main()
