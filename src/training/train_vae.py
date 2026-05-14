import argparse
import os
import random
import sys
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.vae import (
    EnhancedConditionalDecoder,
    EnhancedConditionalEncoder,
    enhanced_loss_function,
    reparameterize,
)
from training.metrics import evaluate_curve_model
from utils.dataloader import get_dataloader, save_normalization_params
from utils.utils import save_model, visualize_reconstructions


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
        pd.DataFrame(rows).to_csv(os.path.join(checkpoint_dir, "dataset_split_info_vae.csv"), index=False)


def train_vae(train_loader, val_loader, test_loader, args):
    set_seed(args.seed, deterministic=args.deterministic)

    encoder = EnhancedConditionalEncoder(
        z_dim=args.z_dim,
        feature_dim=8,
        latent_regularization=args.latent_reg,
    ).to(args.device)

    decoder = EnhancedConditionalDecoder(
        z_dim=args.z_dim,
        feature_dim=8,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = ExponentialLR(optimizer, gamma=args.lr_gamma)

    use_amp = bool(args.amp and args.device.startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_ctx = (lambda: torch.autocast(device_type="cuda", enabled=use_amp)) if args.device.startswith("cuda") else nullcontext

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, "visualizations"), exist_ok=True)
    _save_split_info(train_loader, val_loader, test_loader, args.checkpoint_dir)

    loss_history = []
    best_val_loss = float("inf")
    early_stop_counter = 0
    lr_history = []

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()

        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_dtw = 0.0

        kl_scale = 1.0
        if args.kl_warmup_epochs > 0:
            kl_scale = min(1.0, float(epoch + 1) / float(args.kl_warmup_epochs))
        kl_weight_eff = args.kl_weight * kl_scale

        for features, curves in tqdm(train_loader, desc=f"[Train CVAE] Epoch {epoch + 1}/{args.epochs}"):
            features = features.to(args.device, non_blocking=True)
            curves = curves.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                mu, logvar = encoder(curves, features)
                z = reparameterize(mu, logvar)
                recon = decoder(z, features)
                loss, recon_loss, kl_loss, dtw_loss = enhanced_loss_function(
                    recon,
                    curves,
                    mu,
                    logvar,
                    gamma=kl_weight_eff,
                    dtw_weight=args.dtw_weight,
                    shape_weight=args.shape_weight,
                    dtw_sample_size=args.dtw_sample_size,
                    dtw_window=args.dtw_window,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_recon += float(recon_loss)
            train_kl += float(kl_loss)
            train_dtw += float(dtw_loss)

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        train_dtw /= len(train_loader)

        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_dtw = 0.0

        with torch.no_grad():
            for features, curves in val_loader:
                features = features.to(args.device, non_blocking=True)
                curves = curves.to(args.device, non_blocking=True)

                with amp_ctx():
                    mu, logvar = encoder(curves, features)
                    z = mu
                    recon = decoder(z, features)
                    loss, recon_loss, kl_loss, dtw_loss = enhanced_loss_function(
                        recon,
                        curves,
                        mu,
                        logvar,
                        gamma=kl_weight_eff,
                        dtw_weight=args.dtw_weight,
                        shape_weight=args.shape_weight,
                        dtw_sample_size=args.dtw_sample_size,
                        dtw_window=args.dtw_window,
                    )

                val_loss += float(loss.item())
                val_recon += float(recon_loss)
                val_kl += float(kl_loss)
                val_dtw += float(dtw_loss)

        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)
        val_dtw /= len(val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)

        loss_history.append(
            (
                epoch + 1,
                train_loss,
                val_loss,
                train_recon,
                val_recon,
                train_kl,
                val_kl,
                train_dtw,
                val_dtw,
                kl_weight_eff,
                current_lr,
            )
        )

        print(
            f"CVAE Epoch {epoch + 1}/{args.epochs}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Recon: {train_recon:.4f}, Val Recon: {val_recon:.4f}, "
            f"KL_w: {kl_weight_eff:.4f}, LR: {current_lr:.2e}"
        )

        if (epoch + 1) % args.vis_interval == 0:
            visualize_reconstructions(epoch + 1, encoder, decoder, val_loader, args.device, os.path.join(args.checkpoint_dir, "visualizations"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_model(encoder, os.path.join(args.checkpoint_dir, "best_encoder.pt"))
            save_model(decoder, os.path.join(args.checkpoint_dir, "best_decoder.pt"))
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop_patience and args.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % args.save_interval == 0:
            save_model(encoder, os.path.join(args.checkpoint_dir, f"encoder_epoch_{epoch + 1}.pt"))
            save_model(decoder, os.path.join(args.checkpoint_dir, f"decoder_epoch_{epoch + 1}.pt"))

    save_model(encoder, os.path.join(args.checkpoint_dir, "final_encoder.pt"))
    save_model(decoder, os.path.join(args.checkpoint_dir, "final_decoder.pt"))

    loss_df = pd.DataFrame(
        loss_history,
        columns=[
            "Epoch",
            "Train_Loss",
            "Val_Loss",
            "Train_Recon",
            "Val_Recon",
            "Train_KL",
            "Val_KL",
            "Train_DTW",
            "Val_DTW",
            "KL_Weight_Effective",
            "LR",
        ],
    )
    loss_df.to_csv(os.path.join(args.checkpoint_dir, "vae_loss_history.csv"), index=False)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(loss_df["Epoch"], loss_df["Train_Loss"], label="Train Loss", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_Loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("CVAE Training Loss")
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
    plt.ylabel("KL")
    plt.title("KL Divergence")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(loss_df["Epoch"], loss_df["Train_DTW"], label="Train DTW", linewidth=2)
    plt.plot(loss_df["Epoch"], loss_df["Val_DTW"], label="Val DTW", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("DTW")
    plt.title("DTW Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, "vae_loss_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(lr_history) + 1), lr_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, "learning_rate_schedule.png"), dpi=300)
    plt.close()

    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_encoder.pt"), map_location=args.device))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_decoder.pt"), map_location=args.device))
    encoder.eval()
    decoder.eval()

    def _predict_fn(features, curves):
        mu, _ = encoder(curves, features)
        return decoder(mu, features)

    summary_rows = []
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        m = evaluate_curve_model(loader, args.device, _predict_fn)
        summary_rows.append({"Split": split_name, **m})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.checkpoint_dir, "vae_eval_metrics.csv"), index=False)
    with open(os.path.join(args.checkpoint_dir, "vae_eval_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(summary_df.to_string(index=False))
        f.write("\n")

    print("\n[VAE Evaluation]\n" + summary_df.to_string(index=False))
    return encoder, decoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train enhanced CVAE model")

    parser.add_argument("--feature_csv", type=str, default="./data/input/selected_pairs.csv")
    parser.add_argument("--curve_dir", type=str, default="./data/output")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--latent_reg", type=str, default="tanh")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_gamma", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--kl_warmup_epochs", type=int, default=40)
    parser.add_argument("--dtw_weight", type=float, default=0.1)
    parser.add_argument("--shape_weight", type=float, default=0.1)
    parser.add_argument("--dtw_sample_size", type=int, default=32)
    parser.add_argument("--dtw_window", type=int, default=6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

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
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, test_loader, dataset = get_dataloader(
        feature_csv_path=args.feature_csv,
        curve_folder_path=args.curve_dir,
        batch_size=args.batch_size,
        shuffle=True,
        use_saved_norm=False,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    if val_loader is None:
        raise ValueError("val_split must be > 0 for VAE training.")

    save_normalization_params(dataset.get_normalization_params(), args.checkpoint_dir)
    train_vae(train_loader, val_loader, test_loader, args)


if __name__ == "__main__":
    main()
