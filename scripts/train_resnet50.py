from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts._genimage_common import load_config
from src.data.dataset import GenImageDataset
from src.data.transforms import get_eval_transform, get_train_transform
from src.models.resnet50_baseline import ResNet50Baseline
from src.training.seed import set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a ResNet50 baseline on GenImage manifest images.")
    p.add_argument("--config", default="configs/resnet50.yaml")
    p.add_argument("--manifest")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--epochs", type=int)
    p.add_argument("--device")
    p.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet weights for offline smoke runs.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = int(cfg.get("project", {}).get("seed", cfg.get("seed", 42)))
    set_seed(seed)
    manifest = args.manifest or cfg.get("paths", {}).get("manifest_csv") or cfg.get("data", {}).get("manifest_csv")
    if not manifest:
        raise SystemExit("--manifest or config manifest path is required")
    output_dir = Path(args.output_dir or cfg.get("paths", {}).get("checkpoint_dir", "artifacts/checkpoints/resnet50_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or cfg.get("project", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    image_size = int(cfg.get("data", {}).get("image_size", 512))
    batch_size = int(cfg.get("data", {}).get("batch_size", 16))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("train", {}).get("epochs", 1))
    train_ds = GenImageDataset(manifest, "train", transform=get_train_transform(image_size))
    val_ds = GenImageDataset(manifest, "val", transform=get_eval_transform(image_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(cfg.get("data", {}).get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=int(cfg.get("data", {}).get("num_workers", 0)))
    pretrained = bool(cfg.get("model", {}).get("pretrained", True)) and not args.no_pretrained
    model = ResNet50Baseline(pretrained=pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("train", {}).get("learning_rate", 1e-4)), weight_decay=float(cfg.get("train", {}).get("weight_decay", 1e-4)))
    best_val_loss = float("inf")
    log_rows = []
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []
        for images, labels, _meta in tqdm(train_loader, desc=f"resnet50 train {epoch}"):
            images = images.to(device); targets = labels.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).squeeze(-1), targets)
            loss.backward(); optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_losses = []
        model.eval()
        with torch.inference_mode():
            for images, labels, _meta in val_loader:
                images = images.to(device); targets = labels.float().to(device)
                val_losses.append(float(criterion(model(images).squeeze(-1), targets).detach().cpu()))
        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        payload = {"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "val_loss": val_loss, "pretrained": pretrained}
        torch.save(payload, output_dir / "last_checkpoint.pt")
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(payload, output_dir / "best_checkpoint.pt")
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader(); writer.writerows(log_rows)
    (output_dir / "metrics.json").write_text(json.dumps({"best_val_loss": best_val_loss, "epochs_ran": epochs}, indent=2), encoding="utf-8")
    (output_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"wrote checkpoints to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
