from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.metrics import binary_metrics
from src.training.utils import EarlyStopping, set_seed


def train_feature_model(model: nn.Module, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, output_dir: str | Path, config: dict[str, Any], *, best_metric: str = 'roc_auc') -> dict[str, Any]:
    set_seed(int(config.get('seed', 42)))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(config.get('device', 'cpu')))
    model.to(device)
    batch_size = int(config.get('batch_size', 32))
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()), batch_size=batch_size, shuffle=True)
    val_tensor = torch.from_numpy(val_x).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get('learning_rate', 1e-4)), weight_decay=float(config.get('weight_decay', 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(config.get('epochs', 1))))
    loss_fn = nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(int(config.get('early_stopping_patience', 5)), mode='max')
    rows = []
    best_score = -1.0
    metrics: dict[str, Any] = {}
    for epoch in range(1, int(config.get('epochs', 1)) + 1):
        model.train(); losses=[]
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get('gradient_clip', 1.0)))
            opt.step(); losses.append(float(loss.detach().cpu()))
        sched.step(); model.eval()
        with torch.inference_mode():
            probs=torch.sigmoid(model(val_tensor)).detach().cpu().numpy()
        metrics = binary_metrics(val_y, probs)
        score = metrics.get(best_metric)
        numeric_score = float(score) if isinstance(score, (int, float)) else -1.0
        row={'epoch': epoch, 'train_loss': float(np.mean(losses)) if losses else 0.0, **metrics}
        rows.append(row)
        if numeric_score >= best_score:
            best_score = numeric_score
            torch.save({'model_state_dict': model.state_dict(), 'config': config, 'metrics': metrics}, out/'best_checkpoint.pt')
        torch.save({'model_state_dict': model.state_dict(), 'config': config, 'metrics': metrics}, out/'last_checkpoint.pt')
        if stopper.step(numeric_score):
            break
    (out/'config_snapshot.json').write_text(json.dumps(config, indent=2, sort_keys=True), encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')
    with (out/'training_log.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['epoch'])
        writer.writeheader(); writer.writerows(rows)
    return metrics
