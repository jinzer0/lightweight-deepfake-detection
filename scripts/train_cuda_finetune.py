from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any, TypeVar, cast

import numpy as np
import yaml
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is a progress nicety
    tqdm = None

from _path import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.manifest import read_manifest, validate_manifest_rows  # noqa: E402


PREDICTION_COLUMNS = ["path", "sample_id", "label", "pred_label", "prob_fake", "score", "split"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
T = TypeVar("T")


@dataclass(frozen=True)
class TrialConfig:
    trial_id: str
    epochs: int
    batch_size: int
    head_lr: float
    encoder_lr: float
    weight_decay: float
    dropout: float
    unfreeze_last_n: int


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    best_epoch: int
    threshold: float
    val_accuracy: float
    val_roc_auc: float
    val_f1_fake: float
    test_accuracy: float
    test_roc_auc: float
    train_loss: float
    config: TrialConfig


class ManifestImageDataset:
    def __init__(self, rows: list[dict[str, str]], *, image_size: int, train: bool) -> None:
        self.rows = rows
        self.image_size = int(image_size)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, str, str, str]:
        torch = _torch()
        row = self.rows[index]
        path = Path(row["root"]) / row["rel_path"]
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            if self.train and random.random() < 0.5:
                rgb = ImageOps.mirror(rgb)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return (tensor - mean) / std, label, row["sample_id"], row["rel_path"], row["split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA fine-tune a 512x512 torchvision image classifier with a small hyperparameter grid.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest v1 CSV with real=0 fake=1 labels.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for fine-tuning artifacts.")
    parser.add_argument("--model_arch", choices=["resnet18", "resnet50"], default="resnet18", help="Torchvision classifier backbone.")
    parser.add_argument("--image_size", type=int, default=512, help="Square input image size. Default is 512.")
    parser.add_argument("--device", choices=["cuda", "auto"], default="cuda", help="Training device. cuda refuses CPU fallback.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per hyperparameter trial.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training and evaluation batch size.")
    parser.add_argument("--max_trials", type=int, default=4, help="Maximum number of built-in hyperparameter trials to run.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA AMP when available.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
        print(f"best_trial={result.trial_id}")
        print(f"val_accuracy={result.val_accuracy:.6f} val_roc_auc={result.val_roc_auc:.6f} test_accuracy={result.test_accuracy:.6f} test_roc_auc={result.test_roc_auc:.6f}")
        print(f"wrote artifacts: {args.output_dir}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"cuda fine-tuning failed: {error}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> TrialResult:
    torch = _torch()
    _seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    rows = read_manifest(args.manifest)
    validate_manifest_rows(rows, strict=True)
    splits = {split: [row for row in rows if row["split"] == split] for split in ["train", "val", "test"]}
    _validate_splits(splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_configs = _trial_grid(int(args.epochs), int(args.batch_size))[: int(args.max_trials)]
    all_results: list[TrialResult] = []
    best_result: TrialResult | None = None
    best_state: dict[str, Any] | None = None

    for trial in _progress(trial_configs, desc="Fine-tune trials", unit="trial"):
        print(f"running {trial.trial_id}: {asdict(trial)}")
        module = _build_model(str(args.model_arch), trial.dropout, trial.unfreeze_last_n).to(device)
        optimizer = _optimizer(module, trial)
        criterion = torch.nn.BCEWithLogitsLoss()
        train_loader = _loader(splits["train"], int(args.image_size), trial.batch_size, True, int(args.num_workers))
        val_loader = _loader(splits["val"], int(args.image_size), trial.batch_size, False, int(args.num_workers))
        test_loader = _loader(splits["test"], int(args.image_size), trial.batch_size, False, int(args.num_workers))
        scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device == "cuda"))
        trial_best: TrialResult | None = None
        trial_best_state: dict[str, Any] | None = None
        epoch_iter = _progress(range(1, trial.epochs + 1), desc=f"{trial.trial_id} epochs", unit="epoch")
        for epoch in epoch_iter:
            train_loss = _train_epoch(module, train_loader, optimizer, criterion, scaler, device, bool(args.amp), desc=f"{trial.trial_id} train e{epoch}")
            val_prediction = _predict(module, val_loader, device, bool(args.amp), desc=f"{trial.trial_id} val e{epoch}")
            threshold = _best_threshold(val_prediction["labels"], val_prediction["prob_fake"])
            test_prediction = _predict(module, test_loader, device, bool(args.amp), desc=f"{trial.trial_id} test e{epoch}")
            val_metrics = _single_metrics(val_prediction["labels"], val_prediction["prob_fake"], threshold)
            test_metrics = _single_metrics(test_prediction["labels"], test_prediction["prob_fake"], threshold)
            candidate = TrialResult(
                trial_id=trial.trial_id,
                best_epoch=epoch,
                threshold=threshold,
                val_accuracy=val_metrics["accuracy"],
                val_roc_auc=val_metrics["roc_auc"],
                val_f1_fake=val_metrics["f1_fake"],
                test_accuracy=test_metrics["accuracy"],
                test_roc_auc=test_metrics["roc_auc"],
                train_loss=train_loss,
                config=trial,
            )
            _set_progress_postfix(epoch_iter, loss=train_loss, val_acc=candidate.val_accuracy, val_auc=candidate.val_roc_auc, test_acc=candidate.test_accuracy)
            print(f"{trial.trial_id} epoch={epoch} loss={train_loss:.6f} val_acc={candidate.val_accuracy:.6f} val_auc={candidate.val_roc_auc:.6f} test_acc={candidate.test_accuracy:.6f}")
            if trial_best is None or _is_better(candidate, trial_best):
                trial_best = candidate
                trial_best_state = copy.deepcopy(module.state_dict())
        if trial_best is None or trial_best_state is None:
            raise RuntimeError(f"trial {trial.trial_id} did not produce a result")
        all_results.append(trial_best)
        if best_result is None or _is_better(trial_best, best_result):
            best_result = trial_best
            best_state = trial_best_state

    if best_result is None or best_state is None:
        raise RuntimeError("no hyperparameter trial completed")
    module = _build_model(str(args.model_arch), best_result.config.dropout, best_result.config.unfreeze_last_n).to(device)
    module.load_state_dict(best_state)
    all_prediction = _predict(module, _loader(rows, int(args.image_size), best_result.config.batch_size, False, int(args.num_workers)), device, bool(args.amp), desc="Predict all splits")
    predictions = _prediction_rows(rows, all_prediction, best_result.threshold)
    metrics = _metrics_by_split(rows, all_prediction, best_result.threshold)
    _write_outputs(args, best_result, all_results, metrics, predictions, best_state, device)
    return best_result


def _progress(iterable: Iterable[T], *, desc: str, unit: str, total: int | None = None, leave: bool = True) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


def _set_progress_postfix(progress: Iterable[object], **metrics: object) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if set_postfix is None:
        return
    set_postfix({key: _format_progress_value(value) for key, value in metrics.items()})


def _format_progress_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def _build_model(model_arch: str, dropout: float, unfreeze_last_n: int) -> Any:
    torch = _torch()
    models = _torchvision_models()
    if model_arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_layers = ["layer4", "layer3", "layer2", "layer1"][: max(0, int(unfreeze_last_n))]
    for name, parameter in model.named_parameters():
        if any(name.startswith(layer_name) for layer_name in trainable_layers):
            parameter.requires_grad_(True)
    in_features = int(model.fc.in_features)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(float(dropout)), torch.nn.Linear(in_features, 1))
    return model


def _trial_grid(epochs: int, batch_size: int) -> list[TrialConfig]:
    return [
        TrialConfig("trial_01", epochs, batch_size, 1e-3, 1e-5, 1e-4, 0.10, 1),
        TrialConfig("trial_02", epochs, batch_size, 5e-4, 3e-5, 1e-2, 0.20, 1),
        TrialConfig("trial_03", epochs, batch_size, 1e-3, 1e-4, 1e-2, 0.20, 2),
        TrialConfig("trial_04", epochs, batch_size, 3e-4, 1e-4, 5e-2, 0.10, 2),
        TrialConfig("trial_05", epochs, batch_size, 1e-3, 0.0, 1e-4, 0.10, 0),
    ]


def _optimizer(module: Any, trial: TrialConfig) -> Any:
    torch = _torch()
    head_params = list(module.fc.parameters())
    encoder_params = [parameter for name, parameter in module.named_parameters() if parameter.requires_grad and not name.startswith("fc.")]
    groups: list[dict[str, Any]] = [{"params": head_params, "lr": trial.head_lr}]
    if encoder_params and trial.encoder_lr > 0:
        groups.append({"params": encoder_params, "lr": trial.encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=trial.weight_decay)


def _train_epoch(module: Any, loader: Any, optimizer: Any, criterion: Any, scaler: Any, device: str, amp: bool, *, desc: str) -> float:
    torch = _torch()
    module.train()
    total_loss = 0.0
    total = 0
    for images, labels, *_ in _progress(loader, desc=desc, total=len(loader), unit="batch", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=amp and device == "cuda"):
            logits = module(images).squeeze(-1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
    return total_loss / max(total, 1)


def _predict(module: Any, loader: Any, device: str, amp: bool, *, desc: str) -> dict[str, np.ndarray | list[str]]:
    torch = _torch()
    module.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sample_ids: list[str] = []
    paths: list[str] = []
    splits: list[str] = []
    with torch.no_grad():
        for images, batch_labels, batch_sample_ids, batch_paths, batch_splits in _progress(loader, desc=desc, total=len(loader), unit="batch", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, enabled=amp and device == "cuda"):
                logits = module(images).squeeze(-1)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64))
            labels.append(batch_labels.detach().cpu().numpy().astype(np.int64))
            sample_ids.extend(str(value) for value in batch_sample_ids)
            paths.extend(str(value) for value in batch_paths)
            splits.extend(str(value) for value in batch_splits)
    return {"labels": np.concatenate(labels), "prob_fake": np.concatenate(probabilities), "sample_ids": sample_ids, "paths": paths, "splits": splits}


def _best_threshold(labels: np.ndarray, prob_fake: np.ndarray) -> float:
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        accuracy = float(accuracy_score(labels, (prob_fake >= threshold).astype(np.int64)))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def _single_metrics(labels: np.ndarray, prob_fake: np.ndarray, threshold: float) -> dict[str, float]:
    pred_label = (prob_fake >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_label, labels=[0, 1], average="binary", pos_label=1, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, pred_label)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "roc_auc": _safe_roc_auc(labels, prob_fake),
        "average_precision": _safe_average_precision(labels, prob_fake),
    }


def _metrics_by_split(rows: list[dict[str, str]], prediction: dict[str, np.ndarray | list[str]], threshold: float) -> dict[str, Any]:
    labels = cast(np.ndarray, prediction["labels"])
    prob_fake = cast(np.ndarray, prediction["prob_fake"])
    splits = np.asarray([row["split"] for row in rows], dtype=object)
    metrics: dict[str, Any] = {"sample_count": int(labels.shape[0]), "threshold": float(threshold), "probability_supported": True, "decision_score_only": False, "ranking_metric_input": "prob_fake", "overall": _single_metrics(labels, prob_fake, threshold), "splits": {}}
    for split in ["train", "val", "test"]:
        mask = splits == split
        metrics["splits"][split] = _single_metrics(labels[mask], prob_fake[mask], threshold)
        metrics["splits"][split]["sample_count"] = int(np.sum(mask))
    return metrics


def _prediction_rows(rows: list[dict[str, str]], prediction: dict[str, np.ndarray | list[str]], threshold: float) -> list[dict[str, str]]:
    labels = cast(np.ndarray, prediction["labels"])
    prob_fake = cast(np.ndarray, prediction["prob_fake"])
    pred_label = (prob_fake >= threshold).astype(np.int64)
    return [
        {"path": row["rel_path"], "sample_id": row["sample_id"], "label": str(int(labels[index])), "pred_label": str(int(pred_label[index])), "prob_fake": f"{float(prob_fake[index]):.17g}", "score": f"{float(prob_fake[index]):.17g}", "split": row["split"]}
        for index, row in enumerate(rows)
    ]


def _write_outputs(args: argparse.Namespace, best_result: TrialResult, all_results: list[TrialResult], metrics: dict[str, Any], predictions: list[dict[str, str]], state: dict[str, Any], device: str) -> None:
    torch = _torch()
    output_dir = Path(args.output_dir)
    torch.save({"model_state_dict": state, "best_trial": asdict(best_result), "model_arch": args.model_arch, "image_size": int(args.image_size)}, output_dir / "best_checkpoint.pt")
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "tuning_summary.json", {"best_trial": asdict(best_result), "trials": [asdict(result) for result in all_results]})
    _write_csv(output_dir / "tuning_results.csv", [asdict(result) | asdict(result.config) for result in all_results])
    _write_predictions(output_dir / "predictions.csv", predictions)
    _write_yaml(
        output_dir / "config.yaml",
        {"mode": "cuda_512_resnet_finetune", "model_arch": args.model_arch, "image_size": int(args.image_size), "device": device, "threshold": float(best_result.threshold), "probability_supported": True, "decision_score_only": False, "streamlit_probability_eligible": False, "seed": int(args.seed), "manifest_path": str(args.manifest), "best_trial": asdict(best_result), "created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv},
    )


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def _write_yaml(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(data, file_obj, sort_keys=False)


def _loader(rows: list[dict[str, str]], image_size: int, batch_size: int, shuffle: bool, num_workers: int) -> Any:
    torch = _torch()
    return torch.utils.data.DataLoader(ManifestImageDataset(rows, image_size=image_size, train=shuffle), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def _resolve_device(device: str) -> str:
    torch = _torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _validate_splits(splits: dict[str, list[dict[str, str]]]) -> None:
    for split, rows in splits.items():
        labels = {row["label"] for row in rows}
        if labels != {"0", "1"}:
            raise ValueError(f"split {split} must contain both labels, got {sorted(labels)}")


def _safe_roc_auc(labels: np.ndarray, prob_fake: np.ndarray) -> float:
    if set(np.unique(labels).astype(int).tolist()) != {0, 1}:
        return float("nan")
    return float(roc_auc_score(labels, prob_fake))


def _safe_average_precision(labels: np.ndarray, prob_fake: np.ndarray) -> float:
    if set(np.unique(labels).astype(int).tolist()) != {0, 1}:
        return float("nan")
    return float(average_precision_score(labels, prob_fake))


def _is_better(candidate: TrialResult, incumbent: TrialResult) -> bool:
    return (candidate.val_roc_auc, candidate.val_accuracy, candidate.val_f1_fake, -candidate.train_loss) > (incumbent.val_roc_auc, incumbent.val_accuracy, incumbent.val_f1_fake, -incumbent.train_loss)


def _seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for CUDA fine-tuning") from exc
    return torch


def _torchvision_models() -> Any:
    try:
        import torchvision.models as models
    except ModuleNotFoundError as exc:
        raise RuntimeError("torchvision is required for CUDA fine-tuning") from exc
    return models


if __name__ == "__main__":
    raise SystemExit(main())
