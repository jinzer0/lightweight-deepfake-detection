from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.features.cache import load_split_features
from src.models.clip_classifier import ClipClassifier
from src.models.frequency_classifier import FrequencyClassifier
from src.models.fusion_mlp import FusionMLP
from src.training.trainer import train_feature_model


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Train frequency-only GenImage classifier.')
    p.add_argument('--config', default='configs/frequency_only.yaml')
    p.add_argument('--feature_dir', default=None)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--epochs', type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    feature_dir=args.feature_dir or cfg.get('paths',{}).get('feature_dir','artifacts/features')
    output_dir=args.output_dir or cfg.get('paths',{}).get('checkpoint_dir','artifacts/checkpoints/frequency_only')
    train_x, train_y = load_split_features(feature_dir, 'frequency', 'train')
    val_x, val_y = load_split_features(feature_dir, 'frequency', 'val')
    model=FrequencyClassifier(input_dim=int(train_x.shape[1]), hidden_dim=int(cfg.get('classifier',{}).get('hidden_dim',256)), dropout=float(cfg.get('classifier',{}).get('dropout',0.2)))
    train_cfg={**cfg.get('train',{}), 'seed': cfg.get('project',{}).get('seed',42), 'device': cfg.get('project',{}).get('device','cpu'), 'batch_size': cfg.get('data',{}).get('batch_size',32)}
    if args.epochs is not None:
        train_cfg['epochs']=args.epochs
    metrics=train_feature_model(model, train_x.astype(np.float32), train_y, val_x.astype(np.float32), val_y, output_dir, train_cfg, best_metric='roc_auc')
    print(metrics)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
