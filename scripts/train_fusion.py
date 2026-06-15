from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.features.cache import assert_aligned_feature_rows, load_split_feature_bundle
from src.models.clip_classifier import ClipClassifier
from src.models.frequency_classifier import FrequencyClassifier
from src.models.fusion_mlp import FusionMLP
from src.training.trainer import train_feature_model


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Train fusion GenImage classifier.')
    p.add_argument('--config', default='configs/fusion.yaml')
    p.add_argument('--feature_dir', default=None)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--epochs', type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    feature_dir=args.feature_dir or cfg.get('paths',{}).get('feature_dir','artifacts/features')
    output_dir=args.output_dir or cfg.get('paths',{}).get('checkpoint_dir','artifacts/checkpoints/fusion')
    clip_train, train_y, clip_train_rows = load_split_feature_bundle(feature_dir, 'clip', 'train')
    freq_train, freq_train_y, freq_train_rows = load_split_feature_bundle(feature_dir, 'frequency', 'train')
    clip_val, val_y, clip_val_rows = load_split_feature_bundle(feature_dir, 'clip', 'val')
    freq_val, freq_val_y, freq_val_rows = load_split_feature_bundle(feature_dir, 'frequency', 'val')
    assert_aligned_feature_rows(clip_train_rows, freq_train_rows)
    assert_aligned_feature_rows(clip_val_rows, freq_val_rows)
    if not np.array_equal(train_y, freq_train_y) or not np.array_equal(val_y, freq_val_y):
        raise ValueError('clip and frequency labels are not aligned')
    train_x = np.concatenate([clip_train, freq_train], axis=1)
    val_x = np.concatenate([clip_val, freq_val], axis=1)
    model=FusionMLP(clip_dim=int(clip_train.shape[1]), freq_dim=int(freq_train.shape[1]), hidden_dim=int(cfg.get('classifier',{}).get('hidden_dim',512)), dropout=float(cfg.get('classifier',{}).get('dropout',0.2)))
    train_cfg={**cfg.get('train',{}), 'seed': cfg.get('project',{}).get('seed',42), 'device': cfg.get('project',{}).get('device','cpu'), 'batch_size': cfg.get('data',{}).get('batch_size',32)}
    if args.epochs is not None:
        train_cfg['epochs']=args.epochs
    metrics=train_feature_model(model, train_x.astype(np.float32), train_y, val_x.astype(np.float32), val_y, output_dir, train_cfg, best_metric='roc_auc')
    print(metrics)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
