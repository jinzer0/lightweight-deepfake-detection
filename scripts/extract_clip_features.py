from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts._genimage_common import labels, load_config, manifest_rows
from src.data.dataset import ImageMetadataDataset
from src.features.cache import save_split_features
from src.features.clip_features import extract_clip_features, load_clip_model_and_preprocess


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Extract frozen open_clip image features for GenImage manifest rows.')
    p.add_argument('--config', default='configs/clip_only.yaml')
    p.add_argument('--manifest', required=False)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--split', choices=['train','val','test'], default='train')
    p.add_argument('--device', default=None)
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv); cfg=load_config(args.config)
    manifest=args.manifest or cfg.get('paths',{}).get('manifest_csv') or cfg.get('paths',{}).get('dataset_csv')
    if not manifest:
        raise SystemExit('--manifest or paths.manifest_csv is required')
    device=torch.device(args.device or cfg.get('project',{}).get('device','cpu'))
    model, preprocess = load_clip_model_and_preprocess(cfg, device)
    ds=ImageMetadataDataset(manifest, args.split, transform=preprocess, return_metadata=True)
    loader=DataLoader(ds, batch_size=int(cfg.get('data',{}).get('batch_size',16)), shuffle=False, num_workers=int(cfg.get('data',{}).get('num_workers',0)))
    features, y, meta=extract_clip_features(model, loader, device, normalize=bool(cfg.get('clip',{}).get('normalize_feature', True)))
    out=args.output_dir or cfg.get('paths',{}).get('feature_dir','artifacts/features')
    save_split_features(out, feature_type='clip', split=args.split, features=features, labels=y, rows=manifest_rows(manifest,args.split), metadata={'model_name':cfg.get('clip',{}).get('model_name','ViT-L-14')})
    print(f'wrote {features.shape[0]} CLIP features to {out}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
