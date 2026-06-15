from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from scripts._genimage_common import labels, load_config, manifest_rows, path_for
from src.features.cache import save_split_features
from src.features.frequency_features import extract_genimage_frequency_features, genimage_frequency_dim


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Extract GenImage frequency feature caches.')
    p.add_argument('--config', default='configs/frequency_only.yaml')
    p.add_argument('--manifest', required=False)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--split', choices=['train','val','test'], default='train')
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv); cfg=load_config(args.config)
    manifest=args.manifest or cfg.get('paths',{}).get('manifest_csv') or cfg.get('paths',{}).get('dataset_csv')
    if not manifest:
        raise SystemExit('--manifest or paths.manifest_csv is required')
    rows=manifest_rows(manifest, args.split)
    feats=np.asarray([extract_genimage_frequency_features(path_for(row), cfg) for row in rows], dtype=np.float32)
    if feats.size == 0:
        feats=np.empty((0, genimage_frequency_dim(cfg)), dtype=np.float32)
    out=args.output_dir or cfg.get('paths',{}).get('feature_dir','artifacts/features')
    save_split_features(out, feature_type='frequency', split=args.split, features=feats, labels=labels(rows), rows=rows, metadata={'extractor':'dct_fft_radial_block_stats'})
    print(f'wrote {len(rows)} frequency features to {out}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
