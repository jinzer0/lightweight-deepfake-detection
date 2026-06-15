# AI-GEN Image Detector

## 1. Project Scope
AI-GEN Image Detector is an experimental GenImage real-vs-AI image detector. It preserves label polarity `real=0`, `fake=1`, and all probabilities are `P(fake)`.

## 2. Limitations
This project is not a forensic, legal, or production moderation system. Missing checkpoints are reported as `not run`; the code must not fabricate training metrics or detector performance.

## 3. Environment
Use the `ml_termproj` conda environment for Python commands.

```bash
conda run -n ml_termproj python -m pytest -q
```

## 4. Data Layout
GenImage-style folders are scanned recursively. Paths containing real/original folders become `real`; paths containing fake/generated/AI folders become `fake`.

## 5. Manifest Builder
Build a manifest with required columns `path,label,class_name,generator,source_split,split`.

```bash
conda run -n ml_termproj python scripts/build_manifest.py --data_root data/raw/genimage --output data/metadata/genimage_manifest.csv --split_strategy random
```

## 6. Split Strategies
`random` creates train/val/test rows per class and generator. `generator_holdout` assigns selected or inferred generators to test.

## 7. Configs
Use `configs/resnet50.yaml`, `configs/clip_only.yaml`, `configs/frequency_only.yaml`, and `configs/fusion.yaml` for the requested workflows.

## 8. Frequency Features
Frequency features include DCT radial spectrum, FFT radial spectrum, high-frequency ratio, low/mid/high energy ratios, and block-DCT statistics.

## 9. CLIP Features
CLIP extraction uses frozen `open_clip`, tries `hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K` first, falls back to `ViT-L-14` with `laion2B-s32B-b82K`, and L2-normalizes features.

## 10. Feature Caches
Split caches are written to `artifacts/features` as `.npy` arrays with `feature_index.csv` metadata.

## 11. Models
Implemented model files include ResNet50 baseline, CLIP-only classifier, frequency-only classifier, and an 8-block residual fusion MLP.

## 12. Training
Training scripts use AdamW, weight decay, scheduler, gradient clipping, early stopping, and save best checkpoint, last checkpoint, config snapshot, metrics JSON, and training log CSV.

## 13. Evaluation
`conda run -n ml_termproj python scripts/evaluate.py --help` works offline. Missing checkpoints/caches are marked `not run`.

## 14. Robustness
`robustness_test.py` covers JPEG 95/75/50, resize 0.5/0.25, Gaussian blur 1/2, and center-crop-resize rows; missing artifacts are `not run`.

## 15. Demo
The demo service lives in `src/demo`. `app_gradio.py --help` does not launch the server. With missing checkpoints, predictions return neutral/not-run branch scores instead of claiming trained performance.
