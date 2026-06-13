# AI-GEN Image Detector

AI-GEN Image Detector is a machine learning term project for experimental real vs AI-generated image detection. The current workflow uses a canonical `dataset.csv`, cached `.npy` features, PyTorch MLP checkpoints, a shared `DetectorService`, and a Streamlit demo at `src/app/app.py`.

This is not a forensic or legal evidence tool. `label=0` means real, `label=1` means fake or AI-generated, and every probability reported by the project is a fake or AI-generated probability. Treat results as experiment outputs, not proof about an image.

## Folder Structure

```text
configs/default.yaml          Main project config
data/raw/                     Local image files, including dummy data
data/metadata/dataset.csv     Canonical metadata table
artifacts/features/           .npy feature caches and metadata copies
artifacts/checkpoints/        PyTorch .pt model checkpoints
artifacts/reports/            Metrics, predictions, and CSV reports
artifacts/figures/            Visualizations and Streamlit uploads
src/data/                     Metadata, dummy data, datasets, transforms
src/features/                 Frequency and optional CLIP feature extraction
src/train/                    PyTorch frequency, CLIP, and fusion trainers
src/eval/                     Evaluation and robustness commands
src/inference/                DetectorService for single-image prediction
src/app/app.py                Streamlit demo
scripts/                      Legacy compatibility wrappers only
outputs/                      Legacy output location only
app/streamlit_app.py          Legacy Streamlit entrypoint only
```

## Setup

Use Python 3.10 or newer. The project has been run with a conda environment named `ml_termproj`.

```bash
conda activate ml_termproj
pip install -r requirements.txt
```

If you don't already have the conda environment, create it first with your local Python version, then run the two commands above. The mandatory quick start below is CPU-safe and doesn't require CUDA or CLIP model downloads.

## Config

The default config is `configs/default.yaml`.

Important fields:

```yaml
project:
  seed: 42
  device: cuda
paths:
  dataset_csv: data/metadata/dataset.csv
  raw_data_dir: data/raw
  feature_dir: artifacts/features
  checkpoint_dir: artifacts/checkpoints
  report_dir: artifacts/reports
  figure_dir: artifacts/figures
frequency:
  method: dct
  image_size: 224
  radial_bins: 64
clip:
  model_name: ViT-B-32
  pretrained: openai
classifier:
  type: mlp
train:
  epochs: 20
```

`project.device` is resolved at runtime. The frequency-only quick start remains usable on CPU-only machines. CLIP paths may need installed `open_clip_torch`, cached model weights, network access, and a working CPU or CUDA PyTorch setup.

## Canonical `dataset.csv` Schema

The primary metadata file is `data/metadata/dataset.csv`. It must use this exact column order:

```text
image_id, filepath, label, class_name, dataset, generator, split, width, height, ext
```

Rules enforced by `src.data.validate_metadata`:

- `label=0` means real and `class_name=real`.
- `label=1` means fake or AI-generated and `class_name=fake`.
- `split` must be one of `train`, `val`, or `test`.
- `filepath` must point to an existing image.
- `image_id` and `filepath` must be unique.
- Width and height must be positive integers.

## Mandatory Quick Start, CPU Frequency Only

This is the canonical smoke path. It creates a tiny dummy dataset, validates metadata, caches frequency features, trains the frequency-only PyTorch MLP, evaluates it, runs robustness, and starts the Streamlit demo. It doesn't require CIFAKE, GenImage, CUDA, CLIP weights, or network access.

```bash
python -m src.data.make_dummy_dataset \
  --num_real 30 \
  --num_fake 30 \
  --output_dir data/raw/dummy \
  --csv data/metadata/dataset.csv

python -m src.data.validate_metadata \
  --csv data/metadata/dataset.csv

python -m src.features.cache_features \
  --config configs/default.yaml \
  --feature_type frequency \
  --split train

python -m src.features.cache_features \
  --config configs/default.yaml \
  --feature_type frequency \
  --split val

python -m src.features.cache_features \
  --config configs/default.yaml \
  --feature_type frequency \
  --split test

python -m src.train.train_frequency \
  --config configs/default.yaml

python -m src.eval.evaluate \
  --config configs/default.yaml \
  --model frequency_only \
  --split test

python -m src.eval.robustness \
  --config configs/default.yaml \
  --model frequency_only \
  --split test

streamlit run src/app/app.py
```

In the demo sidebar, keep the default `frequency_only` model for this smoke path. Upload a JPG, JPEG, or PNG after the checkpoint exists. The app saves uploads under `artifacts/figures/uploads/` and displays fake or AI-generated probability, final decision, confidence, branch scores when available, and frequency visualizations.

## Dummy Dataset Generation

Generate deterministic dummy RGB PNG images and `dataset.csv` metadata:

```bash
python -m src.data.make_dummy_dataset \
  --num_real 30 \
  --num_fake 30 \
  --output_dir data/raw/dummy \
  --csv data/metadata/dataset.csv
```

Optional arguments include `--width`, `--height`, and `--seed`. Dummy data proves the workflow is connected. It doesn't measure real detector quality.

## Metadata Validation

Validate `dataset.csv` before caching features:

```bash
python -m src.data.validate_metadata \
  --csv data/metadata/dataset.csv
```

The command prints split, label, generator, and dataset counts. It exits nonzero on missing files, duplicate IDs, bad labels, class polarity mistakes, invalid splits, or wrong column order.

## Feature Caches

Feature caches are written under `artifacts/features/<feature_type>/` as `.npy` arrays plus a metadata CSV:

```text
artifacts/features/frequency/train_features.npy
artifacts/features/frequency/train_labels.npy
artifacts/features/frequency/train_meta.csv
artifacts/features/clip/train_features.npy
artifacts/features/clip/train_labels.npy
artifacts/features/clip/train_meta.csv
```

Create frequency caches for each split:

```bash
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
```

Frequency extraction uses the config's DCT settings by default. Cache loading validates row counts, labels, split values, duplicate `image_id` values, finite feature values, and metadata alignment.

## Optional CLIP Caches

CLIP is optional. Skip this section when working offline, when `open_clip_torch` isn't installed, or when model weights can't be downloaded or found in cache.

```bash
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test
```

The default CLIP config is `ViT-B-32` with `pretrained: openai`. These commands may contact model hosting services unless the weights already exist locally. A failed CLIP cache doesn't block the frequency-only workflow.

## Training

All primary trainers use cached `.npy` features and write PyTorch checkpoints to `artifacts/checkpoints/`.

Frequency-only training, required for the CPU smoke path:

```bash
python -m src.train.train_frequency \
  --config configs/default.yaml
```

CLIP-only training, optional and requires CLIP caches:

```bash
python -m src.train.train_clip \
  --config configs/default.yaml
```

Fusion training, optional and requires both CLIP and frequency caches aligned by `image_id`:

```bash
python -m src.train.train_fusion \
  --config configs/default.yaml
```

Expected checkpoint names:

```text
artifacts/checkpoints/frequency_only.pt
artifacts/checkpoints/clip_only.pt
artifacts/checkpoints/fusion.pt
```

The checkpoints contain model state plus metadata such as `feature_type`, `model_name`, `input_dim`, `hidden_dim`, `threshold`, and a config snapshot.

## Evaluation

Evaluate any trained model on a cached split:

```bash
python -m src.eval.evaluate \
  --config configs/default.yaml \
  --model frequency_only \
  --split test
```

Optional model names are `clip_only` and `fusion` when their caches and checkpoints exist:

```bash
python -m src.eval.evaluate --config configs/default.yaml --model clip_only --split test
python -m src.eval.evaluate --config configs/default.yaml --model fusion --split test
```

Evaluation writes JSON and CSV reports under `artifacts/reports/`:

```text
artifacts/reports/frequency_only_test_metrics.json
artifacts/reports/frequency_only_test_predictions.csv
artifacts/reports/frequency_only_test_per_generator_metrics.csv
artifacts/reports/model_comparison.csv
```

`pred_prob` is the fake or AI-generated probability from `torch.sigmoid(logit)`. `pred_label` is assigned with the checkpoint threshold, default `0.5`.

## Robustness

Run corruption robustness on a trained checkpoint:

```bash
python -m src.eval.robustness \
  --config configs/default.yaml \
  --model frequency_only \
  --split test
```

The default config checks JPEG quality, resize, and blur settings. The command writes:

```text
artifacts/reports/frequency_only_robustness_metrics.csv
```

CLIP-only and fusion robustness are optional and may load CLIP at runtime:

```bash
python -m src.eval.robustness --config configs/default.yaml --model clip_only --split test
python -m src.eval.robustness --config configs/default.yaml --model fusion --split test
```

## Streamlit Demo

Start the current demo with:

```bash
streamlit run src/app/app.py
```

The default sidebar selection is `frequency_only`, which matches the mandatory quick start. `clip_only` and `fusion` are selectable only after their checkpoints and CLIP runtime path are ready. The app routes prediction through `DetectorService`, saves uploaded files to `artifacts/figures/uploads/`, and creates spectrum visualizations in `artifacts/figures/`.

## Artifact Locations

Primary artifacts live here:

```text
artifacts/features/      .npy feature arrays, label arrays, split metadata
artifacts/checkpoints/   PyTorch checkpoints: frequency_only.pt, clip_only.pt, fusion.pt
artifacts/reports/       Metrics JSON, predictions CSV, per-generator CSV, robustness CSV
artifacts/figures/       Spectrum images, radial spectrum plots, uploaded demo files
```

The old `outputs/` directory, manifest-v1 files, `.pt` feature caches, sklearn model artifacts, `python scripts/...` commands, and `app/streamlit_app.py` belong to the deprecated compatibility workflow. They are not the primary path for this refactor.

## Full Optional CLIP Path

Only run this path when CLIP dependencies and weights are available:

```bash
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test
python -m src.train.train_clip --config configs/default.yaml
python -m src.eval.evaluate --config configs/default.yaml --model clip_only --split test
```

Fusion needs both branches:

```bash
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test
python -m src.train.train_fusion --config configs/default.yaml
python -m src.eval.evaluate --config configs/default.yaml --model fusion --split test
python -m src.eval.robustness --config configs/default.yaml --model fusion --split test
```

No remote RTX, CUDA, or full CLIP completion is claimed by this README. Add that claim only when real evidence files record the command, exit code, PyTorch and CUDA versions, device name, OpenCLIP version, and produced artifacts.

## Limitations

This detector is experimental. It shouldn't be used as legal, forensic, academic misconduct, or policy enforcement evidence.

Known limits:

- Dummy data verifies the command path only. It says nothing about real-world accuracy.
- Results depend on the source dataset, split, generator mix, preprocessing, and class balance.
- CIFAKE-style or small local data doesn't prove generalization to Midjourney, DALL-E, SDXL, FLUX, camera images, screenshots, or edited images.
- Frequency features can be sensitive to JPEG compression, resizing, blur, cropping, and other processing.
- CLIP paths are optional and can fail offline because of dependency, network, or model-cache issues.
- A `0.5` threshold is the default checkpoint threshold, not a universal truth boundary.
- Per-generator metrics can be undefined for one-class groups. In that case ROC AUC is reported as null with a warning in the implementation.

## Final Smoke Path

For final local smoke before handing off to T16, use the CPU frequency-only sequence:

```bash
python -m src.data.make_dummy_dataset --num_real 30 --num_fake 30 --output_dir data/raw/dummy --csv data/metadata/dataset.csv
python -m src.data.validate_metadata --csv data/metadata/dataset.csv
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
python -m src.train.train_frequency --config configs/default.yaml
python -m src.eval.evaluate --config configs/default.yaml --model frequency_only --split test
python -m src.eval.robustness --config configs/default.yaml --model frequency_only --split test
streamlit run src/app/app.py
```
