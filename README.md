# AI-GEN Image Detector

## 목차

- [1. Project Overview](#1-project-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Environment Setup](#3-environment-setup)
- [4. Dataset Preparation](#4-dataset-preparation)
- [5. Feature Cache Generation](#5-feature-cache-generation)
- [6. Model Training](#6-model-training)
- [7. Model Evaluation](#7-model-evaluation)
- [8. Evaluation Results and Artifact Paths](#8-evaluation-results-and-artifact-paths)
- [9. streamlit web Demo](#9-streamlit-web-demo)
- [10. 한번에 실행하기](#10-run-in-one-command)

## 1. Project Overview

## AI-GEN Image Detector

딥페이크 이미지 탐지 시스템.

## 핵심 구조

> 공간 도메인 CLIP feature, 주파수 도메인 DCT/FFT feature, 그리고 두 feature를 결합하는 fusion classifier

---

## 2. Repository Structure

| Path | Description |
| --- | --- |
| `README.md` | clean clone 기준 실행 안내서 |
| `environment.yml` | `ml_termproj` conda 환경 생성 파일 |
| `requirements.txt` | Python package 목록 |
| `configs/default.yaml` | 기본 runtime config, demo 및 feature cache 기준 config |
| `configs/frequency_only.yaml` | frequency-only 학습/평가 config |
| `configs/clip_only.yaml` | CLIP-only 학습/평가 config |
| `configs/fusion.yaml` | fusion 학습/평가 config |
| `configs/synthbuster_ood.yaml` | Synthbuster OOD feature/evaluation config |
| `scripts/prepare_genimage_subset.py` | Hugging Face Tiny-GenImage를 512x512 JPG, metadata CSV, manifest CSV로 materialize |
| `scripts/prepare_synthbuster_ood.py` | Zenodo Synthbuster zip을 OOD test-only dataset으로 전처리 |
| `scripts/train_cuda_finetune.py` | manifest-v1 CSV 기반 torchvision ResNet fine-tuning script |
| `src/data/` | dataset.csv/manifest validation, dataset loader, transforms |
| `src/features/cache_features.py` | CLIP/frequency `.npy` feature cache 생성 entrypoint |
| `src/train/train_frequency.py` | cached frequency feature 기반 MLP 학습 |
| `src/train/train_clip.py` | cached CLIP feature 기반 MLP 학습 |
| `src/train/train_fusion.py` | aligned CLIP + frequency cache 기반 fusion classifier 학습 |
| `src/eval/evaluate.py` | cached-feature model 평가 및 metrics/prediction report 생성 |
| `src/eval/robustness.py` | dataset.csv image corruption robustness 평가 entrypoint |
| `src/eval/robustness_runner.py` | manifest/checkpoint 기반 robustness runner |
| `src/inference/detector_service.py` | Streamlit demo가 사용하는 prediction boundary |
| `src/app/app.py` | 현재 Streamlit demo entrypoint |
| `app/streamlit_app.py` | deprecated compatibility Streamlit wrapper |
| `src/visualization/plot_frequency_spectrum.py` | 단일 이미지 spectrum/radial spectrum PNG 생성 |
| `artifacts/features/` | 생성된 CLIP/frequency `.npy` feature cache 저장 위치 |
| `artifacts/checkpoints/` | `frequency_only.pt`, `clip_only.pt`, `fusion.pt` checkpoint 저장 위치 |
| `artifacts/reports/` | 현재 repo에 남아 있는 legacy/report artifact 위치 |
| `outputs/metrics/` | 현재 config 기준 학습/평가/robustness report 저장 위치 |
| `outputs/plots/` | robustness plot 및 spectrum plot 저장 위치 |

config path 기준은 두 가지로 나뉜다. `configs/default.yaml`의 `paths.dataset_csv`는 canonical metadata CSV인 `data/metadata/genimage_tiny_full_dataset.csv`를 가리키며, feature cache 생성, demo, `src.eval.robustness` image corruption 평가에 사용한다.

`configs/frequency_only.yaml`, `configs/clip_only.yaml`, `configs/fusion.yaml`의 `paths.dataset_csv`는 `outputs/genimage_tiny_full/manifest.csv`를 가리키며, cached-feature 학습/평가 config로 사용한다.

---

## 3. Environment Setup

### 3.1 Recommended Environment

git clone 후 프로젝트 루트에서 conda 환경을 생성한다.

#### 1. git clone

```bash```
git clnoe https://github.com/jinzer0/lightweight-deepfake-detection.git
```

#### 2. 환경 구성

```bash
conda env create -f environment.yml
conda activate ml_termproj
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pip install -r requirements.txt
```

### 3.3 데이터셋 구성

#### 1. GenImage

Hugginface dataset "TheKernel01/Tiny-GenImage" 사용

#### 2. Synthbuster

https://zenodo.org/records/10066460 에서 다운로드

---

## 4. Dataset Preparation

### 4.1 Label and CSV Schema

라벨 규칙

| Class | Label | Probability meaning |
| --- | --- | --- |
| `real` | `0` | `P(fake)`가 낮을수록 real |
| `fake` | `1` | `P(fake)`가 높을수록 AI-generated |

metadata CSV column:

```text
image_id,filepath,label,class_name,dataset,generator,split,width,height,ext
```

manifest-v1 CSV는 `src/data/manifest.py` 기준

```text
sample_id,base_sample_id,rel_path,root,label,class_name,source,source_split,split,width,height,sha256,file_size,mtime,status
```

### 4.2 Tiny-GenImage Download and Preprocessing

```bash
conda run --live-stream -n ml_termproj python scripts/prepare_genimage_subset.py --clean
```

| Item | Path / Value |
| --- | --- |
| Hugging Face dataset id | `TheKernel01/Tiny-GenImage` |
| Materialized image root | `data/genimage_tiny_full_512` |
| Image size | `512x512` |
| Metadata CSV | `data/metadata/genimage_tiny_full_dataset.csv` |
| Manifest CSV | `outputs/genimage_tiny_full/manifest.csv` |
| Seed | `42` |

script가 수행하는 전처리 절차

1. `datasets.load_dataset()`으로 Tiny-GenImage를 로드한다.
2. 모든 이미지를 RGB로 변환하고 `512x512` JPG로 저장한다.
3. Hugging Face source `train` split은 project `train`으로 사용한다.
4. Hugging Face source `validation` split은 seed `42`로 project `val`/`test`에 deterministic하게 나눈다.
5. metadata CSV와 manifest CSV를 validation schema에 맞게 쓴다.

> 데이터 구조

```text
data/
  genimage_tiny_full_512/
    train/
    val/
    test/
  metadata/
    genimage_tiny_full_dataset.csv
outputs/
  genimage_tiny_full/
    manifest.csv
```

생성 후 metadata를 검증한다.

```bash
conda run --live-stream -n ml_termproj python -m src.data.validate_metadata \
  --csv data/metadata/genimage_tiny_full_dataset.csv
```

### 4.3 Synthbuster OOD Download and Preprocessing

Synthbuster는 Zenodo record `https://zenodo.org/records/10066460`에서 다운로드한 뒤 `./data/raw` 위치한다. 파일명은 `synthbuster.zip`

> Synthbuster donwload

```bash
mkdir -p data/raw
curl -L "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1" \
  -o data/raw/synthbuster.zip
```

> Synthbuseter preprocessing script

```bash
conda run --live-stream -n ml_termproj python scripts/prepare_synthbuster_ood.py \
  --zip_path data/raw/synthbuster.zip \
  --output_dir data/synthbuster \
  --metadata_csv data/metadata/synthbuster_ood_dataset.csv \
  --manifest outputs/synthbuster_ood/manifest.csv \
  --clean \
  --copy_docs
```

script가 수행하는 전처리 절차

1. zip 내부 `synthbuster/<generator>/<image>` 구조에서 image 파일만 찾는다.
2. 모든 sample을 `split=test`, `label=1`, `class_name=fake`로 기록한다.
3. generator 이름을 안전한 directory name으로 변환한다.
4. 이미지를 `data/synthbuster/test/<generator>/fake/` 아래로 추출한다.
5. `readme`, `licence`, `prompts` 문서는 `--copy_docs` 사용 시 `data/synthbuster/_docs/` 아래에 복사한다.
6. metadata CSV와 manifest CSV를 validation schema에 맞게 쓴다.

> 데이터셋 저장 구조

```text
data/
  raw/
    synthbuster.zip
  synthbuster/
    test/
      <generator>/
        fake/
    _docs/
  metadata/
    synthbuster_ood_dataset.csv
outputs/
  synthbuster_ood/
    manifest.csv
```

생성 후 metadata 검증

```bash
conda run --live-stream -n ml_termproj python -m src.data.validate_metadata \
  --csv data/metadata/synthbuster_ood_dataset.csv
```

---

## 5. Feature Cache Generation

Tiny-GenImage frequency cache 생성

```bash
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
```

Tiny-GenImage CLIP cache 생성

```bash
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test
```

Synthbuster OOD cache test split만 생성

```bash
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/synthbuster_ood.yaml --feature_type frequency --split test
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/synthbuster_ood.yaml --feature_type clip --split test
```

cache를 삭제 후 재생성하려면 생성된 feature directory 삭제

```bash
rm -rf artifacts/features/clip artifacts/features/frequency
rm -rf artifacts/features_synthbuster_ood/clip artifacts/features_synthbuster_ood/frequency
```

## 6. Model Training

현재 main cached-feature 학습 entrypoint는 `--config`만 받는다. epoch, batch size, learning rate, checkpoint/report path는 config YAML에서 읽는다.

| Model | Command | Checkpoint path | Main config |
| --- | --- | --- | --- |
| `frequency_only` | `python -m src.train.train_frequency --config configs/frequency_only.yaml` | `artifacts/checkpoints/frequency_only.pt` | `configs/frequency_only.yaml` |
| `clip_only` | `python -m src.train.train_clip --config configs/clip_only.yaml` | `artifacts/checkpoints/clip_only.pt` | `configs/clip_only.yaml` |
| `fusion` | `python -m src.train.train_fusion --config configs/fusion.yaml` | `artifacts/checkpoints/fusion.pt` | `configs/fusion.yaml` |

> 모델 학습 명령어

```bash
conda run --live-stream -n ml_termproj python -m src.train.train_frequency --config configs/frequency_only.yaml
conda run --live-stream -n ml_termproj python -m src.train.train_clip --config configs/clip_only.yaml
conda run --live-stream -n ml_termproj python -m src.train.train_fusion --config configs/fusion.yaml
```

> 산출물 위치

```text
artifacts/checkpoints/frequency_only.pt
artifacts/checkpoints/clip_only.pt
artifacts/checkpoints/fusion.pt
outputs/metrics/frequency_only_train_log.csv
outputs/metrics/frequency_only_val_metrics.json
outputs/metrics/clip_only_train_log.csv
outputs/metrics/clip_only_val_metrics.json
outputs/metrics/fusion_train_log.csv
outputs/metrics/fusion_val_metrics.json
```

### ResNet(CNN 기반 detector) Fine-tuning

```bash
conda run --live-stream -n ml_termproj python scripts/train_cuda_finetune.py \
  --manifest outputs/genimage_tiny_full/manifest.csv \
  --output_dir outputs/genimage_tiny_full_finetune \
  --device cuda \
  --image_size 512 \
  --model_arch resnet18 \
  --epochs 6 \
  --batch_size 64 \
  --max_trials 4 \
  --num_workers 8 \
  --seed 42
```

---

## 7. Model Evaluation

평가 산출물: Accuracy, Precision, Recall, F1-score, ROC-AUC, prediction CSV, per-generator metrics, model comparison

```bash
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/frequency_only.yaml --model frequency_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/clip_only.yaml --model clip_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/fusion.yaml --model fusion --split test
```

> 산출물 위치

```text
outputs/metrics/<model>_test_metrics.json
outputs/metrics/<model>_test_predictions.csv
outputs/metrics/<model>_test_per_generator_metrics.csv
outputs/metrics/model_comparison.csv
```

### Robustness Evaluation

```bash
conda run --live-stream -n ml_termproj python -m src.eval.robustness --config configs/default.yaml --model frequency_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.robustness --config configs/default.yaml --model clip_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.robustness --config configs/default.yaml --model fusion --split test
```

robustness condition은 `clean`, `jpeg_95`, `jpeg_75`, `jpeg_50`, `resize_0.5`, `resize_0.25`, `blur_1.0`, `blur_2.0` 형태로 구성

> 산출물 위치

```text
outputs/metrics/<model>_robustness_metrics.csv
```

### Synthbuster OOD Evaluation

```bash
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model frequency_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model clip_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model fusion --split test
```

> 산출물 위치

```text
outputs/metrics/synthbuster_ood/<model>_test_metrics.json
outputs/metrics/synthbuster_ood/<model>_test_predictions.csv
outputs/metrics/synthbuster_ood/<model>_test_per_generator_metrics.csv
outputs/metrics/synthbuster_ood/model_comparison.csv
```

---

## 8. Evaluation Results and Artifact Paths

평가 결과 및 체크포인트 등 아티팩트 위치

| Artifact | Path | Generated by |
| --- | --- | --- |
| Frequency cache | `artifacts/features/frequency/{train,val,test}_features.npy` | `src.features.cache_features --feature_type frequency` |
| CLIP cache | `artifacts/features/clip/{train,val,test}_features.npy` | `src.features.cache_features --feature_type clip` |
| Frequency scaler | `artifacts/scalers/frequency_scaler.pkl` | `src.train.train_frequency` |
| Frequency checkpoint | `artifacts/checkpoints/frequency_only.pt` | `src.train.train_frequency` |
| CLIP checkpoint | `artifacts/checkpoints/clip_only.pt` | `src.train.train_clip` |
| Fusion checkpoint | `artifacts/checkpoints/fusion.pt` | `src.train.train_fusion` |
| Validation metrics | `outputs/metrics/<model>_val_metrics.json` | training |
| Train log | `outputs/metrics/<model>_train_log.csv` | training |
| Test metrics | `outputs/metrics/<model>_test_metrics.json` | `src.eval.evaluate` |
| Test predictions | `outputs/metrics/<model>_test_predictions.csv` | `src.eval.evaluate` |
| Per-generator metrics | `outputs/metrics/<model>_test_per_generator_metrics.csv` | `src.eval.evaluate` |
| Model comparison | `outputs/metrics/model_comparison.csv` | `src.eval.evaluate` |
| Robustness metrics | `outputs/metrics/<model>_robustness_metrics.csv` | `src.eval.robustness` |
| Robustness runner CSV | `outputs/metrics/robustness_metrics.csv` | `scripts/robustness_test.py` |
| Robustness runner JSON | `outputs/metrics/robustness_metrics.json` | `scripts/robustness_test.py` |
| Robustness bar plot | `outputs/plots/robustness_barplot.png` | `scripts/robustness_test.py` |
| Synthbuster OOD metrics | `outputs/metrics/synthbuster_ood/<model>_test_metrics.json` | `src.eval.evaluate --config configs/synthbuster_ood.yaml` |
| Demo upload copies | `artifacts/figures/uploads/` | `streamlit run src/app/app.py` |
| Demo spectrum figures | `outputs/plots/` 또는 config `paths.figure_dir` | inference/visualization code |

## 9. streamlit web Demo

web dashboard 형식으로 모델 추론하기 위한 데모

```bash
conda run --live-stream -n ml_termproj streamlit run src/app/app.py
```

> 데모에서 확인 가능한 결과

| Display | Meaning |
| --- | --- |
| AI-generated probability | `P(fake)` |
| Real / AI label | threshold 기준 최종 판정 |
| Confidence level | `configs/default.yaml`의 confidence margin 기준 |
| Branch scores | `clip_score`, `frequency_score`, `fusion_score` |
| Spectrum image | DCT/FFT spectrum 시각화 |
| Radial spectrum graph | radial spectrum profile |

###

## 10. Run in One-Command

git clone부터 데이터셋 준비, 모델 학습, 평가까지 한번에 실행

```bash
git clone https://github.com/jinzer0/lightweight-deepfake-detection.git
cd lightweight-deepfake-detection

conda env create -f environment.yml
conda activate ml_termproj

# Verify environment
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Prepare Tiny-GenImage dataset and metadata/manifest
python scripts/prepare_genimage_subset.py --clean
python -m src.data.validate_metadata --csv data/metadata/genimage_tiny_full_dataset.csv

# Build feature caches
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test

# Train cached-feature models
python -m src.train.train_frequency --config configs/frequency_only.yaml
python -m src.train.train_clip --config configs/clip_only.yaml
python -m src.train.train_fusion --config configs/fusion.yaml

# Evaluate models
python -m src.eval.evaluate --config configs/frequency_only.yaml --model frequency_only --split test
python -m src.eval.evaluate --config configs/clip_only.yaml --model clip_only --split test
python -m src.eval.evaluate --config configs/fusion.yaml --model fusion --split test

# Robustness smoke test with max_samples
python scripts/robustness_test.py \
  --config configs/fusion.yaml \
  --models fusion,clip_only,frequency_only \
  --max_samples 32 \
  --output_csv outputs/metrics/robustness_metrics.csv \
  --output_json outputs/metrics/robustness_metrics.json \
  --plot_path outputs/plots/robustness_barplot.png

# Download and prepare Synthbuster OOD dataset
mkdir -p data/raw
curl -L "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1" \
  -o data/raw/synthbuster.zip
python scripts/prepare_synthbuster_ood.py \
  --zip_path data/raw/synthbuster.zip \
  --output_dir data/synthbuster \
  --metadata_csv data/metadata/synthbuster_ood_dataset.csv \
  --manifest outputs/synthbuster_ood/manifest.csv \
  --clean \
  --copy_docs
python -m src.data.validate_metadata --csv data/metadata/synthbuster_ood_dataset.csv

# Build OOD caches and evaluate OOD
python -m src.features.cache_features --config configs/synthbuster_ood.yaml --feature_type frequency --split test
python -m src.features.cache_features --config configs/synthbuster_ood.yaml --feature_type clip --split test
python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model frequency_only --split test
python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model clip_only --split test
python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model fusion --split test

# Check outputs
find artifacts outputs -maxdepth 4 -type f | sort
```
