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
- [9. Demo / Inference](#9-demo--inference)
- [10. Reproducibility](#10-reproducibility)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Limitations](#12-limitations)
- [13. One-command Reproduction Checklist](#13-one-command-reproduction-checklist)

## 1. Project Overview

`AI-GEN Image Detector`는 JPG/PNG 이미지 입력에 대해 Real / AI-generated 여부와 AI 생성 확률 `P(fake)`를 출력하는 탐지 시스템이다. 라벨 polarity는 전체 코드에서 고정되어 있으며 `real=0`, `fake=1`이다.

핵심 구조는 공간 도메인 CLIP feature, 주파수 도메인 DCT/FFT feature, 그리고 두 feature를 결합하는 fusion classifier이다. Streamlit demo는 AI 생성 확률, Real/AI 최종 판정, confidence level, branch score, spectrum image, radial spectrum graph를 표시한다.

교수자/평가자용 실행 흐름은 다음 순서다.

```text
환경 생성 -> 데이터 준비/전처리 -> feature cache 생성 -> 모델 학습 -> 모델 평가 -> robustness/OOD 평가 -> 결과 확인 -> demo 실행
```

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

config path 기준은 두 가지로 나뉜다. `configs/default.yaml`의 `paths.dataset_csv`는 canonical metadata CSV인 `data/metadata/genimage_tiny_full_dataset.csv`를 가리키며, feature cache 생성, demo, `src.eval.robustness` image corruption 평가에 사용한다. 반면 `configs/frequency_only.yaml`, `configs/clip_only.yaml`, `configs/fusion.yaml`의 `paths.dataset_csv`는 `outputs/genimage_tiny_full/manifest.csv`를 가리키며, cached-feature 학습/평가 config로 사용한다.

## 3. Environment Setup

### 3.1 Recommended Environment

clean clone 후 프로젝트 루트에서 conda 환경을 생성한다.

```bash
conda env create -f environment.yml
conda activate ml_termproj
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

이미 `ml_termproj` 환경이 있다면 다음 명령으로 dependency를 동기화한다.

```bash
conda activate ml_termproj
pip install -r requirements.txt
```

이 저장소에서 검증한 Python 명령은 다음처럼 `conda run`으로 실행할 수 있다.

```bash
conda run --live-stream -n ml_termproj python -m pytest -q
```

### 3.2 CPU-only / GPU Notes

GPU가 있으면 CLIP feature extraction과 training이 빠르다. CUDA 버전과 PyTorch 설치 버전이 맞지 않으면 `torch.cuda.is_available()`가 `False`가 되고 CPU로 실행된다.

교수자 PC에 GPU가 없는 경우에도 frequency feature cache, frequency-only 학습, cached-feature 평가, Streamlit demo의 `frequency_only` 모델은 CPU에서 실행 가능하다. CLIP/fusion 경로는 OpenCLIP weight 다운로드와 CLIP feature cache가 필요하므로 네트워크와 실행 시간이 더 필요하다.

현재 main cached-feature training entrypoint(`src.train.train_frequency`, `src.train.train_clip`, `src.train.train_fusion`)에는 `--quick`, `--limit`, `--sample`, `--max-samples`, `--epochs` CLI 옵션이 없다. 빠른 학습 명령은 없음. 빠른 검증은 `--help`, metadata validation, 단일 split feature cache 생성으로 수행한다.

### 3.3 External Credentials

Tiny-GenImage는 Hugging Face `datasets.load_dataset("TheKernel01/Tiny-GenImage")`로 로드한다. 공개 dataset이면 token 없이 동작한다. Hugging Face 인증이 필요한 환경에서는 token을 README나 코드에 적지 말고 로컬에서만 로그인한다.

```bash
hf auth login
hf auth whoami
```

Synthbuster는 Zenodo record `https://zenodo.org/records/10066460`의 공개 zip 파일을 사용한다. Kaggle과 WandB는 현재 실행 경로에 필요하지 않다.

## 4. Dataset Preparation

### 4.1 Label and CSV Schema

라벨 규칙은 고정이다.

| Class | Label | Probability meaning |
| --- | --- | --- |
| `real` | `0` | `P(fake)`가 낮을수록 real |
| `fake` | `1` | `P(fake)`가 높을수록 AI-generated |

canonical metadata CSV column은 `src/data/validate_metadata.py` 기준으로 다음 순서다.

```text
image_id,filepath,label,class_name,dataset,generator,split,width,height,ext
```

manifest-v1 CSV는 `src/data/manifest.py` 기준으로 다음 column을 사용한다.

```text
sample_id,base_sample_id,rel_path,root,label,class_name,source,source_split,split,width,height,sha256,file_size,mtime,status
```

### 4.2 Tiny-GenImage Download and Preprocessing

현재 주 데이터셋은 Hugging Face dataset `TheKernel01/Tiny-GenImage`이다. 다음 script가 dataset download, image resize/materialize, metadata CSV 생성, manifest CSV 생성을 한 번에 수행한다.

```bash
conda run --live-stream -n ml_termproj python scripts/prepare_genimage_subset.py --clean
```

기본값은 다음과 같다.

| Item | Path / Value |
| --- | --- |
| Hugging Face dataset id | `TheKernel01/Tiny-GenImage` |
| Materialized image root | `data/genimage_tiny_full_512` |
| Image size | `512x512` |
| Metadata CSV | `data/metadata/genimage_tiny_full_dataset.csv` |
| Manifest CSV | `outputs/genimage_tiny_full/manifest.csv` |
| Seed | `42` |

script가 수행하는 전처리는 다음과 같다.

1. `datasets.load_dataset()`으로 Tiny-GenImage를 로드한다.
2. 모든 이미지를 RGB로 변환하고 `512x512` JPG로 저장한다.
3. Hugging Face source `train` split은 project `train`으로 사용한다.
4. Hugging Face source `validation` split은 seed `42`로 project `val`/`test`에 deterministic하게 나눈다.
5. metadata CSV와 manifest CSV를 validation schema에 맞게 쓴다.

기본 예상 구조는 다음과 같다.

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

Synthbuster는 Zenodo record `https://zenodo.org/records/10066460`에서 다운로드한다. Zenodo metadata 기준 파일명은 `synthbuster.zip`, 크기는 약 12.4 GB이며 synthetic image 9개 generator별 1000장으로 구성된다. 이 저장소의 Synthbuster 경로는 OOD test-only dataset이다.

```bash
mkdir -p data/raw
curl -L "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1" \
  -o data/raw/synthbuster.zip
```

다운로드 후 전처리한다.

```bash
conda run --live-stream -n ml_termproj python scripts/prepare_synthbuster_ood.py \
  --zip_path data/raw/synthbuster.zip \
  --output_dir data/synthbuster \
  --metadata_csv data/metadata/synthbuster_ood_dataset.csv \
  --manifest outputs/synthbuster_ood/manifest.csv \
  --clean \
  --copy_docs
```

script가 수행하는 전처리는 다음과 같다.

1. zip 내부 `synthbuster/<generator>/<image>` 구조에서 image 파일만 찾는다.
2. 모든 sample을 `split=test`, `label=1`, `class_name=fake`로 기록한다.
3. generator 이름을 안전한 directory name으로 변환한다.
4. 이미지를 `data/synthbuster/test/<generator>/fake/` 아래로 추출한다.
5. `readme`, `licence`, `prompts` 문서는 `--copy_docs` 사용 시 `data/synthbuster/_docs/` 아래에 복사한다.
6. metadata CSV와 manifest CSV를 validation schema에 맞게 쓴다.

기본 예상 구조는 다음과 같다.

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

생성 후 metadata를 검증한다.

```bash
conda run --live-stream -n ml_termproj python -m src.data.validate_metadata \
  --csv data/metadata/synthbuster_ood_dataset.csv
```

### 4.4 CIFAKE Status

현재 repository instruction은 CIFAKE-specific files/configs를 재도입하지 말라고 명시한다. 현재 main workflow는 Tiny-GenImage와 Synthbuster OOD이다. CIFAKE Kaggle download 명령은 이 README의 실행 경로에 포함하지 않는다.

## 5. Feature Cache Generation

feature cache는 `.npy` feature matrix, `.npy` label array, `.csv` metadata로 저장된다. 기본 저장 규칙은 다음과 같다.

```text
artifacts/features/<feature_type>/<split>_features.npy
artifacts/features/<feature_type>/<split>_labels.npy
artifacts/features/<feature_type>/<split>_meta.csv
```

Tiny-GenImage frequency cache를 생성한다.

```bash
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
```

Tiny-GenImage CLIP cache를 생성한다. 이 명령은 OpenCLIP model weight를 사용할 수 있어야 한다.

```bash
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test
```

Synthbuster OOD cache는 test split만 생성한다.

```bash
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/synthbuster_ood.yaml --feature_type frequency --split test
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/synthbuster_ood.yaml --feature_type clip --split test
```

cache를 삭제 후 재생성하려면 생성된 feature directory를 삭제하고 같은 명령을 다시 실행한다.

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

Quick smoke training 명령은 없음. 현재 main training CLI에는 `--epochs`, `--limit`, `--max-samples` 옵션이 없다. 학습 전 CLI 검증은 다음으로 수행한다.

```bash
conda run --live-stream -n ml_termproj python -m src.train.train_frequency --help
conda run --live-stream -n ml_termproj python -m src.train.train_clip --help
conda run --live-stream -n ml_termproj python -m src.train.train_fusion --help
```

Full training은 다음 순서로 실행한다.

```bash
conda run --live-stream -n ml_termproj python -m src.train.train_frequency --config configs/frequency_only.yaml
conda run --live-stream -n ml_termproj python -m src.train.train_clip --config configs/clip_only.yaml
conda run --live-stream -n ml_termproj python -m src.train.train_fusion --config configs/fusion.yaml
```

학습이 끝나면 다음 파일이 생성된다.

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

### Optional ResNet Fine-tuning

`scripts/train_cuda_finetune.py`는 cached-feature classifier가 아니라 manifest image를 직접 읽는 torchvision ResNet fine-tuning path이다. CUDA가 있는 환경에서 사용한다.

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

이 path의 checkpoint는 `outputs/genimage_tiny_full_finetune/best_checkpoint.pt`이며 Streamlit demo의 `frequency_only`, `clip_only`, `fusion` artifact와 호환되는 파일이 아니다.

## 7. Model Evaluation

cached-feature 평가 entrypoint는 Accuracy, Precision, Recall, F1-score, ROC-AUC, prediction CSV, per-generator metrics, model comparison CSV를 생성한다.

```bash
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/frequency_only.yaml --model frequency_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/clip_only.yaml --model clip_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/fusion.yaml --model fusion --split test
```

출력 위치는 config의 `paths.report_dir`이다. 현재 모델별 config는 `outputs/metrics`를 사용한다.

```text
outputs/metrics/<model>_test_metrics.json
outputs/metrics/<model>_test_predictions.csv
outputs/metrics/<model>_test_per_generator_metrics.csv
outputs/metrics/model_comparison.csv
```

### Robustness Evaluation

현재 image corruption robustness entrypoint는 `src.eval.robustness`이다. config의 `robustness.jpeg_qualities`, `robustness.resize_scales`, `robustness.blur_sigmas`를 사용한다.

```bash
conda run --live-stream -n ml_termproj python -m src.eval.robustness --config configs/default.yaml --model frequency_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.robustness --config configs/default.yaml --model clip_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.robustness --config configs/default.yaml --model fusion --split test
```

robustness condition은 `clean`, `jpeg_95`, `jpeg_75`, `jpeg_50`, `resize_0.5`, `resize_0.25`, `blur_1.0`, `blur_2.0` 형태로 config에서 구성된다. 결과는 다음 파일로 저장된다.

```text
outputs/metrics/<model>_robustness_metrics.csv
```

manifest/checkpoint 기반 robustness runner도 존재한다. 이 runner에는 `--max_samples`가 있으므로 긴 robustness 실행 전 smoke test에 사용할 수 있다.

```bash
conda run --live-stream -n ml_termproj python scripts/robustness_test.py \
  --config configs/fusion.yaml \
  --models fusion,clip_only,frequency_only \
  --max_samples 32 \
  --output_csv outputs/metrics/robustness_metrics.csv \
  --output_json outputs/metrics/robustness_metrics.json \
  --plot_path outputs/plots/robustness_barplot.png
```

### Synthbuster OOD Evaluation

Synthbuster OOD는 `configs/synthbuster_ood.yaml`과 `artifacts/features_synthbuster_ood`를 사용한다. 이 dataset은 현재 전처리 script 기준 `fake` test-only dataset이다.

```bash
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model frequency_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model clip_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model fusion --split test
```

Synthbuster OOD 결과는 다음 위치에 저장된다.

```text
outputs/metrics/synthbuster_ood/<model>_test_metrics.json
outputs/metrics/synthbuster_ood/<model>_test_predictions.csv
outputs/metrics/synthbuster_ood/<model>_test_per_generator_metrics.csv
outputs/metrics/synthbuster_ood/model_comparison.csv
```

## 8. Evaluation Results and Artifact Paths

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

평가 완료 후 결과 파일을 확인한다.

```bash
find artifacts outputs -maxdepth 4 -type f | sort
```

## 9. Demo / Inference

현재 demo entrypoint는 `src/app/app.py`이다.

```bash
conda run --live-stream -n ml_termproj streamlit run src/app/app.py
```

deprecated wrapper인 `app/streamlit_app.py`는 compatibility entrypoint이며 새 실행 명령으로는 사용하지 않는다.

demo는 다음 artifact를 config에서 찾는다.

```text
artifacts/checkpoints/frequency_only.pt
artifacts/checkpoints/clip_only.pt
artifacts/checkpoints/fusion.pt
artifacts/scalers/frequency_scaler.pkl
```

화면에는 다음 값이 표시된다.

| Display | Meaning |
| --- | --- |
| AI-generated probability | `P(fake)` |
| Real / AI label | threshold 기준 최종 판정 |
| Confidence level | `configs/default.yaml`의 confidence margin 기준 |
| Branch scores | `clip_score`, `frequency_score`, `fusion_score` |
| Spectrum image | DCT/FFT spectrum 시각화 |
| Radial spectrum graph | radial spectrum profile |

단일 이미지 spectrum/radial spectrum PNG만 생성하려면 다음 명령을 사용한다.

```bash
conda run --live-stream -n ml_termproj python src/visualization/plot_frequency_spectrum.py \
  --image data/genimage_tiny_full_512/test/<generator>/<class>/<image>.jpg \
  --config configs/default.yaml \
  --output-dir outputs/plots
```

현재 별도 CLI inference script는 없다. inference는 Streamlit demo와 `src.inference.detector_service.DetectorService`를 통해 수행한다.

## 10. Reproducibility

| Item | Value |
| --- | --- |
| Seed | `42` |
| Image size | `512` |
| Main config | `configs/default.yaml` |
| Model configs | `configs/frequency_only.yaml`, `configs/clip_only.yaml`, `configs/fusion.yaml` |
| OOD config | `configs/synthbuster_ood.yaml` |
| Tiny-GenImage split | source `train` -> project `train`, source `validation` -> deterministic `val`/`test` |
| Label polarity | `real=0`, `fake=1` |
| Metrics | accuracy, precision, recall, f1, roc_auc |
| Key packages | `torch`, `open_clip_torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `pillow`, `pyyaml`, `joblib`, `matplotlib`, `streamlit`, `datasets`, `tqdm` |

동일한 결과를 재현하려면 데이터 전처리, feature cache 생성, training, evaluation을 같은 config와 seed로 다시 실행한다.

```bash
conda run --live-stream -n ml_termproj python scripts/prepare_genimage_subset.py --seed 42 --clean
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
conda run --live-stream -n ml_termproj python -m src.train.train_frequency --config configs/frequency_only.yaml
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/frequency_only.yaml --model frequency_only --split test
```

전체 학습 runtime과 VRAM 사용량은 machine, CUDA/PyTorch build, OpenCLIP weight cache 상태에 따라 달라진다. 현재 README에는 검증되지 않은 시간이나 VRAM 수치를 고정값으로 적지 않는다.

## 11. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()`가 `False` | CPU 실행은 가능하다. GPU가 필요하면 CUDA와 호환되는 PyTorch build를 설치한 뒤 `python -c "import torch; print(torch.cuda.is_available())"`로 확인한다. |
| Hugging Face dataset download 실패 | `hf auth login` 후 다시 `scripts/prepare_genimage_subset.py --clean`을 실행한다. token은 repo에 저장하지 않는다. |
| `dataset path not found` 또는 metadata validation 실패 | `python scripts/prepare_genimage_subset.py --clean` 실행 후 `python -m src.data.validate_metadata --csv data/metadata/genimage_tiny_full_dataset.csv`를 실행한다. |
| Synthbuster zip 없음 | `curl -L "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1" -o data/raw/synthbuster.zip`를 실행한다. |
| checkpoint path mismatch | `artifacts/checkpoints/<model>.pt`가 있는지 확인하고 없으면 해당 training command를 먼저 실행한다. |
| CLIP/OpenCLIP model download 실패 | 네트워크 연결과 Hugging Face 인증을 확인한다. CPU-only smoke는 `frequency_only` 경로로 먼저 수행한다. |
| feature cache missing | `python -m src.features.cache_features --config configs/default.yaml --feature_type <clip|frequency> --split <train|val|test>`를 먼저 실행한다. |
| scaler missing | `artifacts/scalers/frequency_scaler.pkl`이 없으면 `python -m src.train.train_frequency --config configs/frequency_only.yaml`를 실행한다. |
| CUDA out of memory | config의 `data.batch_size`를 줄이고 다시 실행한다. main training CLI에는 batch size override 옵션이 없으므로 YAML을 수정해야 한다. |
| matplotlib 한글 폰트 문제 | 현재 plot label은 영어 중심이다. 한글 label을 추가한 경우 로컬 matplotlib font 설정을 확인한다. |
| Streamlit port 충돌 | `streamlit run src/app/app.py --server.port 8502`처럼 빈 port를 지정한다. |

## 12. Limitations

이 프로젝트의 성능은 제한된 데이터셋 기준 성능이며 모든 AI 생성 이미지를 완벽히 탐지한다는 의미가 아니다. 특정 generator, resize, JPEG compression, blur 조건에서 성능이 달라질 수 있다.

학습하지 않은 생성 모델에 대한 성능은 OOD test 결과로만 판단해야 한다. Synthbuster OOD 전처리는 현재 fake-only test dataset을 만들기 때문에, real-vs-fake balanced test와 해석이 다르다.

frequency artifact는 JPEG/resize/blur 후처리에 약해질 수 있다. CLIP/fusion path는 OpenCLIP weight, cached feature, checkpoint가 모두 있어야 실행된다. clean clone에 pretrained checkpoint가 포함되지 않는 경우 모델 학습을 다시 수행해야 한다.

## 13. One-command Reproduction Checklist

아래 순서는 clean clone 기준 전체 실행 checklist이다. `<repo-url>`과 `<repo-name>`은 제출 repository 주소와 directory 이름으로 바꾼다.

```bash
git clone <repo-url>
cd <repo-name>

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

# Run demo
streamlit run src/app/app.py
```
