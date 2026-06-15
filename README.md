# AI-GEN Image Detector

GenImage/Tiny-GenImage 기반 real-vs-AI image detector 실험 프로젝트입니다. 라벨 polarity는 항상 `real=0`, `fake=1`이며, 모든 확률은 `P(fake)`를 의미합니다. 이 프로젝트는 연구/발표용 benchmark pipeline이며 법적 또는 포렌식 판정기가 아닙니다.

## 1. Current Pipeline

현재 GenImage pipeline은 다음 순서로 운영합니다.

```text
Tiny-GenImage materialize
  -> metadata/manifest CSV 생성
  -> CLIP/frequency feature cache 생성
  -> frequency_only / clip_only / fusion 학습
  -> fusion robustness 평가
  -> metrics CSV/JSON + plot 생성
```

현재 주력 config는 모델별 config입니다.

```text
configs/frequency_only.yaml
configs/clip_only.yaml
configs/fusion.yaml
configs/resnet50.yaml
```

이 config들은 현재 workspace 기준으로 다음 실제 경로를 가리킵니다.

```text
manifest_csv:   outputs/genimage_tiny_full/manifest.csv
dataset_csv:    outputs/genimage_tiny_full/manifest.csv
feature_dir:    artifacts/features
checkpoint_dir: artifacts/checkpoints
report_dir:     outputs/metrics
scaler_dir:     artifacts/scalers
```

`configs/default.yaml`은 feature cache 생성용 기준 config입니다. `dataset_csv`는 실제 이미지 `filepath`가 들어 있는 `data/metadata/genimage_tiny_full_dataset.csv`를 봅니다. 모델별 config는 robustness와 학습 산출물 위치 기준으로 `outputs/genimage_tiny_full/manifest.csv`도 함께 사용하며, robustness runner가 manifest의 `root` + `rel_path`를 복구해서 읽습니다.

## 2. Environment

모든 Python 명령은 `ml_termproj` conda 환경에서 실행합니다.

```bash
conda activate ml_termproj
pip install -r requirements.txt
```

또는 다음처럼 `conda run`을 사용합니다.

```bash
conda run --live-stream -n ml_termproj python -m pytest -q
```

Tiny-GenImage 준비에는 `datasets`가 필요합니다. CLIP feature extraction과 fusion robustness에는 `open_clip_torch` 및 CLIP weight/cache가 필요합니다.

## 3. Manifest And Metadata Generation

Tiny-GenImage 전체 준비 명령은 다음입니다.

```bash
conda run --live-stream -n ml_termproj python scripts/prepare_genimage_subset.py --clean
```

기본 산출물은 다음입니다.

```text
data/genimage_tiny_full_512/                 # 512x512 materialized images
data/metadata/genimage_tiny_full_dataset.csv # metadata CSV, filepath 기반
outputs/genimage_tiny_full/manifest.csv      # manifest-v1, root/rel_path 기반
```

현재 split은 deterministic하게 구성됩니다. Tiny-GenImage source `train`은 project `train`으로 사용하고, source `validation`은 `val`과 `test`로 나눕니다. 현재 manifest 크기는 다음 기준입니다.

```text
train: 28,000
val:    3,500
test:   3,500
```

## 4. Feature Cache

Feature cache는 원본 이미지를 매 epoch마다 다시 열고 CLIP/frequency feature를 재추출하지 않기 위해 만듭니다. 현재 학습은 `.npy` feature matrix와 label/meta 파일을 읽습니다.

```text
artifacts/features/clip/{train,val,test}_features.npy
artifacts/features/clip/{train,val,test}_labels.npy
artifacts/features/clip/{train,val,test}_meta.csv
artifacts/features/frequency/{train,val,test}_features.npy
artifacts/features/frequency/{train,val,test}_labels.npy
artifacts/features/frequency/{train,val,test}_meta.csv
```

새로 cache를 만들 때는 `configs/default.yaml`을 사용합니다.

```bash
conda run --live-stream -n ml_termproj python -m src.features.cache_features \
  --config configs/default.yaml \
  --feature_type frequency \
  --split train

conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test

conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test
```

현재 fusion 입력 차원은 다음과 같습니다.

```text
CLIP feature:      768
Frequency feature: 64
Fusion input:      832
```

## 5. Training

Frequency-only, CLIP-only, fusion은 cached feature를 사용해 PyTorch MLP 계열 classifier를 학습합니다.

```bash
conda run --live-stream -n ml_termproj python -m src.train.train_frequency --config configs/frequency_only.yaml
conda run --live-stream -n ml_termproj python -m src.train.train_clip --config configs/clip_only.yaml
conda run --live-stream -n ml_termproj python -m src.train.train_fusion --config configs/fusion.yaml
```

주요 산출물은 다음입니다.

```text
artifacts/checkpoints/frequency_only.pt
artifacts/checkpoints/clip_only.pt
artifacts/checkpoints/fusion.pt
outputs/metrics/fusion_train_log.csv
outputs/metrics/fusion_val_metrics.json
```

Fusion 학습은 frequency train feature로 `StandardScaler`를 fit하고 저장합니다.

```text
artifacts/scalers/frequency_scaler.pkl
```

새로 학습된 `artifacts/checkpoints/fusion.pt`의 `config_snapshot.paths.frequency_scaler_path`에는 이 scaler 경로가 기록됩니다. Robustness 평가에서는 checkpoint에 기록된 scaler만 로드합니다.

## 6. ResNet CUDA Fine-Tune

ResNet baseline은 manifest-v1 CSV를 직접 읽는 CUDA fine-tune script를 사용합니다.

```bash
conda run --live-stream -n ml_termproj python scripts/train_cuda_finetune.py \
  --manifest outputs/genimage_tiny_full/manifest.csv \
  --output_dir outputs/genimage_tiny_full_finetune \
  --device cuda \
  --image_size 512 \
  --model_arch resnet18 \
  --epochs 20 \
  --batch_size 64 \
  --max_trials 1 \
  --num_workers 8 \
  --seed 42
```

이 checkpoint는 Streamlit/fusion feature workflow용 `artifacts/checkpoints/fusion.pt`와 다른 형식입니다.

```text
outputs/genimage_tiny_full_finetune/best_checkpoint.pt
```

## 7. Robustness Evaluation

현재 실제 robustness evaluator는 `src/eval/robustness_runner.py`입니다. `scripts/robustness_test.py`는 이 runner를 호출하는 compatibility wrapper입니다.

```bash
conda run --live-stream -n ml_termproj python scripts/robustness_test.py \
  --config configs/fusion.yaml \
  --models fusion,clip_only,frequency_only \
  --max_samples 50
```

지원 corruption은 다음입니다.

```text
clean
jpeg_q95
jpeg_q75
jpeg_q50
resize_0.5
resize_0.25
blur_1.0
blur_2.0
center_crop_resize
```

주요 산출물은 다음입니다.

```text
outputs/metrics/robustness_metrics.csv
outputs/metrics/robustness_metrics.json
outputs/plots/robustness_barplot.png
```

현재 검증된 모델별 robustness 비교 결과는 `fusion`, `clip_only`, `frequency_only` 각 9개 row씩 총 27개 row가 모두 `status=ok`입니다. `frequency_only`와 `fusion`은 64차원 frequency scaler(`artifacts/scalers/frequency_scaler.pkl`)를 사용합니다.

| Model | clean acc | JPEG q50 acc | resize 0.25 acc | blur 2.0 acc |
|---|---:|---:|---:|---:|
| fusion | 1.00 | 0.90 | 0.54 | 0.54 |
| clip_only | 0.98 | 0.76 | 0.58 | 0.64 |
| frequency_only | 0.94 | 0.94 | 0.50 | 0.50 |

## 8. Evaluation And Reports

Feature model 평가 module은 cached feature와 checkpoint를 읽어 metrics/prediction report를 생성합니다.

```bash
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/frequency_only.yaml --model frequency_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/clip_only.yaml --model clip_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/fusion.yaml --model fusion --split test
```

Compatibility script인 `scripts/evaluate.py`는 lightweight comparison output을 만듭니다. 필요한 artifact가 없으면 해당 row를 `not run`으로 남깁니다.

```bash
conda run --live-stream -n ml_termproj python scripts/evaluate.py --config configs/fusion.yaml --split test
```


## 9. Synthbuster OOD Evaluation

Synthbuster는 현재 Tiny-GenImage 학습에 사용하지 않은 OOD fake-only benchmark입니다. 다운로드한 zip은 다음 명령으로 현재 pipeline용 test-only dataset으로 materialize합니다.

```bash
conda run --live-stream -n ml_termproj python scripts/prepare_synthbuster_ood.py --copy_docs
```

기본 산출물은 다음입니다.

```text
data/synthbuster/                              # extracted OOD images
data/metadata/synthbuster_ood_dataset.csv     # filepath 기반 metadata CSV, split=test
outputs/synthbuster_ood/manifest.csv          # manifest-v1, split=test
configs/synthbuster_ood.yaml                  # OOD feature/eval config
```

현재 Synthbuster 준비 결과는 9개 generator × 1,000장, 총 9,000장입니다. 모두 fake label(`label=1`, `class_name=fake`)이며 split은 `test`만 사용합니다. 따라서 ROC-AUC처럼 real/fake 양쪽 class가 필요한 ranking metric은 `null`이 될 수 있고, 이 경우 accuracy/recall은 fake detection rate로 해석합니다.

OOD cached-feature 평가는 학습된 checkpoint는 그대로 쓰고, feature cache와 report만 별도 경로에 저장합니다.

```bash
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/synthbuster_ood.yaml --feature_type frequency --split test
conda run --live-stream -n ml_termproj python -m src.features.cache_features --config configs/synthbuster_ood.yaml --feature_type clip --split test

conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model frequency_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model clip_only --split test
conda run --live-stream -n ml_termproj python -m src.eval.evaluate --config configs/synthbuster_ood.yaml --model fusion --split test
```

Synthbuster manifest를 직접 쓰는 image-level robustness runner도 사용할 수 있습니다. 전체 9,000장 × corruption 조합은 오래 걸리므로 먼저 `--max_samples`로 smoke test를 권장합니다.

```bash
conda run --live-stream -n ml_termproj python scripts/robustness_test.py \
  --config configs/synthbuster_ood.yaml \
  --manifest outputs/synthbuster_ood/manifest.csv \
  --data_root data/synthbuster \
  --models fusion,clip_only,frequency_only \
  --max_samples 50 \
  --output_csv outputs/metrics/synthbuster_ood/robustness_metrics.csv \
  --output_json outputs/metrics/synthbuster_ood/robustness_metrics.json \
  --plot_path outputs/plots/synthbuster_ood/robustness_barplot.png
```

## 10. Demo

Streamlit demo:

```bash
conda run --live-stream -n ml_termproj streamlit run src/app/app.py
```

Gradio demo:

```bash
conda run --live-stream -n ml_termproj python src/demo/app_gradio.py --config configs/fusion.yaml
```

Demo 출력은 benchmark 조건에서 학습된 실험적 confidence score입니다. 법적/절대적 판정으로 사용하면 안 됩니다.

단일 이미지의 DCT/FFT spectrum과 inference 설정과 동일한 radial spectrum plot만 파일로 뽑으려면 다음 명령을 사용합니다. `frequency.method`, `frequency.image_size`, `frequency.radial_bins`, `log_scale`, `normalize_feature`는 config 값을 따릅니다.

```bash
conda run --live-stream -n ml_termproj python -m src.visualization.plot_frequency_spectrum \
  --config configs/default.yaml \
  --image data/genimage_tiny_subset/real/real/genimage_real_real_000049.jpg \
  --output-dir outputs/plots
```

기본 산출물은 다음 형태입니다.

```text
outputs/plots/<image_stem>_<token>_spectrum.png
outputs/plots/<image_stem>_<token>_radial.png
```

## 11. Current Artifact Map

현재 주요 artifact는 다음 위치에 있습니다.

| Path | Meaning |
|---|---|
| `data/genimage_tiny_full_512/` | materialized Tiny-GenImage 512x512 images |
| `data/metadata/genimage_tiny_full_dataset.csv` | filepath 기반 metadata CSV |
| `outputs/genimage_tiny_full/manifest.csv` | root/rel_path 기반 manifest-v1 CSV |
| `data/synthbuster/` | extracted Synthbuster OOD images |
| `data/metadata/synthbuster_ood_dataset.csv` | Synthbuster OOD filepath 기반 metadata CSV |
| `outputs/synthbuster_ood/manifest.csv` | Synthbuster OOD manifest-v1 CSV |
| `artifacts/features/` | CLIP/frequency feature cache |
| `artifacts/checkpoints/` | `frequency_only.pt`, `clip_only.pt`, `fusion.pt` |
| `artifacts/scalers/frequency_scaler.pkl` | fusion/frequency robustness용 frequency scaler |
| `outputs/metrics/` | train/eval/robustness metrics CSV/JSON |
| `outputs/plots/` | robustness plot 등 plot outputs |

## 12. Verification Commands

현재 pipeline sanity check에 사용한 명령은 다음입니다.

```bash
conda run --live-stream -n ml_termproj python scripts/robustness_test.py --config configs/fusion.yaml --models fusion --max_samples 50 --dry_run
conda run --live-stream -n ml_termproj python scripts/robustness_test.py --config configs/fusion.yaml --models fusion --max_samples 50
conda run --live-stream -n ml_termproj python -m pytest -q tests/
```

최근 확인 결과는 다음과 같습니다.

```text
fusion robustness rows: 9 ok
fusion scaler_status:   9 loaded
pytest:                 103 passed, 1 warning
```

## 13. Limitations

- 이 모델은 Tiny-GenImage/GenImage split 기준 실험 모델이며 범용 detector가 아닙니다.
- CLIP 기반 경로는 CLIP weight/cache availability에 의존합니다.
- Feature cache 재생성 시 metadata CSV와 manifest CSV의 차이를 혼동하면 이미지 path 오류가 날 수 있습니다.
- `outputs/`에는 legacy 실험 산출물도 포함될 수 있으므로 최신 pipeline 판단에는 위에 명시한 config와 artifact map을 우선 확인합니다.
- `scripts/*` 중 일부는 compatibility wrapper입니다. 가능한 경우 `python -m src...` 경로를 우선 사용합니다.
