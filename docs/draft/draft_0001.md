당신은 “AI-GEN Image Detector” 프로젝트의 코딩 에이전트입니다.
아래 요구사항을 기준으로 실제 실행 가능한 코드베이스를 구현하세요.

# 프로젝트 목표

사용자가 JPG/PNG 이미지를 입력하면 해당 이미지가 실제 촬영 이미지인지 AI 생성 이미지인지 확률값으로 분류하는 웹 기반 탐지기를 만든다.

핵심 구조는 다음과 같다.

Input Image
→ Preprocessing
→ Frozen openCLIP image encoder
→ CLIP feature vector
→ FFT/DCT hand-crafted frequency feature extractor
→ Frequency feature vector
→ Feature normalization
→ Concatenation
→ Logistic Regression / SVM / small MLP
→ Real / AI-generated probability

# 최우선 구현 전략

1. CIFAKE subset으로 빠르게 baseline을 완성한다.
2. openCLIP ViT-B/32 image encoder는 frozen feature extractor로만 사용한다.
3. CLIP-only, Frequency-only, CLIP+Frequency fusion 세 가지 실험을 모두 구현한다.
4. feature는 반드시 `.pt` 또는 `.npy`로 cache한다.
5. classifier는 먼저 Logistic Regression과 Linear SVM을 구현하고, 이후 small MLP를 추가한다.
6. JPEG / resize / blur robustness test를 구현한다.
7. 전체 실험 결과를 CSV/JSON/plot으로 저장한다.
8. 마지막에 Streamlit 또는 FastAPI 기반 간단한 이미지 업로드 데모를 만든다.

# 테스트 및 코드 실행 환경

현재 코드를 작성중인 환경은 CUDA GPU가 없는 환경이나, 실제 코드를 실행하고 테스트하는 환경은 CUDA GPU RTX5090이 있다. 만약 필요한 경우 ~/.ssh/config를 참고하고, `ssh tml-server` 로 접속해 코드를 실행하고 테스트한다. remote 서버내 작업 경로는 `~/codes/ml-project`이다. 따라서 테스트가 필요한 경우, rsync를 통해 tml-server:~/codes/ml-project 로 코드 및 디렉토리를 동기화한 뒤, ssh로 접속해서 코드를 실행하고 테스트한다.


# 기술 스택

Python 3.10
torch                     2.9.1+cu128
torchaudio                2.9.1+cu128
torchmetrics              1.9.0
torchvision               0.24.1+cu128
conda environment ml_termproj 사용: conda activate ml_termproj

필수 라이브러리:

* torch
* torchvision
* open_clip_torch
* datasets
* numpy
* scipy
* scikit-learn
* pillow
* opencv-python
* pandas
* matplotlib
* tqdm
* joblib
* pyyaml
* streamlit 또는 fastapi

가능하면 `requirements.txt` 과 `pyproject.toml`을 작성하라.

# 모델 선택

## Spatial branch

기본 backbone:

* `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
* open_clip 라이브러리 사용
* image encoder만 사용
* 모든 backbone parameter는 frozen
* `model.eval()` 사용
* feature extraction 시 `torch.no_grad()` 사용
* 출력 feature는 L2 normalization 수행

구현 예시 구조:

```python
import open_clip

model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
)
tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
```

단, tokenizer는 image-only detector에서는 필요 없으므로 실제 코드에서는 쓰지 않아도 된다.

## Frequency branch

처음부터 learnable frequency network를 만들지 말고, hand-crafted feature로 시작한다.

다음 feature를 구현하라.

### FFT feature

입력 이미지를 RGB로 읽은 뒤 grayscale 또는 luminance 채널로 변환한다.

구현할 feature:

* log magnitude FFT spectrum
* radial power spectrum
* radial spectrum 64-bin 또는 128-bin
* low-frequency energy
* mid-frequency energy
* high-frequency energy
* high-frequency energy ratio
* low/high energy ratio
* spectral slope
* spectrum peak count
* spectrum peak prominence mean/std 가능하면 구현

### DCT feature

가능하면 `scipy.fftpack.dct` 또는 `cv2.dct`를 사용한다.

구현할 feature:

* whole-image DCT coefficient statistics
* 8x8 block-wise DCT statistics
* low/mid/high frequency coefficient energy
* coefficient mean/std/max
* high-frequency DCT energy ratio

최종 frequency feature dimension은 너무 크게 만들지 말고 100~250차원 내외로 유지하라.

# 데이터셋

## 1차 실험

CIFAKE subset 사용.

기본 설정:

* real 1000장
* fake 1000장
* train/val/test = 70:15:15
* seed 고정
* class-balanced split 유지

데이터 로딩은 두 가지 방식을 모두 지원하라.

1. Hugging Face datasets에서 로드
2. local directory에서 로드

local directory 구조 예시:

```text
data/
  cifake/
    real/
      xxx.png
      ...
    fake/
      yyy.png
      ...
```

또는 다음 구조도 지원 가능하게 하라.

```text
data/
  cifake/
    train/
      real/
      fake/
    test/
      real/
      fake/
```

데이터셋 field 이름이나 label 구조가 예상과 다를 수 있으므로, HF dataset loader는 방어적으로 작성하라.
불명확하면 local directory loader를 우선 안정적으로 구현하라.

## 2차 확장

GenImage는 전체 다운로드하지 말 것.
기본 코드에는 GenImage full download를 자동으로 넣지 말고, 추후 generator-balanced subset을 받을 수 있는 확장 구조만 만들어라.

# 코드 구조

다음과 같은 구조로 구현하라.

```text
ai-gen-image-detector/
  README.md
  requirements.txt
  configs/
    default.yaml
    cifake_clip_only.yaml
    cifake_frequency_only.yaml
    cifake_fusion.yaml
  src/
    data/
      dataset.py
      split.py
      transforms.py
    features/
      clip_extractor.py
      frequency_extractor.py
      cache.py
    models/
      classifiers.py
      mlp.py
    train/
      train_sklearn.py
      train_mlp.py
    eval/
      metrics.py
      robustness.py
      plots.py
    inference/
      predictor.py
    utils/
      seed.py
      io.py
      logging.py
  scripts/
    prepare_cifake_subset.py
    extract_clip_features.py
    extract_frequency_features.py
    train_classifier.py
    evaluate.py
    run_robustness.py
    run_all_experiments.py
  app/
    streamlit_app.py
  outputs/
    .gitkeep
  tests/
    test_frequency_features.py
    test_feature_shapes.py
```

기존 코드베이스가 있다면 위 구조를 참고하되, 기존 코드를 불필요하게 대규모 변경하지 말고 새 모듈을 추가하는 방식으로 진행하라.

# 반드시 구현할 CLI

다음 명령이 동작해야 한다.

## 1. CIFAKE subset 준비

```bash
python scripts/prepare_cifake_subset.py \
  --data_root data/cifake \
  --output_root data/cifake_subset \
  --num_real 1000 \
  --num_fake 1000 \
  --seed 42
```

## 2. CLIP feature 추출

```bash
python scripts/extract_clip_features.py \
  --data_root data/cifake_subset \
  --output_path outputs/features/clip_vit_b32.pt \
  --model_name "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K" \
  --batch_size 64 \
  --device cuda
```

## 3. Frequency feature 추출

```bash
python scripts/extract_frequency_features.py \
  --data_root data/cifake_subset \
  --output_path outputs/features/frequency_fft_dct.pt \
  --radial_bins 64
```

## 4. CLIP-only classifier 학습

```bash
python scripts/train_classifier.py \
  --feature_paths outputs/features/clip_vit_b32.pt \
  --mode clip_only \
  --classifier logistic_regression \
  --output_dir outputs/experiments/clip_only_lr
```

## 5. Frequency-only classifier 학습

```bash
python scripts/train_classifier.py \
  --feature_paths outputs/features/frequency_fft_dct.pt \
  --mode frequency_only \
  --classifier logistic_regression \
  --output_dir outputs/experiments/frequency_only_lr
```

## 6. Fusion classifier 학습

```bash
python scripts/train_classifier.py \
  --feature_paths outputs/features/clip_vit_b32.pt outputs/features/frequency_fft_dct.pt \
  --mode fusion \
  --classifier logistic_regression \
  --output_dir outputs/experiments/fusion_lr
```

## 7. MLP 학습

```bash
python scripts/train_classifier.py \
  --feature_paths outputs/features/clip_vit_b32.pt outputs/features/frequency_fft_dct.pt \
  --mode fusion \
  --classifier mlp \
  --output_dir outputs/experiments/fusion_mlp
```

## 8. Robustness test

```bash
python scripts/run_robustness.py \
  --data_root data/cifake_subset \
  --model_dir outputs/experiments/fusion_lr \
  --output_dir outputs/robustness/fusion_lr \
  --tests jpeg resize blur
```

## 9. 전체 실험 실행

```bash
python scripts/run_all_experiments.py \
  --data_root data/cifake_subset \
  --output_dir outputs/full_run \
  --device cuda
```

## 10. 웹 데모 실행

```bash
streamlit run app/streamlit_app.py
```

# Feature cache format

Feature cache는 재현성과 디버깅을 위해 다음 정보를 포함해야 한다.

```python
{
    "features": torch.Tensor or np.ndarray,
    "labels": torch.Tensor or np.ndarray,
    "paths": list[str],
    "split": list[str],  # train / val / test
    "feature_type": "clip" or "frequency",
    "metadata": {
        "model_name": "...",
        "radial_bins": 64,
        "created_at": "...",
        "seed": 42
    }
}
```

CLIP feature와 frequency feature를 fusion할 때는 반드시 `paths` 순서가 일치하는지 검증하라.
일치하지 않으면 path 기준으로 alignment를 수행하라.

# Classifier 구현

다음 classifier를 지원하라.

1. Logistic Regression
2. Linear SVM
3. RBF SVM, 선택 구현
4. Small MLP

## Logistic Regression

* scikit-learn 사용
* StandardScaler 적용
* class_weight="balanced" 옵션은 선택 가능
* max_iter 충분히 크게 설정
* probability output 필요

## SVM

* LinearSVC 또는 SVC
* probability가 필요하면 SVC(probability=True) 또는 calibration 사용
* Linear SVM을 우선 구현

## MLP

작은 데이터셋이므로 매우 작은 구조로 제한하라.

추천 구조:

```text
Input dim
→ Linear(input_dim, 256)
→ ReLU
→ Dropout(0.3)
→ Linear(256, 64)
→ ReLU
→ Dropout(0.2)
→ Linear(64, 1)
→ Sigmoid or BCEWithLogitsLoss
```

학습 설정:

* BCEWithLogitsLoss
* AdamW
* weight_decay = 1e-4
* early stopping
* batch size 32 또는 64
* epoch 50 이하
* validation ROC-AUC 기준 best checkpoint 저장

# Normalization 규칙

CLIP feature:

* L2 normalization 우선 적용

Frequency feature:

* StandardScaler 적용
* train split에서 fit
* val/test에는 transform만 적용

Fusion feature:

* CLIP branch와 frequency branch를 각각 정규화한 뒤 concat
* concat 후 추가 StandardScaler를 적용할지 옵션으로 제공

# 평가 지표

반드시 다음 지표를 계산하라.

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* Average Precision 또는 PR-AUC
* Confusion matrix
* ROC curve
* PR curve

결과 저장:

```text
outputs/experiments/{experiment_name}/
  metrics.json
  predictions.csv
  confusion_matrix.png
  roc_curve.png
  pr_curve.png
  model.joblib or model.pt
  scaler.joblib
  config.yaml
```

`predictions.csv` 컬럼:

```text
path,label,pred_label,prob_fake,split
```

# Robustness test

다음 변형을 test split에 적용한 뒤 기존 classifier로 재평가하라.

## JPEG compression

quality:

* 95
* 75
* 50
* 30

## Resize

다음 변형을 적용하라.

* shorter side 224 유지
* 224 → 160 → 224
* 224 → 128 → 224

## Blur

Gaussian blur:

* sigma 0.5
* sigma 1.0
* sigma 2.0

각 변형별로 metrics를 저장하라.

```text
outputs/robustness/{experiment_name}/
  robustness_metrics.csv
  robustness_summary.png
```

성능 저하량도 계산하라.

```text
clean_accuracy - corrupted_accuracy
clean_auc - corrupted_auc
```

# 시각화

다음 시각화를 가능하면 구현하라.

1. CLIP feature PCA 2D
2. Frequency feature PCA 2D
3. Fusion feature PCA 2D
4. ROC curve
5. PR curve
6. Confusion matrix
7. Robustness degradation bar plot
8. 샘플 이미지별 FFT spectrum visualization

PCA/UMAP 중 PCA는 반드시 구현하고, UMAP은 optional로 둔다.

# Streamlit 데모

`app/streamlit_app.py`를 구현하라.

기능:

* JPG/PNG 이미지 업로드
* 선택 가능한 모델:

  * CLIP-only
  * Frequency-only
  * Fusion
* real/fake probability 출력
* `prob_fake` 출력
* threshold 기본값 0.5
* threshold slider 제공
* 예측 결과:

  * “Likely AI-generated”
  * “Likely real”
* 가능하면 FFT radial spectrum plot 표시
* 주의 문구 표시:

```text
This detector is an experimental model trained on limited benchmark data.
It should not be used as definitive evidence that an image is real or AI-generated.
```

# README 작성

README에는 다음을 포함하라.

1. 프로젝트 개요
2. Architecture diagram 텍스트 버전
3. 설치 방법
4. 데이터셋 준비 방법
5. feature 추출 방법
6. classifier 학습 방법
7. robustness test 실행 방법
8. Streamlit 데모 실행 방법
9. 결과 해석 방법
10. 한계점

특히 한계점에는 다음을 명시하라.

* CIFAKE subset만으로 범용 탐지기 성능을 주장할 수 없다.
* CIFAKE는 generator 다양성이 낮다.
* frequency feature는 JPEG, resize, blur에 취약할 수 있다.
* 모델 출력 확률은 실험적 confidence score이지 법적/절대적 판정이 아니다.
* 실제 DALL-E, Midjourney, SDXL, FLUX 등에 일반화하려면 GenImage, Synthbuster, ForenSynths 등 외부 데이터셋 평가가 필요하다.

# 구현 우선순위

아래 순서대로 작업하라.

## Step 1. 프로젝트 골격 생성

* 폴더 구조 생성
* requirements 작성
* config yaml 작성
* seed utility 작성
* logging utility 작성

## Step 2. Dataset loader 구현

* local directory loader
* train/val/test split
* balanced sampling
* path/label/split metadata 관리

## Step 3. CLIP feature extraction 구현

* openCLIP ViT-B/32 로드
* frozen feature extraction
* batch inference
* L2 normalization
* feature cache 저장

## Step 4. Frequency feature extraction 구현

* FFT radial spectrum
* high-frequency ratio
* DCT statistics
* feature cache 저장
* unit test로 feature shape 검증

## Step 5. Classifier 학습 구현

* Logistic Regression
* Linear SVM
* MLP
* StandardScaler
* metrics 저장
* model/scaler 저장

## Step 6. Fusion pipeline 구현

* CLIP/frequency feature alignment
* branch-wise normalization
* concat
* classifier 학습
* CLIP-only / frequency-only / fusion 성능 비교

## Step 7. Robustness test 구현

* JPEG
* resize
* blur
* clean 대비 성능 하락량 저장

## Step 8. 시각화 구현

* confusion matrix
* ROC curve
* PR curve
* PCA plot
* robustness plot
* FFT spectrum plot

## Step 9. Streamlit demo 구현

* 이미지 업로드
* feature extraction
* model loading
* probability 출력
* FFT visualization
* limitation warning 출력

## Step 10. README와 실행 예시 정리

* 모든 주요 명령어 문서화
* 재현 가능한 실험 순서 작성
* 결과 파일 위치 설명

# Acceptance Criteria

다음 조건을 모두 만족해야 한다.

1. `python scripts/run_all_experiments.py`가 에러 없이 실행된다.
2. CLIP-only, frequency-only, fusion 결과가 각각 생성된다.
3. 각 실험마다 `metrics.json`, `predictions.csv`, `roc_curve.png`, `confusion_matrix.png`가 생성된다.
4. feature cache를 재사용해 classifier만 반복 학습할 수 있다.
5. CLIP feature와 frequency feature concat 시 path alignment 검증이 있다.
6. robustness test 결과가 CSV로 저장된다.
7. Streamlit app에서 단일 이미지 업로드 후 fake probability가 출력된다.
8. README만 보고 다른 사람이 설치와 실행을 따라 할 수 있다.
9. 큰 데이터셋 GenImage 전체를 자동 다운로드하지 않는다.
10. 모델 성능을 범용 탐지기로 과장하는 문구를 넣지 않는다.

# 코딩 스타일

* 함수와 클래스에 type hint를 넣어라.
* 주요 함수에는 docstring을 작성하라.
* 불필요하게 복잡한 abstraction을 만들지 마라.
* 실험 재현성을 위해 seed를 고정하라.
* GPU가 없으면 CPU에서도 최소 동작해야 한다.
* 에러 메시지는 사용자가 원인을 알 수 있게 작성하라.
* feature shape, label count, split count를 로그로 출력하라.
* 경로는 hardcoding하지 말고 CLI argument와 config로 받게 하라.

# 최종 산출물

작업 완료 후 다음을 보고하라.

1. 생성/수정한 파일 목록
2. 실행 가능한 주요 명령어
3. 구현된 모델 종류
4. 구현된 feature 종류
5. 저장되는 결과 파일 구조
6. 아직 구현하지 못한 항목
7. 다음 개선 제안

가장 중요한 목표는 “거창한 논문 재현”이 아니라, 학부 프로젝트 범위에서 실제로 돌아가고 실험 비교가 가능한 AI-generated image detector pipeline을 완성하는 것이다.
