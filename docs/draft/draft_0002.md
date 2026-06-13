# AI-GEN Image Detector 구현 프롬프트

너는 Python/PyTorch 기반 머신러닝 프로젝트를 구현하는 코딩 에이전트다.  
현재 프로젝트는 기계학습 팀프로젝트이며, 주제는 **AI-GEN Image Detector**이다.

목표는 사용자가 JPG 또는 PNG 이미지를 업로드하면 해당 이미지가 실제 촬영 이미지인지 AI 생성 이미지인지 확률값으로 판단하는 시스템을 구현하는 것이다.

최종 시스템은 다음 정보를 제공해야 한다.

- AI-generated probability
  
- 최종 판정: Real 또는 AI
  
- confidence level: high / medium / low
  
- CLIP branch score
  
- frequency branch score
  
- fusion score
  
- DCT/FFT spectrum visualization
  
- radial spectrum graph
  
- 모델별 성능 비교 결과: CLIP-only, Frequency-only, Fusion
  
- robustness test 결과: JPEG compression, resize, blur
  

코드베이스를 이에 맞게 수정하고 리팩토링하라.

---

## 1. 핵심 구현 원칙

반드시 아래 원칙을 지켜라.

1. 모든 이미지 데이터 관리는 `dataset.csv`를 기준으로 한다.
  
2. 학습 코드는 이미지 폴더 구조를 직접 가정하지 않는다.
  
3. `dataset.csv`에 저장된 `filepath`, `label`, `class_name`, `dataset`, `generator`, `split` 정보를 사용한다.
  
4. real label은 `0`, fake label은 `1`로 고정한다.
  
5. feature extraction과 classifier training을 분리한다.
  
6. CLIP image encoder는 frozen feature extractor로만 사용한다.
  
7. feature는 `.npy`와 metadata `.csv`로 cache한다.
  
8. 모델 구현 순서는 `CLIP-only → Frequency-only → Fusion`으로 한다.
  
9. 평가 코드는 모든 모델에 공통으로 재사용 가능해야 한다.
  
10. demo app은 모델 내부를 직접 호출하지 말고 `DetectorService`를 통해 예측해야 한다.
  
11. 구현이 끝날 때마다 실행 가능한 command와 smoke test를 제공한다.
  
12. 코드에는 최소한의 타입 힌트, docstring, 에러 메시지를 포함한다.
  
13. 이번 구현 목표는 classifier fine-tuning이 가능한 end to end model 아키텍처 구현이다.
  
14. 데이터가 아직 없을 수 있으므로, 실제 데이터 없이도 구조 검증이 가능한 dummy metadata/sample generator 또는 test fixture를 제공한다.
  
15. 사람이 발표 직전에 울지 않도록 README에 실행 순서를 명확히 적는다.
  

---

## 2. 기술 스택 및 개발 환경

반드시 아래 환경과 기술을 사용하라.

conda environment ml_termproj 사용: `conda activate ml_termproj`

필요한 패키지 및 모듈 역시 ml_termproj conda 환경에 설치

requirements.txt 작성으로 의존성 관리

CLIP의 경우 open_clip_torch 사용

---

## 3. 최종 프로젝트 구조

아래 구조를 생성하라.

```text
ai-gen-image-detector/
  README.md
  requirements.txt
  .gitignore
  configs/
    default.yaml

  data/
    raw/
      .gitkeep
    metadata/
      dataset.csv.example
    processed/
      .gitkeep

  src/
    __init__.py

    data/
      __init__.py
      dataset.py
      transforms.py
      validate_metadata.py
      make_split.py
      make_dummy_dataset.py

    features/
      __init__.py
      clip_features.py
      frequency_features.py
      cache_features.py

    models/
      __init__.py
      mlp_classifier.py
      fusion_classifier.py

    train/
      __init__.py
      train_clip.py
      train_frequency.py
      train_fusion.py

    eval/
      __init__.py
      metrics.py
      evaluate.py
      robustness.py

    inference/
      __init__.py
      detector_service.py

    visualization/
      __init__.py
      spectrum.py
      radial_spectrum.py

    app/
      __init__.py
      app.py

    utils/
      __init__.py
      config.py
      seed.py
      io.py
      logger.py

  artifacts/
    features/
      .gitkeep
    checkpoints/
      .gitkeep
    reports/
      .gitkeep
    figures/
      .gitkeep

  tests/
    test_metadata.py
    test_dataset.py
    test_frequency_features.py
    test_models.py
```

`.gitignore`에는 아래를 포함하라.

```gitignore
__pycache__/
*.pyc
.env
.venv/
data/raw/*
data/processed/*
artifacts/features/*
artifacts/checkpoints/*
artifacts/reports/*
artifacts/figures/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!artifacts/features/.gitkeep
!artifacts/checkpoints/.gitkeep
!artifacts/reports/.gitkeep
!artifacts/figures/.gitkeep
```

---

## 4. `dataset.csv` 설계

중앙 metadata 파일은 다음 위치에 둔다.

```text
data/metadata/dataset.csv
```

단, 실제 데이터가 없을 수 있으므로 repository에는 예시 파일을 둔다.

```text
data/metadata/dataset.csv.example
```

필수 컬럼은 다음과 같다.

```text
image_id
filepath
label
class_name
dataset
generator
split
width
height
ext
```

컬럼 의미:

| column | type | required | example | description |
| --- | --- | --- | --- | --- |
| image_id | str | yes | cifake_real_000001 | 이미지 고유 ID |
| filepath | str | yes | data/raw/cifake/real/000001.png | 이미지 경로 |
| label | int | yes | 0 또는 1 | real=0, fake=1 |
| class_name | str | yes | real 또는 fake | 사람이 읽는 class 이름 |
| dataset | str | yes | CIFAKE, GenImage, COCO | 데이터셋 출처 |
| generator | str | yes | real_camera, cifake_diffusion, stable_diffusion | 생성 모델 또는 real source |
| split | str | yes | train, val, test | 데이터 split |
| width | int | recommended | 224 | 원본 이미지 너비 |
| height | int | recommended | 224 | 원본 이미지 높이 |
| ext | str | recommended | jpg, png | 이미지 확장자 |

`dataset.csv.example` 예시:

```csv
image_id,filepath,label,class_name,dataset,generator,split,width,height,ext
cifake_real_000001,data/raw/cifake/real/000001.png,0,real,CIFAKE,real_camera,train,32,32,png
cifake_real_000002,data/raw/cifake/real/000002.png,0,real,CIFAKE,real_camera,val,32,32,png
cifake_real_000003,data/raw/cifake/real/000003.png,0,real,CIFAKE,real_camera,test,32,32,png
cifake_fake_000001,data/raw/cifake/fake/000001.png,1,fake,CIFAKE,cifake_diffusion,train,32,32,png
cifake_fake_000002,data/raw/cifake/fake/000002.png,1,fake,CIFAKE,cifake_diffusion,val,32,32,png
cifake_fake_000003,data/raw/cifake/fake/000003.png,1,fake,CIFAKE,cifake_diffusion,test,32,32,png
```

중요한 규칙:

- `label=0`이면 `class_name=real`
  
- `label=1`이면 `class_name=fake`
  
- `split`은 반드시 `train`, `val`, `test` 중 하나
  
- 같은 `image_id` 중복 금지
  
- 같은 `filepath` 중복 금지
  
- `filepath` 파일 존재 여부를 validation에서 검사
  
- generator별 count를 출력
  
- split별 real/fake count를 출력
  
- 나중에 GenImage를 추가해도 같은 `dataset.csv`에 행을 추가하는 방식으로 확장
  

---

## 5. `configs/default.yaml` 작성

다음 설정 파일을 작성하라.

```yaml
project:
  name: ai_gen_image_detector
  seed: 42
  device: cuda

paths:
  dataset_csv: data/metadata/dataset.csv
  raw_data_dir: data/raw
  feature_dir: artifacts/features
  checkpoint_dir: artifacts/checkpoints
  report_dir: artifacts/reports
  figure_dir: artifacts/figures

data:
  image_size: 224
  batch_size: 32
  num_workers: 4
  label_map:
    real: 0
    fake: 1
  splits:
    train: train
    val: val
    test: test

clip:
  model_name: ViT-B-32
  pretrained: openai
  output_dim: 512
  freeze: true
  normalize_feature: true

frequency:
  method: dct
  image_size: 224
  grayscale: true
  radial_bins: 64
  log_scale: true
  normalize_feature: true

classifier:
  type: mlp
  hidden_dim: 256
  dropout: 0.2
  num_classes: 1

train:
  epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 5
  loss: bce_with_logits

eval:
  threshold: 0.5
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc
  per_generator: true
  save_predictions: true

robustness:
  jpeg_qualities: [95, 75, 50]
  resize_scales: [0.5]
  blur_sigmas: [1.0, 2.0]

demo:
  confidence:
    high_margin: 0.25
    medium_margin: 0.10
```

`src/utils/config.py`에서 YAML을 읽는 함수를 구현하라.

```python
def load_config(config_path: str) -> dict:
    ...
```

`src/utils/seed.py`에서 seed 고정 함수를 구현하라.

```python
def set_seed(seed: int) -> None:
    ...
```

---

## 6. 데이터 검증 구현

`src/data/validate_metadata.py`를 구현하라.

실행 예시:

```bash
python -m src.data.validate_metadata --csv data/metadata/dataset.csv
```

검사할 항목:

1. 필수 컬럼 존재 여부
  
2. `image_id` 중복 여부
  
3. `filepath` 중복 여부
  
4. `filepath` 실제 파일 존재 여부
  
5. `label`이 0 또는 1인지
  
6. `class_name`이 real 또는 fake인지
  
7. label과 class_name이 일치하는지
  
8. split이 train/val/test 중 하나인지
  
9. split별 sample count 출력
  
10. split별 label count 출력
  
11. generator별 sample count 출력
  
12. dataset별 sample count 출력
  

에러 발생 시 명확한 메시지를 출력하라.

예:

```text
Missing required columns: ['generator']
Invalid labels found: {2}
Missing files: ['data/raw/cifake/real/000001.png']
Duplicated image_id found: ['cifake_real_000001']
```

---

## 7. dummy dataset 생성기 구현

실제 이미지 데이터가 없을 수도 있으므로 `src/data/make_dummy_dataset.py`를 구현하라.

역할:

- 작은 RGB PNG 이미지를 생성한다.
  
- real/fake dummy 이미지를 만든다.
  
- `data/metadata/dataset.csv`를 자동 생성한다.
  
- train/val/test split을 만든다.
  
- 테스트와 smoke test가 바로 돌아가게 한다.
  

실행 예시:

```bash
python -m src.data.make_dummy_dataset --num_real 30 --num_fake 30 --output_dir data/raw/dummy --csv data/metadata/dataset.csv
```

생성 결과:

```text
data/raw/dummy/real/real_000001.png
data/raw/dummy/fake/fake_000001.png
data/metadata/dataset.csv
```

dummy CSV 예시:

```csv
image_id,filepath,label,class_name,dataset,generator,split,width,height,ext
dummy_real_000001,data/raw/dummy/real/real_000001.png,0,real,DUMMY,real_dummy,train,224,224,png
dummy_fake_000001,data/raw/dummy/fake/fake_000001.png,1,fake,DUMMY,dummy_generator,train,224,224,png
```

---

## 8. Dataset/DataLoader 구현

`src/data/dataset.py`에 `ImageMetadataDataset`을 구현하라.

요구사항:

- `csv_path`
  
- `split`
  
- `transform`
  
- `return_metadata=True`
  
- PIL로 이미지 로드
  
- RGB 변환
  
- label은 int로 반환
  
- metadata dict 반환
  

반환 형식:

```python
image, label, metadata
```

metadata는 최소한 아래 값을 포함해야 한다.

```python
{
    "image_id": ...,
    "filepath": ...,
    "class_name": ...,
    "dataset": ...,
    "generator": ...,
    "split": ...
}
```

`src/data/transforms.py`를 구현하라.

요구사항:

- `get_train_transform(image_size: int)`
  
- `get_eval_transform(image_size: int)`
  
- CLIP normalize 값을 사용
  
- train transform은 resize, random horizontal flip, tensor, normalize
  
- eval transform은 resize, tensor, normalize
  

CLIP normalize 값:

```python
mean=[0.48145466, 0.4578275, 0.40821073]
std=[0.26862954, 0.26130258, 0.27577711]
```

테스트:

```bash
pytest tests/test_dataset.py
```

DataLoader smoke test에서 다음을 확인하라.

```text
images.shape == [B, 3, 224, 224]
labels.shape == [B]
metadata["image_id"] exists
```

---

## 9. CLIP feature extractor 구현

`src/features/clip_features.py`를 구현하라.

기본 라이브러리는 `open_clip_torch`를 사용한다.

요구사항:

- config에서 `clip.model_name`, `clip.pretrained`를 읽는다.
  
- CLIP image encoder를 로드한다.
  
- `freeze: true`면 모든 parameter `requires_grad=False`
  
- eval mode로 설정한다.
  
- dataloader에서 image batch를 받아 feature를 추출한다.
  
- feature는 CPU numpy array로 반환한다.
  
- `normalize_feature: true`면 L2 normalize한다.
  
- 추출 중 gradient 계산을 하지 않는다.
  

함수 예시:

```python
def load_clip_model(config: dict, device: str):
    ...

def extract_clip_features(model, dataloader, device: str, normalize: bool = True):
    ...
    return features, labels, meta_df
```

주의:

- `open_clip`의 preprocessing과 현재 transform이 충돌하지 않도록 처리한다.
  
- 우선은 `transforms.py`의 CLIP normalize를 사용한다.
  
- CLIP 모델 로딩 실패 시 명확한 에러 메시지를 출력한다.
  

---

## 10. Frequency feature extractor 구현

`src/features/frequency_features.py`를 구현하라.

처음에는 DCT 기반으로 구현한다. FFT도 option으로 지원 가능하게 하되, 기본값은 `dct`이다.

요구사항:

- PIL image 또는 torch tensor 입력을 처리할 수 있게 한다.
  
- RGB 이미지를 grayscale로 변환한다.
  
- `image_size`로 resize한다.
  
- DCT 또는 FFT를 계산한다.
  
- magnitude 또는 coefficient energy를 구한다.
  
- `log_scale: true`면 `log1p`를 적용한다.
  
- radial spectrum feature를 만든다.
  
- `radial_bins=64`면 64차원 vector를 반환한다.
  
- normalize 옵션을 제공한다.
  

함수 예시:

```python
def image_to_grayscale_array(image, image_size: int) -> np.ndarray:
    ...

def compute_dct_spectrum(gray: np.ndarray) -> np.ndarray:
    ...

def compute_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    ...

def radial_average(spectrum: np.ndarray, bins: int) -> np.ndarray:
    ...

def extract_frequency_feature(image, config: dict) -> np.ndarray:
    ...
```

DCT는 `scipy.fftpack.dct` 또는 `scipy.fft.dctn`을 사용한다.  
FFT는 `numpy.fft.fft2`, `numpy.fft.fftshift`를 사용한다.

테스트:

```bash
pytest tests/test_frequency_features.py
```

테스트 조건:

```text
feature.shape == [64]
feature contains finite values
feature has no NaN
feature has no inf
```

---

## 11. Feature cache 구현

`src/features/cache_features.py`를 구현하라.

실행 예시:

```bash
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test

python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
```

저장 구조:

```text
artifacts/features/
  clip/
    train_features.npy
    train_labels.npy
    train_meta.csv
    val_features.npy
    val_labels.npy
    val_meta.csv
    test_features.npy
    test_labels.npy
    test_meta.csv

  frequency/
    train_features.npy
    train_labels.npy
    train_meta.csv
    val_features.npy
    val_labels.npy
    val_meta.csv
    test_features.npy
    test_labels.npy
    test_meta.csv
```

저장 시 shape를 출력하라.

예:

```text
Saved clip train features: artifacts/features/clip/train_features.npy
features shape: (1400, 512)
labels shape: (1400,)
meta shape: (1400, 6)
```

---

## 12. 모델 구현

`src/models/mlp_classifier.py`를 구현하라.

요구사항:

- Binary classification용 MLP
  
- output logit 하나
  
- `BCEWithLogitsLoss`와 호환
  
- 입력 차원 configurable
  

예시 구조:

```python
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
```

`src/models/fusion_classifier.py`도 구현하라.

- 기본적으로 `MLPClassifier`를 reuse해도 된다.
  
- CLIP feature와 frequency feature concat을 전제로 한다.
  
- input_dim = clip_dim + freq_dim
  

---

## 13. 학습 코드 구현

### 13.1 CLIP-only 학습

`src/train/train_clip.py`를 구현하라.

입력:

```text
artifacts/features/clip/train_features.npy
artifacts/features/clip/train_labels.npy
artifacts/features/clip/val_features.npy
artifacts/features/clip/val_labels.npy
```

출력:

```text
artifacts/checkpoints/clip_only.pt
artifacts/reports/clip_only_train_log.csv
artifacts/reports/clip_only_val_metrics.json
```

실행:

```bash
python -m src.train.train_clip --config configs/default.yaml
```

요구사항:

- feature `.npy` 로드
  
- torch TensorDataset 생성
  
- MLPClassifier 학습
  
- BCEWithLogitsLoss 사용
  
- AdamW optimizer 사용
  
- validation ROC-AUC 또는 loss 기준 best checkpoint 저장
  
- early stopping 구현
  
- train loss, val loss, val accuracy, val f1, val roc_auc 기록
  

---

### 13.2 Frequency-only 학습

`src/train/train_frequency.py`를 구현하라.

입력:

```text
artifacts/features/frequency/train_features.npy
artifacts/features/frequency/train_labels.npy
artifacts/features/frequency/val_features.npy
artifacts/features/frequency/val_labels.npy
```

출력:

```text
artifacts/checkpoints/frequency_only.pt
artifacts/reports/frequency_only_train_log.csv
artifacts/reports/frequency_only_val_metrics.json
```

실행:

```bash
python -m src.train.train_frequency --config configs/default.yaml
```

---

### 13.3 Fusion 학습

`src/train/train_fusion.py`를 구현하라.

입력:

```text
artifacts/features/clip/train_features.npy
artifacts/features/frequency/train_features.npy
artifacts/features/clip/train_labels.npy

artifacts/features/clip/val_features.npy
artifacts/features/frequency/val_features.npy
artifacts/features/clip/val_labels.npy
```

요구사항:

- CLIP feature와 frequency feature를 image_id 기준으로 정렬해야 한다.
  
- 단순히 array 순서가 같다고 무조건 가정하지 말고, meta.csv의 `image_id`를 기준으로 alignment를 검증한다.
  
- train_meta.csv의 image_id 순서가 다르면 안전하게 merge/reorder한다.
  
- label mismatch가 있으면 에러를 발생시킨다.
  
- concat feature 생성 후 MLPClassifier 학습
  

출력:

```text
artifacts/checkpoints/fusion.pt
artifacts/reports/fusion_train_log.csv
artifacts/reports/fusion_val_metrics.json
```

실행:

```bash
python -m src.train.train_fusion --config configs/default.yaml
```

---

## 14. 평가 코드 구현

`src/eval/metrics.py`를 구현하라.

지원 metric:

- accuracy
  
- precision
  
- recall
  
- f1
  
- roc_auc
  
- confusion matrix
  

함수 예시:

```python
def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    ...
```

`src/eval/evaluate.py`를 구현하라.

실행 예시:

```bash
python -m src.eval.evaluate --config configs/default.yaml --model clip_only --split test
python -m src.eval.evaluate --config configs/default.yaml --model frequency_only --split test
python -m src.eval.evaluate --config configs/default.yaml --model fusion --split test
```

출력:

```text
artifacts/reports/clip_only_test_metrics.json
artifacts/reports/clip_only_test_predictions.csv

artifacts/reports/frequency_only_test_metrics.json
artifacts/reports/frequency_only_test_predictions.csv

artifacts/reports/fusion_test_metrics.json
artifacts/reports/fusion_test_predictions.csv
```

prediction CSV 컬럼:

```text
image_id
filepath
label
pred_prob
pred_label
class_name
dataset
generator
split
model_name
```

per-generator metric도 저장하라.

```text
artifacts/reports/fusion_test_per_generator_metrics.csv
```

컬럼:

```text
model_name
generator
count
accuracy
precision
recall
f1
roc_auc
```

ROC-AUC 계산이 불가능한 경우, 예를 들어 generator 내부에 label이 하나뿐인 경우에는 `roc_auc = null` 또는 `NaN`으로 저장하고 warning을 출력하라.

---

## 15. Robustness test 구현

`src/eval/robustness.py`를 구현하라.

대상 corruption:

1. JPEG compression
  
  - quality 95
    
  - quality 75
    
  - quality 50
    
2. Resize
  
  - scale 0.5로 축소 후 원래 크기로 재확대
3. Gaussian blur
  
  - sigma 1.0
    
  - sigma 2.0
    

실행:

```bash
python -m src.eval.robustness --config configs/default.yaml --model fusion --split test
```

출력:

```text
artifacts/reports/fusion_robustness_metrics.csv
```

컬럼:

```text
model_name
corruption
severity
accuracy
precision
recall
f1
roc_auc
```

구현 방식:

- test split 이미지 파일을 직접 로드한다.
  
- corruption을 적용한다.
  
- 해당 모델에 맞는 feature를 즉석 추출한다.
  
- model inference를 수행한다.
  
- metric을 계산한다.
  

주의:

- CLIP-only 모델은 CLIP feature만 추출한다.
  
- Frequency-only 모델은 frequency feature만 추출한다.
  
- Fusion 모델은 CLIP feature와 frequency feature를 모두 추출해서 concat한다.
  

---

## 16. Visualization 구현

`src/visualization/spectrum.py`를 구현하라.

기능:

```python
def save_spectrum_image(image, output_path: str, method: str = "dct") -> str:
    ...
```

- grayscale 변환
  
- DCT 또는 FFT spectrum 계산
  
- log magnitude 적용
  
- matplotlib로 저장
  
- output path 반환
  

`src/visualization/radial_spectrum.py`를 구현하라.

기능:

```python
def save_radial_spectrum_plot(feature: np.ndarray, output_path: str) -> str:
    ...
```

- radial feature vector를 line plot으로 저장
  
- x축은 frequency bin
  
- y축은 normalized energy
  
- output path 반환
  

matplotlib 사용 시 불필요한 스타일 지정은 하지 않는다.

---

## 17. DetectorService 구현

`src/inference/detector_service.py`를 구현하라.

목표:

```python
detector = DetectorService(config_path="configs/default.yaml", model_name="fusion")
result = detector.predict("sample.png")
```

반환 형식:

```python
{
    "ai_prob": 0.82,
    "pred_label": "AI",
    "confidence": "high",
    "clip_score": 0.76,
    "frequency_score": 0.88,
    "fusion_score": 0.82,
    "spectrum_path": "artifacts/figures/sample_spectrum.png",
    "radial_spectrum_path": "artifacts/figures/sample_radial.png"
}
```

요구사항:

- config 로드
  
- device 결정
  
- checkpoint 로드
  
- CLIP extractor 로드
  
- frequency extractor 준비
  
- 단일 이미지 전처리
  
- model별 inference 지원
  
  - `clip_only`
    
  - `frequency_only`
    
  - `fusion`
    
- `sigmoid(logit)`으로 fake probability 계산
  
- threshold 0.5 기준 pred_label 결정
  
  - prob >= 0.5 → `AI`
    
  - prob < 0.5 → `Real`
    
- confidence 계산
  

confidence 기준:

```python
margin = abs(prob - 0.5)

if margin >= 0.25:
    confidence = "high"
elif margin >= 0.10:
    confidence = "medium"
else:
    confidence = "low"
```

branch score 처리:

- clip_only 모델이면 `clip_score=ai_prob`, `frequency_score=None`, `fusion_score=None`
  
- frequency_only 모델이면 `clip_score=None`, `frequency_score=ai_prob`, `fusion_score=None`
  
- fusion 모델이면 가능하면 별도 clip_only/frequency_only checkpoint도 로드해서 branch score를 제공한다.
  
- branch checkpoint가 없으면 branch score는 `None`으로 두고 warning을 출력한다.
  

---

## 18. Streamlit demo 구현

`src/app/app.py`를 구현하라.

실행:

```bash
streamlit run src/app/app.py
```

기능 요구사항:

- JPG, JPEG, PNG 업로드
  
- 단건 이미지 업로드 우선 구현
  
- 업로드 이미지 미리보기
  
- `DetectorService.predict()` 호출
  
- AI-generated probability 출력
  
- Real/AI 판정 출력
  
- confidence level 출력
  
- CLIP score 출력
  
- frequency score 출력
  
- fusion score 출력
  
- spectrum image 출력
  
- radial spectrum graph 출력
  
- 제한 문구 출력
  

제한 문구:

```text
이 결과는 제한된 데이터셋 기준의 탐지 결과이며, 모든 AI 생성 이미지를 완벽하게 판별한다는 의미는 아닙니다.
```

---

## 19. README 작성

`README.md`에 다음 내용을 포함하라.

1. 프로젝트 소개
  
2. 폴더 구조
  
3. 설치 방법
  
4. config 설명
  
5. dataset.csv schema
  
6. dummy dataset 생성 방법
  
7. metadata validation 방법
  
8. feature cache 생성 방법
  
9. CLIP-only 학습 방법
  
10. Frequency-only 학습 방법
  
11. Fusion 학습 방법
  
12. 평가 방법
  
13. robustness test 방법
  
14. demo 실행 방법
  
15. 산출물 위치
  
16. 제한 사항
  

README 실행 순서 예시:

```bash
pip install -r requirements.txt

python -m src.data.make_dummy_dataset \
  --num_real 30 \
  --num_fake 30 \
  --output_dir data/raw/dummy \
  --csv data/metadata/dataset.csv

python -m src.data.validate_metadata \
  --csv data/metadata/dataset.csv

python -m src.features.cache_features \
  --config configs/default.yaml \
  --feature_type clip \
  --split train

python -m src.features.cache_features \
  --config configs/default.yaml \
  --feature_type clip \
  --split val

python -m src.features.cache_features \
  --config configs/default.yaml \
  --feature_type clip \
  --split test

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

python -m src.train.train_clip \
  --config configs/default.yaml

python -m src.train.train_frequency \
  --config configs/default.yaml

python -m src.train.train_fusion \
  --config configs/default.yaml

python -m src.eval.evaluate \
  --config configs/default.yaml \
  --model fusion \
  --split test

python -m src.eval.robustness \
  --config configs/default.yaml \
  --model fusion \
  --split test

streamlit run src/app/app.py
```

---

## 20. 테스트 구현

pytest 기반 테스트를 작성하라.

### `tests/test_metadata.py`

검사:

- example CSV required columns
  
- invalid label detection
  
- duplicate image_id detection
  
- invalid split detection
  

### `tests/test_dataset.py`

검사:

- dummy dataset 생성
  
- ImageMetadataDataset 로드
  
- image tensor shape
  
- label type
  
- metadata keys
  

### `tests/test_frequency_features.py`

검사:

- dummy PIL image 입력
  
- DCT feature shape
  
- FFT feature shape
  
- radial bins = 64
  
- NaN/inf 없음
  

### `tests/test_models.py`

검사:

- MLPClassifier input `[B, D]`
  
- output shape `[B]`
  
- BCEWithLogitsLoss 계산 가능
  

실행:

```bash
pytest
```

---

## 21. 구현 순서

반드시 아래 순서대로 구현하라.

### Step 1. 프로젝트 구조 생성

- 폴더 생성
  
- `__init__.py` 생성
  
- `.gitignore` 생성
  
- `requirements.txt` 생성
  

### Step 2. Config utility 구현

- `configs/default.yaml`
  
- `src/utils/config.py`
  
- `src/utils/seed.py`
  
- `src/utils/io.py`
  
- `src/utils/logger.py`
  

### Step 3. Metadata pipeline 구현

- `dataset.csv.example`
  
- `make_dummy_dataset.py`
  
- `validate_metadata.py`
  

### Step 4. Dataset/DataLoader 구현

- `dataset.py`
  
- `transforms.py`
  
- dataset test
  

### Step 5. Frequency feature 구현

- `frequency_features.py`
  
- spectrum visualization
  
- radial spectrum visualization
  
- tests
  

### Step 6. CLIP feature 구현

- `clip_features.py`
  
- open_clip 로딩
  
- feature extraction
  

### Step 7. Feature cache 구현

- `cache_features.py`
  
- clip/frequency 저장
  

### Step 8. Model 구현

- `mlp_classifier.py`
  
- `fusion_classifier.py`
  

### Step 9. Training 구현

- `train_clip.py`
  
- `train_frequency.py`
  
- `train_fusion.py`
  

### Step 10. Evaluation 구현

- `metrics.py`
  
- `evaluate.py`
  
- prediction CSV 저장
  
- per-generator metrics 저장
  

### Step 11. Robustness 구현

- JPEG
  
- resize
  
- blur
  
- robustness metrics CSV
  

### Step 12. Inference 구현

- `detector_service.py`

### Step 13. Demo 구현

- Streamlit app

### Step 14. README 정리

- 실행 순서
  
- 파일 구조
  
- 주의사항
  

### Step 15. 전체 smoke test

최소한 dummy dataset으로 아래 command들이 끝까지 돌아가야 한다.

```bash
python -m src.data.make_dummy_dataset --num_real 30 --num_fake 30 --output_dir data/raw/dummy --csv data/metadata/dataset.csv
python -m src.data.validate_metadata --csv data/metadata/dataset.csv
python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
python -m src.train.train_frequency --config configs/default.yaml
python -m src.eval.evaluate --config configs/default.yaml --model frequency_only --split test
pytest
```

CLIP은 모델 다운로드가 필요할 수 있으므로 네트워크가 없는 환경에서는 skip 가능하게 처리하라. 단, 코드 구조는 완성되어 있어야 한다.

---

## 22. Acceptance Criteria

구현 완료 조건은 다음과 같다.

1. `pytest`가 통과한다.
  
2. dummy dataset 생성이 가능하다.
  
3. metadata validation이 가능하다.
  
4. Dataset/DataLoader가 정상 작동한다.
  
5. DCT frequency feature extraction이 가능하다.
  
6. frequency feature cache 생성이 가능하다.
  
7. Frequency-only classifier 학습과 평가가 가능하다.
  
8. CLIP 환경이 준비된 경우 CLIP feature cache 생성이 가능하다.
  
9. CLIP-only classifier 학습과 평가가 가능하다.
  
10. CLIP/frequency feature가 모두 있을 경우 fusion classifier 학습과 평가가 가능하다.
  
11. evaluation 결과가 JSON/CSV로 저장된다.
  
12. per-generator metric이 저장된다.
  
13. robustness test 결과가 CSV로 저장된다.
  
14. DetectorService가 단일 이미지 predict 결과 dict를 반환한다.
  
15. Streamlit app이 실행된다.
  
16. README만 보고 초보자가 실행 순서를 따라갈 수 있다.
  

---

## 23. 구현 시 주의사항

- 실제 CIFAKE/GenImage 데이터 다운로드 자동화는 이번 범위에 포함하지 않는다.
  
- 데이터셋 경로는 `dataset.csv`를 통해서만 읽는다.
  
- 모델 코드에서 특정 폴더 구조나 위치를 hard-code하지 않는다.
  
- CLIP model fine-tuning은 하지 않는다.
  
- feature extraction과 classifier training을 분리한다.
  
- fusion 학습 시 image_id alignment를 반드시 검증한다.
  
- `label=1`은 AI-generated/fake 확률로 해석한다.
  
- 출력 확률은 항상 fake probability이다.
  
- threshold는 config에서 관리한다.
  
- 모델 성능이 낮아도 pipeline이 안정적으로 작동하는 것을 우선한다.
  

- 에러 메시지는 사람이 이해할 수 있게 작성한다. 사람이 이해 못 하는 에러는 사실상 암호문이다.
  

---

## 24. 최종 보고

구현이 끝나면 다음 형식으로 요약하라.

```text
Implemented:
- ...
- ...

How to run:
1. ...
2. ...

Generated files:
- ...

Known limitations:
- ...

Next recommended tasks:
- ...
```