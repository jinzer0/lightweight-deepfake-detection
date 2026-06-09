# AI-GEN Image Detector

머신러닝 텀 프로젝트를 위한 실험용 이미지 탐지기입니다. 이 프로젝트는 CIFAKE 형식의 이미지를 real 또는 fake로 분류하기 위해 사람이 설계한 주파수 특징, 고정된 openCLIP ViT-B/32 이미지 임베딩, 그리고 두 특징을 결합한 fusion 특징을 비교합니다.

이 앱과 스크립트는 연구용 도구이며, 포렌식 도구가 아닙니다. prob_fake는 학습된 모델이 내는 실험적 신뢰도 점수입니다. 확정적인 증거가 아니며, 어떤 이미지가 실제 사진인지 AI 생성 이미지인지 판단하는 법적 증거로 사용해서는 안 됩니다.

## 아키텍처

    로컬 CIFAKE 형식 폴더
      real/ 및 fake/ 이미지
            |
            v
    scripts/prepare_cifake_subset.py
      결정적 split, labels, hashes, dimensions를 담은 manifest v1 CSV
            |
            +-------------------------------+
            |                               |
            v                               v
    scripts/extract_frequency_features.py   scripts/extract_clip_features.py
      FFT/DCT feature_cache_v1 .pt           openCLIP ViT-B/32 feature_cache_v1 .pt
            |                               |
            +---------------+---------------+
                            v
    scripts/train_classifier.py
      frequency_only, clip_only, 또는 fusion
      LogisticRegression 또는 Linear SVM
                            |
                            v
    scripts/evaluate.py 및 scripts/validate_artifacts.py
      metrics.json, predictions.csv, ROC, PR, confusion matrix
                            |
                            +--------------------+
                            |                    |
                            v                    v
    scripts/run_robustness.py           app/streamlit_app.py
      JPEG, resize, blur 검사            artifact 기반 JPG/PNG 데모

이 구현에서는 label을 real=0, fake=1로 고정합니다. 캐시는 manifest 순서와 metadata에 맞춰 검증되므로, 오래되었거나 행이 어긋난 feature가 조용히 재사용되지 않고 즉시 실패합니다.

## 설치

깨끗한 환경에서 Python 3.10 이상을 사용하세요.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

CUDA로 CLIP 특징을 추출하려면 나머지 requirements를 설치하거나 테스트하기 전에 대상 CUDA runtime과 맞는 PyTorch 빌드를 설치하세요. CPU quick 경로에는 CUDA가 필요하지 않습니다.

## CIFAKE 로컬 데이터 준비

스크립트는 real과 fake라는 class 폴더가 있는 로컬 CIFAKE 형식 디렉터리를 사용합니다. 폴더명은 대소문자를 구분하지 않습니다. manifest 코드는 train/real, train/fake, test/real, test/fake처럼 중첩된 source split도 지원합니다.

예시 구조:

    data/cifake/
      train/
        real/
        fake/
      test/
        real/
        fake/

결정적인 manifest를 만듭니다:

    python scripts/prepare_cifake_subset.py --data_root data/cifake --output_manifest outputs/manifests/cifake_manifest.csv --seed 42

작은 로컬 subset이 필요하면 class별 개수를 제한합니다:

    python scripts/prepare_cifake_subset.py --data_root data/cifake --output_manifest outputs/manifests/cifake_small_manifest.csv --num_real 500 --num_fake 500 --seed 42

기존 manifest를 다시 만들지 않고 검증합니다:

    python scripts/prepare_cifake_subset.py --validate_manifest outputs/manifests/cifake_manifest.csv

## Phase A 빠른 실행

먼저 CPU에서 부담 없이 돌릴 수 있는 smoke 경로를 실행하세요:

    python scripts/run_all_experiments.py --quick

기본적으로 이 명령은 outputs/run_all_experiments/quick_data 아래에 아주 작은 synthetic 이미지를 만들고, manifest를 만든 뒤, frequency feature를 추출합니다. 이어서 frequency-only LogisticRegression을 학습하고, artifact를 평가하고, 출력물을 검증합니다. 명시적으로 요청하지 않으면 CLIP은 건너뜁니다.

선택 사항인 quick CLIP smoke:

    python scripts/run_all_experiments.py --quick --quick_include_clip

이 명령은 openCLIP 모델을 다운로드할 수 있으며, 네트워크, 패키지, 모델 캐시 문제로 실패할 수 있습니다. quick synthetic run은 전체 실행 흐름이 연결되어 있는지 확인하는 증거로만 보세요. CIFAKE 성능이나 실제 환경에서의 detector 타당성을 증명하지는 않습니다.

## 전체 로컬 실험 흐름

실제 로컬 CIFAKE subset에서 캐시를 준비하고 각 모델을 학습합니다. 아래 명령은 현재 script help text와 일치합니다.

### Frequency Feature 추출

    python scripts/extract_frequency_features.py --manifest outputs/manifests/cifake_manifest.csv --output_cache outputs/caches/cifake_frequency.pt --seed 42

frequency vector는 224x224 luminance preprocessing, FFT radial features, whole-image DCT summaries, block DCT summaries를 사용합니다.

### CLIP Feature 추출

    python scripts/extract_clip_features.py --manifest outputs/manifests/cifake_manifest.csv --output_cache outputs/caches/cifake_clip.pt --batch_size 32 --device auto --seed 42

작은 smoke extraction을 실행하려면:

    python scripts/extract_clip_features.py --manifest outputs/manifests/cifake_manifest.csv --output_cache outputs/caches/cifake_clip_smoke.pt --max_samples 2 --batch_size 2 --device auto --smoke --write_blocker .sisyphus/evidence/clip_smoke_blocker.txt

기본 model은 hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K입니다.

### Classifier 학습

Frequency-only LogisticRegression:

    python scripts/train_classifier.py --manifest outputs/manifests/cifake_manifest.csv --frequency_cache outputs/caches/cifake_frequency.pt --output_dir outputs/experiments/frequency_only_logistic_regression --mode frequency_only --classifier logistic_regression --seed 42

CLIP-only LogisticRegression:

    python scripts/train_classifier.py --manifest outputs/manifests/cifake_manifest.csv --clip_cache outputs/caches/cifake_clip.pt --output_dir outputs/experiments/clip_only_logistic_regression --mode clip_only --classifier logistic_regression --seed 42

Fusion LogisticRegression:

    python scripts/train_classifier.py --manifest outputs/manifests/cifake_manifest.csv --frequency_cache outputs/caches/cifake_frequency.pt --clip_cache outputs/caches/cifake_clip.pt --output_dir outputs/experiments/fusion_logistic_regression --mode fusion --classifier logistic_regression --seed 42

같은 mode에서 Linear SVM도 사용할 수 있습니다:

    python scripts/train_classifier.py --manifest outputs/manifests/cifake_manifest.csv --frequency_cache outputs/caches/cifake_frequency.pt --output_dir outputs/experiments/frequency_only_linear_svm --mode frequency_only --classifier linear_svm --seed 42

Linear SVM artifact는 decision score만 제공합니다. predict_proba를 제공하지 않으므로, 나중에 calibration을 추가하지 않는 한 Streamlit probability inference에는 사용할 수 없습니다.

### 평가 및 Artifact 검증

    python scripts/evaluate.py --experiment_dir outputs/experiments/frequency_only_logistic_regression --validate

metrics를 새로 만들지 않고 검증만 하려면:

    python scripts/validate_artifacts.py --experiment_dir outputs/experiments/frequency_only_logistic_regression

evaluator는 metrics.json, predictions.csv, confusion_matrix.png, roc_curve.png, pr_curve.png를 쓰거나 확인합니다.

### 전체 자동 실행

로컬 data root를 사용하는 frequency-only full mode 실행:

    python scripts/run_all_experiments.py --data_root data/cifake --output_root outputs/full_run

CLIP-only와 fusion 실험을 포함하려면:

    python scripts/run_all_experiments.py --data_root data/cifake --output_root outputs/full_run --include_clip

리소스가 제한된 환경에서 범위를 정한 로컬 run을 실행하려면 --max_samples_per_class를 사용하세요.

## Robustness 검사

clean-trained frequency LogisticRegression artifact에서 JPEG, resize, blur robustness 검사를 실행합니다:

    python scripts/run_robustness.py --manifest outputs/manifests/cifake_manifest.csv --experiment_dir outputs/experiments/frequency_only_logistic_regression --output_dir outputs/robustness/frequency_only_logistic_regression --quick

robustness 코드는 clean-trained scaler와 classifier를 재사용합니다. 손상된 이미지로 다시 학습하지 않습니다. 출력물은 robustness_metrics.csv와 robustness_summary.png입니다.

## Streamlit 데모

로컬 데모를 실행합니다:

    streamlit run app/streamlit_app.py

config.yaml, model.joblib, scaler.joblib이 들어 있는 artifact directory를 사용하세요. 현재 predictor는 frequency-only LogisticRegression probability artifact를 지원합니다. JPG, JPEG, PNG 업로드를 받을 수 있으며 fake probability threshold의 기본값은 0.5입니다.

앱에는 다음 경고가 표시됩니다:

    이 detector는 제한된 benchmark data로 학습한 실험용 model입니다.
    이미지가 실제 사진인지 AI 생성 이미지인지 판단하는 확정적 증거로 사용해서는 안 됩니다.

## Artifact 위치

자주 쓰는 경로:

    outputs/manifests/                         manifest CSV 파일
    outputs/caches/                            feature_cache_v1 .pt 파일
    outputs/experiments/<experiment_name>/     학습된 model artifact와 plot
    outputs/robustness/<experiment_name>/      robustness CSV 및 PNG 파일
    outputs/run_all_experiments/               기본 quick smoke 출력물

experiment directory에는 다음 파일이 들어 있습니다:

    config.yaml
    model.joblib
    scaler.joblib
    metrics.json
    predictions.csv
    confusion_matrix.png
    roc_curve.png
    pr_curve.png

predictions.csv는 probability를 낼 수 있는 LogisticRegression artifact에서 prob_fake를 사용합니다. decision score만 있는 artifact는 prob_fake를 비워 두고 score로 ranking metrics를 보고합니다.

## 결과 해석

출력물은 실험 증거로만 사용하세요:

- prob_fake는 실험적 모델 신뢰도 점수이며, 확정적 증거나 법적 증거가 아닙니다.
- Accuracy, ROC AUC, PR AUC는 manifest split과 source data에 따라 달라집니다.
- Threshold 0.5는 고정 기본값이며, test split에 맞춰 튜닝한 threshold가 아닙니다.
- Linear SVM score는 margin이며, calibration된 probability가 아닙니다.
- Quick synthetic output은 script 실행 흐름을 확인할 뿐, dataset 성능을 증명하지 않습니다.

run을 비교하기 전에 항상 metrics.json, predictions.csv, plot, run summary JSON, cache metadata를 확인하세요. label polarity, class balance, split membership, cache freshness도 함께 확인해야 합니다.

## 한계

CIFAKE subset 결과만으로는 일반적인 실제 환경 detector에 대한 주장을 뒷받침할 수 없습니다. CIFAKE는 generator 다양성이 낮고, 작은 로컬 subset은 증거 범위를 더 좁힙니다.

Frequency feature는 JPEG compression, resizing, blur, 그리고 비슷한 preprocessing 변화에 민감할 수 있습니다. 안정성을 이야기하기 전에 robustness 검사가 필요합니다.

DALL-E, Midjourney, SDXL, FLUX 및 다른 generator로 일반화하려면 GenImage, Synthbuster, ForenSynths 같은 외부 dataset이 필요합니다. CIFAKE나 synthetic quick run이 이런 dataset을 대체하지 않습니다.

아래에 적힌 증거가 run log에 남아 있지 않다면 remote CUDA 및 full CLIP run은 완료된 것으로 보지 않습니다. 이 README만 보고 CUDA 완료를 주장하지 마세요.

## Phase C 보류 항목

아래 항목은 의도적으로 보류했습니다:

- checkpointing, reload tests, calibration checks를 포함한 MLP classifier.
- dataset ID와 schema를 확인한 뒤 진행할 HF loader hardening.
- RBF SVM.
- UMAP visualizations.
- Advanced peak prominence features.
- 전체 dataset 다운로드 없이 작성하는 GenImage extension note.
- 더 많은 artifact가 생긴 뒤 다듬을 Streamlit model selector.
- 선택 사항인 pyproject.toml.

## 위험 및 차단 요인

다음 문제가 발생하면 run log나 notepad에 기록하세요:

- CIFAKE data source가 없거나 예상한 로컬 폴더 구조와 다릅니다.
- CLIP model download가 실패합니다.
- remote credential이 없습니다.
- CUDA package version이 remote driver 또는 runtime과 맞지 않습니다.
- stale cache reuse가 감지되었거나 의심됩니다.
- label inversion이 의심됩니다.
- train/test leakage가 감지되었거나 의심됩니다.
- robustness runtime이 사용 가능한 machine에 비해 너무 커집니다.

## 원격 RTX5090 CUDA 실행 가이드

원격 실행 정보:

    host: tml-server
    remote path: ~/codes/ml-project
    conda env: ml_termproj

로컬 data, outputs, virtualenv, cache artifact를 제외하고 source code를 동기화합니다:

    rsync -av --exclude data/ --exclude outputs/ --exclude .venv/ --exclude __pycache__/ --exclude .pytest_cache/ --exclude .mypy_cache/ --exclude .ruff_cache/ ./ tml-server:~/codes/ml-project/

원격 shell과 environment를 시작합니다:

    ssh tml-server
    cd ~/codes/ml-project
    conda activate ml_termproj

실행 환경 증거를 수집합니다:

    python -c "import importlib.metadata, torch; print('torch.__version__=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('open_clip_torch=', importlib.metadata.version('open_clip_torch'))"

CIFAKE data가 원격 machine에 준비된 뒤 아주 작은 CUDA CLIP smoke를 실행합니다:

    python scripts/prepare_cifake_subset.py --data_root data/cifake --output_manifest outputs/manifests/remote_cifake_smoke.csv --num_real 2 --num_fake 2 --seed 42

    python scripts/extract_clip_features.py --manifest outputs/manifests/remote_cifake_smoke.csv --output_cache outputs/caches/remote_clip_smoke.pt --max_samples 4 --batch_size 2 --device cuda --smoke --write_blocker .sisyphus/evidence/remote_clip_smoke_blocker.txt

더 큰 원격 run은 먼저 sample count를 제한해서 시작합니다:

    python scripts/run_all_experiments.py --data_root data/cifake --output_root outputs/remote_full_run --include_clip --max_samples_per_class 1000

원격 CUDA 실행에서 수집해야 할 증거 항목:

- 정확한 command와 exit code가 포함된 command log.
- torch.__version__.
- CUDA availability와 device name.
- open_clip_torch version.
- 생성된 .pt file의 cache metadata, feature type, model ID, manifest hash, row count, device 포함.
- 각 experiment directory의 metrics와 artifact list.

이 repository는 현재 원격 CUDA smoke나 full run이 완료되었다고 주장하지 않습니다. 위 증거가 실제로 존재할 때만 그 주장을 추가하세요.
