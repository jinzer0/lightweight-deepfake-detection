# **\[기계학습 텀프로젝트\]**

# **중간 보고서**

# AI-GEN Image Detector

# 3팀

# **텀 프로젝트 작품명**

AI-GEN Image Detector

# **팀 구성원 소개**

| 이름 | 학번 | 역할 | 세부 담당 | 비고 |
| ----- | ----- | ----- | ----- | ----- |
| 구교영 | 202111244 | 팀장 / PM | 일정 관리, 보고서 및 발표자료 통합, 전체 구조 설계 | 팀장 |
| 이지민 | 202211346 | 데이터 담당 | CIFAKE 및 GenImage 데이터셋 수집, real/fake subset 구성, train/validation/test split, 데이터 전처리 및 metadata CSV 생성  |  |
| 김진영 | 202211286 | 모델 담당 | end to end detector model 설계, baseline classifier 구현, fusion classifier 설계 및 학습 |  |
| 바상체렝 | 202413539  | 주파수 / 평가 담당 | DCT feature extractor 구현, spectrum 시각화, 성능 평가, robustness test |  |

# **작품소개**

최근 이미지 생성 모델은 매우 빠르게 발전하고 있다. Stable Diffusion, Midjourney, DALL-E 계열 모델처럼 사람 눈으로 실제 사진과 구분하기 어려운 이미지를 생성하는 도구가 널리 사용되면서, 이미지가 포함된 뉴스, SNS 게시물, 광고, 과제, 리뷰 등의 신뢰성을 판단하기 어려워지고 있다. 업로드된 프로젝트 제안서도 이 문제를 핵심 동기로 제시하며, 단순히 “보기에는 자연스럽다”는 인간의 주관적 판단만으로 이미지 진위를 가리기 어렵다는 점을 강조한다.   
본 프로젝트의 목표는 사용자가 JPG 또는 PNG 이미지를 업로드하면, 해당 이미지가 실제 촬영 이미지인지 AI 생성 이미지인지 확률값으로 출력하는 탐지 시스템을 개발하는 것이다. 최종 결과는 AI-generated probability, 최종 판정, 신뢰도 등급, 주파수 스펙트럼 시각화, branch별 score를 함께 제공한다. 

# **핵심 문제**

## **1\. 학습 데이터에 없는 새로운 생성 모델에 대한 일반화 문제**

AI 생성 이미지 탐지 모델은 학습에 사용한 생성 모델의 이미지에는 높은 정확도를 보일 수 있지만, 학습에 포함되지 않은 새로운 생성 모델의 이미지에서는 성능이 떨어질 수 있다. 예를 들어 Stable Diffusion 이미지로 학습한 분류기가 DALL-E, Flux, Midjourney 계열 이미지에는 잘 동작하지 않을 수 있다.

| 해결방안 | 내용 |
| :---: | ----- |
| 생성 모델 단위 train/test split | 단순 random split이 아니라 생성 모델 단위로 학습용과 테스트용을 분리한다. |
| CLIP feature 사용 | ImageNet 또는 특정 생성 모델에만 과도하게 맞는 CNN feature보다 대규모 사전학습된 CLIP feature를 활용한다. |
| Cross-generator test 수행 | 학습에 사용하지 않은 generator를 test set으로 두어 일반화 성능을 확인한다. |
| Per-generator accuracy 측정 | 생성 모델별 성능을 따로 계산해 어떤 generator에 약한지 분석한다. |

## **2\. 주파수 도메인 아티팩트 활용 문제**

핵심 문제 정의

고도화된 AI 생성 이미지는 RGB 공간에서 사람이 보기에는 자연스럽게 보일 수 있다. 따라서 단순히 이미지의 픽셀이나 의미적 특징만으로는 real/fake를 구분하기 어렵다. 반면 생성 모델은 업샘플링, 디코더 구조, convolution 연산 과정에서 자연 이미지와 다른 주파수 패턴을 남길 수 있다.

## **2.1 후처리로 인한 탐지 성능 저하 문제**

핵심 문제 정의

실제 환경의 이미지는 원본 그대로 사용되지 않는다. SNS 업로드 과정에서 JPEG 압축, 리사이즈, 크롭, 블러, 스크린샷 변환 등이 발생할 수 있다. 이러한 후처리는 주파수 도메인 아티팩트를 약화시키거나 변형시킬 수 있다.

| 해결방안 | 내용 |
| :---: | ----- |
| JPEG compression test | JPEG quality 95, 75, 50 조건에서 성능 변화를 측정한다. |
| Resize test | 이미지를 축소 후 재확대하여 탐지 성능 변화를 확인한다. |
| Blur test | Gaussian blur 적용 후 성능 변화를 확인한다. |
| Fusion model 사용 | 주파수 feature만 의존하지 않고 CLIP feature와 결합한다. |
| Confidence level 출력 | 확률이 애매한 경우 낮은 신뢰도로 표시하여 과도한 확신을 피한다. |

## **3\. 제한된 컴퓨팅 자원 문제**

핵심 문제 정의

대규모 데이터셋 전체를 사용해 큰 모델을 scratch부터 학습하거나 전체 fine-tuning하는 것은 학부 프로젝트 환경에서 부담이 크다. GPU 시간, 저장공간, 학습 안정성 문제가 발생할 수 있다.

| 해결방안 | 내용 |
| :---: | ----- |
| Frozen CLIP 사용 | CLIP image encoder는 학습하지 않고 feature extractor로만 사용한다. |
| 작은 classifier 사용 | 작은 MLP를 선택하고 필요 시 Logistic Regression, SVM 중 현실적인 모델을 병용한다. |
| Feature cache 저장 | CLIP feature와 frequency feature를 .npy 또는 .pt 파일로 저장해 반복 학습 시간을 줄인다. |
| GenImage subset 사용 | 전체 데이터셋 대신 generator별 균형 sampling한 subset으로 먼저 실험한다. |
| baseline 단계적 구현 | CLIP-only → Frequency-only → Fusion 순서로 구현한다. |

## **4\. Black-box 판정 문제**

사용자에게 단순히 “AI-generated: 0.82”만 보여주면 왜 그런 판단이 나왔는지 이해하기 어렵다. 탐지 모델이 실제로 어떤 근거를 사용했는지도 설명하기 어렵다.

| 해결방안 | 내용 |
| :---: | ----- |
| Branch score 제공 | CLIP branch score와 frequency branch score를 각각 출력한다. |
| DCT/FFT spectrum 제공 | 입력 이미지의 주파수 스펙트럼을 시각화한다. |
| Radial spectrum graph 제공 | 주파수 대역별 에너지 변화를 그래프로 표시한다. |
| 모델 비교 결과 제공 | CLIP-only, Frequency-only, Fusion 모델 결과를 함께 비교한다. |
| Confidence level 제공 | 높음 / 중간 / 낮음 형태로 신뢰도를 구분한다. |

# **SW 설계**

- ## **기능 요구사항**

  - 이미지 업로드 기능: 사용자는 로컬 PC에서 검사할 이미지 파일(JPG, PNG)을 단건 또는 다건으로 업로드할 수 있어야 한다.  
  - 실시간 AI 판별 기능: 시스템은 업로드된 이미지를 분석하여 'AI 생성 이미지일 확률(%)'과 '최종 진위 판정 결과(Real/AI)'를 화면에 출력해야 한다.  
  - 신뢰도 등급 제공 기능: 시스템은 확률값에 따라 판정에 대한 신뢰도 등급(높음/중간/낮음)을 계산하여 사용자에게 제공해야 한다.  
  - 판정 근거 시각화(XAI) 기능: 시스템은 주파수 도메인의 방사형 스펙트럼(Radial Spectrum)을 렌더링하여 출력해야 한다. 

- ## **비기능 요구사항**

  - 정확도: Held-out 테스트셋 기준 75% 이상의 정확도 또는 ROC-AUC 0.70 이상의 정확도, 미학습 생성 모델 테스트에서는 70% 이상의 정확도 또는 ROC-AUC 0.65 이상의 정확도를 보장해야 함   
    

- ## **데이터 파이프라인**

  - 본 프로젝트는 모델 학습과 평가를 위해 real/fake 이미지 데이터를 수집하고, 라벨링 및 split 정보를 metadata CSV로 관리한다. 현재는 CIFAKE 데이터셋을 이용해 real 1000장, fake 1000장으로 구성된 1차 subset을 만들었으며, real=0, fake=1로 라벨링하였다. train/validation/test는 70:15:15 비율로 분리하였고, dataset.csv에 filepath, label, class\_name, generator, split 정보를 저장하였다. 이후 GenImage subset으로 확장하여 generator 단위 split을 적용할 계획이다.   
    

- ## **사용 사례**

  - 사용 사례 다이어그램  
    ![][image1]

- ## **SW Architecture**

  - 배치도(Deployment diagram)  
  - ![][image2] 

    

- 논리적인 SW 구조도(High-Level Logical SW Architecture)  
- ![][image3] 

- ## **상세 SW(SW 서브시스템 또는 SW 모듈별)**

  - 클래스 다이어그램  
    ![][image4]

# **최종 산출물**

- ## **최종 산출물 구성도**

  ![][image5]![][image6]  
    
    
  ![][image7]

- ## **Application Layer**

  - 활용한 SW:

| 활용 SW | 활용 목적 |
| :---: | ----- |
| GenImage | AI 생성 이미지 탐지 학습 및 평가 |
| CIFAKE | AI 생성 이미지 탐지 학습 및 평가 |
| COCO | 실사 이미지 탐지 학습 및 평가 |
| CNNDetection | CNN 기반 baseline 참고 |
| ClipBased-Synthetic Image Detection | CLIP 기반 synthetic image detection 참고 |
| Synthbuster | Fourier artifact 기반 탐지 참고 |
| DIRE | diffusion image detection 확장 baseline 참고 |
| Google Colab | GPU 기반 학습 및 실험 |
| WandB (Optional) | 실험 로그 관리, 선택 사용 |
| ConvNeXt, Swin-T | CNN Feature Extractor |

- ## **Platform Layer**

  - Environment  
    Python 3.10.18  
    Pytorch 2.7.1+cu118(사용 GPU 따라 버전 상이)

- ## **Infra Layer**

  - Ubuntu 24.04 LTS  
  - Machine: Google Colab / Intel i9-13900K, RTX 4090, VRAM 24GB, DRAM 64GB

- ## **최종 산출물 구성도에 대한 설명**

  최종 산출물은 사용자가 웹 화면에서 이미지를 업로드하거나 CLI 명령어를 통해 AI 생성 이미지 확률과 판단 근거를 제공하는 탐지 시스템이다.  
  Application Layer는 실제 프로젝트에서 직접 구현하는 코드 모듈로 구성된다. app.py는 사용자 인터페이스를 담당하고, detector\_service.py는 전처리, feature extraction, classifier inference, visualization을 하나의 흐름으로 연결한다. clip\_features.py는 CLIP image encoder를 사용해 공간 도메인 feature를 추출하고, frequency\_features.py는 FFT/DCT 기반 주파수 feature를 추출한다. 두 feature는 fusion\_classifier.py에서 결합되어 최종 AI 생성 확률을 출력한다.

# **Risk Analysis \+ Risk Reduction Plan**

- ## **예상이 되는 문제들, 극복해야할 어려움들**


| 예상 위험 | 설명 |
| :---: | ----- |
| 특정 생성 모델 과적합 | 특정 generator 이미지에만 잘 동작할 수 있다. |
| 새로운 생성 모델에 대한 성능 저하 | 학습에 없는 DALL-E, Flux, Midjourney 계열 이미지에서 성능이 낮을 수 있다. |
| 주파수 특징 약화 | JPEG 압축, resize, blur 후처리로 frequency artifact가 사라질 수 있다. |
| GPU 부족 | 큰 CNN/ViT 모델 fine-tuning은 학습 시간이 길고 실패 가능성이 높다. |
| 데이터 용량 문제 | GenImage, CIFAKE 전체 데이터셋은 크기가 커서 저장 및 처리 부담이 있다. |
| False positive 문제 | 실제 사진을 AI 생성 이미지로 잘못 판정할 수 있다. |
| Black-box 문제 | 단순 확률 출력만으로는 판단 근거를 설명하기 어렵다. |
| 데모 배포 문제 | 서버 환경, 패키지 버전, GPU 사용 가능 여부에 따라 데모가 불안정할 수 있다. |
| 범용성 과장 문제 | 모든 AI 생성 이미지를 완벽히 잡는 것처럼 오해될 수 있다. |
| 데이터 불균형 문제 | generator별 이미지 수가 다르면 특정 class에 편향될 수 있다. |


- ## **해결 계획**


| 문제 | 해결 계획 |
| :---: | ----- |
| 특정 생성 모델 과적합 | 생성 모델 단위 train/test split을 적용한다. |
| 새로운 생성 모델 성능 저하 | Cross-generator test와 per-generator accuracy를 반드시 측정한다. |
| 주파수 특징 약화 | JPEG, resize, blur augmentation 및 robustness test를 수행한다. |
| GPU 부족 | Frozen CLIP feature와 Logistic Regression 또는 작은 MLP를 사용한다. |
| 데이터 용량 문제 | GenImage 전체가 아니라 generator별 subset으로 먼저 실험한다. |
| False positive 문제 | validation set 기준 threshold를 조정하고 confidence level을 제공한다. |
| Black-box 문제 | DCT spectrum, radial spectrum, branch score를 함께 제공한다. |
| 데모 배포 문제 | local demo, Colab demo, Streamlit/Gradio demo를 모두 준비한다. |
| 범용성 과장 문제 | “제한된 데이터셋 기준 성능”임을 보고서와 발표에서 명시한다. |
| 데이터 불균형 문제 | real/fake class와 generator별 sample 수를 균형 있게 구성한다. |


# **Success Criteria**

1. Held-out 테스트셋 기준 75% 이상의 정확도 또는 ROC-AUC 0.70 이상의 정확도 \- 정상적인 모델 학습이 되었는가?

2. CIFAKE, GenImage 데이터셋이 아닌 다른 미학습 생성 모델 테스트에서는 70% 이상의 정확도 또는 ROC-AUC 0.65 이상의 정확도 \- 학습되지 않은 데이터셋에서도 최소한의 성능이 보장되는가?

# **상세 역할 분담**

| 구교영 | 프로젝트 일정 관리, 논리적/물리적 아키텍처 설계, Fusion 모델 전략 수립, 최종 자료 검토 |
| :---: | :---- |
| 김진영 | end to end Deepfake Image Detection Pipeline 설계, CNN 및 FFT Classifier, fusion classifier  neural network간 연결, 테스트 및 학습 |
| 이지민 | CIFAKE 및 GenImage 데이터셋 수집, real/fake subset 구성, train/validation/test split, dataset.csv 생성, 데이터 전처리 파이프라인 구축  |
| 바상체렝 | DCT 기반 주파수 feature extractor를 구현하고, 스펙트럼 시각화로 판단 근거를 제공하며, 모델 성능 평가와 robustness test를 담당한다. |
| 공통 | 각 담당 파트 PPT 제작, 보고서 작성 |

# **스케줄 Update**

| 9주차 | 제안서 발표, 관련 논문 조사, 데이터셋 후보 정리 |
| :---- | :---- |
| 10주차 | GenImage 데이터셋 구성 및 구조 파악 |
| 11주차 | 이미지 전처리, train/validation/test split 설계 |
| 12주차 | CLIP 기반 feature extractor 구현, CLIP-only baseline 모델 학습 |
| 13주차 | 중간 발표, DCT 기반 frequency feature extractor 구현, Frequency-only baseline 실험 |
| 14주차 | CLIP feature와 DCT feature를 결합한 fusion feature 구성, MLP classifier 학습 |
| 15주차 | 성능 평가 및 분석: Accuracy, F1-score, ROC-AUC, confusion matrix, robustness test |
| 16주차 | demo 완성, 결과 시각화 정리, 최종 보고서 및 발표자료 작성, 최종 발표 |

# **참고자료 리스트**

GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image(2023.07)  
[https://arxiv.org/abs/2306.08571](https://arxiv.org/abs/2306.08571)

Raising the Bar of AI-generated Image Detection with CLIP(2024.04)  
[https://arxiv.org/abs/2312.00195](https://arxiv.org/abs/2312.00195)

ClipBased-SyntheticImageDetection(2024.11)  
[https://github.com/grip-unina/ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)

CNN-generated images are surprisingly easy to spot… for now(2019.12)  
[https://arxiv.org/abs/1912.11035](https://arxiv.org/abs/1912.11035)

CNN Detection(2024.07)  
[https://github.com/peterwang512/CNNDetection](https://github.com/peterwang512/CNNDetection)

Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions(2020.03)  
[https://arxiv.org/abs/2003.01826](https://arxiv.org/abs/2003.01826)

Synthbuster: Towards Detection of Diffusion Model Generated Images(2025.07)  
[https://github.com/qbammey/synthbuster](https://github.com/qbammey/synthbuster)

DIRE for Diffusion-Generated Image Detection(2023.03)  
[https://arxiv.org/abs/2303.09295](https://arxiv.org/abs/2303.09295)

GenImage-Dataset/GenImage(2024.04)  
[https://github.com/GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage)

Bird, J. J. CIFAKE: Real and AI-Generated Synthetic Images. Kaggle Dataset. (2023.03)  
[https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) 

DIRE(2024.07)  
[https://github.com/ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE)


