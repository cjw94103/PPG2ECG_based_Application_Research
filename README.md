# Introduction
<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*gSvhoJry6FKQxbXIzbPlzA.png" width="50%" height="50%" align="center"/>
본 연구는 스마트워치, 펄스옥시미터에서 나오는 PPG 신호를 ECG로 변환하고 데이터의 모달리티(Modality)를 확장하여 3개의 태스크 (6종 부정맥 분류, 스트레스 여부 분류, HR 추정)의 모델 성능 개선을 실험적 검증하고 
이용 가능성에 대한 개념적 가이드라인을 제시하기 위하여 수행된 탐색적 연구입니다.

PPG는 기타 생체 신호에 비해 측정이 편리하고 접근성이 좋은 장점을 가지고 있어 적절한 신호 처리 알고리즘으로 전처리 파이프라인을 잘 구성하면 사용자에게 편리하면서 건강 관리에 대한 양질의 서비스를 제공할 수 있습니다.
각 태스크의 개요는 아래와 같습니다.

## PPG2ECG
![image](https://github.com/cjw94103/PPG2ECG_based_Application_Research/assets/45551860/c924ed8f-3c25-457b-b563-ef5ae8eeaf02)

스마트워치, 펄스옥시미터 등의 디바이스에서 추출된 PPG 신호를 GAN 기반의 모델을 통해 Synthetic ECG (유사 II 리드 심전도)로 변환하는 모듈입니다. CycleGAN의 1D 버전을 사용하였으며 PPG2ECG에서 생성된 Synthetic ECG는 다른 PPG 기반의 Task에서 PPG와 함께 사용되어 성능 향상을 위한 프롬프트의 역할을 수행합니다.

## Arryhthmia Classification
![image](https://github.com/cjw94103/PPG2ECG_based_Application_Research/assets/45551860/6fe6970a-7eb2-4207-b73d-4593ac7bbc95)

PPG와 PPG2ECG 모델에서 생성된 Synthetic ECG를 함께 입력하여 6종의 부정맥 (Sinus Rhythm, Premature Ventricular Contraction, Premature Atrial Contraction, Ventricular Tachycardia, SupraVentricular Tachycardia, Atrial Fibrillation)을 분류하는 모델을 학습합니다. 분류 모델은 VGG16의 Variant를 1D 버전으로 구현하여 사용되며 multi-class classification의 Formulation으로 학습 수행합니다.

## HR Estimation
![image](https://github.com/cjw94103/PPG2ECG_based_Application_Research/assets/45551860/92ad5f7c-51cc-43d6-8c67-57d780ccfde3)

PPG2ECG 모델에서 생성된 Synthetic ECG를 기반으로 Peak Detection을 통한 HR을 추정 알고리즘을 개발합니다. 원 데이터의 ECG 신호에서 측정한 HR을 Ground-Truth로 사용하여 PPG와 Synthetic ECG에서 생성된 각 HR의 MAE를 통해 오차를 확인합니다.

## Stress Classification
![image](https://github.com/cjw94103/PPG2ECG_based_Application_Research/assets/45551860/659543af-1540-4d0c-a906-8754d174d484)

PPG와 PPG2ECG 모델에서 생성된 Synthetic ECG를 함께 입력하여 스트레스 여부에 대한 이진 분류 (정상 상태, 스트레스 상태)을 분류하는 인공지능 모델을 학습합니다. 분류 모델은 VGG16의 Variant를 1D 버전으로 구현하여 사용되며 multi-class classification의 Formulation으로 학습 수행합니다.
