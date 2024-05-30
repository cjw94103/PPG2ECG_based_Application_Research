## 1. Dataset Preparation
"Multiclass Arrhythmia Detection and Classification from Photoplethysmography Signals Using a Deep Convolutional Neural Network"에서 제공한 데이터셋을 다운로드 합니다. 228명의 데이터 중 91명분의 validation, test set만 공개되어 있어 91명분의 데이터에서 train, validation, test를 6:2:2로 split합니다. https://github.com/zdzdliu/PPGArrhythmiaDetection 링크에서 데이터를 다운받아 주세요.

데이터셋을 다운로드 받은 후 압축을 풀고 아래와 같은 폴더 구조로 설정해주세요.
```python
00_Data
├── 02_Arrhythmia_Classification
│   ├── 01_Original
│   ├── 02_Formatting

```
01_Original 폴더에는 다운로드 받은 원 데이터에 대한 압축을 푼 파일들이 있어야 하고, 02_Formatting에는 01_01_DataPreprocessing.ipynb를 실행시키면 전처리 된 데이터가 저장됩니다.
자세한 내용은 01_01_DataPreprocessing.ipynb 코드를 참고해주세요.
## 2. Model
### VGG16

<img src="https://velog.velcdn.com/images%2Fchoonsik_mom%2Fpost%2F2a8c2e30-9f96-4dcc-97e5-c7e1289af137%2Fimage.png"  width="75%" height="75%"/>

Arrhythmia_Classification는 VGG16의 1D 버전 구현을 사용합니다. VGG16의 원래 구현은 Convolution feature를 1D로 flatten하고 FC layer로 전달하지만 연산 효율을 위하여 Global Average Pooling 후 Classifier로 넘기는 구조로 구현합니다.

### Train
먼저 config 파일을 만들어야 합니다. make_config.ipynb 코드를 참고하여 config 파일을 만들어주세요. 코드의 config 폴더 안에 VGG16_OnlyPPG.json, VGG16_PPGECG.json 파일을 참고해주시면 되겠습니다. VGG16_OnlyPPG.json은 학습 데이터로 PPG만을 사용하는 경우입니다. VGG16_PPGECG.json은 PPG를 입력으로 받아 PPG2ECG GAN에서 생성된 Synthetic ECG를 Channel axis로 Concatenate하여 2채널로 학습시킨 경우입니다. train_dist_OnlyPPG.py, train_dist_PPGECG.py는 pytorch의 'gloo' 기반의 DistributedDataParallel을 이용하여 구현되었습니다. Single, Multi-GPU에서 모두 구동되니 사용하시면 됩니다. 아래와 같은 명령어를 사용하면 학습할 수 있습니다.
- 학습 데이터로 PPG만을 사용하는 경우
```python
python train_dist_OnlyPPG.py --config_path /path/your/config_path
```

- PPG와 Synthetic ECG를 모두 사용하는 경우
```python
python train_dist_PPGECG.py --config_path /path/your/config_path
```
## 3. 학습 결과
학습 결과는 OnlyPPG, PPGECG의 경우를 모두 학습하여 성능 비교를 통해 Synthetic ECG를 학습 데이터에 포함시키는 것이 모델의 성능 개선에 도움이 된다라는 것을 검증하기 위하여 Class (부정맥)별 Precision, Recall, F1-score를 산출합니다.
### OnlyPPG
|클래스|Precision|Recall|F1-Score|
|------|---|---|---|
|Sinus Rhythm|0.9698|0.9741|0.9719|
|PVC|0.6440|0.6153|0.6293|
|PAC|0.4631|0.3778|0.4161|
|VT|0.6409|0.6240|0.6323|
|SVT|0.7428|0.8775|0.8046|
|AFIB|0.8634|0.8631|0.8046|
|Average|0.7207|0.8631|0.7196|
### PPGECG
|클래스|Precision|Recall|F1-Score|
|------|---|---|---|
|Sinus Rhythm|0.9743|0.9819|0.9781|
|PVC|0.7094|0.7879|0.7466|
|PAC|0.5117|0.4132|0.4572|
|VT|0.8007|0.4898|0.6078|
|SVT|0.6458|0.9420|0.7663|
|AFIB|0.8660|0.7980|0.8306|
|Average|0.7513|0.7355|0.7311|

PPGECG의 경우 OnlyPPG보다 macro F1 기준 약 1.15%의 성능 향상을 보인다.
