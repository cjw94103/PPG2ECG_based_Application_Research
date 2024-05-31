## 1. Dataset Preparation
데이터셋은 Synchronized PPG, ECG가 있는 BIDMC, CapnoBase, DaLia, WESAD 데이터셋을 사용합니다. 다운로드 링크는 아래와 같습니다.

- BIDMC : https://physionet.org/content/bidmc/1.0.0/
- CapnoBase : https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT
- DaLia : [https://outbox.eait.uq.edu.au/uqdliu3/uqvitalsignsdataset/index.html](https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP)
- WESAD : https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx
  
데이터셋을 다운로드 받은 후 압축을 풀고 아래와 같은 폴더 구조로 설정해주세요.
```python
00_Data
├── 01_PPG2ECG
│   ├── 01_Original
│   ├───── 01_BIDMC
│   ├───── 02_CapnoBase
│   ├───── 03_DaLia
│   └───── 04_WESAD
│   ├── 02_Formatting
│   ├───── 01_BIDMC
│   ├───── 02_CapnoBase
│   ├───── 03_DaLia
│   └───── 04_WESAD
```

## 2. HR Estimation 방법
Synchronized PPG, ECG가 있는 BIDMC, CapnoBase, DaLia, WESAD 데이터셋에서 원 데이터의 ECG 신호에서 측정한 HR을 Ground-Truth로 사용합니다. PPG, ECG를 이용한 HR 계산은 neurokit library를 사용하며 PPG를 이용한 HR 계산에는 "elgandi", ECG를 이용한 HR 계산에는 "nabian2018" method를 사용합니다. Ground-Truth HR과 PPG, Synthetic ECG에서 계산한 HR의 오차를 구하여 성능을 비교합니다. 코드는 직관적으로 실행하여 결과를 확인할 수 있도록 모두 .ipynb 형식으로 제공됩니다.

## 3. MAE 결과
원 데이터의 ECG 신호에서 측정한 HR을 Ground-Truth로 사용하여 PPG와 Synthetic ECG에서 생성된 각 HR을 Peak Detection 알고리즘을 통해 계산하고 PPG, SYNECG의 경우로 나누어 Mean Absolute Error (MAE)를 계산합니다.

|데이터셋|신호|MAE|
|------|---|---|
|BIDMC|PPG|2.6468|
||SYNECG|2.5718|
|CapnoBase|PPG|0.6754|
||SYNECG|0.7656|
|DaLia|PPG|14.3673|
||SYNECG|12.7747|
|WESAD|PPG|11.8635|
||SYNECG|9.4286|

모든 데이터셋에서 SYNECG를 통한 HR 계산이 MAE 오차가 더 적습니다.
