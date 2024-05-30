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
01_Original 폴더에는 다운로드 받은 원 데이터에 대한 압축을 푼 파일들이 있어야 하고, 02_Formatting에는 01_01_DataPreprocessing.ipynb를 실행시키면 전처리 된 데이터가 저장됩니다.
자세한 내용은 01_01_DataPreprocessing.ipynb 코드를 참고해주세요.
## 2. Model
### CycleGAN
![image](https://github.com/cjw94103/PPG2ECG_based_Application_Research/assets/45551860/289631bf-813d-4a5d-b53f-dab76d8975de)

PPG2ECG는 CycleGAN의 1D 버전 구현을 사용합니다. CycleGAN은 image-to-image translation의 대표적 모델로 paired example 없이 $X$라는 domain으로부터 얻은 이미지를 target domain $Y$로 translation하는 방법입니다.
CycleGAN의 목표는 Adversarial Loss를 통해, $G(x)$로부터의 이미지 데이터의 분포와 $Y$로부터의 이미지 데이터의 분포를 구별할 수 없도록 forward mapping $G:X \to Y$을 학습하고 constraint를 위해 inverse mapping $F:Y \to X$를 학습합니다.
image translation을 위하여 inverse mapping $F(G(x))$가 $x$와 같아지도록 Cycle Consistency Loss를 사용합니다.
추가적으로 이미지 $x$와 생성된 이미지 $x'$, 이미지 $y$와 생성된 이미지 $y'$이 같아지도록 강제하는 Identity Loss를 사용합니다. 여기서는 domain $X$를 PPG로, $Y$를 ECG로 취급하여 CycleGAN을 학습합니다.

### Train
먼저 config 파일을 만들어야 합니다. make_config.ipynb 코드를 참고하여 config 파일을 만들어주세요. 코드의 config 폴더 안에 CycleGAN_PPG2ECG.json 파일을 참고해주시면 되겠습니다. train_dist.py는 pytorch의 'gloo' 기반의 DistributedDataParallel을 이용하여 구현되었습니다. Single, Multi-GPU에서 모두 구동되니 사용하시면 됩니다. 아래와 같은 명령어를 사용하면 학습할 수 있습니다.
```python
python train_dist.py --config_path /path/your/config_path
```
## 3. Inference
학습이 완료되면 inference.ipynb를 참고하여 학습 완료된 모델의 가중치를 로드하여 추론을 수행할 수 있습니다.
## 4. 학습 결과
### Quantitative Evaluation

### Qualitative Evaluation
