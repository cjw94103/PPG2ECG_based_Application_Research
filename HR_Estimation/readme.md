## 3. MAE 결과
원 데이터의 ECG 신호에서 측정한 HR을 Ground-Truth로 사용하여 PPG와 Synthetic ECG에서 생성된 각 HR을 Peak Detection 알고리즘을 통해 계산하고 PPG, SYNECG의 경우로 나누어 Mean Absolute Error (MAE)를 계산합니다. 모든 데이터셋에서 SYNECG를 통한 HR 계산이 MAE 오차가 더 적습니다.
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
