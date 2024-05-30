# 1. Introduction
<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*gSvhoJry6FKQxbXIzbPlzA.png" width="50%" height="50%" align="center"/>
본 연구는 스마트워치, 펄스옥시미터에서 나오는 PPG 신호를 ECG로 변환하고 데이터의 모달리티(Modality)를 확장하여 3개의 태스크 (6종 부정맥 분류, 스트레스 여부 분류, HR 추정)의 모델 성능 개선을 실험적 검증하고 
이용 가능성에 대한 개념적 가이드라인을 제시하기 위하여 수행된 탐색적 연구입니다.

PPG는 기타 생체 신호에 비해 측정이 편리하고 접근성이 좋은 장점을 가지고 있어 적절한 신호 처리 알고리즘으로 전처리 파이프라인을 잘 구성하면 사용자에게 편리하면서 건강 관리에 대한 양질의 서비스를 제공할 수 있습니다.
