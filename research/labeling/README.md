# research/labeling — 학습 데이터 구축 당시 원본 코드

2025년 프로젝트 당시 본인이 작성해 실행한 학습 데이터 구축 코드입니다. **당시 원본**이며, 기록 보존용이라 이 레포에서 실행하지 않습니다. 노트북은 교정 문장이 들어 있을 수 있는 출력을 지우고 코드만 남겼습니다. 입력 데이터(고객 로그)는 공개하지 않습니다.

| 순서 | 파일 | 원본 | 하는 일 |
|---|---|---|---|
| 1 | `01_service_log_to_csv.ipynb` | [`experiments/2_난이도분류기/01_교정기록_추출_라벨링입력생성`](../experiments/2_난이도분류기/01_교정기록_추출_라벨링입력생성.ipynb) | 서비스 로그 `문장교정기록.json`(4,115행)에서 원문·교정문·사용자가 고른 강도·분야를 뽑아 정제(1,064행) → `munch_notlabeled.csv` |
| 2 | `02_label_service_log_v4.py` | 제출 라벨링 스크립트 V4 (문장교정기록) | 서비스 로그를 Gemini로 1~5점 채점. **강도·분야를 함께 제시** → v1 |
| 3 | `03_label_correction_pairs_v3.py` | 제출 라벨링 스크립트 V3 | 스타트업이 준 교정 문장 쌍(`munch_data/*.csv`, 원문·교정문만)을 Gemini로 채점. **강도 없이** 원문·교정문만 제시 → v2 raw |
| 4 | `04_reverse_engineer_intensity.ipynb` | [`experiments/2_난이도분류기/10_말뭉치_강도역산_SequenceMatcher`](../experiments/2_난이도분류기/10_말뭉치_강도역산_SequenceMatcher.ipynb) | 라벨링된 v2에 SequenceMatcher 비율로 강도를 역산(>0.9 WEAK, >0.6 MODERATE, 나머지 STRONG), 분야 NONE → v2 augmented |
| 5 | `05_standardize_data.py` | `standardize_data.py` | v1(결측 10행 제거, 1,053) + v2 augmented(6,648) → `dataset_master.csv` 7,701행 |

API 키는 환경변수(`GEMINI_API_KEY`)로만 읽습니다. 과정과 한계는 [docs/labeling.md](../../docs/labeling.md)에 있습니다.
