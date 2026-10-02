# Provenance

어떤 결과가 **언제, 어떤 코드로, 어떤 입력에서** 나왔는지 기록합니다. 원본 자료(고객 로그·라벨·예측 CSV·가중치)는 비공개이며 이 레포에 포함하지 않습니다.

## 1. 출처 계보

| 위치 (로컬, 비공개) | 성격 | 이 레포에서의 취급 |
|---|---|---|
| `bookend-yeardream-proj` | 프로젝트 당시 원본 (노트북, `apps/eval`, `libs/core`, `data/`) | 1차 근거. 읽기 전용 |
| `backup-bookend` | 교육 제출용 소스·모델 백업 | 제출 V8 노트북·가중치의 근거 |
| `smart-router-minimal` | 프로젝트 이후 사용자 백엔드 학습 시도 | 참고만 |
| `SmartRouter_Pipeline` | 이후 LLM이 만든 후속본 | 당시 구현 근거 아님 |
| 이 레포 | LLM 사후 재구성본 → 2026-09 AI 보조로 재정리한 최종본 | 유일한 최종 진입점 |

`research/`는 모두 당시 원본 코드입니다(출력 제거). `research/labeling/`은 학습 데이터 구축 코드, `research/experiments/`는 분석·모델링 실험 노트북 21개입니다([색인](../research/experiments/README.md)). 이전에 `research/`에 있던 노트북 8개는 당시 Gemini 2.5 Pro가 실험 노트북을 바탕으로 재구성한 사후본이라 2026-10-02에 삭제했습니다(git 이력에 남아 있음).

## 2. 결과 유형 구분

| 유형 | 의미 | 해당 결과 |
|---|---|---|
| 저장 결과 재계산 | 당시 저장된 CSV를 이 레포 코드로 다시 집계 | `evaluate`, `judge-report` |
| 실제 체크포인트 추론 | 로컬 가중치로 이번에 다시 추론 | `reinfer` 1156/1156 결정 일치 |
| 원 로그 재병합 | raw JSON을 명시적 키로 다시 병합 | `audit-data` |
| 새 학습 | 없음 | — |
| 유료 API 재호출 | 없음. 생성·Judge는 당시 결과만 사용 | — |

## 3. 핵심 파일 SHA-256

| 파일 | SHA-256 |
|---|---|
| `models/smart_router_v3/model.safetensors` (= 제출 checkpoint-205 = 원본 = 제출 백업) | `a641eddbe545975fd5c31f97d4d79f1ee796f09ff79683f3f1303598e284fb04` |
| `data/processed/full_merged_v3.csv` | `06de38eb5dad4de1d8553329cc44a4f5a1ad5609df6339213f06d25b13c2c01b` |
| `data/5. test_results_routed.csv` | `5a7d7cca08366b39e305342299290e58337ef6949837d86fe9509d5091fedfc2` |
| `data/8. evaluation_results.csv` | `7d05f2aae8f8e6b669a6a074c5237ae0d96b350b9a0fb77a42064bce7a4de1d0` |
| 원본 `experiments/1_서비스로그_분석/02_로그3종_병합.ipynb` (출력 포함 원본) | `6d554047c564b90e776a269f8b8d6b173b99572666a2f9281a95832c6ed27817` |
| 원본 `experiments/1_서비스로그_분석/03_이벤트_클렌징_v3생성.ipynb` (출력 포함 원본) | `ee903a2fade22a03d08198583924d7cf1c4dde5889db5d266c39d9bc74eadd6a` |
| 제출 V8 노트북 | `e052796d7da8f8b7b097d62d183b5e95337bcd0071f2ecfc29df23dbcc275a2a` |

## 4. 단계별 근거 연결

| 단계 | 당시 코드 (원본 경로) | 산출물 |
|---|---|---|
아래 코드 경로는 모두 이 레포의 `research/` 기준입니다(`apps/eval/`만 비공개 원본 레포).

| 단계 | 당시 코드 | 산출물 |
|---|---|---|
| 로그 병합 (팀원 주도) | `experiments/1_서비스로그_분석/02_로그3종_병합` → `03_이벤트_클렌징_v3생성` | `full_merged_v3.csv` (62097×86). 입력인 v2 파일이 없어 완전 재생성은 불가 |
| 이벤트 EDA·초기 ML | `experiments/1_서비스로그_분석/04`~`06` | RF/LightGBM/CatBoost, 중요도 (라우터 계보와 별개) |
| 서비스 로그 추출 | `labeling/01_service_log_to_csv` (= `experiments/2_난이도분류기/01`) (`문장교정기록.json` 4115행 → 1064행) | `munch_notlabeled.csv` |
| 난이도 라벨링 v1 | `labeling/02_label_service_log_v4.py` (강도·분야 제시) | `munch_labeled_FINAL_190139.csv` → `1. source_v1_legacy.csv` |
| 난이도 라벨링 v2 | `labeling/03_label_correction_pairs_v3.py` (`munch_data/*.csv`, 원문·교정문만 제시) | `munch_labeled_RESUMED_200125.csv` → `2. source_v2_raw.csv` |
| 강도·분야 역산 (v2, 라벨링 후) | `labeling/04_reverse_engineer_intensity` (= `experiments/2_난이도분류기/10`) | `3. source_v2_augmented.csv` |
| 학습 데이터 | `labeling/05_standardize_data.py` | `4. dataset_master.csv` 7701행 |
| 분류기 학습·트래픽 시뮬레이션 | `experiments/2_난이도분류기/11_태그입력_RoBERTa_전체데이터_최종` (V8, 제출본과 데이터 경로·출력 문구만 다름) | checkpoint-205, `5. test_results_routed.csv` 1156행 |
| 교정 생성 | `apps/eval/src/smartrouter_eval/run_inference_azure.py` (gpt-5-nano / gpt-5-mini) | `7. inference_results.csv` |
| Judge 평가 | `apps/eval/src/smartrouter_eval/test_inference_azure.py` (o4-mini, 100건, seed 42) | `8. evaluation_results.csv`, `8. evaluation_report.txt` |

원본 `data/readme.md`는 v1/v2의 출처 설명이 서로 뒤바뀌어 있습니다. 코드의 입출력 파일명과 컬럼(v1은 `intensity`·`field` 포함, v2는 `original`·`corrected`만)을 기준으로 바로잡았습니다.

`experiments/3_라우팅평가/01_테스트셋_라우팅결과_저장`·`02_교정생성_mock_미실행` 노트북의 생성 함수는 `time.sleep`으로 흉내만 내는 mock입니다. 실제 생성은 `apps/eval` 스크립트로 수행했습니다(사용자 진술 및 스크립트·산출물 연결로 확인).

V8 노트북의 시뮬레이션 셀은 `./model/smart_router_v3_final_merged`를 로드합니다. 이 폴더는 현재 원본·백업 어디에도 없습니다. 다만 그 셀의 출력(Hard 650 / Easy 506 / FN 68)은 저장 CSV 및 현재 가중치의 전체 재추론 결과와 일치합니다.

## 5. 이 레포 재구성 이력 (2026-09, AI 보조)

- 변경 전 커밋: `7411311`. 변경 전 파일 해시는 세션 기록(`apply/sessions/smartrouter_rebuild/result.md`)에 있습니다.
- 2026-09-29: 공개 이력에 남아 있던 노트북 출력(교정 문장 포함 가능)을 모든 커밋에서 제거했습니다(`git filter-branch`, 노트북 외 변경 없음, 작성자·날짜 보존). 그 결과 `7411311`은 `25fbd79`로 바뀌었습니다. 재작성 전 전체 이력은 비공개 로컬 백업(git bundle)으로 보존했습니다.
- Bedrock/Azure 생성 서비스, 대시보드, Docker/Airflow/Terraform, ONNX 스크립트, 과거 차트 2장은 `legacy/`로 옮겼습니다. 기본 실행 경로에서 제외되었으며 삭제하지 않았습니다.
- `.gitignore`의 `models/` 규칙이 `src/smartrouter/models/classifier.py`까지 무시하던 문제를 `/models/`로 고쳤습니다.
