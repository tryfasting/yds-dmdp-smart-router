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

`research/` 8개 노트북은 사후 재구성본입니다. 원본의 같은 이름 노트북과 파일 바이트가 모두 다릅니다. 당시 실행 기록이 아니라 설명용 자료로 취급합니다.

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
| 원본 `01_데이터 병합.ipynb` | `6d554047c564b90e776a269f8b8d6b173b99572666a2f9281a95832c6ed27817` |
| 원본 `00_데이터 클렌징.ipynb` | `ee903a2fade22a03d08198583924d7cf1c4dde5889db5d266c39d9bc74eadd6a` |
| 제출 V8 노트북 | `e052796d7da8f8b7b097d62d183b5e95337bcd0071f2ecfc29df23dbcc275a2a` |

## 4. 단계별 근거 연결

| 단계 | 당시 코드 (원본 경로) | 산출물 |
|---|---|---|
| 로그 병합 (팀원 주도) | `notebooks/experiments/01_데이터 병합.ipynb` → `00_데이터 클렌징.ipynb` | `full_merged_v3.csv` (62097×86). 입력인 v2 파일이 없어 완전 재생성은 불가 |
| 이벤트 EDA·초기 ML | 제출 `02_데이터전처리`, `05_머신러닝모델링_및_검증` | RF/LightGBM/CatBoost, 중요도 (라우터 계보와 별개) |
| 난이도 라벨링 | 제출 `학습데이터 만들기_V3` (Gemini, score≥4 → Hard) | `1. source_v1_legacy.csv`, `2. source_v2_raw.csv` |
| 강도·분야 역산 | 메타데이터 역공학 노트북 (SequenceMatcher) | `3. source_v2_augmented.csv` |
| 학습 데이터 | 병합 | `4. dataset_master.csv` 7701행 |
| 분류기 학습 | 제출 V8 RoBERTa 노트북 | checkpoint-205 |
| 트래픽 시뮬레이션 | 원본 `06_모델링_지표llm-as-judge_유선종_V8_RoBERTa_with_AllData.ipynb` | `5. test_results_routed.csv` 1156행 |
| 교정 생성 | `apps/eval/src/smartrouter_eval/run_inference_azure.py` (gpt-5-nano / gpt-5-mini) | `7. inference_results.csv` |
| Judge 평가 | `apps/eval/src/smartrouter_eval/test_inference_azure.py` (o4-mini, 100건, seed 42) | `8. evaluation_results.csv`, `8. evaluation_report.txt` |

원본 `07_모델_평가`·`08_분류된_문장으로_교정 실행` 노트북의 생성 함수는 `time.sleep`으로 흉내만 내는 mock입니다. 실제 생성은 `apps/eval` 스크립트로 수행했습니다(사용자 진술 및 스크립트·산출물 연결로 확인).

V8 노트북의 시뮬레이션 셀은 `./model/smart_router_v3_final_merged`를 로드합니다. 이 폴더는 현재 원본·백업 어디에도 없습니다. 다만 그 셀의 출력(Hard 650 / Easy 506 / FN 68)은 저장 CSV 및 현재 가중치의 전체 재추론 결과와 일치합니다.

## 5. 이 레포 재구성 이력 (2026-09, AI 보조)

- 변경 전 커밋: `7411311`. 변경 전 파일 해시는 세션 기록(`apply/sessions/smartrouter_rebuild/result.md`)에 있습니다.
- 2026-09-29: 공개 이력에 남아 있던 노트북 출력(교정 문장 포함 가능)을 모든 커밋에서 제거했습니다(`git filter-branch`, 노트북 외 변경 없음, 작성자·날짜 보존). 그 결과 `7411311`은 `25fbd79`로 바뀌었습니다. 재작성 전 전체 이력은 비공개 로컬 백업(git bundle)으로 보존했습니다.
- Bedrock/Azure 생성 서비스, 대시보드, Docker/Airflow/Terraform, ONNX 스크립트, 과거 차트 2장은 `legacy/`로 옮겼습니다. 기본 실행 경로에서 제외되었으며 삭제하지 않았습니다.
- `.gitignore`의 `models/` 규칙이 `src/smartrouter/models/classifier.py`까지 무시하던 문제를 `/models/`로 고쳤습니다.
