# ==============================================================================
# MLOps 배치 파이프라인: Airflow DAG 스켈레톤 (면접 대비 주석용)
# ==============================================================================
# [배치 시나리오 설명]
# 1. extract_logs: 매일 밤 12시, 하루 동안 발생한 서비스 로그 데이터를 수집합니다.
# 2. evaluate_performance: 수집된 문장교정 기록(수락 여부 등)을 기반으로 RoBERTa 라우팅 정확도/F1-Score를 평가합니다.
# 3. auto_tune_threshold: F1-Score가 저하되었거나 비즈니스 요구사항(비용 절감 목표 변경 등)이 있을 때,
#    F1-Score 기준 최적 임계값(Threshold)을 재계산하여 config 파일이나 DB 환경 변수를 자동으로 업데이트합니다.
# ==============================================================================

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# --- 작업 태스크용 가상 파이썬 함수 정의 ---

def extract_logs(**kwargs):
    """
    일별 서비스 이벤트 로그 및 교정 수락 데이터를 수집하는 태스크
    """
    print("🔄 [Step 1] PostgreSQL/DynamoDB에서 오늘의 서비스 이벤트 로그를 추출합니다...")
    # SQL query: SELECT * FROM service_logs WHERE date = yesterday
    return "/tmp/raw_logs_yesterday.csv"

def evaluate_performance(**kwargs):
    """
    전날의 라우팅 정확도 및 Precision/Recall/F1-Score 분석
    """
    ti = kwargs['ti']
    log_path = ti.xcom_pull(task_ids='extract_service_logs')
    print(f"📊 [Step 2] {log_path} 데이터를 바탕으로 RoBERTa 분류 성능 평가지표를 계산합니다...")
    # 계산 결과 가상의 지표 출력
    f1_score = 0.81
    print(f"✅ Current F1-Score evaluated: {f1_score}")
    return f1_score

def auto_tune_threshold(**kwargs):
    """
    F1-Score 추이를 분석하여, 필요 시 임계값(Threshold)을 최적화 튜닝하는 태스크
    """
    ti = kwargs['ti']
    current_f1 = ti.xcom_pull(task_ids='evaluate_routing_performance')
    print(f"⚙️ [Step 3] 평가된 F1-Score ({current_f1})를 기반으로 최적 임계값을 재계산합니다...")
    
    # 가상의 튜닝 조건 분기
    if current_f1 < 0.80:
        new_threshold = 0.76  # 정밀도를 높이기 위해 조정
        print(f"⚠️ 성능 경고: 임계값을 {new_threshold}로 업데이트합니다 (기본값: 0.78).")
        # Config Server나 SSM Parameter Store의 환경 변수를 업데이트하는 API 호출 로직 위치
    else:
        print("🟢 성능이 양호합니다. 기존 임계값(0.78)을 유지합니다.")

# --- Airflow DAG 설정 ---

default_args = {
    'owner': 'yusj',
    'depends_on_past': False,
    'start_date': datetime(2026, 7, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'smartrouter_mlops_pipeline',
    default_args=default_args,
    description='Daily ML routing log evaluation and automatic threshold tuning pipeline',
    schedule_interval='@daily',  # 매일 밤 12시 실행
    catchup=False,
) as dag:

    # 1. 로그 데이터 추출 태스크
    task_extract = PythonOperator(
        task_id='extract_service_logs',
        python_callable=extract_logs,
    )

    # 2. 성능 지표 평가 태스크
    task_eval = PythonOperator(
        task_id='evaluate_routing_performance',
        python_callable=evaluate_performance,
    )

    # 3. 임계값 자동 최적화 태스크
    task_tune = PythonOperator(
        task_id='auto_tune_threshold',
        python_callable=auto_tune_threshold,
    )

    # 태스크 흐름 정의 (순차적 실행)
    task_extract >> task_eval >> task_tune
