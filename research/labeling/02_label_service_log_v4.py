import pandas as pd
import json
import time
import os
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from google import genai
from google.genai import types

# =============================================================================
# [Configuration] 설정
# =============================================================================

# 1. 입력 파일 경로 (이전 단계에서 생성한 파일)
INPUT_DATA_PATH = "./data/munch_notlabeled.csv"

# 2. 이어하기 설정 (테스트이므로 None)
RESUME_CHECKPOINT_PATH = None

# 3. [NEW] 테스트 모드 설정 (안전 장치)
TEST_MODE = False   # True: 5개만 맛보기 실행 / False: 전체 1063개 실행
TEST_LIMIT = 5     # 테스트할 개수

# 4. 모델 릴레이 순서
MODEL_QUEUE = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash"
]

# 5. 모델별 속도 설정
MODEL_CONFIGS = {
    "gemini-2.5-flash-lite": {"sleep_time": 4.5},
    "gemini-2.0-flash-lite": {"sleep_time": 2.5},
    "gemini-2.0-flash":      {"sleep_time": 4.5},
    "gemini-2.5-flash":      {"sleep_time": 6.5},
}

BATCH_SIZE = 5
SAVE_INTERVAL = 20

# =============================================================================

# API 키 설정
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY 환경변수가 없습니다.")
else:
    client = genai.Client(api_key=api_key)

current_model_idx = 0
current_model_name = MODEL_QUEUE[0]

def get_current_config():
    cfg = MODEL_CONFIGS.get(current_model_name, {"sleep_time": 5.0})
    return cfg['sleep_time']

def switch_model():
    global current_model_idx, current_model_name
    current_model_idx += 1
    if current_model_idx >= len(MODEL_QUEUE):
        print("❌ 모든 모델 소진.")
        return False
    current_model_name = MODEL_QUEUE[current_model_idx]
    tqdm.write(f"\n🔄 Switched to: {current_model_name}")
    return True

def get_gemini_judgment_batch(batch_data: List[Dict]) -> List[Dict]:
    global current_model_name
    
    # 응답 스키마
    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "id": {"type": "INTEGER"},
                "reasoning": {"type": "STRING"},
                "score": {"type": "INTEGER"},
                "difficulty_label": {"type": "INTEGER"}
            },
            "required": ["id", "reasoning", "score", "difficulty_label"]
        }
    }
    
    gen_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=response_schema
    )

    # 프롬프트 구성
    input_text = ""
    for item in batch_data:
        field_val = item.get('field', 'General')
        if pd.isna(field_val) or str(field_val).strip() == "": field_val = "General"

        input_text += f"""
        [Case ID: {item['id']}]
        - Context: Intensity='{item['intensity']}', Field='{field_val}'
        - Input Sentence: "{item['input_sentence']}"
        - Corrected Output: "{item['final_corrected']}"
        -------------------------
        """

    prompt = f"""
    Role: AI Linguistic Expert & Model Architect.
    Task: Assess the "Intelligence Level" required to perform the correction for the following {len(batch_data)} cases.

    [Evaluation Criteria]
    - If user asked for 'Strong' intensity but result is simple (typos), score LOW.
    - Only score HIGH if correction required deep reasoning, tone shift, or creativity.

    [Scoring Rubric (1-5)]
    - 1 (Very Easy): Mechanical fixes. -> Flash
    - 2 (Easy): Simple swaps. -> Flash
    - 3 (Moderate): Reordering. -> Boundary
    - 4 (Hard): Structural change, formal tone. -> Pro
    - 5 (Very Hard): Rewrite, deep inference. -> Pro

    [Label Logic]
    - Score >= 4 -> difficulty_label: 1 (Hard)
    - Score <= 3 -> difficulty_label: 0 (Easy)

    [Input Data]
    {input_text}

    [Output Requirement]
    Return a JSON LIST of objects.
    """

    while True:
        try:
            response = client.models.generate_content(
                model=current_model_name, 
                contents=prompt,
                config=gen_config
            )
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if switch_model():
                    time.sleep(2)
                    continue
                else: return []
            else:
                print(f"❌ Error: {e}")
                return []

def _map_results_to_df(df, results_map):
    for idx, row in df.iterrows():
        t_id = row['id']
        if t_id in results_map:
            res = results_map[t_id]
            df.at[idx, 'reasoning'] = res.get('reasoning')
            df.at[idx, 'score'] = res.get('score')
            df.at[idx, 'difficulty_label'] = res.get('difficulty_label')

def process_resume(df, resume_path):
    if resume_path is None: return {}
    # (테스트 모드에서는 로직 생략, 필요시 기존 코드 참조)
    return {}

def main_process():
    # 1. 데이터 로드
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"❌ 파일 없음: {INPUT_DATA_PATH}")
        return
    
    print(f"📂 Loading: {INPUT_DATA_PATH}")
    df = pd.read_csv(INPUT_DATA_PATH)
    
    # 컬럼 체크
    if 'field' not in df.columns: df['field'] = 'General'
    if 'reasoning' not in df.columns:
        df['reasoning'] = None
        df['score'] = None
        df['difficulty_label'] = None
    
    df['id'] = range(len(df))

    # 2. 저장 경로 설정
    now_str = datetime.now().strftime("%H%M%S")
    
    if TEST_MODE:
        print(f"\n🧪 [TEST MODE ON] 전체 데이터 중 앞의 {TEST_LIMIT}개만 처리합니다.")
        checkpoint_path = f"./data/munch_ckpt_TEST_{now_str}.csv"
        final_save_path = f"./data/munch_labeled_TEST_{now_str}.csv"
    else:
        checkpoint_path = f"./data/munch_ckpt_AUTO_{now_str}.csv"
        final_save_path = f"./data/munch_labeled_FINAL_{now_str}.csv"

    # 3. 처리할 데이터 준비
    records = df[['id', 'input_sentence', 'final_corrected', 'intensity', 'field']].to_dict('records')
    
    # [핵심] 테스트 모드일 경우 데이터 자르기
    if TEST_MODE:
        records = records[:TEST_LIMIT]

    results_map = {}
    
    print(f"🚀 Processing Start | Target: {len(records)} items")

    # 4. 루프 시작
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Progress"):
        batch = records[i : i + BATCH_SIZE]
        
        batch_results = get_gemini_judgment_batch(batch)
        
        if not batch_results and current_model_idx >= len(MODEL_QUEUE):
            print("🛑 중단: 모델 한도 초과")
            break

        for res in batch_results:
            if res.get('id') is not None:
                results_map[res['id']] = res
        
        time.sleep(get_current_config())

        # 중간 저장
        if (i // BATCH_SIZE + 1) % SAVE_INTERVAL == 0:
            _map_results_to_df(df, results_map)
            # 테스트 모드면 전체 df가 아니라 해당 부분만 저장해도 됨 (여기선 전체 저장 유지)
            df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')

    # 5. 최종 저장
    _map_results_to_df(df, results_map)
    
    # 테스트 모드면 결과가 있는 행만 남겨서 저장 (깔끔하게 보기 위해)
    if TEST_MODE:
        df_final = df.dropna(subset=['score'])
    else:
        df_final = df

    df_final = df_final.drop(columns=['id'])
    df_final.to_csv(final_save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 완료! 파일 저장됨: {final_save_path}")
    if 'difficulty_label' in df_final.columns:
        print(df_final[['input_sentence', 'score', 'reasoning']].head(TEST_LIMIT))

if __name__ == "__main__":
    main_process()