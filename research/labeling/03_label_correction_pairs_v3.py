import pandas as pd
import json
import time
import os
import glob
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from google import genai
from google.genai import types

# =============================================================================
# [Configuration] 모델 릴레이(Relay) 및 이어하기 설정
# =============================================================================

# 1. 이어하기 설정 (중요!)
# 멈췄던 체크포인트 파일명을 여기에 적어주세요. (사용자 제공 파일명 반영됨)
RESUME_CHECKPOINT_PATH = "./data/munch_labeled_FINAL_190558_ckpt.csv"

# 2. 모델 릴레이 순서 (Queue)
# 첫 번째 모델이 한도 초과(429)되면, 자동으로 다음 모델로 넘어갑니다.
# gemini-2.5-flash-lite (RPD 1,000) -> 2.5-flash -> 2.0-flash-lite 순서 추천
MODEL_QUEUE = [
    "gemini-2.5-flash-lite",  # 1타자: RPD 1,000회 (가장 넉넉)
    "gemini-2.5-flash",       # 2타자: RPD 250회
    "gemini-2.0-flash-lite",  # 3타자: RPD 200회
    "gemini-1.5-flash"        # 4타자: 예비용
]

# 3. 모델별 RPM/Sleep 설정
MODEL_CONFIGS = {
    "gemini-2.5-flash-lite": {"sleep_time": 4.5}, # 15 RPM
    "gemini-2.5-flash":      {"sleep_time": 6.5}, # 10 RPM (좀 느림)
    "gemini-2.0-flash-lite": {"sleep_time": 2.5}, # 30 RPM
    "gemini-1.5-flash":      {"sleep_time": 4.5}, # 15 RPM
    "gemini-2.0-flash":      {"sleep_time": 5.5}  # 15 RPM
}

BATCH_SIZE = 5
SAVE_INTERVAL = 20  # 100건마다 저장

# =============================================================================

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 전역 변수로 현재 모델 상태 관리
current_model_idx = 0
current_model_name = MODEL_QUEUE[0]

def get_current_config():
    """현재 모델에 맞는 Sleep Time 반환"""
    cfg = MODEL_CONFIGS.get(current_model_name, {"sleep_time": 5.0})
    return cfg['sleep_time']

def switch_model():
    """다음 모델로 교체하는 함수"""
    global current_model_idx, current_model_name
    current_model_idx += 1
    
    if current_model_idx >= len(MODEL_QUEUE):
        print("❌ 모든 모델의 한도를 소진했습니다. 내일 다시 시도하세요.")
        return False # 더 이상 교체할 모델 없음
    
    current_model_name = MODEL_QUEUE[current_model_idx]
    tqdm.write(f"\n🔄 [Model Switch] Quota exceeded. Switching to NEXT Model: {current_model_name}")
    tqdm.write(f"   - New Sleep Time: {get_current_config()}s")
    return True

def get_gemini_judgment_batch(batch_data: List[Dict]) -> List[Dict]:
    """
    Gemini API 호출 (자동 모델 스위칭 포함)
    """
    global current_model_name

    # 1. 응답 스키마 정의
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

    # 2. 입력 데이터 직렬화
    input_text = ""
    for item in batch_data:
        input_text += f"""
        [ID: {item['id']}]
        - Original: "{item['original']}"
        - Corrected: "{item['corrected']}"
        -------------------------
        """

    # 3. [복구 완료] 상세 프롬프트 (Rubric & Logic 완벽 포함)
    prompt = f"""
    Role: Linguistic Expert Judge.
    Task: Evaluate the difficulty of AI text correction for the following {len(batch_data)} cases.

    [Scoring Rubric (1-5 Scale)]
    - Score 1 (Very Easy): Mechanical fixes (typos, punctuation).
    - Score 2 (Easy): Particles, simple word swaps.
    - Score 3 (Moderate): Basic reordering, minor structural changes.
    - Score 4 (Hard): Tone shift, major structural changes, formality adjustments.
    - Score 5 (Very Hard): Total reconstruction, abstract rewriting, nuance-heavy changes.

    [Label Logic]
    - Score >= 4 -> difficulty_label: 1 (Hard)
    - Score <= 3 -> difficulty_label: 0 (Easy)

    [Input Data Batch]
    {input_text}

    [Output Requirement]
    Return a JSON LIST containing an object for EACH case. Ensure the 'id' matches the Input Data.
    """

    # 4. API 호출 및 모델 스위칭 로직
    while True:
        try:
            response = client.models.generate_content(
                model=current_model_name, 
                contents=prompt,
                config=gen_config
            )
            return json.loads(response.text)

        except Exception as e:
            error_msg = str(e)
            # 429 (Resource Exhausted) 에러 발생 시 모델 교체 시도
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                # 모델 교체 시도
                if switch_model():
                    time.sleep(2) # 교체 후 잠시 대기 후 재시도 (continue)
                    continue
                else:
                    return [] # 모든 모델 소진
            else:
                # 429가 아닌 다른 에러는 즉시 리턴 (데이터 문제 등)
                print(f"❌ Non-recoverable Error on {current_model_name}: {e}")
                return []

def _map_results_to_df(df, results_map):
    """결과를 DataFrame에 매핑"""
    for idx, row in df.iterrows():
        t_id = row['id']
        if t_id in results_map:
            res = results_map[t_id]
            df.at[idx, 'reasoning'] = res.get('reasoning')
            df.at[idx, 'score'] = res.get('score')
            df.at[idx, 'difficulty_label'] = res.get('difficulty_label')

def process_resume(df, resume_path):
    """체크포인트 로드 및 이어하기 준비"""
    results_map = {}
    
    if os.path.exists(resume_path):
        print(f"📂 Loading checkpoint from: {resume_path}")
        try:
            df_ckpt = pd.read_csv(resume_path)
            # 완료된 데이터(점수가 있는 것)만 맵에 등록
            completed_rows = df_ckpt.dropna(subset=['score'])
            for _, row in completed_rows.iterrows():
                # id 컬럼이 있는 경우에만 로드
                if 'id' in row and pd.notna(row['id']): 
                    r_id = int(row['id'])
                    results_map[r_id] = {
                        'id': r_id,
                        'reasoning': row.get('reasoning'),
                        'score': row.get('score'),
                        'difficulty_label': row.get('difficulty_label')
                    }
            print(f"✅ Resuming... {len(results_map)} items already finished.")
        except Exception as e:
            print(f"⚠️ Failed to read checkpoint: {e}. Starting from scratch.")
    else:
        print("⚠️ Checkpoint file not found. Starting from scratch.")
    
    return results_map

def main_process():
    # 1. 원본 데이터 로드
    munch_folder_path = './data/munch_data/'
    csv_files = glob.glob(os.path.join(munch_folder_path, '*.csv'))
    print(f"📂 Found {len(csv_files)} CSV files in '{munch_folder_path}'")

    munch_list = []
    for f in csv_files:
        munch_list.append(pd.read_csv(f, header=None, names=['original', 'corrected']))
    df = pd.concat(munch_list, ignore_index=True)
    
    # 2. 데이터프레임 초기화
    if 'reasoning' not in df.columns:
        df['reasoning'] = None
        df['score'] = None
        df['difficulty_label'] = None
    
    # ID 생성 (체크포인트와 매칭을 위해 필수)
    df['id'] = range(len(df)) 

    # 3. 이어하기 데이터 로드
    results_map = process_resume(df, RESUME_CHECKPOINT_PATH)
    
    # 체크포인트 내용을 현재 df에 미리 반영
    _map_results_to_df(df, results_map)

    # 4. 처리할 데이터 준비
    records = df[['id', 'original', 'corrected']].to_dict('records')
    
    print(f"🚀 Relay Processing Start | First Model: {current_model_name}")
    print(f"   - Total Rows: {len(df)}")
    print(f"   - Remaining: {len(df) - len(results_map)}")
    
    # 파일명 생성
    now_str = datetime.now().strftime("%H%M%S")
    final_save_path = f"./data/munch_labeled_RESUMED_{now_str}.csv"
    checkpoint_path = final_save_path.replace(".csv", "_ckpt.csv")

    # Loop Start
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Total Progress"):
        batch = records[i : i + BATCH_SIZE]
        
        # [중요] 이미 완료된 배치인지 확인 (Skip Logic)
        # 배치의 첫 번째 아이템 ID가 results_map에 있으면 이미 한 것으로 간주
        if batch[0]['id'] in results_map:
            continue
            
        # API 호출
        batch_results = get_gemini_judgment_batch(batch)
        
        # 모든 모델 한도 초과 시 종료
        if not batch_results and current_model_idx >= len(MODEL_QUEUE):
            print("🛑 Process stopped due to quota exhaustion.")
            break

        # 결과 저장
        for res in batch_results:
            r_id = res.get('id')
            if r_id is not None:
                results_map[r_id] = res
        
        # 현재 모델의 속도에 맞춰 대기
        time.sleep(get_current_config()) 

        # 중간 저장 (Checkpoint)
        current_batch_idx = i // BATCH_SIZE
        if (current_batch_idx + 1) % SAVE_INTERVAL == 0:
            _map_results_to_df(df, results_map)
            df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')

    # 5. 최종 저장
    print("Saving final results...")
    _map_results_to_df(df, results_map)
    
    # ID 컬럼 제거 후 저장
    df = df.drop(columns=['id'])
    df.to_csv(final_save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ All Done! Saved to {final_save_path}")
    if 'difficulty_label' in df.columns:
        print(df['difficulty_label'].value_counts())

if __name__ == "__main__":
    main_process()