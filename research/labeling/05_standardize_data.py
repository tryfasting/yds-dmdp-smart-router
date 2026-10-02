# scripts/standardize_data.py
# 데이터셋 파일명, 컬럼 이름 통일 후 통합하는 코드(1회성 코드)

import os
import pandas as pd

# ---------------------------------------------------------
# [설정] 경로 및 파일명 정의
# ---------------------------------------------------------
BASE_DIR = "data"

# Input & Output Files
INPUT_FILES = {
    "v1": "munch_labeled_FINAL_190139.csv",
    "v2_raw": "munch_labeled_RESUMED_200125.csv",
    "v2_aug": "munch_labeled_RESUMED_200125_revised.csv"
}

OUTPUT_FILES = {
    "v1": "source_v1_legacy.csv",
    "v2_raw": "source_v2_raw.csv",
    "v2_aug": "source_v2_augmented.csv",
    "master": "dataset_master.csv"
}

MAPPINGS = {
    "v1": {
        "input_sentence": "input_text",
        "final_corrected": "ground_truth",
        "intensity": "meta_intensity",
        "field": "meta_field",
        "difficulty_label": "label_difficulty",
        "reasoning": "reasoning",
        "score": "score"
    },
    "v2_raw": {
        "original": "input_text",
        "corrected": "ground_truth",
        "difficulty_label": "label_difficulty",
        "reasoning": "reasoning",
        "score": "score"
    },
    "v2_aug": {
        "input_sentence": "input_text",
        "corrected": "ground_truth",
        "intensity": "meta_intensity",
        "field": "meta_field",
        "difficulty_label": "label_difficulty",
        "reasoning": "reasoning",
        "score": "score"
    }
}


def load_and_standardize(file_key):
    """개별 파일을 로드하여 컬럼명을 표준화하고 저장합니다."""
    input_path = os.path.join(BASE_DIR, INPUT_FILES[file_key])
    output_path = os.path.join(BASE_DIR, OUTPUT_FILES[file_key])
    mapping = MAPPINGS[file_key]

    if not os.path.exists(input_path):
        print(f"⚠️ Warning: Input file not found: {input_path}")
        return None

    df = pd.read_csv(input_path)

    # 컬럼 표준화
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
    df_standardized = df[list(rename_dict.keys())].rename(columns=rename_dict)

    # 개별 파일 저장
    df_standardized.to_csv(output_path, index=False)
    return df_standardized


def create_master_dataset():
    print("\n🚀 [Step 1] Standardizing Individual Files (Score included)...")

    # 1. 파일 표준화
    df_v1 = load_and_standardize("v1")
    _ = load_and_standardize("v2_raw")  # 아카이빙용
    df_v2_aug = load_and_standardize("v2_aug")

    if df_v1 is None or df_v2_aug is None:
        print("❌ Error: Essential datasets are missing.")
        return

    print("\n🚀 [Step 2] Creating Master Dataset (Merge to match 7701)...")

    # 2. 전처리 (Merge 전 개별 처리)
    # v1에서 difficulty_label이 없는 10개만 제거 (1063 -> 1053)
    len_v1_before = len(df_v1)
    df_v1 = df_v1.dropna(subset=['label_difficulty'])
    print(f"   - v1 cleansing: {len_v1_before} -> {len(df_v1)} (Dropped {len_v1_before - len(df_v1)} null labels)")

    # 3. 출처 마킹
    df_v1['source_origin'] = 'v1_legacy'
    df_v2_aug['source_origin'] = 'v2_augmented'

    # 4. 병합 (Concatenate)
    # ignore_index=True로 새로운 인덱스 생성
    df_master = pd.concat([df_v1, df_v2_aug], ignore_index=True)

    # 5. 데이터 클렌징 (텍스트 공백 제거만 수행, 행 삭제 X)
    print("🧹 Cleaning Text format (No row dropping)...")
    for col in ['input_text', 'ground_truth']:
        df_master[col] = df_master[col].astype(str).str.strip()

    for col in ['meta_intensity', 'meta_field']:
        if col in df_master.columns:
            df_master[col] = df_master[col].astype(str).str.upper().str.strip()

    # 라벨 및 점수 타입 변환
    df_master['label_difficulty'] = df_master['label_difficulty'].astype(int)
    # score는 float 유지 (필요 시 int 변환 가능하지만 1.5점 등이 있을 수 있으므로)
    if 'score' in df_master.columns:
        df_master['score'] = df_master['score'].fillna(0.0).astype(float)

    # 6. 최종 저장
    master_path = os.path.join(BASE_DIR, OUTPUT_FILES["master"])
    df_master.to_csv(master_path, index=False)

    print(f"\n✅ Master Dataset Created Successfully!")
    print(f"   - Path: {master_path}")
    print(f"   - Total Rows: {len(df_master)} (Expected: 7701)")
    print(f"   - Columns: {list(df_master.columns)}")

    if len(df_master) == 7701:
        print("   🎉 PERFECT MATCH with original training set!")
    else:
        print(f"   ⚠️ Count Mismatch! Difference: {len(df_master) - 7701}")



if __name__ == "__main__":
    create_master_dataset()