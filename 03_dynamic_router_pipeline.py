import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class SmartRouter:
    """
    [SmartRouter Pipeline: Step 3]
    비즈니스 환경에 맞춰 튜닝 가능한 임계값(Threshold) 기반 동적 라우팅 파이프라인
    (기존 07_모델_평가 및 07_임계값_튜닝 통합 리팩토링본)
    """
    def __init__(self, model_path: str, threshold: float = 0.78):
        print(f"Loading Router Model from {model_path}...")
        self.tokenizer = RobertaTokenizer.from_pretrained('klue/roberta-base')
        self.classifier = RobertaForSequenceClassification.from_pretrained(model_path)
        
        # 비즈니스 상황(비용 중시 vs 품질 중시)에 따라 유연하게 조절 가능한 임계값
        # F1 Score가 가장 높았던 0.78을 기본값(Default)으로 설정
        self.threshold = threshold 

    def analyze_difficulty(self, text: str) -> float:
        """입력된 문장의 교정 난이도 확률(Probability) 계산"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            hard_prob = probs[0][1].item() # '어려움' 클래스에 속할 확률
            
        return hard_prob

    def route_request(self, user_text: str):
        """임계값을 기준으로 모델 분기 (Routing)"""
        hard_prob = self.analyze_difficulty(user_text)
        
        # 확률이 Threshold 이상이면 '어려운 문장'으로 판단하여 고성능 모델로 우회
        if hard_prob >= self.threshold:
            print(f"[Router] Score: {hard_prob:.2f} >= {self.threshold} -> 대형 모델(Heavy LLM)로 라우팅")
            return self._call_heavy_model(user_text)
        else:
            print(f"[Router] Score: {hard_prob:.2f} < {self.threshold} -> 경량 모델(Gemini Flash)로 라우팅")
            return self._call_light_model(user_text)

    def _call_light_model(self, text: str):
        # 기존 일괄 처리되던 경량 모델 호출 로직 (비용 절감)
        return "Result from Gemini Flash"
        
    def _call_heavy_model(self, text: str):
        # 이탈 방지를 위한 고품질 교정 모델 호출 로직 (품질 보장)
        return "Result from Heavy Model (e.g., GPT-4 / EXAONE)"

def tune_threshold(validation_results):
    """
    Precision과 Recall의 Trade-off를 분석하여 최적의 Threshold를 찾는 시뮬레이션 함수
    """
    print("\n[Threshold 튜닝 시뮬레이션]")
    # 실제 노트북에서 수행했던 F1-score 기반 임계값 탐색 로직 요약
    print("Threshold 0.50 -> Recall 높으나 비용 증가 (F1: 0.71)")
    print("Threshold 0.78 -> 비용과 품질의 최적 균형점 도출 (F1: 0.78) **선택**")
    print("Threshold 0.90 -> 비용 최소화되나 고객 이탈 방어 실패 가능성 (F1: 0.65)")

if __name__ == "__main__":
    # 포트폴리오 시연용 코드
    # router = SmartRouter("./models/smartrouter_final", threshold=0.78)
    # router.route_request("간단한 오탈자 수정해주세요.")
    # router.route_request("전문적인 의학 논문 교열이 필요합니다. 문맥을 다듬어주세요.")
    pass
