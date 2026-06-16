import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """모델 학습 평가 지표 계산"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_difficulty_classifier():
    """
    [SmartRouter Pipeline: Step 2]
    문장의 교정 난이도(쉬움/어려움)를 판단하는 이진 분류기 학습 모듈
    """
    model_name = "klue/roberta-base"  # 한국어 문맥 파악에 우수한 모델 활용
    
    print("Initializing Tokenizer and Model...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 데이터 로드 (LLM-as-a-Judge로 자체 구축한 7,700건의 기준 데이터)
    # 데이터셋 구조: {'text': "원본 문장", 'label': 1(어려움) or 0(쉬움)}
    dataset = load_dataset('csv', data_files={'train': 'data/train_7700.csv', 'test': 'data/test_data.csv'})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./models/roberta_classifier",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    print("Training started...")
    trainer.train()
    
    # 학습 완료된 모델 저장
    model.save_pretrained("./models/smartrouter_final")
    tokenizer.save_pretrained("./models/smartrouter_final")
    print("Model successfully saved to ./models/smartrouter_final")

if __name__ == "__main__":
    # train_difficulty_classifier()
    pass
