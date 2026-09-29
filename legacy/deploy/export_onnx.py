import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    model_path = r"c:\Workspace\01-projects\SmartRouter_Pipeline\models\smart_router_v3"
    onnx_path = os.path.join(model_path, "model.onnx")

    print("📂 Loading PyTorch model and tokenizer...")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path {model_path} does not exist!")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Create dummy input for tracing (batch_size=1, sequence_length=128)
    dummy_text = "[STRONG] [THESIS] 이 문장은 테스트용 문장입니다."
    inputs = tokenizer(
        dummy_text, 
        return_tensors="pt", 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"🚀 Exporting PyTorch model to ONNX: {onnx_path}...")
    
    # Export the model
    # RoBERTa classification model accepts input_ids and attention_mask
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14
    )
    print("✅ ONNX Export Successful!")

if __name__ == "__main__":
    main()
