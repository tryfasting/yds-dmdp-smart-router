import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = os.getenv("ROUTER_MODEL_PATH", str(BASE_DIR / "models" / "smart_router_v3"))
MODEL_NAME = "klue/roberta-base"

# --- Router Configuration ---
LABEL_HARD = 1
ROUTER_THRESHOLD = 0.25  # Optimized via Threshold Tuning (Max F1 balance point)

# --- AWS Bedrock Configuration ---
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Model IDs (AWS Bedrock)
MODEL_NANO = "anthropic.claude-3-haiku-20240307-v1:0"       # Cost-Effective / Fast
MODEL_MINI = "anthropic.claude-3-5-sonnet-20240620-v1:0"   # High-Quality / Deep Context
MODEL_JUDGE = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Evaluation Judge

# Pricing Table per 1M tokens (USD)
PRICING = {
    MODEL_NANO: {"input": 0.25, "output": 1.25},
    MODEL_MINI: {"input": 3.00, "output": 15.00},
}

# --- Prompt System Components ---
COMMON_CONSTRAINT = (
    "\n[CRITICAL OUTPUT RULES]\n"
    "1. Output ONLY the corrected version of the input sentence.\n"
    "2. Do NOT provide explanations, greetings, or conversational filler.\n"
    "3. Do NOT expand the content beyond the original meaning. Keep the length similar.\n"
    "4. If the input is empty or meaningless, return it as is."
)

INTENSITY_PROMPTS = {
    "WEAK": (
        "Role: Proofreader.\n"
        "Goal: Fix grammatical errors, typos, and punctuation only.\n"
        "Constraint: Keep the sentence structure and tone exactly the same."
    ),
    "MODERATE": (
        "Role: Editor.\n"
        "Goal: Improve flow, clarity, and readability.\n"
        "Constraint: Make it smoother but preserve the core meaning and length."
    ),
    "STRONG": (
        "Role: Senior Editor.\n"
        "Goal: Rewrite for professional impact and elegance.\n"
        "Constraint: Use sophisticated vocabulary and structure, but do NOT add new information."
    )
}

FIELD_PROMPTS = {
    "NONE": "Tone: Natural and neutral.",
    "EMAIL": "Tone: Professional, polite, and clear. Suitable for business correspondence.",
    "ARTICLE": "Tone: Engaging, informative, and journalistic. Keep the reader hooked.",
    "THESIS": "Tone: Academic, objective, and formal. Use precise terminology and avoid colloquialisms.",
    "REPORT": "Tone: Concise, factual, and business-like. Focus on clarity and data delivery.",
    "MARKETING": "Tone: Persuasive, catchy, and customer-focused. Highlight benefits and appeal to emotion.",
    "CUSTOMER_SERVICE": "Tone: Empathetic, polite, and solution-oriented. Use soft and respectful language."
}

# --- Evaluation Template (G-Eval) ---
GEVAL_PROMPT_TEMPLATE = """
You are an expert evaluator for an AI writing assistant.

### Task
Evaluate the 'AI Generated Correction' based on User Intent (Intensity) and Context (Field).

### Input Data
- Original Sentence: "{source_text}"
- User Intent (Intensity): {intensity}
- Context Field: {field}

### AI Generated Correction
"{generated_text}"

### Evaluation Criteria (Score 1-5)
1. Intent Accuracy: Does it match the requested Intensity ({intensity})?
2. Context Appropriateness: Does the tone match the Field ({field})?
3. Quality: Is the result natural and error-free?

### Output Format
Return ONLY a JSON object:
{{
    "score": <float, 1.0 to 5.0>,
    "reasoning": "<short explanation>"
}}
"""
