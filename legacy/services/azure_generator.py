import json
from typing import Optional, Tuple
from smartrouter.core.logger import logger
from smartrouter.core import config
from smartrouter.services.base import BaseGenerator

class AzureOpenAIGenerator(BaseGenerator):
    """
    Client wrapper for Azure OpenAI Service.
    Implements the BaseGenerator interface for Multi-Cloud support.
    Used historically in the project to interact with GPT-5 Nano/Mini models.
    """

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.mock_mode = False
        
        # We wrap the import here so that if the user hasn't installed openai,
        # it doesn't break the whole app unless they actually try to use Azure.
        try:
            from openai import OpenAI
            
            # Using dummy fallback values if not set, as this is primarily
            # for architectural proof in the interview setting.
            azure_key = api_key or getattr(config, "AZURE_API_KEY", "dummy-key")
            azure_endpoint = endpoint or getattr(config, "AZURE_ENDPOINT", "https://dummy.openai.azure.com/")
            
            if azure_key == "dummy-key":
                logger.warning("⚠️ [AzureOpenAIGenerator] AZURE_API_KEY not set. Running in MOCK Mode.")
                self.mock_mode = True
            
            self.client = OpenAI(
                base_url=azure_endpoint,
                api_key=azure_key
            )
            logger.info("✅ [AzureOpenAIGenerator] Azure OpenAI Client initialized successfully.")
        except ImportError:
            logger.error("❌ [AzureOpenAIGenerator] 'openai' library not installed. Please install it to use Azure.")
            self.mock_mode = True
        except Exception as e:
            logger.warning(f"⚠️ [AzureOpenAIGenerator] Failed to initialize Azure client: {e}")
            self.mock_mode = True

    def _build_system_instruction(self, intensity: str, field: str) -> str:
        base_prompt = config.INTENSITY_PROMPTS.get(
            intensity, config.INTENSITY_PROMPTS.get("MODERATE", "")
        )
        field_key = field.upper() if field else "NONE"
        field_prompt = config.FIELD_PROMPTS.get(
            field_key, config.FIELD_PROMPTS.get("NONE", "")
        )
        return f"{base_prompt}\n\n[Context Requirement]\n{field_prompt}\n{config.COMMON_CONSTRAINT}"

    def get_correction(self, model_name: str, text: str, intensity: str, field: str = "NONE") -> str:
        """
        Invokes Azure OpenAI model (GPT-5 Nano/Mini) to correct the sentence.
        """
        system_instruction = self._build_system_instruction(intensity, field)

        if self.mock_mode:
            return f"[Azure Mock: {model_name}] Corrected: {text}"

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": text}
                ],
                max_tokens=1000 
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            logger.error(f"❌ [AzureOpenAIGenerator] API Error (Model: {model_name}): {e}", exc_info=True)
            return f"[Azure Fallback] Corrected: {text}"

    def get_evaluation(self, source_text: str, generated_text: str, intensity: str, field: str = "NONE") -> Tuple[float, str]:
        """
        Evaluates using Azure OpenAI's JSON mode.
        """
        prompt = config.GEVAL_PROMPT_TEMPLATE.format(
            source_text=source_text,
            intensity=intensity,
            field=field,
            generated_text=generated_text
        )

        if self.mock_mode:
            return 4.3, "[Mock Azure Judge] Multi-cloud evaluation demo."

        try:
            # We assume config.MODEL_JUDGE could be an Azure model name here, or we hardcode.
            judge_model = getattr(config, "AZURE_MODEL_JUDGE", "gpt-5-mini")
            
            response = self.client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000, 
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from Judge model")

            result = json.loads(content)
            score = float(result.get('score', 4.0))
            reasoning = result.get('reasoning', "Evaluation complete.")
            
            return score, reasoning
            
        except Exception as e:
            logger.error(f"❌ [AzureOpenAIGenerator] Evaluation failed: {e}", exc_info=True)
            return 4.0, f"Fallback Azure Eval (Err: {e})"
