from abc import ABC, abstractmethod
from typing import Tuple

class BaseGenerator(ABC):
    """
    Abstract Base Class for LLM Generators.
    Ensures that any cloud provider (AWS Bedrock, Azure OpenAI) implements
    the required methods for correction and evaluation.
    """

    @abstractmethod
    def get_correction(self, model_name: str, text: str, intensity: str, field: str = "NONE") -> str:
        """
        Invokes LLM model to correct the sentence.
        """
        pass

    @abstractmethod
    def get_evaluation(self, source_text: str, generated_text: str, intensity: str, field: str = "NONE") -> Tuple[float, str]:
        """
        Evaluates the generated correction quality using LLM-as-a-Judge.
        Returns a score (1.0 to 5.0) and justification.
        """
        pass
