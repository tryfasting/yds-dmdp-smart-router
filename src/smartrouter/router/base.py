from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseRouter(ABC):
    """
    Abstract Base Class for Smart Router decision makers.
    Enables plug-and-play extensions for alternative classifiers or heuristic rules.
    """
    
    @abstractmethod
    def predict(self, text: str, intensity: str, field: str) -> Dict[str, Any]:
        """
        Evaluate input text and return routing metadata.
        
        Args:
            text (str): Input sentence to evaluate.
            intensity (str): Desired revision intensity (WEAK, MODERATE, STRONG).
            field (str): Context domain (EMAIL, THESIS, etc.).
            
        Returns:
            Dict[str, Any]: Routing decision containing 'target_model', 'prob_hard', 'decision', etc.
        """
        pass
