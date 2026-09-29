from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRouter(ABC):
    """
    Abstract base class for routing decision makers.
    Lets the RoBERTa router and the intensity-rule baseline be compared through one interface.
    """

    @abstractmethod
    def predict(self, text: str, intensity: str, field: str) -> Dict[str, Any]:
        """
        Evaluate a request and return routing metadata.

        Returns:
            Dict[str, Any]: at least 'tier' ("light" or "heavy") and 'router'.
        """
