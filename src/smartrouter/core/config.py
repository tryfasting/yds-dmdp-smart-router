import os
from pathlib import Path

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
# Submitted V8 checkpoint (root weights == checkpoint-205, SHA-256 a641edd...).
MODEL_PATH = Path(os.getenv("ROUTER_MODEL_PATH", str(BASE_DIR / "models" / "smart_router_v3")))

# --- Router Configuration ---
LABEL_HARD = 1
MAX_LENGTH = 128
# Chosen on the same split that was used for checkpoint selection (optimistic; see docs/decisions.md).
ROUTER_THRESHOLD = 0.25

INTENSITIES = ("WEAK", "MODERATE", "STRONG")
FIELDS = ("NONE", "EMAIL", "ARTICLE", "THESIS", "REPORT", "MARKETING", "CUSTOMER_SERVICE")
DEFAULT_INTENSITY = "WEAK"
DEFAULT_FIELD = "NONE"

# Abstract routing tiers. The historical experiment mapped light -> gpt-5-nano, heavy -> gpt-5-mini.
TIER_LIGHT = "light"
TIER_HEAVY = "heavy"
