"""injectionguard - Prompt injection detection for LLM applications and MCP servers."""

__version__ = "0.1.0"

from injectionguard.types import Detection, DetectionResult, ThreatLevel
from injectionguard.detector import Detector, detect, is_safe

__all__ = ["Detector", "DetectionResult", "ThreatLevel", "Detection", "detect", "is_safe"]
