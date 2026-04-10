"""injectionguard - Prompt injection detection for LLM applications and MCP servers."""

__version__ = "0.2.0"

from injectionguard.types import Detection, DetectionResult, ThreatLevel
from injectionguard.detector import Detector, detect, is_safe
from injectionguard.rate import (
    SlidingWindowDetector, RateConfig, DetectionEvent,
    score_injection_indicators, format_rate_report,
)
from injectionguard.scoring import (
    ConfidenceLevel, ConfidenceScore, ConfidenceScorer, format_confidence_report,
)

__all__ = [
    "Detector", "DetectionResult", "ThreatLevel", "Detection",
    "detect", "is_safe",
    "SlidingWindowDetector", "RateConfig", "DetectionEvent",
    "score_injection_indicators", "format_rate_report",
    "ConfidenceLevel", "ConfidenceScore", "ConfidenceScorer", "format_confidence_report",
]


def _lazy_middleware():
    from injectionguard.middleware import InjectionGuardMiddleware
    return InjectionGuardMiddleware
