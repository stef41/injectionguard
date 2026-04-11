"""injectionguard - Prompt injection detection for LLM applications and MCP servers."""

__version__ = "0.4.0"

from injectionguard.types import Detection, DetectionResult, ThreatLevel
from injectionguard.detector import Detector, detect, is_safe
from injectionguard.rate import (
    SlidingWindowDetector, RateConfig, DetectionEvent,
    score_injection_indicators, format_rate_report,
)
from injectionguard.scoring import (
    ConfidenceLevel, ConfidenceScore, ConfidenceScorer, format_confidence_report,
)
from injectionguard.canary import CanarySystem, CanaryToken, CanaryMatch, CanaryReport
from injectionguard.drift import (
    DriftResult, ConversationTurn, compute_drift, detect_conversation_drift,
)
from injectionguard.strategies.perplexity import (
    PerplexityReport, PerplexityWindow, analyze_perplexity, compute_perplexity,
    sliding_window_perplexity,
)

__all__ = [
    "Detector", "DetectionResult", "ThreatLevel", "Detection",
    "detect", "is_safe",
    "SlidingWindowDetector", "RateConfig", "DetectionEvent",
    "score_injection_indicators", "format_rate_report",
    "ConfidenceLevel", "ConfidenceScore", "ConfidenceScorer", "format_confidence_report",
    "CanarySystem", "CanaryToken", "CanaryMatch", "CanaryReport",
    "DriftResult", "ConversationTurn", "compute_drift", "detect_conversation_drift",
    "PerplexityReport", "PerplexityWindow", "analyze_perplexity", "compute_perplexity",
    "sliding_window_perplexity",
]


def _lazy_middleware():
    from injectionguard.middleware import InjectionGuardMiddleware
    return InjectionGuardMiddleware
