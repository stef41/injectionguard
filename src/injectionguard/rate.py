"""Rate-based detection for gradual prompt injection escalation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RateConfig:
    """Configuration for sliding window rate detection."""

    window_size: int = 10
    threshold: float = 0.6
    decay_factor: float = 0.9
    min_samples: int = 3


@dataclass
class DetectionEvent:
    """A single detection event in the sliding window."""

    timestamp: float
    text: str
    score: float
    flagged: bool

    def __str__(self) -> str:
        status = "FLAGGED" if self.flagged else "ok"
        return f"[{status}] score={self.score:.3f} t={self.timestamp:.1f}"


# --- Pattern scoring ---

_IMPERATIVE_PATTERNS = [
    (r'\b(?:ignore|disregard|forget|override)\b', 0.3),
    (r'\b(?:you\s+must|you\s+should|you\s+will|you\s+are\s+now)\b', 0.2),
    (r'\b(?:do\s+not|don\'t)\s+follow\b', 0.25),
    (r'\b(?:instead|actually|correction)\b', 0.1),
]

_ROLE_PLAY_PATTERNS = [
    (r'\b(?:act|behave|respond|pretend|imagine)\s+as\b', 0.25),
    (r'\b(?:you\s+are\s+(?:a|an|the)\s+)', 0.2),
    (r'\b(?:new\s+(?:role|persona|identity|character))\b', 0.25),
    (r'\b(?:switch|change)\s+(?:to|into)\b.*\bmode\b', 0.2),
]

_OVERRIDE_PATTERNS = [
    (r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)', 0.5),
    (r'(?:system|admin|root)\s*(?:prompt|access|override)', 0.35),
    (r'(?:jailbreak|bypass|disable)\s+(?:safety|filter|restriction|guardrail)', 0.5),
    (r'(?:reveal|show|dump|repeat)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)', 0.35),
]

_EXFILTRATION_PATTERNS = [
    (r'(?:send|transmit|post|forward)\s+.*\b(?:data|info|conversation)\b', 0.3),
    (r'\b(?:http|https|ftp)://', 0.15),
]

_ALL_PATTERNS = (
    _IMPERATIVE_PATTERNS
    + _ROLE_PLAY_PATTERNS
    + _OVERRIDE_PATTERNS
    + _EXFILTRATION_PATTERNS
)


def score_injection_indicators(text: str) -> float:
    """Score a single text for injection patterns.

    Returns a float between 0.0 (benign) and 1.0 (highly suspicious).
    """
    total = 0.0
    for pattern, weight in _ALL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            total += weight
    # Clamp to [0, 1]
    return min(total, 1.0)


class SlidingWindowDetector:
    """Detect gradual injection escalation over a sliding window of inputs."""

    def __init__(self, config: Optional[RateConfig] = None) -> None:
        self.config = config or RateConfig()
        self._events: list[DetectionEvent] = []

    def feed(self, text: str, timestamp: Optional[float] = None) -> DetectionEvent:
        """Add a new input, score it, and check for escalation.

        Returns the DetectionEvent for this input.
        """
        if timestamp is None:
            timestamp = time.time()

        score = score_injection_indicators(text)
        agg = self._aggregate_score(score)
        flagged = agg >= self.config.threshold and len(self._events) + 1 >= self.config.min_samples

        event = DetectionEvent(
            timestamp=timestamp,
            text=text,
            score=score,
            flagged=flagged,
        )
        self._events.append(event)

        # Trim to window size
        if len(self._events) > self.config.window_size:
            self._events = self._events[-self.config.window_size:]

        return event

    def current_score(self) -> float:
        """Return the current aggregate threat score (0-1)."""
        if not self._events:
            return 0.0
        return self._aggregate_score(self._events[-1].score)

    def is_escalating(self) -> bool:
        """Return True if the threat level is increasing over the window."""
        events = self._window_events()
        if len(events) < 2:
            return False
        mid = len(events) // 2
        first_half = sum(e.score for e in events[:mid]) / mid
        second_half = sum(e.score for e in events[mid:]) / (len(events) - mid)
        return second_half > first_half

    def reset(self) -> None:
        """Clear all events from the window."""
        self._events.clear()

    def history(self) -> list[DetectionEvent]:
        """Return all recorded events."""
        return list(self._events)

    def window_summary(self) -> dict:
        """Return a summary of the current window state."""
        events = self._window_events()
        if not events:
            return {
                "window_size": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "flagged_count": 0,
                "escalating": False,
                "current_score": 0.0,
            }
        scores = [e.score for e in events]
        return {
            "window_size": len(events),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "flagged_count": sum(1 for e in events if e.flagged),
            "escalating": self.is_escalating(),
            "current_score": self.current_score(),
        }

    # --- internal helpers ---

    def _window_events(self) -> list[DetectionEvent]:
        """Return events within the window."""
        return self._events[-self.config.window_size:]

    def _aggregate_score(self, latest_score: float) -> float:
        """Compute a decay-weighted aggregate of recent scores plus the latest."""
        events = self._window_events()
        if not events:
            return latest_score

        weight = 1.0
        total_score = 0.0
        total_weight = 0.0

        # Latest score gets highest weight
        total_score += latest_score * weight
        total_weight += weight

        for event in reversed(events):
            weight *= self.config.decay_factor
            total_score += event.score * weight
            total_weight += weight

        return min(total_score / total_weight, 1.0) if total_weight > 0 else 0.0


def format_rate_report(events: list[DetectionEvent]) -> str:
    """Format a list of detection events into a human-readable report."""
    if not events:
        return "No events recorded."

    flagged = sum(1 for e in events if e.flagged)
    lines = [f"Rate detection report: {len(events)} event(s), {flagged} flagged", ""]
    for i, event in enumerate(events, 1):
        lines.append(f"  {i}. {event}")
    return "\n".join(lines)
