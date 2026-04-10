"""Shared types for injectionguard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ThreatLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


LEVEL_ORDER = [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]


@dataclass
class Detection:
    """A single injection detection."""

    strategy: str
    pattern: str
    threat_level: ThreatLevel
    message: str
    offset: int = 0

    def __str__(self):
        return f"[{self.threat_level.value}] {self.strategy}: {self.message}"


@dataclass
class DetectionResult:
    """Result of scanning text for prompt injections."""

    text: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return len(self.detections) == 0

    @property
    def threat_level(self) -> ThreatLevel:
        if not self.detections:
            return ThreatLevel.NONE
        max_idx = max(LEVEL_ORDER.index(d.threat_level) for d in self.detections)
        return LEVEL_ORDER[max_idx]

    @property
    def is_critical(self) -> bool:
        return self.threat_level == ThreatLevel.CRITICAL

    def __str__(self):
        if self.is_safe:
            return "\u2713 No injection detected"
        lines = [f"\u26a0 {len(self.detections)} injection pattern(s) detected (threat: {self.threat_level.value}):"]
        for d in self.detections:
            lines.append(f"  - {d}")
        return "\n".join(lines)
