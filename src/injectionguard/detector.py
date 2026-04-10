"""Core prompt injection detection engine."""

from __future__ import annotations

import re
from typing import Optional

from injectionguard.types import Detection, DetectionResult, ThreatLevel, LEVEL_ORDER
from injectionguard.strategies.heuristic import check_heuristic
from injectionguard.strategies.encoding import check_encoding
from injectionguard.strategies.structural import check_structural


ALL_STRATEGIES = [
    check_heuristic,
    check_encoding,
    check_structural,
]


class Detector:
    """Main prompt injection detector."""

    def __init__(
        self,
        strategies: Optional[list] = None,
        threshold: ThreatLevel = ThreatLevel.LOW,
        allow_list: Optional[list[str]] = None,
        block_list: Optional[list[str]] = None,
    ):
        self.strategies = strategies or ALL_STRATEGIES
        self.threshold = threshold
        self.allow_list: list[str] = allow_list or []
        self.block_list: list[str] = block_list or []

    def scan(self, text: str) -> DetectionResult:
        """Scan text for prompt injection patterns."""
        result = DetectionResult(text=text)

        # Block list: always flag these patterns
        for pattern in self.block_list:
            for match in re.finditer(re.escape(pattern), text, re.IGNORECASE):
                result.detections.append(Detection(
                    strategy="blocklist",
                    pattern=pattern,
                    threat_level=ThreatLevel.CRITICAL,
                    message=f"Blocklisted pattern: '{pattern}'",
                    offset=match.start(),
                ))

        for strategy in self.strategies:
            try:
                detections = strategy(text)
                result.detections.extend(detections)
            except Exception:
                pass

        threshold_idx = LEVEL_ORDER.index(self.threshold)
        result.detections = [
            d for d in result.detections
            if LEVEL_ORDER.index(d.threat_level) >= threshold_idx
        ]

        # Allow list: remove detections that match allowed patterns
        if self.allow_list:
            result.detections = [
                d for d in result.detections
                if not any(allowed.lower() in d.message.lower() for allowed in self.allow_list)
                and d.strategy != "blocklist"  # never allow blocklisted
                or d.strategy == "blocklist"
            ]

        return result

    def is_safe(self, text: str) -> bool:
        """Quick check if text appears safe."""
        return self.scan(text).is_safe

    def scan_mcp_output(self, tool_name: str, output: str) -> DetectionResult:
        """Scan MCP tool output for injection attempts targeting the calling agent."""
        result = self.scan(output)

        mcp_patterns = [
            (r'<\s*(?:system|assistant|user)\s*>', ThreatLevel.HIGH, "XML role tag in tool output"),
            (r'\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>', ThreatLevel.HIGH, "LLM instruction tags in tool output"),
            (r'Human:|Assistant:|System:', ThreatLevel.MEDIUM, "Conversation role markers in tool output"),
        ]

        for pattern, level, message in mcp_patterns:
            for match in re.finditer(pattern, output, re.IGNORECASE):
                result.detections.append(Detection(
                    strategy="mcp",
                    pattern=pattern,
                    threat_level=level,
                    message=f"{message} (tool: {tool_name})",
                    offset=match.start(),
                ))

        return result

    def scan_batch(self, texts: list[str]) -> list[DetectionResult]:
        """Scan multiple texts."""
        return [self.scan(t) for t in texts]


def detect(text: str) -> DetectionResult:
    """Convenience: scan text for injections."""
    return Detector().scan(text)


def is_safe(text: str) -> bool:
    """Convenience: check if text is safe."""
    return Detector().is_safe(text)
