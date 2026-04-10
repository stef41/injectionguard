"""Detect structural injection patterns in text."""

from __future__ import annotations

import re
from injectionguard.types import Detection, ThreatLevel

_TOKEN_PATTERNS = [
    (r'<\|system\|>', ThreatLevel.CRITICAL, "OpenAI system marker"),
    (r'<\|user\|>', ThreatLevel.HIGH, "OpenAI user marker"),
    (r'<\|assistant\|>', ThreatLevel.HIGH, "OpenAI assistant marker"),
    (r'<\|im_start\|>\s*system', ThreatLevel.CRITICAL, "ChatML system injection"),
    (r'<\|im_start\|>\s*user', ThreatLevel.HIGH, "ChatML user injection"),
    (r'<\|im_start\|>\s*assistant', ThreatLevel.HIGH, "ChatML assistant injection"),
    (r'<\|im_end\|>', ThreatLevel.HIGH, "ChatML end token"),
    (r'<\|endoftext\|>', ThreatLevel.CRITICAL, "End-of-text token"),
    (r'\[INST\]', ThreatLevel.HIGH, "Llama instruction tag"),
    (r'\[/INST\]', ThreatLevel.HIGH, "Llama instruction close tag"),
    (r'<<SYS>>', ThreatLevel.CRITICAL, "Llama system tag"),
    (r'<</SYS>>', ThreatLevel.CRITICAL, "Llama system close tag"),
    (r'<\|begin_of_text\|>', ThreatLevel.CRITICAL, "Begin-of-text token"),
    (r'<\|start_header_id\|>', ThreatLevel.CRITICAL, "Header start token"),
    (r'<\|end_header_id\|>', ThreatLevel.CRITICAL, "Header end token"),
    (r'<\|eot_id\|>', ThreatLevel.CRITICAL, "End-of-turn token"),
]


def check_structural(text: str) -> list[Detection]:
    """Detect structural patterns that suggest injection attempts."""
    detections: list[Detection] = []

    # Special token injection
    for pattern, level, message in _TOKEN_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            detections.append(Detection(
                strategy="structural",
                pattern=pattern,
                threat_level=level,
                message=f"Special token: {message}",
                offset=match.start(),
            ))

    # Excessive newlines (context pushing)
    for match in re.finditer(r'\n{10,}', text):
        detections.append(Detection(
            strategy="structural",
            pattern="excessive-newlines",
            threat_level=ThreatLevel.LOW,
            message=f"Excessive newlines ({len(match.group())}) - possible context pushing",
            offset=match.start(),
        ))

    # Low-entropy padding attack
    if len(text) > 1000:
        for i in range(0, len(text) - 100, 100):
            window = text[i:i + 100]
            if len(set(window)) < 5:
                detections.append(Detection(
                    strategy="structural",
                    pattern="repetition-padding",
                    threat_level=ThreatLevel.LOW,
                    message="Low-entropy text section - possible padding attack",
                    offset=i,
                ))
                break

    # Code block with injection content
    for match in re.finditer(r'```(?:\w*)\n(.*?)```', text, re.DOTALL):
        content = match.group(1).lower()
        injection_kws = ["system prompt", "ignore previous", "you are now", "override instructions"]
        if any(kw in content for kw in injection_kws):
            detections.append(Detection(
                strategy="structural",
                pattern="code-block-injection",
                threat_level=ThreatLevel.MEDIUM,
                message="Code block contains injection-like content",
                offset=match.start(),
            ))

    return detections
