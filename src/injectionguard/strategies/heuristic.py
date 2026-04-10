"""Pattern-based heuristic detection for common injection patterns."""

from __future__ import annotations

import re
from injectionguard.types import Detection, ThreatLevel

INJECTION_PATTERNS = [
    # Direct instruction override
    (r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?|guidelines?)',
     ThreatLevel.CRITICAL, "Direct instruction override attempt"),
    (r'disregard\s+(?:all\s+)?(?:your\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)',
     ThreatLevel.CRITICAL, "Instruction disregard attempt"),
    (r'forget\s+(?:all\s+)?(?:your\s+)?(?:previous|prior|above|your)\s+(?:instructions?|prompts?|rules?|training)',
     ThreatLevel.CRITICAL, "Instruction erasure attempt"),
    (r'override\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instructions?|prompts?|rules?)',
     ThreatLevel.CRITICAL, "Instruction override attempt"),

    # Role manipulation
    (r'you\s+are\s+now\s+(?:a|an|the)\s+',
     ThreatLevel.HIGH, "Role reassignment attempt"),
    (r'(?:act|behave|respond)\s+as\s+(?:if\s+)?(?:you\s+(?:are|were)\s+)?(?:a|an|the)\s+',
     ThreatLevel.MEDIUM, "Role manipulation attempt"),
    (r'(?:pretend|imagine)\s+(?:that\s+)?you\s+(?:are|were|have)',
     ThreatLevel.MEDIUM, "Role pretending attempt"),
    (r'(?:switch|change)\s+(?:to|into)\s+(?:a|an|the)?\s*\w+\s+mode',
     ThreatLevel.HIGH, "Mode switch attempt"),
    (r'new\s+(?:system\s+)?(?:instructions?|prompt|persona|identity)',
     ThreatLevel.HIGH, "Identity override attempt"),

    # System prompt extraction
    (r'(?:show|reveal|display|output|print|repeat|tell\s+me)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?)',
     ThreatLevel.HIGH, "System prompt extraction attempt"),
    (r'what\s+(?:are|were)\s+your\s+(?:original\s+)?(?:instructions?|rules?|guidelines?|prompt)',
     ThreatLevel.HIGH, "System prompt extraction attempt"),
    (r'(?:copy|paste|echo|dump)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)',
     ThreatLevel.HIGH, "System prompt dump attempt"),

    # Data exfiltration
    (r'(?:send|transmit|post|exfiltrate|forward)\s+(?:this|the|all|my)\s+(?:\w+\s+){0,3}(?:data|information|conversation|messages?)\s+to',
     ThreatLevel.CRITICAL, "Data exfiltration attempt"),

    # Tool/function manipulation
    (r'(?:call|invoke|execute|run|trigger)\s+(?:the\s+)?(?:tool|function|api|endpoint)\s+',
     ThreatLevel.MEDIUM, "Tool invocation injection"),

    # Jailbreak patterns
    (r'(?:DAN|jailbreak|bypass|override)\s+(?:mode|prompt|filter|safety|restriction)',
     ThreatLevel.CRITICAL, "Jailbreak attempt"),
    (r'(?:disable|remove|ignore|bypass)\s+(?:your|all|the)?\s*(?:safety|filter|restriction|guardrail|content\s+policy)',
     ThreatLevel.CRITICAL, "Safety bypass attempt"),
    (r'(?:do\s+anything\s+now|no\s+restrictions?\s+mode)',
     ThreatLevel.CRITICAL, "Unrestricted mode attempt"),

    # Continuation/completion hijacking
    (r'(?:continue|complete)\s+(?:the|this)\s+(?:response|output|text)\s+(?:with|by|as)',
     ThreatLevel.MEDIUM, "Response hijacking attempt"),

    # Boundary/delimiter attacks
    (r'---+\s*(?:system|end|new)\s*---+',
     ThreatLevel.HIGH, "Delimiter boundary attack"),
    (r'={3,}\s*(?:system|end|new)\s*={3,}',
     ThreatLevel.HIGH, "Delimiter boundary attack"),
]


def check_heuristic(text: str) -> list[Detection]:
    """Check for known injection patterns using regex heuristics."""
    detections: list[Detection] = []

    for pattern, level, message in INJECTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            detections.append(Detection(
                strategy="heuristic",
                pattern=pattern,
                threat_level=level,
                message=message,
                offset=match.start(),
            ))

    return detections
