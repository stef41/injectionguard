"""Detect encoded/obfuscated injection attempts."""

from __future__ import annotations

import base64
import re
from injectionguard.types import Detection, ThreatLevel

_INJECTION_KEYWORDS = [
    "ignore", "previous", "instructions", "system prompt",
    "disregard", "forget", "override", "jailbreak",
    "you are now", "act as", "pretend", "bypass",
    "disable safety", "remove filter",
]

_INVISIBLE_CHARS = {
    '\u200b': "Zero-width space",
    '\u200c': "Zero-width non-joiner",
    '\u200d': "Zero-width joiner",
    '\u2060': "Word joiner",
    '\ufeff': "Zero-width no-break space (BOM)",
    '\u00ad': "Soft hyphen",
    '\u200e': "Left-to-right mark",
    '\u200f': "Right-to-left mark",
    '\u202a': "Left-to-right embedding",
    '\u202b': "Right-to-left embedding",
    '\u202c': "Pop directional formatting",
    '\u202d': "Left-to-right override",
    '\u202e': "Right-to-left override",
}


def check_encoding(text: str) -> list[Detection]:
    """Detect injection hidden in encoded or obfuscated text."""
    detections: list[Detection] = []

    # Base64 encoded injection
    for match in re.finditer(r'[A-Za-z0-9+/]{40,}={0,2}', text):
        b64_text = match.group()
        try:
            decoded = base64.b64decode(b64_text).decode('utf-8', errors='ignore')
            if _contains_injection(decoded):
                detections.append(Detection(
                    strategy="encoding",
                    pattern="base64",
                    threat_level=ThreatLevel.HIGH,
                    message=f"Base64 encoded injection: '{decoded[:60]}...'",
                    offset=match.start(),
                ))
        except Exception:
            pass

    # Invisible Unicode characters
    for char, name in _INVISIBLE_CHARS.items():
        if char in text:
            detections.append(Detection(
                strategy="encoding",
                pattern=f"U+{ord(char):04X}",
                threat_level=ThreatLevel.MEDIUM,
                message=f"Invisible character: {name} (U+{ord(char):04X})",
                offset=text.index(char),
            ))

    # Hex-encoded injection
    for match in re.finditer(r'(?:\\x[0-9a-fA-F]{2}){4,}', text):
        try:
            decoded = bytes.fromhex(
                match.group().replace('\\x', '')
            ).decode('utf-8', errors='ignore')
            if _contains_injection(decoded):
                detections.append(Detection(
                    strategy="encoding",
                    pattern="hex",
                    threat_level=ThreatLevel.HIGH,
                    message=f"Hex-encoded injection: '{decoded[:60]}'",
                    offset=match.start(),
                ))
        except Exception:
            pass

    # URL-encoded injection
    for match in re.finditer(r'(?:%[0-9a-fA-F]{2}){4,}', text):
        try:
            from urllib.parse import unquote
            decoded = unquote(match.group())
            if _contains_injection(decoded):
                detections.append(Detection(
                    strategy="encoding",
                    pattern="url-encoded",
                    threat_level=ThreatLevel.HIGH,
                    message=f"URL-encoded injection: '{decoded[:60]}'",
                    offset=match.start(),
                ))
        except Exception:
            pass

    return detections


def _contains_injection(text: str) -> bool:
    """Check if decoded text contains injection keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _INJECTION_KEYWORDS)
