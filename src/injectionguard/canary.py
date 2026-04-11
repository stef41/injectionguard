"""Cryptographic canary token system for prompt leak detection.

Canary tokens are unique, verifiable markers embedded in system prompts.
If an LLM output contains a canary token, the system prompt was leaked —
either through direct extraction or indirect prompt injection.

Uses HMAC-SHA256 for token generation with configurable secrets.
Supports both visible tokens and zero-width Unicode steganographic tokens.

Reference: Greshake et al. 2023, "Not what you've signed up for:
Compromising Real-World LLM-Integrated Applications with Indirect
Prompt Injection" — canary-based leak detection.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

from injectionguard.types import Detection, ThreatLevel

# ── Zero-width steganography ─────────────────────────────────────────────
# Encode bits using zero-width Unicode characters (invisible in most UIs).
_ZW_SPACE = "\u200b"     # bit 0
_ZW_NON_JOINER = "\u200c"  # bit 1
_ZW_JOINER = "\u200d"    # separator

_BIT_MAP = {_ZW_SPACE: "0", _ZW_NON_JOINER: "1"}
_BIT_REV = {"0": _ZW_SPACE, "1": _ZW_NON_JOINER}


def _bytes_to_zw(data: bytes) -> str:
    """Encode bytes as zero-width Unicode characters."""
    bits = "".join(format(b, "08b") for b in data)
    return "".join(_BIT_REV[bit] for bit in bits)


def _zw_to_bytes(zw: str) -> bytes:
    """Decode zero-width Unicode characters back to bytes."""
    bits = []
    for ch in zw:
        if ch in _BIT_MAP:
            bits.append(_BIT_MAP[ch])
    # Pad to multiple of 8
    while len(bits) % 8 != 0:
        bits.append("0")
    byte_list = []
    for i in range(0, len(bits), 8):
        byte_val = int("".join(bits[i:i + 8]), 2)
        byte_list.append(byte_val)
    return bytes(byte_list)


def _extract_zw_sequences(text: str) -> list[str]:
    """Extract contiguous zero-width character sequences from text."""
    sequences: list[str] = []
    current: list[str] = []
    zw_chars = {_ZW_SPACE, _ZW_NON_JOINER, _ZW_JOINER}
    for ch in text:
        if ch in zw_chars:
            current.append(ch)
        elif current:
            sequences.append("".join(current))
            current = []
    if current:
        sequences.append("".join(current))
    return sequences


# ── Token generation ─────────────────────────────────────────────────────

@dataclass
class CanaryToken:
    """A generated canary token with its metadata."""
    token_id: str          # Hex digest (first 16 bytes of HMAC)
    visible_marker: str    # Human-readable marker string
    invisible_marker: str  # Zero-width encoded marker
    timestamp: float       # When the token was created
    context: str           # What this token protects (e.g., "system_prompt")


@dataclass
class CanaryMatch:
    """A detected canary token in output text."""
    token_id: str
    match_type: str  # "visible" or "invisible"
    offset: int
    context: str


@dataclass
class CanaryReport:
    """Result of scanning for canary token leaks."""
    matches: list[CanaryMatch]
    is_leaked: bool
    detections: list[Detection]


class CanarySystem:
    """Manages canary token generation and detection.

    Usage:
        system = CanarySystem(secret="my-secret-key")
        token = system.generate("system_prompt")

        # Embed token.visible_marker in your system prompt
        prompt = f"You are a helpful assistant. {token.visible_marker}"

        # Later, scan LLM output for leaks
        report = system.scan(llm_output, [token])
        if report.is_leaked:
            print("System prompt was leaked!")
    """

    def __init__(self, secret: str = "injectionguard-default-secret"):
        self._secret = secret.encode("utf-8")

    def generate(self, context: str = "default", nonce: str | None = None) -> CanaryToken:
        """Generate a new canary token.

        Args:
            context: Descriptive label for what this token protects.
            nonce: Optional nonce for deterministic generation (for testing).
                   If None, uses current timestamp.

        Returns:
            CanaryToken with both visible and invisible markers.
        """
        if nonce is None:
            nonce = f"{time.time():.6f}"

        payload = f"{context}:{nonce}".encode("utf-8")
        mac = hmac.new(self._secret, payload, hashlib.sha256).digest()

        token_id = mac[:16].hex()
        # Visible marker: human-readable but distinctive
        short_id = mac[:6].hex()
        visible = f"[[CANARY:{short_id}]]"
        # Invisible marker: zero-width encoded token ID
        invisible = _ZW_JOINER + _bytes_to_zw(mac[:8]) + _ZW_JOINER

        return CanaryToken(
            token_id=token_id,
            visible_marker=visible,
            invisible_marker=invisible,
            timestamp=time.time(),
            context=context,
        )

    def verify(self, token: CanaryToken, context: str, nonce: str) -> bool:
        """Verify that a token was generated with the given parameters.

        This re-derives the HMAC and compares to the stored token_id.
        Uses constant-time comparison to prevent timing attacks.
        """
        payload = f"{context}:{nonce}".encode("utf-8")
        mac = hmac.new(self._secret, payload, hashlib.sha256).digest()
        expected_id = mac[:16].hex()
        return hmac.compare_digest(token.token_id, expected_id)

    def scan(self, text: str, tokens: list[CanaryToken]) -> CanaryReport:
        """Scan text for any canary token leaks.

        Checks for both visible markers (substring match) and invisible
        markers (zero-width character sequence decoding).

        Args:
            text: The text to scan (typically LLM output).
            tokens: List of canary tokens to search for.

        Returns:
            CanaryReport with matches and Detection objects.
        """
        matches: list[CanaryMatch] = []

        # Check visible markers
        for token in tokens:
            idx = text.find(token.visible_marker)
            if idx >= 0:
                matches.append(CanaryMatch(
                    token_id=token.token_id,
                    match_type="visible",
                    offset=idx,
                    context=token.context,
                ))

        # Check invisible markers
        zw_sequences = _extract_zw_sequences(text)
        for token in tokens:
            # Extract the raw zero-width content between joiners
            inv = token.invisible_marker
            inner = inv.strip(_ZW_JOINER)
            for seq in zw_sequences:
                seq_inner = seq.strip(_ZW_JOINER)
                if len(seq_inner) >= len(inner) and inner in seq_inner:
                    # Verify by decoding
                    decoded = _zw_to_bytes(inner)
                    seq_decoded = _zw_to_bytes(seq_inner[:len(inner)])
                    if decoded == seq_decoded:
                        offset = text.find(seq)
                        matches.append(CanaryMatch(
                            token_id=token.token_id,
                            match_type="invisible",
                            offset=offset if offset >= 0 else 0,
                            context=token.context,
                        ))
                        break

        # Build detections
        detections: list[Detection] = []
        for m in matches:
            detections.append(Detection(
                strategy="canary",
                pattern=f"canary_leak_{m.match_type}",
                threat_level=ThreatLevel.CRITICAL,
                message=(
                    f"Canary token leaked ({m.match_type}): "
                    f"context=\"{m.context}\", id={m.token_id[:12]}..."
                ),
                offset=m.offset,
            ))

        return CanaryReport(
            matches=matches,
            is_leaked=len(matches) > 0,
            detections=detections,
        )
