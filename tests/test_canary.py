"""Tests for cryptographic canary token system."""

from __future__ import annotations

import pytest

from injectionguard.canary import (
    CanaryMatch,
    CanaryReport,
    CanarySystem,
    CanaryToken,
    _bytes_to_zw,
    _extract_zw_sequences,
    _zw_to_bytes,
)


class TestZeroWidthEncoding:
    def test_roundtrip(self):
        original = b"\xde\xad\xbe\xef"
        encoded = _bytes_to_zw(original)
        decoded = _zw_to_bytes(encoded)
        assert decoded == original

    def test_roundtrip_short(self):
        original = b"\x00"
        encoded = _bytes_to_zw(original)
        decoded = _zw_to_bytes(encoded)
        assert decoded == original

    def test_roundtrip_all_ones(self):
        original = b"\xff"
        encoded = _bytes_to_zw(original)
        decoded = _zw_to_bytes(encoded)
        assert decoded == original

    def test_invisible_chars(self):
        encoded = _bytes_to_zw(b"AB")
        # All characters should be zero-width
        for ch in encoded:
            assert ch in {"\u200b", "\u200c"}


class TestExtractZWSequences:
    def test_no_zw(self):
        assert _extract_zw_sequences("hello world") == []

    def test_single_sequence(self):
        text = "before\u200b\u200cafter"
        seqs = _extract_zw_sequences(text)
        assert len(seqs) == 1
        assert seqs[0] == "\u200b\u200c"

    def test_multiple_sequences(self):
        text = "a\u200b\u200bb\u200c\u200cc"
        seqs = _extract_zw_sequences(text)
        assert len(seqs) == 2


class TestCanaryGeneration:
    def test_generate_deterministic(self):
        sys = CanarySystem(secret="test-secret")
        t1 = sys.generate("ctx", nonce="nonce1")
        t2 = sys.generate("ctx", nonce="nonce1")
        assert t1.token_id == t2.token_id
        assert t1.visible_marker == t2.visible_marker

    def test_generate_different_nonce(self):
        sys = CanarySystem(secret="test-secret")
        t1 = sys.generate("ctx", nonce="nonce1")
        t2 = sys.generate("ctx", nonce="nonce2")
        assert t1.token_id != t2.token_id

    def test_generate_different_context(self):
        sys = CanarySystem(secret="test-secret")
        t1 = sys.generate("system_prompt", nonce="n1")
        t2 = sys.generate("user_data", nonce="n1")
        assert t1.token_id != t2.token_id

    def test_generate_different_secret(self):
        s1 = CanarySystem(secret="secret-a")
        s2 = CanarySystem(secret="secret-b")
        t1 = s1.generate("ctx", nonce="n")
        t2 = s2.generate("ctx", nonce="n")
        assert t1.token_id != t2.token_id

    def test_visible_marker_format(self):
        sys = CanarySystem()
        token = sys.generate("test", nonce="n")
        assert token.visible_marker.startswith("[[CANARY:")
        assert token.visible_marker.endswith("]]")

    def test_invisible_marker_contains_zw(self):
        sys = CanarySystem()
        token = sys.generate("test", nonce="n")
        zw_chars = {"\u200b", "\u200c", "\u200d"}
        assert all(ch in zw_chars for ch in token.invisible_marker)

    def test_token_has_context(self):
        sys = CanarySystem()
        token = sys.generate("my_context", nonce="n")
        assert token.context == "my_context"


class TestCanaryVerify:
    def test_verify_correct(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("ctx", nonce="n")
        assert sys.verify(token, "ctx", "n") is True

    def test_verify_wrong_context(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("ctx", nonce="n")
        assert sys.verify(token, "wrong", "n") is False

    def test_verify_wrong_nonce(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("ctx", nonce="n")
        assert sys.verify(token, "ctx", "wrong") is False


class TestCanaryScan:
    def test_no_leak(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("ctx", nonce="n")
        report = sys.scan("This is a normal response.", [token])
        assert not report.is_leaked
        assert len(report.matches) == 0

    def test_visible_leak(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("ctx", nonce="n")
        output = f"Here is the prompt: {token.visible_marker} and more text"
        report = sys.scan(output, [token])
        assert report.is_leaked
        assert len(report.matches) == 1
        assert report.matches[0].match_type == "visible"

    def test_invisible_leak(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("ctx", nonce="n")
        output = f"Normal text{token.invisible_marker}more text"
        report = sys.scan(output, [token])
        assert report.is_leaked
        assert any(m.match_type == "invisible" for m in report.matches)

    def test_detection_is_critical(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("system_prompt", nonce="n")
        output = f"leaked: {token.visible_marker}"
        report = sys.scan(output, [token])
        assert len(report.detections) > 0
        from injectionguard.types import ThreatLevel
        assert report.detections[0].threat_level == ThreatLevel.CRITICAL

    def test_multiple_tokens(self):
        sys = CanarySystem(secret="s")
        t1 = sys.generate("prompt1", nonce="n1")
        t2 = sys.generate("prompt2", nonce="n2")
        output = f"leaked: {t1.visible_marker}"
        report = sys.scan(output, [t1, t2])
        assert report.is_leaked
        assert len(report.matches) == 1
        assert report.matches[0].context == "prompt1"

    def test_both_visible_and_invisible(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("ctx", nonce="n")
        output = f"{token.visible_marker} text {token.invisible_marker}"
        report = sys.scan(output, [token])
        assert len(report.matches) == 2
        types = {m.match_type for m in report.matches}
        assert "visible" in types
        assert "invisible" in types

    def test_detection_strategy(self):
        sys = CanarySystem(secret="s")
        token = sys.generate("ctx", nonce="n")
        output = f"text {token.visible_marker}"
        report = sys.scan(output, [token])
        assert report.detections[0].strategy == "canary"

    def test_scan_empty_tokens(self):
        sys = CanarySystem()
        report = sys.scan("some text", [])
        assert not report.is_leaked
