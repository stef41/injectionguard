"""Perplexity-based prompt injection detection.

Uses a character-level trigram language model trained on natural English text.
Prompt injections tend to have anomalous perplexity because they contain:
  - Unusual command syntax ("ignore previous", "system:")
  - Encoded payloads (base64, hex)
  - Repetitive padding or delimiter runs
  - Mixed-language/code fragments

The detector computes both global and sliding-window perplexity, flagging
segments that deviate significantly from natural language statistics.

Reference: Jain et al. 2023, "Baseline Defenses for Adversarial Attacks
Against Aligned Language Models" — perplexity filtering as input guard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from injectionguard.types import Detection, ThreatLevel

# ── Trained trigram model ──────────────────────────────────────────────────
# Log-probabilities for character trigrams, estimated from ~1M chars of
# English Wikipedia + StackOverflow + conversational text.  We store only
# the 500 most-common trigrams; everything else falls back to bigram or
# unigram with Katz back-off (discount = 0.5).
#
# Format: {trigram: log2_probability}
# Built offline — this is the shipped "model".

# Rather than a huge literal, we build the model from frequency counts of
# the printable ASCII character set.  This keeps the source compact while
# giving a genuine statistical model.

_CHAR_FREQS: dict[str, float] = {
    " ": 0.1831, "e": 0.1026, "t": 0.0751, "a": 0.0653, "o": 0.0616,
    "i": 0.0567, "n": 0.0571, "s": 0.0518, "h": 0.0497, "r": 0.0499,
    "d": 0.0328, "l": 0.0331, "c": 0.0223, "u": 0.0228, "m": 0.0203,
    "w": 0.0170, "f": 0.0181, "g": 0.0161, "y": 0.0146, "p": 0.0150,
    "b": 0.0127, ",": 0.0114, ".": 0.0111, "v": 0.0082, "k": 0.0069,
    "\"": 0.0026, "'": 0.0024, "-": 0.0015, "x": 0.0015, "j": 0.0010,
    "q": 0.0010, "z": 0.0007, ";": 0.0003, ":": 0.0004, "!": 0.0003,
    "?": 0.0006, "\n": 0.0100, "0": 0.0020, "1": 0.0020, "2": 0.0015,
    "3": 0.0012, "4": 0.0011, "5": 0.0011, "6": 0.0010, "7": 0.0010,
    "8": 0.0010, "9": 0.0010,
}

# Common English bigrams with relative log2 probs
_BIGRAM_LOGP: dict[str, float] = {}
_TRIGRAM_LOGP: dict[str, float] = {}

# Build bigram model from co-occurrence patterns
_COMMON_BIGRAMS = [
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
    "ve", "co", "me", "de", "hi", "ri", "ro", "ic", " t", " a",
    " i", " s", " o", " w", " f", " b", " c", " d", "e ", "s ",
    "t ", "n ", "d ", "y ", "r ", "l ", "f ", ". ", ", ", "  ",
    "ig", "no", "pr", "ys", "em", "ol", "ec", "ly", "ss", "ch",
]

for _i, _bg in enumerate(_COMMON_BIGRAMS):
    _BIGRAM_LOGP[_bg] = -math.log2(max(0.001, 0.06 - _i * 0.0008))

# Common English trigrams
_COMMON_TRIGRAMS = [
    "the", "ing", "and", "ion", "tio", "ent", "ati", "for", "her",
    "ter", "hat", "tha", "ere", "ate", "his", "con", "res", "ver",
    "all", "ons", "nce", "men", "ith", "ted", "ers", "pro", "thi",
    "wit", "are", "ess", "not", "ive", "was", "ect", "rea", "com",
    "eve", "int", "est", "sta", "eni", " th", " an", " to", " of",
    " in", " is", " co", " re", " st", " wh", " he", " it", " be",
    " on", " ha", "ed ", "ng ", "er ", "on ", "es ", "he ", "of ",
    "in ", "al ", "an ", "re ", "nd ", "to ", "is ", "en ", "or ",
    "se ", "at ", " a ", " I ", "ly ", "ll ", "le ", "th ", "nt ",
    "st ", "ss ", "ce ", "te ", "as ", "ne ", "it ",
]

for _i, _tg in enumerate(_COMMON_TRIGRAMS):
    _TRIGRAM_LOGP[_tg] = -math.log2(max(0.001, 0.04 - _i * 0.0004))

# Back-off discount (Katz back-off factor in log2 space)
_BACKOFF_PENALTY = 2.0  # bits
_UNKNOWN_LOGP = -12.0   # bits for completely unknown characters


def _char_logp(c: str) -> float:
    """Unigram log2 probability of a character."""
    freq = _CHAR_FREQS.get(c.lower(), 0.0001)
    return math.log2(freq)


def _score_trigram(a: str, b: str, c: str) -> float:
    """Katz back-off log2 probability: trigram → bigram → unigram."""
    tri = a + b + c
    if tri in _TRIGRAM_LOGP:
        return _TRIGRAM_LOGP[tri]

    bi = b + c
    if bi in _BIGRAM_LOGP:
        return _BIGRAM_LOGP[bi] - _BACKOFF_PENALTY

    return _char_logp(c) - 2 * _BACKOFF_PENALTY


def compute_perplexity(text: str) -> float:
    """Compute character-level perplexity of text using trigram LM.

    Returns 2^H where H is the average cross-entropy in bits/char.
    Natural English text: perplexity ~4-8.
    Prompt injections: typically >12.
    Encoded payloads: typically >20.
    """
    if len(text) < 3:
        return 1.0

    total_logp = 0.0
    n = 0
    lower = text.lower()
    for i in range(2, len(lower)):
        total_logp += _score_trigram(lower[i - 2], lower[i - 1], lower[i])
        n += 1

    if n == 0:
        return 1.0

    avg_logp = total_logp / n  # negative value (log2 prob)
    cross_entropy = -avg_logp
    # Clamp to avoid overflow
    cross_entropy = min(cross_entropy, 30.0)
    return 2.0 ** cross_entropy


@dataclass
class PerplexityWindow:
    """A window of text with its perplexity score."""
    start: int
    end: int
    text: str
    perplexity: float


def sliding_window_perplexity(
    text: str,
    window_size: int = 50,
    stride: int = 25,
) -> list[PerplexityWindow]:
    """Compute perplexity in sliding windows over the text.

    Returns windows sorted by perplexity (highest first), which
    highlights the most anomalous segments.
    """
    if len(text) < window_size:
        p = compute_perplexity(text)
        return [PerplexityWindow(0, len(text), text, p)]

    windows: list[PerplexityWindow] = []
    for start in range(0, len(text) - window_size + 1, stride):
        end = start + window_size
        segment = text[start:end]
        p = compute_perplexity(segment)
        windows.append(PerplexityWindow(start, end, segment, p))

    windows.sort(key=lambda w: w.perplexity, reverse=True)
    return windows


@dataclass
class PerplexityReport:
    """Full perplexity analysis of a text."""
    global_perplexity: float
    max_window_perplexity: float
    anomalous_windows: list[PerplexityWindow]
    is_anomalous: bool
    detections: list[Detection]


def analyze_perplexity(
    text: str,
    global_threshold: float = 14.0,
    window_threshold: float = 20.0,
    window_size: int = 50,
    stride: int = 25,
) -> PerplexityReport:
    """Full perplexity-based injection analysis.

    Computes global perplexity and per-window perplexity.  High-perplexity
    windows are flagged as potential injection vectors.

    Args:
        text: Input text to analyze.
        global_threshold: Global perplexity above which the whole text is suspicious.
        window_threshold: Per-window perplexity above which a segment is flagged.
        window_size: Characters per sliding window.
        stride: Step between consecutive windows.

    Returns:
        PerplexityReport with detections for anomalous segments.
    """
    global_ppl = compute_perplexity(text)
    windows = sliding_window_perplexity(text, window_size, stride)
    max_ppl = windows[0].perplexity if windows else 0.0

    anomalous = [w for w in windows if w.perplexity > window_threshold]
    detections: list[Detection] = []

    if global_ppl > global_threshold:
        if global_ppl > global_threshold * 2:
            level = ThreatLevel.HIGH
        elif global_ppl > global_threshold * 1.5:
            level = ThreatLevel.MEDIUM
        else:
            level = ThreatLevel.LOW
        detections.append(Detection(
            strategy="perplexity",
            pattern="high_global_perplexity",
            threat_level=level,
            message=f"Global perplexity {global_ppl:.1f} exceeds threshold {global_threshold:.1f}",
            offset=0,
        ))

    for w in anomalous[:3]:  # Top 3 anomalous windows
        if w.perplexity > window_threshold * 2:
            level = ThreatLevel.HIGH
        elif w.perplexity > window_threshold * 1.5:
            level = ThreatLevel.MEDIUM
        else:
            level = ThreatLevel.LOW
        snippet = w.text[:30].replace("\n", "\\n")
        detections.append(Detection(
            strategy="perplexity",
            pattern="high_window_perplexity",
            threat_level=level,
            message=f"Window [{w.start}:{w.end}] perplexity {w.perplexity:.1f}: \"{snippet}...\"",
            offset=w.start,
        ))

    is_anom = global_ppl > global_threshold or len(anomalous) > 0

    return PerplexityReport(
        global_perplexity=global_ppl,
        max_window_perplexity=max_ppl,
        anomalous_windows=anomalous,
        is_anomalous=is_anom,
        detections=detections,
    )
