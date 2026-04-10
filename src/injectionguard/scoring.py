"""Detection confidence scoring for prompt injection analysis."""

from __future__ import annotations

import base64
import re
from collections import defaultdict
from dataclasses import dataclass, field


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

class ConfidenceLevel:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


_DEFAULT_WEIGHTS: dict[str, float] = {
    "imperative_density": 0.15,
    "role_manipulation": 0.25,
    "instruction_override": 0.25,
    "encoding_obfuscation": 0.15,
    "delimiter_injection": 0.20,
}

# ------------------------------------------------------------------
# Factor helpers
# ------------------------------------------------------------------

_IMPERATIVE_STARTERS = re.compile(
    r"(?:^|(?<=\.\s))(?:ignore|forget|disregard|override|stop|do not|don'?t|"
    r"please|tell|say|write|print|output|return|execute|run|act|pretend|"
    r"reveal|show|list|give|answer|respond|repeat|always|never)\b",
    re.IGNORECASE | re.MULTILINE,
)

_ROLE_PATTERNS = re.compile(
    r"\b(?:you are|you'?re|act as|pretend (?:to be|you'?re)|"
    r"imagine you'?re|roleplay as|play the role|assume the (?:role|identity)|"
    r"behave as|from now on you)\b",
    re.IGNORECASE,
)

_OVERRIDE_PATTERNS = re.compile(
    r"\b(?:ignore (?:previous|above|all|prior|earlier)|"
    r"forget (?:previous|above|all|prior|earlier|your|everything)|"
    r"disregard (?:previous|above|all|prior|earlier|your)|"
    r"override (?:previous|your|instructions|system)|"
    r"new instructions|start over|reset (?:instructions|context)|"
    r"do not follow|don'?t follow previous)\b",
    re.IGNORECASE,
)

_DELIMITER_PATTERNS = re.compile(
    r"(?:```(?:system|user|assistant)|<\|(?:im_start|im_end|system|endoftext)\|>|"
    r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|### (?:System|Instruction|Human|Assistant):|"
    r"SYSTEM:|USER:|ASSISTANT:)",
    re.IGNORECASE,
)

_BASE64_BLOCK = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_HEX_BLOCK = re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")
_UNICODE_ESCAPE = re.compile(r"(?:\\u[0-9a-fA-F]{4}){3,}")


def imperative_density(text: str) -> float:
    """Ratio of imperative sentences to total sentence-level fragments."""
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
    if not sentences:
        return 0.0
    hits = sum(1 for s in sentences if _IMPERATIVE_STARTERS.search(s))
    return hits / len(sentences)


def role_manipulation_score(text: str) -> float:
    """Detect role-assumption language like 'you are', 'act as', 'pretend to be'."""
    matches = _ROLE_PATTERNS.findall(text)
    if not matches:
        return 0.0
    return min(1.0, len(matches) * 0.4)


def instruction_override_score(text: str) -> float:
    """Detect override language like 'ignore previous', 'forget everything'."""
    matches = _OVERRIDE_PATTERNS.findall(text)
    if not matches:
        return 0.0
    return min(1.0, len(matches) * 0.5)


def encoding_obfuscation_score(text: str) -> float:
    """Detect base64, hex-encoded, or unicode-escaped payloads."""
    score = 0.0
    if _BASE64_BLOCK.search(text):
        # Check if any base64 block actually decodes to ASCII
        for m in _BASE64_BLOCK.finditer(text):
            try:
                decoded = base64.b64decode(m.group() + "==").decode("utf-8", errors="ignore")
                if any(c.isalpha() for c in decoded):
                    score += 0.5
                    break
            except Exception:
                pass
    if _HEX_BLOCK.search(text):
        score += 0.4
    if _UNICODE_ESCAPE.search(text):
        score += 0.4
    return min(1.0, score)


def delimiter_injection_score(text: str) -> float:
    """Detect system-prompt delimiters injected into user text."""
    matches = _DELIMITER_PATTERNS.findall(text)
    if not matches:
        return 0.0
    return min(1.0, len(matches) * 0.5)


# ------------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------------

@dataclass
class ConfidenceScore:
    """Result of confidence scoring."""

    score: float
    level: str
    factors: list[str] = field(default_factory=list)
    explanation: str = ""


def _level_for_score(score: float) -> str:
    if score >= 0.70:
        return ConfidenceLevel.HIGH
    if score >= 0.40:
        return ConfidenceLevel.MEDIUM
    if score >= 0.15:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.NONE


# ------------------------------------------------------------------
# Scorer
# ------------------------------------------------------------------

_FACTOR_FUNCS: dict[str, callable] = {
    "imperative_density": imperative_density,
    "role_manipulation": role_manipulation_score,
    "instruction_override": instruction_override_score,
    "encoding_obfuscation": encoding_obfuscation_score,
    "delimiter_injection": delimiter_injection_score,
}


class ConfidenceScorer:
    """Score how confident we are that text contains a prompt injection."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights) if weights else dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def score_factors(self, text: str) -> dict[str, float]:
        """Return individual factor scores for *text*."""
        return {name: fn(text) for name, fn in _FACTOR_FUNCS.items()}

    def score(self, text: str) -> ConfidenceScore:
        """Return an overall :class:`ConfidenceScore` for *text*."""
        factors = self.score_factors(text)
        weighted = sum(factors[k] * self.weights.get(k, 0.0) for k in factors)
        total_weight = sum(self.weights.get(k, 0.0) for k in factors) or 1.0
        raw = weighted / total_weight
        active = [k for k, v in factors.items() if v > 0]
        level = _level_for_score(raw)
        explanation_parts: list[str] = []
        for k in active:
            explanation_parts.append(f"{k}={factors[k]:.2f}")
        explanation = f"score={raw:.2f} ({', '.join(explanation_parts)})" if explanation_parts else f"score={raw:.2f}"
        return ConfidenceScore(score=round(raw, 4), level=level, factors=active, explanation=explanation)

    def batch_score(self, texts: list[str]) -> list[ConfidenceScore]:
        """Score multiple texts."""
        return [self.score(t) for t in texts]

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def calibrate(self, labeled_examples: list[tuple[str, bool]]) -> dict[str, float]:
        """Adjust weights from labeled data (text, is_injection) pairs.

        Uses a simple approach: for each factor, compute the average value
        across positives minus the average across negatives. Positive
        deltas get higher weight.

        Returns the updated weights dict.
        """
        positives = [t for t, label in labeled_examples if label]
        negatives = [t for t, label in labeled_examples if not label]

        if not positives or not negatives:
            return dict(self.weights)

        pos_avgs: dict[str, float] = defaultdict(float)
        neg_avgs: dict[str, float] = defaultdict(float)

        for t in positives:
            for k, v in self.score_factors(t).items():
                pos_avgs[k] += v / len(positives)
        for t in negatives:
            for k, v in self.score_factors(t).items():
                neg_avgs[k] += v / len(negatives)

        raw: dict[str, float] = {}
        for k in _FACTOR_FUNCS:
            delta = pos_avgs[k] - neg_avgs[k]
            raw[k] = max(delta, 0.01)

        total = sum(raw.values())
        self.weights = {k: round(v / total, 4) for k, v in raw.items()}
        return dict(self.weights)

    def threshold_at_precision(self, target_precision: float, labeled_examples: list[tuple[str, bool]] | None = None) -> float:
        """Find score threshold achieving *target_precision* on labeled data.

        If no labeled data provided, returns a heuristic threshold.
        """
        if not labeled_examples:
            # Heuristic fallback
            return max(0.0, min(1.0, target_precision * 0.8))

        scored: list[tuple[float, bool]] = [
            (self.score(t).score, label) for t, label in labeled_examples
        ]
        scored.sort(key=lambda x: -x[0])

        best_threshold = 1.0
        tp = 0
        fp = 0
        for s, label in scored:
            if label:
                tp += 1
            else:
                fp += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            if precision >= target_precision:
                best_threshold = s
        return round(best_threshold, 4)


# ------------------------------------------------------------------
# Reporting helper
# ------------------------------------------------------------------

def format_confidence_report(scores: list[ConfidenceScore]) -> str:
    """Format a list of :class:`ConfidenceScore` objects into a human-readable report."""
    lines: list[str] = [f"Confidence Report ({len(scores)} items)", "=" * 40]
    for i, sc in enumerate(scores, 1):
        factors_str = ", ".join(sc.factors) if sc.factors else "none"
        lines.append(f"  [{i}] score={sc.score:.4f}  level={sc.level}  factors=[{factors_str}]")
    # Summary
    by_level: dict[str, int] = defaultdict(int)
    for sc in scores:
        by_level[sc.level] += 1
    lines.append("-" * 40)
    for lvl in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW, ConfidenceLevel.NONE):
        if by_level[lvl]:
            lines.append(f"  {lvl}: {by_level[lvl]}")
    return "\n".join(lines)
