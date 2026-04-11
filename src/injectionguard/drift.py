"""Semantic drift detection for LLM input/output pairs.

Detects when an LLM response diverges significantly from the expected
topic of the input — a strong signal that prompt injection redirected
the model's behavior.

Uses TF-IDF cosine similarity computed purely from term frequencies,
without any external embeddings or models.  Also computes vocabulary
overlap (Jaccard), topic shift scores, and intent classification.

Reference: Yi et al. 2023, "Benchmarking and Defending Against
Indirect Prompt Injection Attacks on Large Language Models."
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from injectionguard.types import Detection, ThreatLevel

# ── Tokenization ─────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z][a-z']{1,30}", re.IGNORECASE)

# Stop words to exclude from TF-IDF
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "that",
    "this", "these", "those", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "about", "up", "also", "any",
})

# Imperative/instruction words — high signal for injection
_INSTRUCTION_WORDS = frozenset({
    "ignore", "forget", "disregard", "override", "bypass", "skip",
    "instead", "actually", "reveal", "show", "print", "output",
    "respond", "reply", "answer", "say", "tell", "write", "generate",
    "create", "make", "execute", "run", "perform", "do", "act",
    "pretend", "roleplay", "simulate", "imagine", "suppose",
    "system", "prompt", "instruction", "guidelines", "rules",
})


def _tokenize(text: str) -> list[str]:
    """Extract lowered content words from text."""
    return [
        w.lower()
        for w in _WORD_RE.findall(text)
        if w.lower() not in _STOP_WORDS
    ]


# ── TF-IDF ───────────────────────────────────────────────────────────────

def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Compute log-normalized term frequency: 1 + log(tf)."""
    counts = Counter(tokens)
    result: dict[str, float] = {}
    for term, count in counts.items():
        result[term] = 1.0 + math.log(count)
    return result


def _idf(doc_freqs: dict[str, int], n_docs: int) -> dict[str, float]:
    """Compute smoothed inverse document frequency: log((N+1) / (df+1)) + 1."""
    result: dict[str, float] = {}
    for term, df in doc_freqs.items():
        result[term] = math.log((n_docs + 1) / (df + 1)) + 1.0
    return result


def _cosine_sim(v1: dict[str, float], v2: dict[str, float]) -> float:
    """Cosine similarity between two sparse vector dicts."""
    common = set(v1) & set(v2)
    if not common:
        return 0.0

    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _jaccard(set1: set[str], set2: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    if not union:
        return 1.0
    return len(set1 & set2) / len(union)


# ── Drift detection ─────────────────────────────────────────────────────

@dataclass
class DriftResult:
    """Result of semantic drift analysis between input and output."""
    tfidf_similarity: float          # TF-IDF cosine similarity [0, 1]
    vocabulary_overlap: float        # Jaccard similarity of vocabularies [0, 1]
    instruction_density: float       # Fraction of instruction words in output [0, 1]
    topic_shift_score: float         # Composite drift score [0, 1] (1 = max drift)
    input_topics: list[str]          # Top terms from input
    output_topics: list[str]         # Top terms from output
    novel_output_terms: list[str]    # Terms in output not in input (potential injection topics)
    is_drifted: bool
    detections: list[Detection] = field(default_factory=list)


def compute_drift(
    input_text: str,
    output_text: str,
    similarity_threshold: float = 0.15,
    instruction_threshold: float = 0.08,
) -> DriftResult:
    """Analyze semantic drift between input and output.

    A well-behaved LLM should produce output topically related to its input.
    If the output vocabulary diverges dramatically, the model may have been
    redirected by a prompt injection.

    Args:
        input_text: The user's input (or system prompt + user input).
        output_text: The LLM's response.
        similarity_threshold: Below this TF-IDF similarity, flag as drifted.
        instruction_threshold: Above this instruction word density, flag.

    Returns:
        DriftResult with similarity metrics and detections.
    """
    input_tokens = _tokenize(input_text)
    output_tokens = _tokenize(output_text)

    # Handle edge cases
    if not input_tokens or not output_tokens:
        return DriftResult(
            tfidf_similarity=0.0 if (input_tokens or output_tokens) else 1.0,
            vocabulary_overlap=0.0 if (input_tokens or output_tokens) else 1.0,
            instruction_density=0.0,
            topic_shift_score=0.0,
            input_topics=[],
            output_topics=[],
            novel_output_terms=[],
            is_drifted=False,
            detections=[],
        )

    # TF-IDF similarity
    # Use both docs as the "corpus" for IDF
    input_set = set(input_tokens)
    output_set = set(output_tokens)
    all_terms = input_set | output_set
    doc_freqs: dict[str, int] = {}
    for term in all_terms:
        df = 0
        if term in input_set:
            df += 1
        if term in output_set:
            df += 1
        doc_freqs[term] = df

    idf_scores = _idf(doc_freqs, 2)

    input_tf = _term_freq(input_tokens)
    output_tf = _term_freq(output_tokens)

    input_tfidf = {t: input_tf[t] * idf_scores.get(t, 0.0) for t in input_tf}
    output_tfidf = {t: output_tf[t] * idf_scores.get(t, 0.0) for t in output_tf}

    tfidf_sim = _cosine_sim(input_tfidf, output_tfidf)
    vocab_overlap = _jaccard(input_set, output_set)

    # Instruction word density in output
    instruction_count = sum(1 for t in output_tokens if t in _INSTRUCTION_WORDS)
    instruction_density = instruction_count / len(output_tokens) if output_tokens else 0.0

    # Top terms (by TF)
    input_top = sorted(input_tf, key=input_tf.get, reverse=True)[:10]  # type: ignore[arg-type]
    output_top = sorted(output_tf, key=output_tf.get, reverse=True)[:10]  # type: ignore[arg-type]

    # Novel terms in output not present in input
    novel = [t for t in output_top if t not in input_set]

    # Composite drift score: weighted combination
    # High score = more drift = more suspicious
    topic_shift = (
        0.4 * (1.0 - tfidf_sim) +
        0.3 * (1.0 - vocab_overlap) +
        0.3 * min(1.0, instruction_density / max(instruction_threshold, 0.01))
    )
    topic_shift = max(0.0, min(1.0, topic_shift))

    is_drifted = tfidf_sim < similarity_threshold or instruction_density > instruction_threshold

    # Build detections
    detections: list[Detection] = []
    if tfidf_sim < similarity_threshold:
        if tfidf_sim < similarity_threshold * 0.5:
            level = ThreatLevel.HIGH
        elif tfidf_sim < similarity_threshold * 0.75:
            level = ThreatLevel.MEDIUM
        else:
            level = ThreatLevel.LOW
        detections.append(Detection(
            strategy="drift",
            pattern="low_tfidf_similarity",
            threat_level=level,
            message=(
                f"Output topic diverges from input "
                f"(TF-IDF sim={tfidf_sim:.3f}, threshold={similarity_threshold:.3f})"
            ),
            offset=0,
        ))

    if instruction_density > instruction_threshold:
        if instruction_density > instruction_threshold * 3:
            level = ThreatLevel.HIGH
        elif instruction_density > instruction_threshold * 2:
            level = ThreatLevel.MEDIUM
        else:
            level = ThreatLevel.LOW
        detections.append(Detection(
            strategy="drift",
            pattern="high_instruction_density",
            threat_level=level,
            message=(
                f"Output contains unusual instruction word density "
                f"({instruction_density:.3f}, threshold={instruction_threshold:.3f})"
            ),
            offset=0,
        ))

    return DriftResult(
        tfidf_similarity=tfidf_sim,
        vocabulary_overlap=vocab_overlap,
        instruction_density=instruction_density,
        topic_shift_score=topic_shift,
        input_topics=input_top,
        output_topics=output_top,
        novel_output_terms=novel,
        is_drifted=is_drifted,
        detections=detections,
    )


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str   # "user" or "assistant"
    text: str


def detect_conversation_drift(
    turns: list[ConversationTurn],
    window: int = 3,
    threshold: float = 0.10,
) -> list[DriftResult]:
    """Analyze drift across a multi-turn conversation.

    Compares each assistant response to the concatenation of the
    preceding `window` user messages.  A sudden drop in similarity
    suggests injection redirection mid-conversation.

    Args:
        turns: List of conversation turns.
        window: Number of preceding user turns to consider as context.
        threshold: Similarity threshold for drift detection.

    Returns:
        List of DriftResult objects, one per assistant turn.
    """
    results: list[DriftResult] = []
    user_history: list[str] = []

    for turn in turns:
        if turn.role == "user":
            user_history.append(turn.text)
        elif turn.role == "assistant" and user_history:
            # Take last `window` user messages as context
            context = " ".join(user_history[-window:])
            drift = compute_drift(context, turn.text, similarity_threshold=threshold)
            results.append(drift)

    return results
