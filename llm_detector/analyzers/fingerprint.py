"""Intrinsic fingerprint detection -- LLM-preferred vocabulary."""

import re

FINGERPRINT_WORDS = [
    # Original 27 (ChatGPT-3.5 era, established in v0.51)
    'delve', 'utilize', 'comprehensive', 'streamline', 'leverage', 'robust',
    'facilitate', 'innovative', 'synergy', 'paradigm', 'holistic', 'nuanced',
    'multifaceted', 'spearhead', 'underscore', 'pivotal', 'landscape',
    'cutting-edge', 'actionable', 'seamlessly', 'noteworthy', 'meticulous',
    'endeavor', 'paramount', 'aforementioned', 'furthermore', 'henceforth',
    # v0.63 additions (Kobak et al. 2024 excess vocabulary, Science Advances)
    'tapestry', 'realm', 'embark', 'foster', 'showcasing',
]

_FINGERPRINT_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in FINGERPRINT_WORDS) + r')\b',
    re.IGNORECASE
)


def run_fingerprint_full(text):
    """Detect LLM fingerprint words in a single pass.

    Returns (score, hit_count, rate, spans) where spans is a list of
    (start_char, end_char, matched_text, 'fingerprint', word) tuples.
    """
    word_count = len(text.split())
    spans = []
    for m in _FINGERPRINT_RE.finditer(text):
        spans.append((m.start(), m.end(), m.group(), 'fingerprint', m.group().lower()))
    hits = len(spans)
    rate = hits / max(word_count / 1000, 1)
    score = min(rate / 5.0, 1.0)
    return score, hits, rate, spans


def run_fingerprint(text):
    """Detect LLM fingerprint words. Returns (score, hit_count, rate)."""
    score, hits, rate, _ = run_fingerprint_full(text)
    return score, hits, rate


def run_fingerprint_spans(text):
    """Return character-level spans for fingerprint word hits.

    Returns list of (start_char, end_char, matched_text, 'fingerprint', word).
    """
    _, _, _, spans = run_fingerprint_full(text)
    return spans
