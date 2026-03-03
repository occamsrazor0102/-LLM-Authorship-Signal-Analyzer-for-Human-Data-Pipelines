#!/usr/bin/env python3
"""
LLM-Generated Task Prompt Detection Pipeline v0.61
═══════════════════════════════════════════════════
Bundled release — Phases 1–5 in single file.

Changes from v0.60 — Phase 5: Semantic & Model-Based Detection

  • LAYER 2.8 — SEMANTIC RESONANCE: Cosine similarity of sentence embeddings
        (all-MiniLM-L6-v2) against AI/human archetype centroids. Catches
        semantically AI-like text even when specific keywords are avoided.
        Ref: Mitchell et al. (2023) "DetectGPT"
  • LOCAL PERPLEXITY: distilgpt2-based perplexity scoring. AI text has low
        perplexity (< 20) on small LMs; human text typically > 35.
        Ref: Gehrmann et al. (2019) "GLTR", Mitchell et al. (2023) "DetectGPT"
  • LAYER 3.1-LOCAL — DNA-GPT PROXY: Zero-LLM divergent continuation analysis.
        Fits a backoff n-gram LM on the prefix, samples K=32 surrogate
        continuations, and measures BScore overlap + NCD + internal self-
        similarity + conditional surprisal + TTR. Automatically activates
        when no API key is provided, so CH3:CONT is never empty.
        Ref: Yang et al. (2024) "DNA-GPT" (ICLR 2024)
        Ref: Li et al. (2004) "The Similarity Metric" (NCD theory)
  • FTFY NORMALIZATION: Robust encoding repair (mojibake, broken surrogates)
        via ftfy before NFKC/homoglyph passes. Handles "confusable" Unicode
        that standard regex misses.
        Ref: Robyn Speer (2019) "ftfy: Fixes Text For You"
  • PDF INPUT SUPPORT: New load_pdf() ingests PDF documents via pypdf.
        Each page → separate task; short pages auto-combined.
  • All v0.60 features retained (calibration, windowing, dual-mode, MIXED).

Layers / Channels:
  NORM      — Text normalization (ftfy, NFKC, homoglyph, whitespace)
  GATE      — Language/fairness support check
  CH1:PROMPT — L0, L2.5, L2.6(VSD/SSI), L2.7(IDI)  [task_prompt primary]
  CH2:STYLE  — L3.0(NSSI), L2.8(Semantic), PPL, L2   [generic_aigt primary]
  CH3:CONT   — L3.1(DNA-GPT) or L3.1-Local(proxy)   [both modes]
  CH4:WINDOW — Sentence-window scoring                [generic_aigt]

Usage:
  python llm_detector_v060.py --text "Text..." [--mode task_prompt|generic_aigt|auto]
  python llm_detector_v060.py <input.xlsx|.csv|.pdf> [--mode auto] [--sheet SHEET]
  python llm_detector_v060.py --text "Text..." --api-key sk-... --provider anthropic
  python llm_detector_v060.py <input.xlsx> --collect baselines.jsonl
  python llm_detector_v060.py --analyze-baselines baselines.jsonl [--baselines-csv out.csv]

Requirements:
  pip install openpyxl pandas
  Optional: pip install spacy                  (improves sentence segmentation)
  Optional: pip install anthropic              (enables Layer 3.1 with Anthropic API)
  Optional: pip install openai                 (enables Layer 3.1 with OpenAI API)
  Optional: pip install sentence-transformers scikit-learn  (enables Layer 2.8 Semantic)
  Optional: pip install transformers torch     (enables local perplexity scoring)
  Optional: pip install ftfy                   (enables robust Unicode normalization)
  Optional: pip install pypdf                  (enables PDF input)
"""

import re, os, argparse, json, math, statistics, threading, zlib, unicodedata
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ── API key: use --api-key CLI flag or env vars ─────────────────────────
# v0.60: Hardcoded key removed. Set OPENAI_API_KEY or ANTHROPIC_API_KEY
# in your environment, or pass --api-key on the command line.


# ── spaCy setup: lightweight sentencizer, no model download ──────────────────
try:
    import spacy
    from spacy.lang.en import English
    _nlp = English()
    _nlp.add_pipe("sentencizer")
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("INFO: spacy not installed. Sentence segmentation will use regex fallback.")
except Exception as e:
    HAS_SPACY = False
    print(f"INFO: spacy sentencizer setup failed ({e}). Using regex fallback.")


# ── ftfy: robust text encoding repair ────────────────────────────────────────
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# ── sentence-transformers: semantic vector analysis (Layer 2.8) ──────────────
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    import numpy as np
    _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')

    # Pre-computed centroids for known AI response styles
    _AI_ARCHETYPES = [
        "As an AI language model, I cannot provide personal opinions.",
        "Here is a comprehensive breakdown of the key factors to consider.",
        "To address this challenge, we must consider multiple perspectives.",
        "This thorough analysis demonstrates the critical importance of the topic.",
        "Furthermore, it is essential to note that this approach ensures alignment.",
        "In conclusion, by leveraging these strategies we can achieve optimal results.",
    ]
    _HUMAN_ARCHETYPES = [
        "honestly idk maybe try restarting it lol",
        "so I went ahead and just hacked together a quick script",
        "tbh the whole thing is kinda janky but it works",
        "yeah no that's totally wrong, here's what actually happened",
        "I messed around with it for a bit and got something working",
    ]
    _AI_CENTROIDS = _EMBEDDER.encode(_AI_ARCHETYPES)
    _HUMAN_CENTROIDS = _EMBEDDER.encode(_HUMAN_ARCHETYPES)
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
except Exception as e:
    HAS_SEMANTIC = False
    print(f"INFO: sentence-transformers setup failed ({e}). Semantic layer disabled.")

# ── transformers: local perplexity scoring ───────────────────────────────────
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch as _torch
    _PPL_MODEL_ID = 'distilgpt2'
    _PPL_MODEL = GPT2LMHeadModel.from_pretrained(_PPL_MODEL_ID)
    _PPL_TOKENIZER = GPT2TokenizerFast.from_pretrained(_PPL_MODEL_ID)
    _PPL_MODEL.eval()
    HAS_PERPLEXITY = True
except ImportError:
    HAS_PERPLEXITY = False
except Exception as e:
    HAS_PERPLEXITY = False
    print(f"INFO: transformers/torch setup failed ({e}). Perplexity scoring disabled.")

# ── pypdf: PDF text extraction ───────────────────────────────────────────────
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# ══════════════════════════════════════════════════════════════════════════════
# TEXT NORMALIZATION (v0.60)
# Pre-pass before all detection layers. Neutralizes cheap evasion attacks.
# Ref: RAID benchmark (Dugan et al. 2024) — formatting perturbations,
#      homoglyphs, and spacing attacks degrade metric-based detectors.
# Ref: MGTBench (He et al. 2023) — paraphrasing sensitivity in rule-based
#      detectors.
# ══════════════════════════════════════════════════════════════════════════════

# Common homoglyph mappings: visually similar Unicode → ASCII
# Only includes safe, unambiguous Latin-script lookalikes.
_HOMOGLYPH_MAP = str.maketrans({
    '\u0410': 'A', '\u0412': 'B', '\u0421': 'C', '\u0415': 'E',  # Cyrillic
    '\u041d': 'H', '\u041a': 'K', '\u041c': 'M', '\u041e': 'O',
    '\u0420': 'P', '\u0422': 'T', '\u0425': 'X',
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x',
    '\u2018': "'", '\u2019': "'", '\u201a': "'",                    # Smart quotes
    '\u201c': '"', '\u201d': '"', '\u201e': '"',
    '\u2032': "'", '\u2033': '"',                                    # Primes
    '\u2014': '--', '\u2013': '-', '\u2012': '-',                   # Dashes
    '\u2026': '...', '\u22ef': '...',                               # Ellipses
    '\uff01': '!', '\uff1f': '?', '\uff0c': ',', '\uff0e': '.',   # Fullwidth
    '\uff1a': ':', '\uff1b': ';',
})

# Zero-width and invisible characters to strip
_INVISIBLE_RE = re.compile(
    '[\u200b\u200c\u200d\u200e\u200f'   # Zero-width space/joiners/marks
    '\u2060\u2061\u2062\u2063\u2064'     # Word joiner, invisible operators
    '\ufeff'                              # BOM / zero-width no-break space
    '\u00ad'                              # Soft hyphen
    '\u034f'                              # Combining grapheme joiner
    '\u180e'                              # Mongolian vowel separator
    '\u2028\u2029'                        # Line/paragraph separators
    ']'
)

# Inter-character spacing: "l i k e  t h i s"
_INTERSPACED_RE = re.compile(r'(?<!\w)(\w) (\w) (\w) (\w)(?!\w)')


def normalize_text(text):
    """
    Normalize text to neutralize common evasion attacks.
    Returns (normalized_text, delta_report).

    delta_report is a dict with:
      - obfuscation_delta: float, fraction of characters changed (0.0–1.0)
      - invisible_chars: int, count of stripped invisible characters
      - homoglyphs: int, count of homoglyph substitutions
      - interspacing_spans: int, count of inter-character spacing regions
      - whitespace_collapsed: bool, whether excess whitespace was removed
      - ftfy_applied: bool, whether ftfy encoding repair was applied
    """
    original = text
    original_len = max(len(text), 1)
    changes = 0
    ftfy_applied = False

    # 0. ftfy encoding repair (mojibake, broken surrogates, etc.)
    #    Runs before all other steps to fix encoding-level corruption.
    if HAS_FTFY:
        pre_ftfy = text
        text = ftfy.fix_text(text)
        ftfy_changes = sum(1 for a, b in zip(pre_ftfy, text) if a != b)
        ftfy_changes += abs(len(pre_ftfy) - len(text))
        changes += ftfy_changes
        ftfy_applied = ftfy_changes > 0

    # 1. Strip invisible/zero-width characters
    invisible_count = len(_INVISIBLE_RE.findall(text))
    text = _INVISIBLE_RE.sub('', text)
    changes += invisible_count

    # 2. NFKC normalization (compatibility decomposition + canonical composition)
    #    Normalizes fullwidth chars, ligatures, superscripts, etc.
    pre_nfkc = text
    text = unicodedata.normalize('NFKC', text)
    nfkc_changes = sum(1 for a, b in zip(pre_nfkc, text) if a != b)
    changes += nfkc_changes

    # 3. Homoglyph folding
    pre_homoglyph = text
    text = text.translate(_HOMOGLYPH_MAP)
    homoglyph_count = sum(1 for a, b in zip(pre_homoglyph, text) if a != b)
    changes += homoglyph_count

    # 4. Inter-character spacing collapse: "l i k e" → "like"
    interspacing_spans = len(_INTERSPACED_RE.findall(text))
    if interspacing_spans > 0:
        # Collapse runs of single-char-space-single-char
        def _collapse_interspaced(m):
            # Find the full span of interspaced characters
            return m.group(0).replace(' ', '')
        # Iteratively collapse (greedy)
        prev = None
        while prev != text:
            prev = text
            text = re.sub(r'(?<!\w)(\w) (?=\w(?:\s\w)*(?!\w))', r'\1', text)
        spacing_changes = len(original) - len(text)  # approximate
        changes += max(spacing_changes, 0)

    # 5. Whitespace collapse (multiple spaces → single, strip leading/trailing)
    pre_ws = text
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    ws_collapsed = (pre_ws != text)
    if ws_collapsed:
        changes += abs(len(pre_ws) - len(text))

    # 6. Control character stripping (C0/C1 except \n \r \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    obfuscation_delta = changes / original_len

    return text, {
        'obfuscation_delta': round(obfuscation_delta, 4),
        'invisible_chars': invisible_count,
        'homoglyphs': homoglyph_count,
        'interspacing_spans': interspacing_spans,
        'whitespace_collapsed': ws_collapsed,
        'ftfy_applied': ftfy_applied,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FAIRNESS / LANGUAGE SUPPORT GATES (v0.60)
# Ref: Liang et al. (2023) "GPT Detectors Are Biased Against Non-Native
#      English Writers" — detectors disproportionately flag non-native text.
# Ref: Wang et al. (2023) "M4 — multilingual detection remains harder."
# ══════════════════════════════════════════════════════════════════════════════

# Top-50 English function words (closed class, highly stable across registers)
_ENGLISH_FUNCTION_WORDS = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'am', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'shall', 'should', 'may', 'might', 'can', 'could', 'must',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'about', 'between', 'through', 'after', 'before',
    'and', 'or', 'but', 'not', 'if', 'that', 'this', 'it', 'he', 'she',
    'they', 'we', 'i', 'you', 'my', 'your', 'his', 'her', 'its', 'our',
    'their', 'who', 'which', 'what', 'there',
])


def check_language_support(text, word_count=None):
    """
    Assess whether text is within the validated English-prose envelope.

    Returns dict:
      - support_level: 'SUPPORTED' | 'REVIEW' | 'UNSUPPORTED'
      - function_word_coverage: float (0.0–1.0)
      - non_latin_ratio: float (0.0–1.0)
      - reason: str
    """
    words = text.lower().split()
    if word_count is None:
        word_count = len(words)

    if word_count < 30:
        return {
            'support_level': 'REVIEW',
            'function_word_coverage': 0.0,
            'non_latin_ratio': 0.0,
            'reason': 'Text too short for reliable English detection',
        }

    # Function-word coverage: what fraction of tokens are English function words
    fw_count = sum(1 for w in words if w in _ENGLISH_FUNCTION_WORDS)
    fw_coverage = fw_count / max(word_count, 1)

    # Script analysis: what fraction of alphabetic chars are non-Latin
    alpha_chars = [c for c in text if c.isalpha()]
    n_alpha = max(len(alpha_chars), 1)
    non_latin = sum(1 for c in alpha_chars
                    if unicodedata.category(c).startswith('L')
                    and not ('\u0041' <= c <= '\u007a' or '\u00c0' <= c <= '\u024f'))
    non_latin_ratio = non_latin / n_alpha

    # Decision logic
    if non_latin_ratio > 0.30:
        level = 'UNSUPPORTED'
        reason = f'High non-Latin script content ({non_latin_ratio:.0%})'
    elif fw_coverage < 0.08:
        level = 'UNSUPPORTED'
        reason = f'Very low English function-word coverage ({fw_coverage:.0%})'
    elif fw_coverage < 0.12:
        level = 'REVIEW'
        reason = f'Low English function-word coverage ({fw_coverage:.0%}) — possible non-native or non-English text'
    elif non_latin_ratio > 0.10:
        level = 'REVIEW'
        reason = f'Mixed-script content ({non_latin_ratio:.0%} non-Latin)'
    else:
        level = 'SUPPORTED'
        reason = 'Text within validated English-prose envelope'

    return {
        'support_level': level,
        'function_word_coverage': round(fw_coverage, 4),
        'non_latin_ratio': round(non_latin_ratio, 4),
        'reason': reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3.0 — N-GRAM SELF-SIMILARITY INDEX (NSSI)
# Offline statistical fingerprinting of LLM expository text
# v0.60: Expanded to 12-signal convergence with information-theoretic signals.
# v0.60: causal_deficit (s9) weight halved (supporting-only demotion).
# ══════════════════════════════════════════════════════════════════════════════

# ── Feature: Formulaic Academic Phrases ──────────────────────────────────────
_NSSI_FORMULAIC_PATTERNS = [
    (r'\bthis\s+(?:report|analysis|paper|study|section|document)\s+(?:provides?|presents?|examines?|dissects?|identifies?|evaluates?|proposes?|outlines?)\b', 1.5),
    (r'\b(?:it\s+is\s+(?:worth|important|imperative|crucial|essential|critical)\s+(?:noting|to\s+note|to\s+acknowledge|to\s+emphasize|to\s+recognize))\b', 2.0),
    (r'\b(?:to\s+address\s+this\s+(?:gap|issue|problem|challenge|limitation|deficiency|concern|shortcoming))\b', 1.5),
    (r'\b(?:perhaps\s+the\s+most\s+(?:\w+\s+)?(?:damning|significant|important|critical|notable|striking|concerning|alarming))\b', 2.0),
    (r'\b(?:(?:while|although)\s+(?:theoretically|conceptually|technically)\s+(?:sound|elegant|promising|valid|robust|appealing))\b', 2.0),
    (r'\b(?:the\s+(?:analysis|evidence|data|results?|findings?)\s+(?:suggests?|reveals?|indicates?|shows?|demonstrates?|confirms?)\s+that)\b', 1.0),
    (r'\b(?:this\s+(?:creates?|represents?|highlights?|underscores?|reveals?|illustrates?|exemplifies?)\s+(?:a|the|an))\b', 1.5),
    (r'\b(?:the\s+(?:primary|core|fundamental|critical|key|central|overarching)\s+(?:challenge|issue|problem|question|limitation|concern|insight|takeaway))\b', 1.0),
    (r'\b(?:in\s+(?:layman.s\s+terms|other\s+words|practical\s+terms|simple\s+terms|real.world\s+(?:terms|scenarios|situations)))\b', 1.5),
    (r'\b(?:defense\s+in\s+depth)\b', 1.0),
    (r'\b(?:arms?\s+race)\b', 0.5),
    (r'\b(?:the\s+(?:era|age|dawn)\s+of)\b', 0.5),
    (r'\b(?:a\s+(?:paradigm|fundamental|seismic|tectonic)\s+shift)\b', 2.0),
    (r'\b(?:the\s+(?:elephant|gorilla)\s+in\s+the\s+room)\b', 1.5),
    (r'\b(?:a\s+double.edged\s+sword)\b', 1.5),
    (r'\b(?:in\s+(?:conclusion|summary|closing),?)\b', 0.5),
    (r'\b(?:the\s+path\s+forward\s+(?:is|requires|demands|involves))\b', 1.5),
    (r'\b(?:(?:unless|until)\s+the\s+(?:community|industry|field|sector)\s+(?:adopts?|embraces?|commits?))\b', 2.0),
    (r'\b(?:the\s+(?:immediate|long.term|strategic)\s+(?:future|imperative|priority|solution)\s+(?:belongs?\s+to|lies?\s+in|requires?))\b', 2.0),
]

# ── Feature: Power Adjectives ────────────────────────────────────────────────
_NSSI_POWER_ADJ = re.compile(
    r'\b(?:comprehensive|exhaustive|rigorous|robust|holistic|systemic|'
    r'fundamental|critical|profound|decisive|catastrophic|perilous|'
    r'unprecedented|groundbreaking|transformative|paradigmatic|'
    r'monumental|pivotal|seminal|nascent|burgeoning|'
    r'overarching|multifaceted|nuanced|granular|bespoke|'
    r'actionable|scalable|tractable|non-trivial|intractable)\b',
    re.I
)

# ── Feature: Discourse Scaffolding ───────────────────────────────────────────
_NSSI_SCARE_QUOTE = re.compile(r'[\u201c\u201d][^\u201c\u201d]{2,40}[\u201c\u201d]|"[^"]{2,40}"')
_NSSI_EM_DASH = re.compile(r'\u2014|--')
_NSSI_PAREN = re.compile(r'\([^)]{12,}\)')
_NSSI_COLON_EXPLAIN = re.compile(r':\s+[A-Z]')

# ── Feature: Demonstrative Monotony ──────────────────────────────────────────
_NSSI_DEMONSTRATIVE = re.compile(
    r'\bthis\s+(?:approach|method|framework|analysis|issue|mechanism|assumption|'
    r'limitation|strategy|technique|variant|disparity|metric|paradigm|'
    r'architecture|pipeline|deficiency|vulnerability|solution|concept|'
    r'pattern|signal|feature|constraint|observation|phenomenon|'
    r'suggests?|indicates?|creates?|ensures?|effectively|underscores?|'
    r'highlights?|represents?|reveals?|demonstrates?|means?|implies?|'
    r'raises?|poses?|necessitates?)\b', re.I
)

# ── Feature: Transition Connector Density ────────────────────────────────────
_NSSI_TRANSITION = re.compile(
    r'\b(?:however|furthermore|consequently|moreover|nevertheless|'
    r'additionally|specifically|crucially|ultimately|conversely|'
    r'notably|importantly|interestingly|remarkably|significantly|'
    r'simultaneously|correspondingly|paradoxically)\b', re.I
)

# ── Feature: Causal Reasoning Deficit ────────────────────────────────────────
_NSSI_CAUSAL = re.compile(
    r'\b(?:because|since|'
    r'so\b(?!\s+(?:that|much|many|far|long|called))|'
    r'if|but|although|though|unless|whereas|'
    r'while(?=\s+\w+\s+(?:is|was|are|were|has|had|do|does|did|can|could|would|should|might|may))|'
    r'therefore|hence|thus|'
    r'think|believe|feel|know|suspect|doubt|wonder|guess|suppose|reckon|'
    r'maybe|perhaps|probably|apparently|presumably)\b', re.I
)


def _nssi_get_sentences(text):
    """Split text into sentences (regex fallback for NSSI)."""
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def run_layer30(text):
    """
    Layer 3.0: N-Gram Self-Similarity Index (NSSI).

    Detects LLM-generated expository text via statistical convergence
    of formulaic phrases, power adjectives, discourse scaffolding,
    demonstrative monotony, causal reasoning deficit, information-theoretic
    entropy measures, and structural regimentation.

    v0.60: Fires on 12-signal convergence (was 9).
    New signals: s10 (burstiness/sent_cv), s11 (zlib compression ratio),
                 s12 (hapax legomena deficit).

    Returns dict with individual feature scores and composite NSSI.
    """
    words = text.split()
    word_count = len(words)
    sentences = _nssi_get_sentences(text)
    n_sents = max(len(sentences), 1)

    # ── Minimum text gate ────────────────────────────────────────────────
    if word_count < 200:
        return {
            'nssi_score': 0.0, 'nssi_signals': 0, 'nssi_active': [],
            'determination': None, 'confidence': 0.0,
            'reason': 'NSSI: text too short for analysis',
            'formulaic_density': 0.0, 'formulaic_weighted': 0.0,
            'power_adj_density': 0.0, 'scare_quote_density': 0.0,
            'emdash_density': 0.0, 'parenthetical_density': 0.0,
            'colon_density': 0.0, 'demonstrative_density': 0.0,
            'transition_density': 0.0, 'causal_density': 0.0,
            'causal_ratio': 0.0, 'this_the_start_rate': 0.0,
            'section_depth': 0, 'sent_length_cv': 0.0,
            'comp_ratio': 0.0, 'hapax_ratio': 0.0,
            'hapax_count': 0, 'unique_words': 0,
            'word_count': word_count, 'sentence_count': n_sents,
        }

    # ── 1. Formulaic phrase density (weighted) ───────────────────────────
    formulaic_raw = 0
    formulaic_weighted = 0.0
    for pattern, weight in _NSSI_FORMULAIC_PATTERNS:
        hits = len(re.findall(pattern, text, re.I))
        formulaic_raw += hits
        formulaic_weighted += hits * weight
    formulaic_density = formulaic_raw / n_sents
    formulaic_w_density = formulaic_weighted / n_sents

    # ── 2. Power adjective density ───────────────────────────────────────
    power_hits = len(_NSSI_POWER_ADJ.findall(text))
    power_density = power_hits / n_sents

    # ── 3. Discourse scaffolding ─────────────────────────────────────────
    scare_quotes = len(_NSSI_SCARE_QUOTE.findall(text))
    emdashes = len(_NSSI_EM_DASH.findall(text))
    parentheticals = len(_NSSI_PAREN.findall(text))
    colon_explains = len(_NSSI_COLON_EXPLAIN.findall(text))

    scare_density = scare_quotes / n_sents
    emdash_density = emdashes / n_sents
    paren_density = parentheticals / n_sents
    colon_density = colon_explains / n_sents

    # ── 4. Demonstrative monotony ────────────────────────────────────────
    demo_hits = len(_NSSI_DEMONSTRATIVE.findall(text))
    demo_density = demo_hits / n_sents

    # ── 5. Transition connector density ──────────────────────────────────
    trans_hits = len(_NSSI_TRANSITION.findall(text))
    trans_density = trans_hits / n_sents

    # ── 5b. Causal reasoning deficit ─────────────────────────────────────
    causal_hits = len(_NSSI_CAUSAL.findall(text))
    causal_density = causal_hits / n_sents
    causal_ratio = (trans_hits + 1) / (causal_hits + 1)

    # ── 6. Sentence-start monotony ───────────────────────────────────────
    starts = [s.split()[0].lower() for s in sentences if s.split()]
    this_the_starts = sum(1 for s in starts if s in ('this', 'the', 'these', 'those'))
    this_the_rate = this_the_starts / n_sents

    # ── 7. Section hierarchy depth ───────────────────────────────────────
    headers = re.findall(r'^(\d+(?:\.\d+)*)\s+', text, re.M)
    section_depth = max((h.count('.') + 1 for h in headers), default=0)

    # ── 8. Sentence length coefficient of variation ──────────────────────
    sent_lens = [len(s.split()) for s in sentences]
    if len(sent_lens) > 2:
        sent_cv = statistics.stdev(sent_lens) / max(statistics.mean(sent_lens), 1)
    else:
        sent_cv = 0.5

    # ── COMPOSITE NSSI SCORE (convergence of 12 signals) ─────────────────
    signals = []

    # s1: Formulaic phrase density
    s1 = min(formulaic_w_density / 0.25, 1.0) if formulaic_w_density >= 0.04 else 0.0
    if s1 > 0: signals.append(('formulaic', s1))

    # s2: Power adjective saturation
    s2 = min(power_density / 0.30, 1.0) if power_density >= 0.08 else 0.0
    if s2 > 0: signals.append(('power_adj', s2))

    # s3: Scare quote density
    s3 = min(scare_density / 0.40, 1.0) if scare_density >= 0.08 else 0.0
    if s3 > 0: signals.append(('scare_quotes', s3))

    # s4: Demonstrative monotony
    s4 = min(demo_density / 0.12, 1.0) if demo_density >= 0.03 else 0.0
    if s4 > 0: signals.append(('demonstratives', s4))

    # s5: Transition connector density
    s5 = min(trans_density / 0.20, 1.0) if trans_density >= 0.05 else 0.0
    if s5 > 0: signals.append(('transitions', s5))

    # s6: Discourse scaffolding (em-dashes + parentheticals + colons)
    scaffold = emdash_density + paren_density + colon_density
    s6 = min(scaffold / 0.60, 1.0) if scaffold >= 0.15 else 0.0
    if s6 > 0: signals.append(('scaffolding', s6))

    # s7: Sentence-start monotony (this/the)
    s7 = min(this_the_rate / 0.35, 1.0) if this_the_rate >= 0.20 else 0.0
    if s7 > 0: signals.append(('start_monotony', s7))

    # s8: Deep section hierarchy
    s8 = min(section_depth / 4.0, 1.0) if section_depth >= 3 else 0.0
    if s8 > 0: signals.append(('hierarchy', s8))

    # s9: Causal reasoning deficit
    # v0.60: Weight halved (supporting-only demotion). Anchored to non-primary
    # source (Faculty Focus blog post). Retain for convergence contribution
    # but prevent it from being a strong individual driver.
    s9 = 0.0
    if trans_hits >= 2 and causal_ratio >= 1.5:
        s9 = min((causal_ratio - 1.0) / 3.0, 1.0) * 0.5  # v0.60: halved
    if s9 > 0: signals.append(('causal_deficit', s9))

    # ── v0.60: New information-theoretic signals ─────────────────────────

    # s10: Operationalized Burstiness (fixing the dead sent_cv code)
    # Human CV is typically > 0.45; AI tends toward high uniformity (< 0.35).
    # Gate: need >= 4 sentences for CV to be statistically meaningful.
    s10 = 0.0
    if n_sents >= 4 and sent_cv <= 0.35:
        s10 = min((0.35 - sent_cv) / 0.15, 1.0)
    if s10 > 0: signals.append(('low_burstiness', s10))

    # s11: Zlib Compression Entropy (zero-model perplexity proxy)
    # AI text is statistically predictable -> compresses better than human text.
    # Ref: Jiang et al. (2023) "Low-Resource Text Classification with Compressors"
    text_bytes = text.encode('utf-8')
    original_len = max(len(text_bytes), 1)
    compressed_len = len(zlib.compress(text_bytes))
    comp_ratio = compressed_len / original_len

    s11 = 0.0
    if comp_ratio <= 0.42 and word_count >= 150:
        s11 = min((0.42 - comp_ratio) / 0.08, 1.0)
    if s11 > 0: signals.append(('high_compressibility', s11))

    # s12: Hapax Legomena (Zipfian lexical sparsity)
    # AI models truncate the vocabulary long tail -> fewer single-occurrence words.
    # Ref: Crothers et al. (2023) "Machine vs Human Authorship: A Lexical Analysis"
    clean_words = [w.strip('.,!?"\'():;').lower() for w in words]
    clean_words = [w for w in clean_words if w]
    word_freqs = Counter(clean_words)
    hapax_count = sum(1 for count in word_freqs.values() if count == 1)
    unique_words = len(word_freqs)
    hapax_ratio = hapax_count / unique_words if unique_words > 0 else 0.0

    s12 = 0.0
    if hapax_ratio <= 0.45 and word_count >= 150:
        s12 = min((0.45 - hapax_ratio) / 0.15, 1.0)
    if s12 > 0: signals.append(('hapax_deficit', s12))

    # ── Convergence scoring (v0.60: recalibrated for 12 signals) ─────────
    n_active = len(signals)
    if n_active == 0:
        nssi_score = 0.0
    else:
        mean_strength = sum(s for _, s in signals) / n_active
        # v0.60: denominator raised from 4.0 to 5.5 to normalize across 12 signals
        convergence = min(n_active / 5.5, 1.0)
        nssi_score = mean_strength * convergence
        # v0.60: bonus threshold raised from 6 to 8 signals
        if n_active >= 8:
            nssi_score = min(nssi_score * 1.3, 1.0)

    # ── Determination (v0.60: thresholds raised for 12-signal pool) ──────
    if nssi_score >= 0.70 and n_active >= 7:        # was: >= 5
        det = 'RED'
        conf = min(0.85, nssi_score)
        reason = f"NSSI convergence (score={nssi_score:.2f}, {n_active} signals)"
    elif nssi_score >= 0.45 and n_active >= 5:      # was: >= 4
        det = 'AMBER'
        conf = min(0.65, nssi_score)
        reason = f"Elevated NSSI (score={nssi_score:.2f}, {n_active} signals)"
    elif nssi_score >= 0.25 and n_active >= 4:      # was: >= 3
        det = 'YELLOW'
        conf = min(0.40, nssi_score)
        reason = f"Moderate NSSI (score={nssi_score:.2f}, {n_active} signals)"
    else:
        det = None
        conf = 0.0
        reason = 'NSSI: insufficient signal convergence'

    return {
        'nssi_score': round(nssi_score, 4), 'nssi_signals': n_active,
        'nssi_active': [(name, round(val, 3)) for name, val in signals],
        'determination': det, 'confidence': round(conf, 4), 'reason': reason,
        'formulaic_density': round(formulaic_density, 4),
        'formulaic_weighted': round(formulaic_w_density, 4),
        'power_adj_density': round(power_density, 4),
        'scare_quote_density': round(scare_density, 4),
        'emdash_density': round(emdash_density, 4),
        'parenthetical_density': round(paren_density, 4),
        'colon_density': round(colon_density, 4),
        'demonstrative_density': round(demo_density, 4),
        'transition_density': round(trans_density, 4),
        'causal_density': round(causal_density, 4),
        'causal_ratio': round(causal_ratio, 4),
        'this_the_start_rate': round(this_the_rate, 4),
        'section_depth': section_depth,
        'sent_length_cv': round(sent_cv, 4),
        # v0.60: new signal diagnostics
        'comp_ratio': round(comp_ratio, 4),
        'hapax_ratio': round(hapax_ratio, 4),
        'hapax_count': hapax_count,
        'unique_words': unique_words,
        'word_count': word_count, 'sentence_count': n_sents,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3.1 — DNA-GPT DIVERGENT CONTINUATION ANALYSIS
# Online detection via LLM API (Anthropic or OpenAI)
# ══════════════════════════════════════════════════════════════════════════════

def _dna_ngrams(tokens, n):
    """Generate n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _dna_bscore(original_tokens, regenerated_tokens, ns=(2, 3, 4), weights=(0.25, 0.50, 0.25)):
    """Compute DNA-GPT BScore: weighted n-gram overlap."""
    scores = []
    for n, w in zip(ns, weights):
        orig_ng = set(_dna_ngrams(original_tokens, n))
        regen_ng = set(_dna_ngrams(regenerated_tokens, n))
        if not orig_ng or not regen_ng:
            scores.append(0.0)
            continue
        overlap = len(orig_ng & regen_ng)
        precision = overlap / len(regen_ng) if regen_ng else 0
        recall = overlap / len(orig_ng) if orig_ng else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        scores.append(f1 * w)
    return sum(scores)


def _dna_truncate_text(text, ratio=0.5):
    """Truncate text at sentence boundary. Returns (prefix, continuation)."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 4:
        words = text.split()
        mid = int(len(words) * ratio)
        return ' '.join(words[:mid]), ' '.join(words[mid:])
    cut = max(2, int(len(sentences) * ratio))
    return ' '.join(sentences[:cut]), ' '.join(sentences[cut:])


def _dna_call_anthropic(prefix, continuation_length, api_key,
                        model='claude-sonnet-4-20250514', n_samples=3, temperature=0.7):
    """Generate continuations using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic  (required for Layer 3.1 with Anthropic)")
    client = anthropic.Anthropic(api_key=api_key)
    continuations = []
    max_tokens = min(max(continuation_length * 2, 200), 4096)
    for _ in range(n_samples):
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user",
                       "content": f"Continue the following text naturally, maintaining the same style, tone, and topic. Do not add any preamble or meta-commentary — just continue writing:\n\n{prefix}"}]
        )
        continuations.append(msg.content[0].text if msg.content else "")
    return continuations


def _dna_call_openai(prefix, continuation_length, api_key,
                     model='gpt-4o-mini', n_samples=3, temperature=0.7):
    """Generate continuations using OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai  (required for Layer 3.1 with OpenAI)")
    client = openai.OpenAI(api_key=api_key)
    continuations = []
    max_tokens = min(max(continuation_length * 2, 200), 4096)
    for _ in range(n_samples):
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user",
                       "content": f"Continue the following text naturally, maintaining the same style, tone, and topic. Do not add any preamble or meta-commentary — just continue writing:\n\n{prefix}"}]
        )
        continuations.append(resp.choices[0].message.content if resp.choices else "")
    return continuations


def run_layer31(text, api_key=None, provider='anthropic', model=None,
                truncation_ratio=0.5, n_samples=3, temperature=0.7):
    """
    Layer 3.1: DNA-GPT Divergent Continuation Analysis.
    Truncates candidate text, regenerates continuations via LLM API,
    measures n-gram overlap (BScore) between original and regenerated.
    """
    word_count = len(text.split())

    if word_count < 150:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: insufficient text',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    if not api_key:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: no API key provided',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    prefix, original_continuation = _dna_truncate_text(text, truncation_ratio)
    if len(original_continuation.split()) < 30:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: continuation too short after truncation',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    orig_tokens = original_continuation.lower().split()
    continuation_word_count = len(orig_tokens)

    if model is None:
        model = 'claude-sonnet-4-20250514' if provider == 'anthropic' else 'gpt-4o-mini'

    try:
        if provider == 'anthropic':
            continuations = _dna_call_anthropic(prefix, continuation_word_count, api_key,
                                                model, n_samples, temperature)
        elif provider == 'openai':
            continuations = _dna_call_openai(prefix, continuation_word_count, api_key,
                                             model, n_samples, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")
    except Exception as e:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': f'DNA-GPT: API call failed ({e})',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    sample_scores = []
    for regen_text in continuations:
        regen_tokens = regen_text.lower().split()
        if len(regen_tokens) < 10:
            continue
        regen_tokens = regen_tokens[:int(len(orig_tokens) * 1.5)]
        bs = _dna_bscore(orig_tokens, regen_tokens)
        sample_scores.append(round(bs, 4))

    if not sample_scores:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: all regenerations failed or too short',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    bscore = statistics.mean(sample_scores)
    bscore_max = max(sample_scores)

    if bscore >= 0.20 and bscore_max >= 0.22:
        det, conf = 'RED', min(0.90, 0.60 + bscore)
        reason = f"DNA-GPT: high continuation overlap (BScore={bscore:.3f}, max={bscore_max:.3f})"
    elif bscore >= 0.12:
        det, conf = 'AMBER', min(0.70, 0.40 + bscore)
        reason = f"DNA-GPT: elevated continuation overlap (BScore={bscore:.3f})"
    elif bscore >= 0.08:
        det, conf = 'YELLOW', min(0.40, 0.20 + bscore)
        reason = f"DNA-GPT: moderate continuation overlap (BScore={bscore:.3f})"
    else:
        det, conf = 'GREEN', max(0.0, 0.10 - bscore)
        reason = f"DNA-GPT: low continuation overlap (BScore={bscore:.3f}) — likely human"

    return {
        'bscore': round(bscore, 4), 'bscore_max': round(bscore_max, 4),
        'bscore_samples': sample_scores, 'determination': det,
        'confidence': round(conf, 4), 'reason': reason,
        'n_samples': len(sample_scores), 'truncation_ratio': truncation_ratio,
        'continuation_words': continuation_word_count, 'word_count': word_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3.1-LOCAL: DNA-GPT PROXY (v0.61)
# Zero-LLM divergent continuation analysis using a backoff n-gram language
# model as surrogate for the generative model.
#
# Preserves the core DNA-GPT scoring mechanic (split→regenerate→overlap) but
# replaces the LLM regeneration with K samples from a classical n-gram LM
# fitted on the prefix.  Additional proxy features:
#   - NCD (Normalized Compression Distance): zlib-based predictability
#   - Internal n-gram self-similarity: first-half → second-half echo
#   - Conditional surprisal: how surprising is the suffix given the prefix?
#   - Repeated n-gram rate: monotonicity signal
#
# Ref: Yang et al. (2024) "DNA-GPT" (ICLR 2024)
# Ref: Li et al. (2004) "The Similarity Metric" (NCD theory)
# ══════════════════════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r'\w+|[^\w\s]')


def _proxy_tokenize(text):
    """Tokenize for n-gram LM. Returns lowercased word/punct tokens."""
    return _TOKEN_RE.findall(text.lower())


class _BackoffNGramLM:
    """Simple backoff n-gram language model for DNA-GPT proxy regeneration.

    Fits on a text corpus (typically just the prefix), then samples
    continuations or scores token log-probabilities.
    """

    def __init__(self, order=5, alpha=0.1):
        self.order = max(order, 1)
        self.alpha = alpha  # Laplace smoothing
        self.tables = [defaultdict(Counter) for _ in range(self.order)]
        self.vocab = set()

    def fit(self, texts):
        """Train on an iterable of text strings."""
        bos = ['<s>'] * (self.order - 1)
        for text in texts:
            toks = bos + _proxy_tokenize(text) + ['</s>']
            self.vocab.update(toks)
            for i in range(self.order - 1, len(toks)):
                for ctx_len in range(self.order):
                    ctx = tuple(toks[i - ctx_len:i]) if ctx_len else ()
                    self.tables[ctx_len][ctx][toks[i]] += 1

    def _counts(self, context):
        """Get counts for context with backoff."""
        max_ctx = min(len(context), self.order - 1)
        for ctx_len in range(max_ctx, -1, -1):
            ctx = tuple(context[-ctx_len:]) if ctx_len else ()
            counts = self.tables[ctx_len].get(ctx)
            if counts:
                return counts
        # Last-resort uniform fallback
        return Counter({t: 1 for t in self.vocab}) if self.vocab else Counter({'</s>': 1})

    def sample_next(self, context):
        """Sample a single next token given context."""
        import random as _random
        counts = self._counts(context)
        items = list(counts.items())
        total = sum(c for _, c in items)
        r = _random.random() * total
        acc = 0.0
        for tok, c in items:
            acc += c
            if acc >= r:
                return tok
        return items[-1][0]

    def logprob(self, token, context):
        """Log-probability of token given context (with Laplace smoothing)."""
        counts = self._counts(context)
        total = sum(counts.values())
        vocab_size = max(len(self.vocab), 1)
        p = (counts.get(token, 0) + self.alpha) / (total + self.alpha * vocab_size)
        return math.log(p)

    def sample_suffix(self, prefix_tokens, length):
        """Generate a continuation of `length` tokens from prefix context."""
        ctx = ['<s>'] * (self.order - 1) + list(prefix_tokens)
        out = []
        for _ in range(length):
            tok = self.sample_next(ctx)
            if tok == '</s>':
                break
            out.append(tok)
            ctx.append(tok)
        return out


def _calculate_ncd(prefix, suffix):
    """Normalized Compression Distance between prefix and suffix.

    Low NCD = highly predictable continuation (AI signal).
    High NCD = divergent continuation (human signal).
    """
    x = prefix.encode('utf-8')
    y = suffix.encode('utf-8')
    xy = x + b' ' + y

    c_x = len(zlib.compress(x))
    c_y = len(zlib.compress(y))
    c_xy = len(zlib.compress(xy))

    denom = max(c_x, c_y)
    if denom == 0:
        return 0.0
    return (c_xy - min(c_x, c_y)) / denom


def _internal_ngram_overlap(prefix_tokens, suffix_tokens, ns=(3, 4)):
    """Fraction of suffix n-grams that appear in prefix (echo effect).

    High overlap = AI-like self-consistency.
    Low overlap = human-like semantic drift.
    """
    if not suffix_tokens:
        return 0.0

    total_weight = 0.0
    weighted_overlap = 0.0

    for n in ns:
        pfx_ng = set(_dna_ngrams(prefix_tokens, n))
        sfx_ng = set(_dna_ngrams(suffix_tokens, n))
        if not sfx_ng:
            continue
        w = n * math.log(n) if n > 1 else 1.0
        overlap = len(pfx_ng & sfx_ng) / len(sfx_ng)
        weighted_overlap += w * overlap
        total_weight += w

    return weighted_overlap / total_weight if total_weight else 0.0


def _repeated_ngram_rate(tokens, n=4):
    """Fraction of n-grams that are repetitions (monotonicity signal)."""
    count = max(0, len(tokens) - n + 1)
    if count == 0:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(count)]
    return 1.0 - len(set(grams)) / len(grams)


def _conditional_surprisal(lm, prefix_tokens, suffix_tokens):
    """Mean negative log-probability of suffix given prefix under LM."""
    ctx = ['<s>'] * (lm.order - 1) + list(prefix_tokens)
    total = 0.0
    for tok in suffix_tokens:
        total -= lm.logprob(tok, ctx)
        ctx.append(tok)
    return total / max(1, len(suffix_tokens))


def _type_token_ratio(tokens):
    """Type-Token Ratio: vocabulary richness. Low = AI-like uniformity."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def run_layer31_local(text, gamma=0.5, K=32, order=5):
    """
    Layer 3.1-Local: Zero-LLM DNA-GPT proxy.

    Splits text at γ, fits backoff n-gram LM on prefix, samples K
    continuations, and measures overlap + divergence features.

    Returns dict matching run_layer31 output schema for drop-in integration.
    """
    word_count = len(text.split())

    if word_count < 80:
        return {
            'bscore': 0.0, 'bscore_samples': [], 'determination': None,
            'confidence': 0.0, 'reason': 'DNA-GPT-Local: insufficient text (<80 words)',
            'n_samples': 0, 'truncation_ratio': gamma, 'word_count': word_count,
            'proxy_features': {},
        }

    # ── Split at sentence boundary (reuse existing helper) ─────────────
    prefix_text, suffix_text = _dna_truncate_text(text, gamma)
    prefix_tokens = _proxy_tokenize(prefix_text)
    suffix_tokens = _proxy_tokenize(suffix_text)

    if len(suffix_tokens) < 20:
        return {
            'bscore': 0.0, 'bscore_samples': [], 'determination': None,
            'confidence': 0.0, 'reason': 'DNA-GPT-Local: suffix too short after split',
            'n_samples': 0, 'truncation_ratio': gamma, 'word_count': word_count,
            'proxy_features': {},
        }

    # ── Fit backoff LM on prefix ──────────────────────────────────────
    lm = _BackoffNGramLM(order=order)
    lm.fit([prefix_text])

    # ── Generate K surrogate continuations and measure overlap ─────────
    sample_scores = []
    for _ in range(K):
        regen = lm.sample_suffix(prefix_tokens, len(suffix_tokens))
        if len(regen) < 10:
            continue
        bs = _dna_bscore(suffix_tokens, regen)
        sample_scores.append(round(bs, 4))

    if not sample_scores:
        sample_scores = [0.0]

    bscore = statistics.mean(sample_scores)
    bscore_max = max(sample_scores)

    # ── Compute proxy features ────────────────────────────────────────
    ncd = _calculate_ncd(prefix_text, suffix_text)
    internal_overlap = _internal_ngram_overlap(prefix_tokens, suffix_tokens)
    cond_surp = _conditional_surprisal(lm, prefix_tokens, suffix_tokens)
    repeat4 = _repeated_ngram_rate(suffix_tokens, 4)
    ttr = _type_token_ratio(suffix_tokens)

    proxy_features = {
        'ncd': round(ncd, 4),
        'internal_overlap': round(internal_overlap, 4),
        'cond_surprisal': round(cond_surp, 4),
        'repeat4': round(repeat4, 4),
        'ttr': round(ttr, 4),
    }

    # ── Composite scoring ─────────────────────────────────────────────
    # Combine BScore (n-gram surrogate overlap) with proxy features.
    # Lower NCD, higher internal_overlap, and higher repeat4 → AI signal.
    # Lower cond_surprisal and lower TTR → AI signal.
    #
    # Composite = weighted combination, z-scored against empirical ranges.
    # These ranges are approximate; production should calibrate on labeled data.

    # NCD: AI typical 0.85–0.95, human typical 0.95–1.05
    ncd_signal = max(0.0, (1.0 - ncd) / 0.15)  # 0→1 as NCD drops from 1.0 to 0.85

    # Internal overlap: AI typical 0.15–0.40, human typical 0.02–0.12
    overlap_signal = max(0.0, min(1.0, (internal_overlap - 0.05) / 0.30))

    # Repeat4: AI typical 0.05–0.20, human typical 0.00–0.05
    repeat_signal = max(0.0, min(1.0, repeat4 / 0.15))

    # TTR: AI typical 0.30–0.50, human typical 0.50–0.70
    ttr_signal = max(0.0, min(1.0, (0.55 - ttr) / 0.20))

    # BScore from surrogate: direct overlap measure
    bscore_signal = min(1.0, bscore / 0.15)

    # Weighted composite (BScore is primary; others are supporting)
    composite = (
        0.30 * bscore_signal +
        0.25 * ncd_signal +
        0.20 * overlap_signal +
        0.10 * repeat_signal +
        0.10 * ttr_signal +
        0.05 * max(0.0, min(1.0, (5.0 - cond_surp) / 3.0))  # low surprisal → AI
    )

    proxy_features['composite'] = round(composite, 4)

    # ── Determination (calibrated to avoid over-flagging) ─────────────
    # These thresholds are conservative; the local proxy is less precise
    # than true LLM-based DNA-GPT, so we only flag strong signals.
    if composite >= 0.60 and (ncd_signal >= 0.4 or overlap_signal >= 0.5):
        det = 'RED'
        conf = min(0.80, 0.50 + composite * 0.30)
        reason = (f"DNA-GPT-Local: high self-consistency "
                  f"(composite={composite:.2f}, NCD={ncd:.3f}, "
                  f"overlap={internal_overlap:.3f})")
    elif composite >= 0.40:
        det = 'AMBER'
        conf = min(0.60, 0.30 + composite * 0.30)
        reason = (f"DNA-GPT-Local: elevated predictability "
                  f"(composite={composite:.2f}, NCD={ncd:.3f})")
    elif composite >= 0.25:
        det = 'YELLOW'
        conf = min(0.35, 0.15 + composite * 0.20)
        reason = (f"DNA-GPT-Local: moderate self-consistency "
                  f"(composite={composite:.2f})")
    else:
        det = None
        conf = 0.0
        reason = (f"DNA-GPT-Local: low predictability "
                  f"(composite={composite:.2f}) — likely human")

    return {
        'bscore': round(bscore, 4),
        'bscore_max': round(bscore_max, 4),
        'bscore_samples': sample_scores,
        'determination': det,
        'confidence': round(conf, 4),
        'reason': reason,
        'n_samples': len(sample_scores),
        'truncation_ratio': gamma,
        'continuation_words': len(suffix_tokens),
        'word_count': word_count,
        'proxy_features': proxy_features,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0: PREAMBLE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

PREAMBLE_PATTERNS = [
    (r"(?i)^\s*[\"']?(got it|sure thing|absolutely|certainly|of course)[.!,\s]", "assistant_ack", "CRITICAL"),
    (r"(?i)^\s*[\"']?here(?:'s| is| are)\s+(your|the|a)\s+(final|updated|revised|complete|rewritten|prompt|task|evaluation)", "artifact_delivery", "CRITICAL"),
    (r"(?i)^\s*[\"']?below is\s+(a\s+)?(rewritten|revised|updated|the|your)", "artifact_delivery", "CRITICAL"),
    (r"(?i)(copy[- ]?paste|ready to use|plug[- ]and[- ]play)", "copy_paste_instruction", "MEDIUM"),
    (r"(?i)(failure[- ]inducing|designed to (test|challenge|trip|catch|induce))", "meta_design", "CRITICAL"),
    (r"(?i)^\s*[\"']?(I'?ve |I have |I'?ll |let me )(created?|drafted?|prepared?|written|designed|built|put together)", "first_person_creation", "CRITICAL"),
    (r"(?i)(natural workplace style|sounds? like a real|human[- ]issued|reads? like a human)", "style_masking", "HIGH"),
    (r"(?i)notes on what I (fixed|changed|cleaned|updated|revised)", "editorial_meta", "HIGH"),
]


def run_layer0(text):
    """Layer 0: Preamble detection. Returns (score, severity, hits)."""
    first_500 = text[:500]
    hits = []
    severity = 'NONE'

    for pat, name, sev in PREAMBLE_PATTERNS:
        search_text = first_500 if name in ('assistant_ack', 'artifact_delivery', 'first_person_creation') else text
        if re.search(pat, search_text):
            hits.append((name, sev))
            if sev == 'CRITICAL':
                severity = 'CRITICAL'
            elif sev == 'HIGH' and severity not in ('CRITICAL',):
                severity = 'HIGH'
            elif sev == 'MEDIUM' and severity == 'NONE':
                severity = 'MEDIUM'

    score = {'CRITICAL': 0.99, 'HIGH': 0.75, 'MEDIUM': 0.50, 'NONE': 0.0}[severity]
    return score, severity, hits


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: INTRINSIC FINGERPRINTS
# ══════════════════════════════════════════════════════════════════════════════

FINGERPRINT_WORDS = [
    'delve', 'utilize', 'comprehensive', 'streamline', 'leverage', 'robust',
    'facilitate', 'innovative', 'synergy', 'paradigm', 'holistic', 'nuanced',
    'multifaceted', 'spearhead', 'underscore', 'pivotal', 'landscape',
    'cutting-edge', 'actionable', 'seamlessly', 'noteworthy', 'meticulous',
    'endeavor', 'paramount', 'aforementioned', 'furthermore', 'henceforth',
]

_FINGERPRINT_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in FINGERPRINT_WORDS) + r')\b',
    re.IGNORECASE
)


def run_layer2(text):
    """Layer 2: Intrinsic fingerprint words. Returns (score, hit_count, rate)."""
    word_count = len(text.split())
    matches = _FINGERPRINT_RE.findall(text)
    hits = len(matches)
    rate = hits / max(word_count / 1000, 1)
    score = min(rate / 5.0, 1.0)
    return score, hits, rate


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2.5: PROMPT-ENGINEERING SIGNATURES
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINT_FRAMES = [
    r'must account for', r'should be visible', r'at least \d+[%$]?',
    r'at or below', r'no more than', r'no \w+ may', r'must have',
    r'should address', r'should be delivered', r'within \d+%',
    r'or higher', r'or lower', r'instead of', r'without exceeding',
    r'about \d+[-–]\d+', r'strictly on',
    r'must include\b', r'must address\b', r'must be \w+',
    r'you may not\b',
    r'may not (?:be|introduce|omit|use|exceed|include)\b',
    r'in this exact\b', r'with exactly \d+',
    r'every \w+ must\b', r'all \w+ must\b',
    r'clearly (?:list|state|describe|identify|document)',
    r'(?:document|report|response) must\b',
    r'(?:following|these) sections',
    r'use \w+ formatting', r'plain language',
    r'no \w+[- ]only\b',
]

META_DESIGN_PATTERNS = [
    r'(?i)workflows? tested',
    r'(?i)acceptance (checklist|criteria)',
    r'(?i)(used for|for) grading',
    r'(?i)SOC \d{2}-?\d{4}',
    r'(?i)expected effort:?\s*\d',
    r'(?i)deliberate (anomalies|errors|issues)',
    r'(?i)checkable artifacts',
    r'(?i)authoritative source of truth',
    r'(?i)scenario anchor date',
    r'(?i)avoid vague language',
    r'(?i)explicit non-functional',
    r'(?i)grounded in\b',
]


def get_sentences(text):
    """Segment text into sentences using spacy sentencizer or regex fallback."""
    if HAS_SPACY:
        doc = _nlp(text)
        return [s.text for s in doc.sents]
    else:
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sents if s.strip()]


def run_layer25(text):
    """Layer 2.5: Prompt-engineering signatures. Returns dict of metrics."""
    sents = get_sentences(text)
    n_sents = max(len(sents), 1)
    word_count = len(text.split())

    total_frames = 0
    distinct_pats = set()
    for pat in CONSTRAINT_FRAMES:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            total_frames += len(matches)
            distinct_pats.add(pat)
    cfd = total_frames / n_sents

    multi_frame = 0
    for sent in sents:
        ct = sum(1 for pat in CONSTRAINT_FRAMES if re.search(pat, sent, re.IGNORECASE))
        if ct >= 2:
            multi_frame += 1
    mfsr = multi_frame / n_sents

    has_role = bool(re.search(r'you (are|work|supervise|manage|serve|lead|oversee)', text[:600], re.IGNORECASE))
    has_deliverable = bool(re.search(r'(submit|deliver|present|provide|create|produce|prepare|generate)\s+(your|the|a|an|exactly)', text, re.IGNORECASE))
    has_closing = bool(re.search(r'(final|should be delivered|all conclusions|base all|submission|deliverable)', text[-300:], re.IGNORECASE))
    fc = int(has_role) + int(has_deliverable) + int(has_closing)

    cond_count = len(re.findall(r'\bif\b[^.]*?,', text, re.IGNORECASE))
    cond_count += len(re.findall(r'\bwhen\b[^.]*?,', text, re.IGNORECASE))
    cond_count += len(re.findall(r'\bunless\b', text, re.IGNORECASE))
    cond_density = cond_count / n_sents

    meta_hits = [pat for pat in META_DESIGN_PATTERNS if re.search(pat, text)]

    contractions = len(re.findall(r"\b\w+'(?:t|re|ve|s|d|ll|m)\b", text, re.IGNORECASE))

    must_count = len(re.findall(r'\bmust\b', text, re.IGNORECASE))
    must_rate = must_count / n_sents

    numbered_criteria = len(re.findall(r'^\s*\d{1,2}[.)]\s+.{20,}', text, re.MULTILINE))

    composite = 0.0
    if cfd >= 0.50: composite += 0.40
    elif cfd >= 0.30: composite += 0.25
    elif cfd >= 0.15: composite += 0.10
    if len(distinct_pats) >= 8: composite += 0.20
    elif len(distinct_pats) >= 5: composite += 0.12
    elif len(distinct_pats) >= 3: composite += 0.05
    if len(meta_hits) >= 3: composite += 0.20
    elif len(meta_hits) >= 1: composite += 0.08
    if fc == 3: composite += 0.10
    if fc >= 2 and len(distinct_pats) >= 8:
        composite += 0.15
    if numbered_criteria >= 15: composite += 0.15
    elif numbered_criteria >= 10: composite += 0.08
    if contractions == 0 and word_count > 500: composite += 0.05

    return {
        'cfd': cfd,
        'distinct_frames': len(distinct_pats),
        'mfsr': mfsr,
        'framing_completeness': fc,
        'conditional_density': cond_density,
        'meta_design_hits': len(meta_hits),
        'meta_design_details': meta_hits,
        'contractions': contractions,
        'must_count': must_count,
        'must_rate': must_rate,
        'numbered_criteria': numbered_criteria,
        'composite': min(composite, 1.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2.6: VOICE-SPECIFICATION DISSONANCE (VSD)
# ══════════════════════════════════════════════════════════════════════════════

CASUAL_MARKERS = [
    'hey', 'ok so', 'ok,', 'dont', 'wont', 'cant', 'gonna', 'gotta',
    'thx', 'pls', 'gimme', 'lemme', 'kinda', 'sorta', 'tho', 'btw',
    'fyi', 'alot', 'ya', 'yep', 'nah', 'nope', 'lol', 'haha',
]

MANUFACTURED_TYPOS = [
    'atached', 'alot', 'recieved', 'seperate', 'occured', 'wierd',
    'definately', 'accomodate', 'occurence', 'independant', 'noticable',
    'occassion', 'tommorow', 'calender', 'begining', 'acheive', 'untill',
    'beleive', 'existance', 'grammer', 'arguement', 'commited',
    'maintainance', 'necesary', 'occuring', 'persue', 'prefered',
    'recomend', 'refered', 'succesful', 'suprise',
]


def _build_marker_pattern(marker):
    tokens = marker.split()
    if len(tokens) > 1:
        escaped = r'\s+'.join(re.escape(t) for t in tokens)
    else:
        escaped = re.escape(marker)
    first_char = tokens[0][0]
    if first_char.isalnum() or first_char == '_':
        leading = r'\b'
    else:
        leading = r'(?<!\w)'
    last_char = tokens[-1][-1]
    if last_char.isalnum() or last_char == '_':
        trailing = r'\b'
    else:
        trailing = r'(?!\w)'
    return leading + escaped + trailing


_CASUAL_RE = [re.compile(_build_marker_pattern(m), re.IGNORECASE) for m in CASUAL_MARKERS]
_TYPO_RE = [re.compile(r'\b' + re.escape(t) + r'\b', re.IGNORECASE) for t in MANUFACTURED_TYPOS]


def run_layer26(text):
    """Layer 2.6: Voice-Specification Dissonance. Returns dict of metrics."""
    words = text.split()
    n_words = len(words)
    per100 = max(n_words / 100, 1)

    casual_count = sum(len(pat.findall(text)) for pat in _CASUAL_RE)
    misspelling_count = sum(len(pat.findall(text)) for pat in _TYPO_RE)

    contractions = len(re.findall(r"\b\w+'(?:t|re|ve|s|d|ll|m)\b", text, re.IGNORECASE))

    em_dashes = len(re.findall(r'(?<!\d)\s?[—–]\s?(?!\d)', text))
    em_dashes += len(re.findall(r' - ', text))

    lowercase_starts = sum(1 for line in text.split('\n') if line.strip() and line.strip()[0].islower())

    # v0.60: Misspelling weight reduced from 5 to 1. Manufactured typos
    # are also common in non-native English writing (Liang et al. 2023).
    # At weight 5, a few legitimate misspellings can inflate voice_score
    # enough to trigger VSD on non-native formal writing.
    voice_score = (casual_count * 5 + misspelling_count * 1 + contractions * 1.5
                   + em_dashes * 1 + lowercase_starts * 0.5) / per100

    camel_cols = len(re.findall(r'[A-Z][a-z]+_[A-Z][a-z_]+', text))
    filenames = len(set(re.findall(
        r'\w+\.(?:csv|xlsx|xls|tsv|json|xml|pdf|docx|doc|pptx|ppt|txt|md|html|py|zip|png|jpg|jpeg|gif|mp4)\b',
        text, re.IGNORECASE)))
    calcs = len(re.findall(
        r'(calculated?|computed?|deriv|formula|multiply|divid|subtract|sum\b|average|ratio|percent|\bnet\b.*[-=])',
        text, re.IGNORECASE))
    tabs = len(re.findall(r'(?i)(tab \d|\btab\b.*[:—-]|sheet \d)', text))
    col_listings = len(re.findall(r'(?:columns?|fields?)\s*[:]\s*\w', text, re.IGNORECASE))
    tech_parens = len(re.findall(
        r'\([^)]*(?:\.\w{2,4}|%|\d+[kKmM]?\b|formula|column|tab)[^)]*\)', text))

    spec_score = (camel_cols * 1.5 + filenames * 2 + calcs * 2 + tabs * 3
                  + col_listings * 3 + tech_parens * 1) / per100

    vsd = voice_score * spec_score

    hedges = len(re.findall(
        r'\b(pretty sure|i think|probably|maybe|might be|seems like|sort of|kind of|'
        r'not sure|i guess|iirc|afaik|if i recall|i believe)\b', text, re.IGNORECASE))

    return {
        'voice_score': voice_score,
        'spec_score': spec_score,
        'vsd': vsd,
        'voice_gated': voice_score > 2.0,
        'casual_markers': casual_count,
        'misspellings': misspelling_count,
        'contractions': contractions,
        'em_dashes': em_dashes,
        'camel_cols': camel_cols,
        'filenames': filenames,
        'calcs': calcs,
        'tabs': tabs,
        'col_listings': col_listings,
        'hedges': hedges,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2.7: INSTRUCTION DENSITY INDEX (IDI)
# ══════════════════════════════════════════════════════════════════════════════

def run_layer27(text):
    """Layer 2.7: Instruction Density Index. Catches formal-exhaustive LLM output."""
    words = text.split()
    n_words = len(words)
    per100 = max(n_words / 100, 1)

    imp_keywords = ['must', 'include', 'create', 'load', 'set', 'show', 'use', 'derive', 'treat', 'mark']
    imperatives = sum(len(re.findall(r'\b' + kw + r'\b', text, re.IGNORECASE)) for kw in imp_keywords)

    cond_keywords = ['if', 'otherwise', 'when', 'unless']
    conditionals = sum(len(re.findall(r'\b' + kw + r'\b', text, re.IGNORECASE)) for kw in cond_keywords)

    binary_specs = len(re.findall(r'\b(?:Yes|No)\b', text))
    missing_handling = len(re.findall(r'\bMISSING\b', text))
    flag_count = len(re.findall(r'\b[Ff]lag\b', text))

    idi = (imperatives * 1.0 + conditionals * 2.0 + binary_specs * 1.5 +
           missing_handling * 3.0 + flag_count * 2.0) / per100

    return {
        'idi': idi,
        'imperatives': imperatives,
        'imp_rate': imperatives / per100,
        'conditionals': conditionals,
        'cond_rate': conditionals / per100,
        'binary_specs': binary_specs,
        'missing_refs': missing_handling,
        'flag_count': flag_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2.8: SEMANTIC RESONANCE (v0.61)
# Cosine similarity of sentence embeddings against AI/human archetype centroids.
# Catches semantically AI-like text even when specific keywords are avoided.
# Ref: Mitchell et al. (2023) "DetectGPT" — semantic density as AI signal.
# ══════════════════════════════════════════════════════════════════════════════

def run_layer28(text):
    """Layer 2.8: Semantic Resonance — embedding proximity to AI archetypes.

    Uses sentence-transformers to encode text and measure cosine similarity
    against pre-computed AI and human archetype centroids.

    Returns dict with:
      - semantic_ai_score: float, max cosine similarity to AI centroids
      - semantic_human_score: float, max cosine similarity to human centroids
      - semantic_delta: float, ai_score - human_score (positive = AI-like)
      - determination: str, 'AMBER'/'YELLOW'/None
      - confidence: float
    """
    if not HAS_SEMANTIC:
        return {
            'semantic_ai_score': 0.0,
            'semantic_human_score': 0.0,
            'semantic_delta': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Semantic layer unavailable (sentence-transformers not installed)',
        }

    words = text.split()
    if len(words) < 30:
        return {
            'semantic_ai_score': 0.0,
            'semantic_human_score': 0.0,
            'semantic_delta': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Semantic layer: text too short',
        }

    # For long texts, chunk into ~200-word segments and average
    chunk_size = 200
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)

    if not chunks:
        chunks = [text]

    vecs = _EMBEDDER.encode(chunks)

    # Max similarity to AI archetypes across all chunks
    ai_sims = _cosine_similarity(vecs, _AI_CENTROIDS)
    max_ai_sim = float(ai_sims.max())
    mean_ai_sim = float(ai_sims.max(axis=1).mean())

    # Max similarity to human archetypes across all chunks
    human_sims = _cosine_similarity(vecs, _HUMAN_CENTROIDS)
    max_human_sim = float(human_sims.max())
    mean_human_sim = float(human_sims.max(axis=1).mean())

    semantic_delta = mean_ai_sim - mean_human_sim

    # Determination thresholds (conservative — embeddings are a supporting signal)
    if mean_ai_sim >= 0.65 and semantic_delta >= 0.15:
        det = 'AMBER'
        conf = min(0.60, mean_ai_sim)
        reason = f"Semantic resonance: AI={mean_ai_sim:.2f}, delta={semantic_delta:.2f}"
    elif mean_ai_sim >= 0.50 and semantic_delta >= 0.08:
        det = 'YELLOW'
        conf = min(0.35, mean_ai_sim)
        reason = f"Semantic resonance: AI={mean_ai_sim:.2f}, delta={semantic_delta:.2f}"
    else:
        det = None
        conf = 0.0
        reason = 'Semantic resonance: below threshold'

    return {
        'semantic_ai_score': round(max_ai_sim, 4),
        'semantic_human_score': round(max_human_sim, 4),
        'semantic_ai_mean': round(mean_ai_sim, 4),
        'semantic_human_mean': round(mean_human_sim, 4),
        'semantic_delta': round(semantic_delta, 4),
        'determination': det,
        'confidence': conf,
        'reason': reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL PERPLEXITY SCORING (v0.61)
# White-box detection: how "surprised" is distilgpt2 by the text?
# AI text has low perplexity (< 20); human text typically > 30-40.
# Ref: GLTR (Gehrmann et al. 2019), DetectGPT (Mitchell et al. 2023)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_perplexity(text):
    """Calculate token-level perplexity of text using distilgpt2.

    Returns dict with:
      - perplexity: float, exp(cross-entropy loss)
      - determination: str, 'AMBER'/'YELLOW'/None
      - confidence: float
    """
    if not HAS_PERPLEXITY:
        return {
            'perplexity': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Perplexity scoring unavailable (transformers/torch not installed)',
        }

    words = text.split()
    if len(words) < 50:
        return {
            'perplexity': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Perplexity: text too short',
        }

    # Truncate to model max length (1024 tokens for GPT-2)
    encodings = _PPL_TOKENIZER(text, return_tensors='pt', truncation=True,
                                max_length=1024)
    input_ids = encodings.input_ids

    if input_ids.size(1) < 10:
        return {
            'perplexity': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Perplexity: too few tokens after encoding',
        }

    with _torch.no_grad():
        outputs = _PPL_MODEL(input_ids, labels=input_ids)
        loss = outputs.loss

    ppl = _torch.exp(loss).item()

    # Determination thresholds
    # AI text on distilgpt2: typically PPL < 20
    # Human text on distilgpt2: typically PPL > 35
    # Gray zone: 20-35
    if ppl <= 15.0:
        det = 'AMBER'
        conf = min(0.65, (20.0 - ppl) / 20.0)
        reason = f"Low perplexity ({ppl:.1f}): highly predictable text"
    elif ppl <= 25.0:
        det = 'YELLOW'
        conf = min(0.35, (30.0 - ppl) / 30.0)
        reason = f"Moderate perplexity ({ppl:.1f}): somewhat predictable"
    else:
        det = None
        conf = 0.0
        reason = f"Normal perplexity ({ppl:.1f}): consistent with human text"

    return {
        'perplexity': round(ppl, 2),
        'determination': det,
        'confidence': conf,
        'reason': reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION: FINAL DETERMINATION
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# TOPIC MASKING (v0.60)
# Mask topical content before stylometric feature extraction so features
# capture style rather than topic.
# ══════════════════════════════════════════════════════════════════════════════

_TOPIC_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
_TOPIC_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
_TOPIC_DATE_RE = re.compile(
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    r'|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
    r'|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*[\s,]+\d{1,2},?\s*\d{4}\b',
    re.I,
)
_TOPIC_FILENAME_RE = re.compile(r'\b\w+\.\w{2,4}\b')
_TOPIC_VERSION_RE = re.compile(r'\bv?\d+\.\d+(?:\.\d+)*\b', re.I)
_TOPIC_NUMBER_RE = re.compile(r'\b\d{3,}\b')
_TOPIC_CAMELCASE_RE = re.compile(r'\b[a-z]+[A-Z]\w+\b|\b[A-Z][a-z]+[A-Z]\w*\b')
_TOPIC_ALLCAPS_RE = re.compile(r'\b[A-Z]{2,}\b')


def mask_topical_content(text):
    """Replace topical tokens with placeholders for stylometric analysis.

    Returns (masked_text, mask_count).
    """
    count = 0
    for pattern, repl in [
        (_TOPIC_URL_RE, ' _URL_ '),
        (_TOPIC_EMAIL_RE, ' _EMAIL_ '),
        (_TOPIC_DATE_RE, ' _DATE_ '),
        (_TOPIC_FILENAME_RE, ' _FILE_ '),
        (_TOPIC_VERSION_RE, ' _VER_ '),
        (_TOPIC_NUMBER_RE, ' _NUM_ '),
        (_TOPIC_CAMELCASE_RE, ' _IDENT_ '),
        (_TOPIC_ALLCAPS_RE, ' _ACRO_ '),
    ]:
        hits = len(pattern.findall(text))
        if hits:
            text = pattern.sub(repl, text)
            count += hits
    return text, count


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC-SCRUBBED STYLOMETRIC FEATURES (v0.60)
# Character n-grams, function-word ratios, punctuation patterns.
# Computed on masked text to reduce topic leakage.
# ══════════════════════════════════════════════════════════════════════════════

def extract_stylometric_features(text, masked_text=None):
    """Extract topic-invariant stylometric features.

    Args:
        text: Original (normalized) text.
        masked_text: Topic-masked text (if None, masks internally).

    Returns dict with features:
        char_ngram_profile: dict of top char 4-gram frequencies
        function_word_ratio: float (fraction of tokens that are function words)
        punct_bigrams: Counter of punctuation bigram patterns
        sent_length_dispersion: float (CV of sentence lengths)
        type_token_ratio: float
        avg_word_length: float
        short_word_ratio: float (words <= 3 chars)
    """
    if masked_text is None:
        masked_text, _ = mask_topical_content(text)

    words = masked_text.lower().split()
    n_words = max(len(words), 1)

    # Character 4-grams (on masked text, lowered)
    lower_masked = masked_text.lower()
    char4 = Counter()
    for i in range(len(lower_masked) - 3):
        gram = lower_masked[i:i+4]
        if not gram.startswith('_'):  # skip placeholder tokens
            char4[gram] += 1
    total_4grams = max(sum(char4.values()), 1)
    # Normalize to frequency profile (top 50)
    char_ngram_profile = {g: c / total_4grams for g, c in char4.most_common(50)}

    # Function word ratio
    fw_count = sum(1 for w in words if w in _ENGLISH_FUNCTION_WORDS)
    function_word_ratio = fw_count / n_words

    # Punctuation bigrams (pairs of consecutive punctuation characters)
    punct_chars = re.findall(r'[^\w\s]', text)
    punct_bigrams = Counter()
    for i in range(len(punct_chars) - 1):
        punct_bigrams[punct_chars[i] + punct_chars[i+1]] += 1

    # Sentence length dispersion
    sentences = get_sentences(text)
    sent_lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(sent_lengths) >= 2:
        mean_sl = statistics.mean(sent_lengths)
        std_sl = statistics.stdev(sent_lengths)
        sent_length_dispersion = std_sl / max(mean_sl, 1)
    else:
        sent_length_dispersion = 0.0

    # Type-token ratio (on original words, not masked)
    orig_words = re.findall(r'\w+', text.lower())
    n_orig = max(len(orig_words), 1)
    type_token_ratio = len(set(orig_words)) / n_orig

    # Average word length
    word_lengths = [len(w) for w in orig_words]
    avg_word_length = statistics.mean(word_lengths) if word_lengths else 0

    # Short word ratio
    short_words = sum(1 for w in orig_words if len(w) <= 3)
    short_word_ratio = short_words / n_orig

    return {
        'char_ngram_profile': char_ngram_profile,
        'function_word_ratio': round(function_word_ratio, 4),
        'punct_bigrams': dict(punct_bigrams.most_common(20)),
        'sent_length_dispersion': round(sent_length_dispersion, 4),
        'type_token_ratio': round(type_token_ratio, 4),
        'avg_word_length': round(avg_word_length, 2),
        'short_word_ratio': round(short_word_ratio, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# WINDOWED SCORING (v0.60)
# Score text in sliding sentence windows to detect mixed human+AI content.
# Ref: M4GT-Bench (Wang et al. 2024) — mixed detection as separate task.
# ══════════════════════════════════════════════════════════════════════════════

def score_windows(text, window_size=5, stride=2):
    """Score text in overlapping sentence windows using lightweight features.

    Returns dict:
        windows: list of {start, end, score, features} per window
        max_window_score: float
        mean_window_score: float
        window_variance: float
        hot_span_length: int (longest contiguous run of high-scoring windows)
        n_windows: int
        mixed_signal: bool (True if variance suggests hybrid text)
    """
    sentences = get_sentences(text)
    if len(sentences) < window_size:
        return {
            'windows': [],
            'max_window_score': 0.0,
            'mean_window_score': 0.0,
            'window_variance': 0.0,
            'hot_span_length': 0,
            'n_windows': 0,
            'mixed_signal': False,
        }

    windows = []
    for start in range(0, len(sentences) - window_size + 1, stride):
        end = start + window_size
        window_text = ' '.join(sentences[start:end])
        window_words = window_text.split()
        n_w = max(len(window_words), 1)

        # Lightweight per-window features (subset of full pipeline):
        # 1. Formulaic phrase density
        formulaic_count = sum(
            len(re.findall(pat, window_text, re.I))
            for pat, _weight in _NSSI_FORMULAIC_PATTERNS
        )
        formulaic_density = formulaic_count / (n_w / 100)

        # 2. Transition connector density
        trans_hits = len(_NSSI_TRANSITION.findall(window_text))
        trans_density = trans_hits / (n_w / 100)

        # 3. Power adjective density
        power_hits = len(_NSSI_POWER_ADJ.findall(window_text))
        power_density = power_hits / (n_w / 100)

        # 4. Function word ratio (high in natural English)
        fw = sum(1 for w in window_words if w.lower() in _ENGLISH_FUNCTION_WORDS)
        fw_ratio = fw / n_w

        # 5. Sentence length CV within window
        w_sent_lengths = [len(s.split()) for s in sentences[start:end] if s.strip()]
        if len(w_sent_lengths) >= 2:
            w_mean = statistics.mean(w_sent_lengths)
            w_std = statistics.stdev(w_sent_lengths)
            w_cv = w_std / max(w_mean, 1)
        else:
            w_cv = 0.5  # neutral

        # Composite window score: higher = more AI-like
        # Low fw_ratio + low CV + high formulaic/transition = AI pattern
        ai_indicators = 0.0
        if formulaic_density > 2.0:
            ai_indicators += min(formulaic_density / 5.0, 0.3)
        if trans_density > 3.0:
            ai_indicators += min(trans_density / 8.0, 0.2)
        if power_density > 1.5:
            ai_indicators += min(power_density / 4.0, 0.2)
        if w_cv < 0.25 and len(w_sent_lengths) >= 3:
            ai_indicators += 0.15  # uniform sentence length
        if fw_ratio < 0.12:
            ai_indicators += 0.15  # unusually low function words

        window_score = min(ai_indicators, 1.0)

        windows.append({
            'start': start,
            'end': end,
            'score': round(window_score, 3),
            'formulaic': round(formulaic_density, 2),
            'transitions': round(trans_density, 2),
            'sent_cv': round(w_cv, 3),
        })

    scores = [w['score'] for w in windows]
    max_score = max(scores) if scores else 0.0
    mean_score = statistics.mean(scores) if scores else 0.0
    variance = statistics.variance(scores) if len(scores) >= 2 else 0.0

    # Hot span: longest contiguous run of windows scoring above threshold
    hot_threshold = 0.30
    hot_span = 0
    current_span = 0
    for s in scores:
        if s >= hot_threshold:
            current_span += 1
            hot_span = max(hot_span, current_span)
        else:
            current_span = 0

    # Mixed signal: high variance indicates some windows are hot, others cold
    mixed_signal = variance >= 0.02 and max_score >= 0.30 and mean_score < 0.50

    return {
        'windows': windows,
        'max_window_score': round(max_score, 3),
        'mean_window_score': round(mean_score, 3),
        'window_variance': round(variance, 4),
        'hot_span_length': hot_span,
        'n_windows': len(windows),
        'mixed_signal': mixed_signal,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE FUSION CHANNELS (v0.58, updated v0.60)
# Each channel scores a semi-independent signal family and returns a
# ChannelResult. determine() then fuses channel results with corroboration.
# ══════════════════════════════════════════════════════════════════════════════

class ChannelResult:
    """Result from a single detection channel."""
    __slots__ = ('channel', 'score', 'severity', 'explanation',
                 'mode_eligibility', 'sub_signals')

    SEVERITIES = ('GREEN', 'YELLOW', 'AMBER', 'RED')
    SEV_ORDER = {'GREEN': 0, 'YELLOW': 1, 'AMBER': 2, 'RED': 3}

    def __init__(self, channel, score=0.0, severity='GREEN', explanation='',
                 mode_eligibility=None, sub_signals=None):
        self.channel = channel
        self.score = score
        self.severity = severity
        self.explanation = explanation
        self.mode_eligibility = mode_eligibility or ['task_prompt', 'generic_aigt']
        self.sub_signals = sub_signals or {}

    @property
    def sev_level(self):
        return self.SEV_ORDER.get(self.severity, 0)

    def __repr__(self):
        return f"CH:{self.channel}={self.severity}({self.score:.2f})"


def _detect_mode(l25, l27, l30, word_count):
    """Auto-detect whether text is a task prompt or generic AI text.

    Heuristic: if prompt-structure signals (CFD, IDI, spec_score) dominate,
    the text is likely a task prompt. If word_count is high and NSSI signals
    are present, it's more likely generic expository text.
    """
    prompt_signal = 0.0
    if l25['composite'] >= 0.15:
        prompt_signal += l25['composite']
    if l27 and l27.get('idi', 0) >= 5:
        prompt_signal += 0.3
    if l25.get('framing_completeness', 0) >= 2:
        prompt_signal += 0.2

    generic_signal = 0.0
    if l30 and l30.get('nssi_signals', 0) >= 3:
        generic_signal += 0.4
    if word_count >= 400:
        generic_signal += 0.2

    if prompt_signal > generic_signal + 0.1:
        return 'task_prompt'
    elif generic_signal > prompt_signal + 0.1:
        return 'generic_aigt'
    else:
        return 'task_prompt'  # default: conservative (higher bar for RED)


def _score_prompt_structure(l0_score, l0_severity, l25, l26, l27, word_count):
    """Channel 1: Prompt-structure signals (task_prompt primary).

    Layers: L0 preamble, L2.5 CFD/MFSR, L2.6 VSD, L2.7 IDI, SSI.
    """
    sub = {}
    score = 0.0
    severity = 'GREEN'
    parts = []

    # L0: preamble
    if l0_severity == 'CRITICAL':
        return ChannelResult(
            'prompt_structure', 0.99, 'RED',
            'Preamble detection (Layer 0: critical hit)',
            mode_eligibility=['task_prompt', 'generic_aigt'],
            sub_signals={'l0': 0.99},
        )
    if l0_score >= 0.50:
        sub['l0'] = l0_score
        score = max(score, l0_score)
        parts.append(f"L0={l0_score:.2f}")

    # L2.5: prompt engineering
    comp = l25['composite']
    sub['l25'] = comp
    if comp >= 0.60:
        score = max(score, comp)
        severity = 'RED'
        parts.append(f"L2.5={comp:.2f}(RED)")
    elif comp >= 0.40:
        score = max(score, comp)
        severity = max(severity, 'AMBER', key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
        parts.append(f"L2.5={comp:.2f}(AMBER)")
    elif comp >= 0.20:
        score = max(score, comp * 0.7)
        severity = max(severity, 'YELLOW', key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
        parts.append(f"L2.5={comp:.2f}(YELLOW)")

    # L2.6: VSD (voice-gated)
    if l26['voice_gated']:
        vsd = l26['vsd']
        sub['vsd_gated'] = vsd
        if vsd >= 50:
            score = max(score, 0.90)
            severity = 'RED'
            parts.append(f"VSD={vsd:.0f}(RED)")
        elif vsd >= 21:
            score = max(score, 0.70)
            if ChannelResult.SEV_ORDER.get(severity, 0) < 2:
                severity = 'AMBER'
            parts.append(f"VSD={vsd:.0f}(AMBER)")

    # L2.7: IDI
    if l27:
        idi = l27['idi']
        sub['idi'] = idi
        if idi >= 12:
            score = max(score, 0.85)
            severity = 'RED'
            parts.append(f"IDI={idi:.0f}(RED)")
        elif idi >= 8:
            score = max(score, 0.65)
            if ChannelResult.SEV_ORDER.get(severity, 0) < 2:
                severity = 'AMBER'
            parts.append(f"IDI={idi:.0f}(AMBER)")

    # SSI (sterile specification)
    ssi_spec_threshold = 5.0 if l26['contractions'] == 0 else 7.0
    ssi_triggered = (
        l26['spec_score'] >= ssi_spec_threshold
        and l26['voice_score'] < 0.5
        and l26['hedges'] == 0
        and word_count >= 150
    )
    if ssi_triggered:
        sub['ssi'] = l26['spec_score']
        if l26['spec_score'] >= 8.0:
            score = max(score, 0.70)
            if ChannelResult.SEV_ORDER.get(severity, 0) < 2:
                severity = 'AMBER'
            parts.append(f"SSI={l26['spec_score']:.0f}(AMBER)")
        else:
            score = max(score, 0.45)
            if ChannelResult.SEV_ORDER.get(severity, 0) < 1:
                severity = 'YELLOW'
            parts.append(f"SSI={l26['spec_score']:.0f}(YELLOW)")

    # VSD ungated (very high)
    if not l26['voice_gated'] and l26['vsd'] >= 100:
        sub['vsd_ungated'] = l26['vsd']
        score = max(score, 0.60)
        if ChannelResult.SEV_ORDER.get(severity, 0) < 2:
            severity = 'AMBER'
        parts.append(f"VSD_ungated={l26['vsd']:.0f}")
    elif not l26['voice_gated'] and l26['vsd'] >= 21:
        sub['vsd_ungated'] = l26['vsd']
        score = max(score, 0.30)
        if ChannelResult.SEV_ORDER.get(severity, 0) < 1:
            severity = 'YELLOW'

    explanation = f"Prompt-structure: {', '.join(parts)}" if parts else 'Prompt-structure: no signals'

    return ChannelResult(
        'prompt_structure', score, severity, explanation,
        mode_eligibility=['task_prompt', 'generic_aigt'],
        sub_signals=sub,
    )


def _score_stylometry(l2_score, l30, l26=None, l28=None, ppl=None):
    """Channel 2: Stylometric signals (generic_aigt primary).

    Layers: L3.0 NSSI, L2.8 Semantic, Perplexity, L2 fingerprints (supporting).
    v0.61: Added l28 (semantic resonance) and ppl (perplexity) signals.
    """
    sub = {}
    score = 0.0
    severity = 'GREEN'
    parts = []

    # L2 fingerprints: supporting-only, never drives severity alone
    if l2_score > 0:
        sub['l2_fingerprints'] = l2_score

    # L3.0 NSSI: primary stylometric signal
    if l30 and l30.get('determination'):
        nssi_det = l30['determination']
        nssi_score = l30.get('nssi_score', 0)
        nssi_signals = l30.get('nssi_signals', 0)
        sub['nssi_score'] = nssi_score
        sub['nssi_signals'] = nssi_signals

        if nssi_det == 'RED':
            score = max(score, min(0.85, l30.get('confidence', 0.80)))
            severity = 'RED'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(RED)")
        elif nssi_det == 'AMBER':
            score = max(score, min(0.65, l30.get('confidence', 0.60)))
            severity = 'AMBER'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(AMBER)")
        elif nssi_det == 'YELLOW':
            score = max(score, min(0.40, l30.get('confidence', 0.30)))
            severity = 'YELLOW'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(YELLOW)")

    # L2.8 Semantic Resonance: supporting signal, can independently reach YELLOW
    if l28 and l28.get('determination'):
        sem_det = l28['determination']
        sem_delta = l28.get('semantic_delta', 0)
        sub['semantic_ai_score'] = l28.get('semantic_ai_mean', 0)
        sub['semantic_delta'] = sem_delta

        if sem_det == 'AMBER':
            # Semantic can boost existing severity or independently reach YELLOW+
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"Sem=AMBER(delta={sem_delta:.2f},boost)")
            else:
                score = max(score, l28.get('confidence', 0.55))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"Sem=AMBER(delta={sem_delta:.2f})")
        elif sem_det == 'YELLOW':
            if severity != 'GREEN':
                score = min(score + 0.05, 1.0)
                parts.append(f"Sem=YELLOW(delta={sem_delta:.2f},supporting)")
            else:
                score = max(score, l28.get('confidence', 0.30))
                severity = 'YELLOW'
                parts.append(f"Sem=YELLOW(delta={sem_delta:.2f})")

    # Perplexity: supporting signal, can independently reach YELLOW
    if ppl and ppl.get('determination'):
        ppl_det = ppl['determination']
        ppl_val = ppl.get('perplexity', 0)
        sub['perplexity'] = ppl_val

        if ppl_det == 'AMBER':
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"PPL={ppl_val:.0f}(AMBER,boost)")
            else:
                score = max(score, ppl.get('confidence', 0.55))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"PPL={ppl_val:.0f}(AMBER)")
        elif ppl_det == 'YELLOW':
            if severity != 'GREEN':
                score = min(score + 0.05, 1.0)
                parts.append(f"PPL={ppl_val:.0f}(YELLOW,supporting)")
            else:
                score = max(score, ppl.get('confidence', 0.30))
                severity = 'YELLOW'
                parts.append(f"PPL={ppl_val:.0f}(YELLOW)")

    # L2 fingerprints add supporting weight if any stylometric signal is active
    if l2_score >= 0.30 and severity != 'GREEN':
        score = min(score + 0.10, 1.0)
        parts.append(f"L2={l2_score:.2f}(supporting)")

    explanation = f"Stylometry: {', '.join(parts)}" if parts else 'Stylometry: no signals'

    return ChannelResult(
        'stylometry', score, severity, explanation,
        mode_eligibility=['generic_aigt'],  # supporting-only in task_prompt mode
        sub_signals=sub,
    )


def _score_continuation(l31):
    """Channel 3: Continuation-based detection (DNA-GPT / DNA-GPT-Local).

    Eligible in both modes.
    v0.61: Supports both API-based and local proxy results.
    """
    sub = {}
    score = 0.0
    severity = 'GREEN'
    parts = []

    if l31 and l31.get('determination'):
        dna_det = l31['determination']
        bscore = l31.get('bscore', 0)
        sub['bscore'] = bscore

        # Track proxy features if present (local mode)
        proxy = l31.get('proxy_features')
        if proxy:
            sub['ncd'] = proxy.get('ncd', 0)
            sub['internal_overlap'] = proxy.get('internal_overlap', 0)
            sub['composite'] = proxy.get('composite', 0)
            label = 'Local'
        else:
            label = 'API'

        if dna_det == 'RED':
            score = min(0.90, l31.get('confidence', 0.80))
            severity = 'RED'
            parts.append(f"BScore={bscore:.3f}({label},RED)")
        elif dna_det == 'AMBER':
            score = min(0.70, l31.get('confidence', 0.60))
            severity = 'AMBER'
            parts.append(f"BScore={bscore:.3f}({label},AMBER)")
        elif dna_det == 'YELLOW':
            score = min(0.40, l31.get('confidence', 0.30))
            severity = 'YELLOW'
            parts.append(f"BScore={bscore:.3f}({label},YELLOW)")

    explanation = f"Continuation: {', '.join(parts)}" if parts else 'Continuation: no signals'

    return ChannelResult(
        'continuation', score, severity, explanation,
        mode_eligibility=['task_prompt', 'generic_aigt'],
        sub_signals=sub,
    )


def _score_windowing(text=None, window_result=None):
    """Channel 4: Sentence-window scoring (v0.60).

    Detects mixed human+AI content via per-window score variance.
    """
    if window_result is None or window_result.get('n_windows', 0) == 0:
        return ChannelResult(
            'windowing', 0.0, 'GREEN',
            'Windowing: insufficient text for windows',
            mode_eligibility=['generic_aigt'],
            sub_signals={},
        )

    sub = {
        'max_window': window_result['max_window_score'],
        'mean_window': window_result['mean_window_score'],
        'variance': window_result['window_variance'],
        'hot_span': window_result['hot_span_length'],
        'n_windows': window_result['n_windows'],
        'mixed_signal': window_result['mixed_signal'],
    }

    score = 0.0
    severity = 'GREEN'
    parts = []

    max_w = window_result['max_window_score']
    mean_w = window_result['mean_window_score']
    variance = window_result['window_variance']
    hot_span = window_result['hot_span_length']
    mixed = window_result['mixed_signal']

    # High max window score indicates at least part of text is AI-like
    if max_w >= 0.60 and hot_span >= 3:
        score = max(score, 0.75)
        severity = 'RED'
        parts.append(f"hot_span={hot_span}(max={max_w:.2f})")
    elif max_w >= 0.45 and hot_span >= 2:
        score = max(score, 0.55)
        severity = 'AMBER'
        parts.append(f"hot_span={hot_span}(max={max_w:.2f})")
    elif max_w >= 0.30:
        score = max(score, 0.30)
        severity = 'YELLOW'
        parts.append(f"max_window={max_w:.2f}")

    # Mixed signal: override severity to MIXED if detected
    if mixed and severity != 'GREEN':
        parts.append(f"MIXED(var={variance:.3f})")

    explanation = f"Windowing: {', '.join(parts)}" if parts else 'Windowing: no signals'

    return ChannelResult(
        'windowing', score, severity, explanation,
        mode_eligibility=['generic_aigt'],
        sub_signals=sub,
    )


def determine(l0_score, l0_severity, l25, l26, l27=None, word_count=0,
              l30=None, l31=None, lang_gate=None, norm_report=None,
              mode='auto', l2_score=0.0, l28=None, ppl=None, **kwargs):
    """
    v0.61: Evidence fusion with channel-based corroboration.

    Args:
        mode: 'task_prompt' | 'generic_aigt' | 'auto'
            task_prompt: Only prompt-structure can independently drive RED.
            generic_aigt: All channels can contribute independently.
            auto: Heuristic selects mode based on text features.
        l28: Layer 2.8 semantic resonance result (optional).
        ppl: Perplexity scoring result (optional).

    Returns:
        (determination, reason, confidence, channel_details)
        channel_details is a dict with channel results for diagnostics.

    Corroboration rules:
        RED: Requires one strong channel (RED) plus one supporting (YELLOW+),
             OR two channels both at AMBER+.
             Exception: L0 CRITICAL preamble → instant RED (no corroboration).
        AMBER: One channel at AMBER, or two at YELLOW+.
        YELLOW: One channel at YELLOW+.
        REVIEW: Signals present but below YELLOW threshold and above GREEN.
        GREEN: No significant signals.
    """
    # ── Mode detection ────────────────────────────────────────────────
    if mode == 'auto':
        mode = _detect_mode(l25, l27, l30, word_count)

    # ── Score all channels ────────────────────────────────────────────
    ch_prompt = _score_prompt_structure(l0_score, l0_severity, l25, l26, l27, word_count)
    ch_style = _score_stylometry(l2_score, l30, l26, l28=l28, ppl=ppl)
    ch_cont = _score_continuation(l31)
    ch_window = _score_windowing(window_result=kwargs.get('window_result'))

    channels = [ch_prompt, ch_style, ch_cont, ch_window]
    channel_details = {
        'mode': mode,
        'channels': {ch.channel: {
            'score': ch.score, 'severity': ch.severity,
            'explanation': ch.explanation, 'mode_eligible': mode in ch.mode_eligibility,
        } for ch in channels},
    }

    # ── Fairness severity cap ────────────────────────────────────────
    severity_cap = None
    if lang_gate and lang_gate.get('support_level') == 'UNSUPPORTED':
        severity_cap = 'YELLOW'
    elif lang_gate and lang_gate.get('support_level') == 'REVIEW':
        severity_cap = 'AMBER'

    def _apply_cap(det, reason, conf):
        if severity_cap is None:
            return det, reason, conf
        sev_order = {'GREEN': 0, 'YELLOW': 1, 'REVIEW': 1, 'AMBER': 2, 'RED': 3}
        if sev_order.get(det, 0) > sev_order.get(severity_cap, 3):
            gate_reason = lang_gate.get('reason', 'language support gate')
            return severity_cap, f"{reason} [capped from {det}: {gate_reason}]", min(conf, 0.40)
        return det, reason, conf

    # ── L0 CRITICAL: instant RED, no corroboration needed ────────────
    if ch_prompt.sub_signals.get('l0') == 0.99 and l0_severity == 'CRITICAL':
        det, reason, conf = _apply_cap('RED', ch_prompt.explanation, 0.99)
        return det, reason, conf, channel_details

    # ── Mode-aware channel filtering ─────────────────────────────────
    # In task_prompt mode, only prompt-structure is "primary" — other
    # channels can support but not independently drive RED.
    if mode == 'task_prompt':
        primary_channels = [ch for ch in channels if 'task_prompt' in ch.mode_eligibility]
        supporting_channels = [ch for ch in channels if 'task_prompt' not in ch.mode_eligibility]
    else:  # generic_aigt
        primary_channels = channels
        supporting_channels = []

    # ── Evidence fusion ──────────────────────────────────────────────
    # Sort by severity level (descending)
    all_active = sorted(
        [ch for ch in channels if ch.severity != 'GREEN'],
        key=lambda c: c.sev_level, reverse=True,
    )
    primary_active = sorted(
        [ch for ch in primary_channels if ch.severity != 'GREEN'],
        key=lambda c: c.sev_level, reverse=True,
    )
    support_active = [ch for ch in supporting_channels if ch.severity != 'GREEN']

    # Count channels at each severity level
    n_red = sum(1 for ch in all_active if ch.severity == 'RED')
    n_amber_plus = sum(1 for ch in all_active if ch.sev_level >= 2)
    n_yellow_plus = sum(1 for ch in all_active if ch.sev_level >= 1)
    n_primary_red = sum(1 for ch in primary_active if ch.severity == 'RED')
    n_primary_amber = sum(1 for ch in primary_active if ch.sev_level >= 2)

    # Build explanation from top channels
    top_explanations = [ch.explanation for ch in all_active[:3]]
    combined_reason = ' + '.join(top_explanations) if top_explanations else 'No significant signals'
    top_score = max((ch.score for ch in all_active), default=0.0)

    # ── RED: strong primary + supporting, or two AMBER+ channels ─────
    if n_primary_red >= 1 and n_yellow_plus >= 2:
        # Strong primary signal corroborated by any other channel
        det, reason, conf = _apply_cap('RED', combined_reason, top_score)
        return det, reason, conf, channel_details

    if n_primary_amber >= 2:
        # Two primary channels both at AMBER or higher
        det, reason, conf = _apply_cap('RED', combined_reason, min(top_score, 0.85))
        return det, reason, conf, channel_details

    if mode == 'task_prompt' and n_primary_red >= 1 and n_yellow_plus == 1:
        # In task_prompt mode: single strong primary signal without corroboration
        # → demote to AMBER (require corroboration for RED)
        det, reason, conf = _apply_cap('AMBER', f"{combined_reason} [single-channel, demoted from RED]", min(top_score, 0.75))
        return det, reason, conf, channel_details

    if mode == 'generic_aigt' and n_red >= 1:
        # In generic_aigt mode: single RED channel sufficient
        # but confidence is lower without corroboration
        if n_yellow_plus >= 2:
            det, reason, conf = _apply_cap('RED', combined_reason, top_score)
        else:
            det, reason, conf = _apply_cap('RED', f"{combined_reason} [single-channel]", min(top_score, 0.75))
        return det, reason, conf, channel_details

    # ── AMBER: one channel at AMBER, or two at YELLOW+ ──────────────
    if n_primary_amber >= 1:
        det, reason, conf = _apply_cap('AMBER', combined_reason, min(top_score, 0.70))
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.60), channel_details
        return det, reason, conf, channel_details

    if n_yellow_plus >= 2:
        det, reason, conf = _apply_cap('AMBER', f"{combined_reason} [multi-channel convergence]", min(top_score, 0.60))
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.55), channel_details
        return det, reason, conf, channel_details

    # Supporting channels at AMBER in task_prompt mode → AMBER
    if mode == 'task_prompt' and any(ch.sev_level >= 2 for ch in support_active):
        support_expl = [ch.explanation for ch in support_active if ch.sev_level >= 2]
        det, reason, conf = _apply_cap('AMBER', f"{' + '.join(support_expl)} [supporting channel]", 0.55)
        return det, reason, conf, channel_details

    # ── YELLOW: one channel at YELLOW+ ──────────────────────────────
    if n_yellow_plus >= 1:
        det, reason, conf = _apply_cap('YELLOW', combined_reason, min(top_score, 0.45))
        # v0.60: Check for MIXED signal from windowing
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.50), channel_details
        return det, reason, conf, channel_details

    # ── Obfuscation delta ────────────────────────────────────────────
    if norm_report and norm_report.get('obfuscation_delta', 0) >= 0.05:
        delta = norm_report['obfuscation_delta']
        det, reason, conf = _apply_cap('YELLOW', f"Text normalization delta ({delta:.1%}) suggests obfuscation", 0.35)
        return det, reason, conf, channel_details

    # ── REVIEW: any channel has non-zero score but below YELLOW ──────
    any_signal = any(ch.score > 0.05 for ch in channels)
    if any_signal:
        weak_parts = [ch.explanation for ch in channels if ch.score > 0.05]
        return 'REVIEW', f"Weak signals below threshold: {' + '.join(weak_parts[:2])}", 0.10, channel_details

    # ── GREEN ────────────────────────────────────────────────────────
    return 'GREEN', 'No significant signals', 0.0, channel_details


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_prompt(text, task_id='', occupation='', attempter='', stage='',
                   run_l3=True, api_key=None, dna_provider='anthropic',
                   dna_model=None, dna_samples=3,
                   ground_truth=None, language=None, domain=None,
                   mode='auto', cal_table=None):
    """Run full v0.61 pipeline on a single prompt. Returns result dict.

    v0.60: Added cal_table for conformal calibration, audit_trail in output.
    v0.61: Added Layer 2.8 (semantic resonance), perplexity scoring.
    """
    word_count = len(text.split())

    # ── v0.60: Normalization pre-pass ────────────────────────────────────
    normalized_text, norm_report = normalize_text(text)

    # ── v0.60: Fairness / language support gate ──────────────────────────
    lang_gate = check_language_support(normalized_text, word_count)

    # All layers run on normalized text
    text_for_analysis = normalized_text

    l0_score, l0_severity, l0_hits = run_layer0(text_for_analysis)
    l2_score, l2_hits, l2_rate = run_layer2(text_for_analysis)
    l25 = run_layer25(text_for_analysis)
    l26 = run_layer26(text_for_analysis)
    l27 = run_layer27(text_for_analysis)

    l30 = None
    if run_l3:
        l30 = run_layer30(text_for_analysis)

    l31 = None
    if run_l3 and api_key:
        l31 = run_layer31(
            text_for_analysis, api_key=api_key, provider=dna_provider,
            model=dna_model, n_samples=dna_samples,
        )
    elif run_l3:
        # v0.61: Fall back to zero-LLM local proxy when no API key
        l31 = run_layer31_local(text_for_analysis)

    # ── v0.61: Semantic resonance (Layer 2.8) ─────────────────────────
    l28 = run_layer28(text_for_analysis)

    # ── v0.61: Local perplexity scoring ───────────────────────────────
    ppl = calculate_perplexity(text_for_analysis)

    # ── v0.60: Topic-scrubbed stylometry ─────────────────────────────
    masked_text, mask_count = mask_topical_content(text_for_analysis)
    stylo_features = extract_stylometric_features(text_for_analysis, masked_text)

    # ── v0.60: Windowed scoring ──────────────────────────────────────
    window_result = score_windows(text_for_analysis)

    det, reason, confidence, channel_details = determine(
        l0_score, l0_severity, l25, l26, l27, word_count,
        l30=l30, l31=l31,
        lang_gate=lang_gate, norm_report=norm_report,
        mode=mode, l2_score=l2_score,
        l28=l28, ppl=ppl,
        window_result=window_result,
    )

    # ── v0.60: Conformal calibration ─────────────────────────────────
    # Compute length_bin for calibration lookup
    if word_count < 100:
        length_bin = 'short'
    elif word_count < 300:
        length_bin = 'medium'
    elif word_count < 800:
        length_bin = 'long'
    else:
        length_bin = 'very_long'

    cal_result = apply_calibration(confidence, cal_table, domain=domain, length_bin=length_bin)

    # ── v0.60: Audit trail ───────────────────────────────────────────
    audit_trail = {
        'pipeline_version': 'v0.61',
        'mode_resolved': channel_details.get('mode', mode),
        'channels': channel_details.get('channels', {}),
        'fairness_gate': {
            'support_level': lang_gate.get('support_level'),
            'fw_coverage': lang_gate.get('function_word_coverage'),
        },
        'normalization': {
            'obfuscation_delta': norm_report.get('obfuscation_delta', 0),
            'invisible_chars': norm_report.get('invisible_chars', 0),
            'homoglyphs': norm_report.get('homoglyphs', 0),
            'ftfy_applied': norm_report.get('ftfy_applied', False),
        },
        'calibration': cal_result,
        # v0.61: Semantic + Perplexity availability
        'semantic_available': HAS_SEMANTIC,
        'perplexity_available': HAS_PERPLEXITY,
    }

    result = {
        'task_id': task_id,
        'occupation': occupation,
        'attempter': attempter,
        'stage': stage,
        'word_count': word_count,
        'determination': det,
        'reason': reason,
        'confidence': confidence,
        # v0.60: Calibrated confidence
        'calibrated_confidence': cal_result['calibrated_confidence'],
        'p_value': cal_result['p_value'],
        'calibration_stratum': cal_result['stratum_used'],
        # v0.58: Mode and channel details
        'mode': channel_details.get('mode', mode),
        'channel_details': channel_details,
        # v0.60: Audit trail
        'audit_trail': audit_trail,
        # v0.60: Normalization
        'norm_obfuscation_delta': norm_report.get('obfuscation_delta', 0.0),
        'norm_invisible_chars': norm_report.get('invisible_chars', 0),
        'norm_homoglyphs': norm_report.get('homoglyphs', 0),
        # v0.60: Fairness gate
        'lang_support_level': lang_gate.get('support_level', 'SUPPORTED'),
        'lang_fw_coverage': lang_gate.get('function_word_coverage', 0.0),
        'lang_non_latin_ratio': lang_gate.get('non_latin_ratio', 0.0),
        # Layer 0
        'l0_score': l0_score,
        'l0_severity': l0_severity,
        'l0_hits': len(l0_hits),
        'l0_details': l0_hits,
        # Layer 2 (diagnostic-only — not used in determine())
        'l2_score': l2_score,
        'l2_fingerprint_hits': l2_hits,
        # Layer 2.5
        'l25_composite': l25['composite'],
        'l25_cfd': l25['cfd'],
        'l25_distinct_frames': l25['distinct_frames'],
        'l25_mfsr': l25['mfsr'],
        'l25_framing': l25['framing_completeness'],
        'l25_conditional_density': l25['conditional_density'],
        'l25_meta_design': l25['meta_design_hits'],
        'l25_contractions': l25['contractions'],
        'l25_must_rate': l25['must_rate'],
        'l25_numbered_criteria': l25['numbered_criteria'],
        # Layer 2.7
        'l27_idi': l27['idi'],
        'l27_imperatives': l27['imperatives'],
        'l27_conditionals': l27['conditionals'],
        'l27_binary_specs': l27['binary_specs'],
        'l27_missing_refs': l27['missing_refs'],
        'l27_flag_count': l27['flag_count'],
        # Layer 2.6
        'l26_voice_score': l26['voice_score'],
        'l26_spec_score': l26['spec_score'],
        'l26_vsd': l26['vsd'],
        'l26_voice_gated': l26['voice_gated'],
        'l26_casual_markers': l26['casual_markers'],
        'l26_misspellings': l26['misspellings'],
        'l26_camel_cols': l26['camel_cols'],
        'l26_calcs': l26['calcs'],
        'l26_hedges': l26['hedges'],
        # SSI (v0.60: contraction-absence is soft bonus, not hard gate)
        'ssi_triggered': (
            l26['spec_score'] >= (5.0 if l26['contractions'] == 0 else 7.0)
            and l26['voice_score'] < 0.5
            and l26['hedges'] == 0
            and word_count >= 150
        ),
        # v0.60: Metadata for baseline collection
        'ground_truth': ground_truth,
        'language': language,
        'domain': domain,
        # v0.60: Windowed scoring
        'window_max_score': window_result.get('max_window_score', 0.0),
        'window_mean_score': window_result.get('mean_window_score', 0.0),
        'window_variance': window_result.get('window_variance', 0.0),
        'window_hot_span': window_result.get('hot_span_length', 0),
        'window_n_windows': window_result.get('n_windows', 0),
        'window_mixed_signal': window_result.get('mixed_signal', False),
        # v0.60: Stylometric features (summary)
        'stylo_fw_ratio': stylo_features.get('function_word_ratio', 0.0),
        'stylo_sent_dispersion': stylo_features.get('sent_length_dispersion', 0.0),
        'stylo_ttr': stylo_features.get('type_token_ratio', 0.0),
        'stylo_avg_word_len': stylo_features.get('avg_word_length', 0.0),
        'stylo_short_word_ratio': stylo_features.get('short_word_ratio', 0.0),
        'stylo_mask_count': mask_count,
    }

    # Layer 2.8 — Semantic Resonance fields (v0.61)
    result.update({
        'l28_semantic_ai_score': l28.get('semantic_ai_score', 0.0),
        'l28_semantic_human_score': l28.get('semantic_human_score', 0.0),
        'l28_semantic_ai_mean': l28.get('semantic_ai_mean', 0.0),
        'l28_semantic_human_mean': l28.get('semantic_human_mean', 0.0),
        'l28_semantic_delta': l28.get('semantic_delta', 0.0),
        'l28_determination': l28.get('determination'),
        'l28_confidence': l28.get('confidence', 0.0),
    })

    # Perplexity fields (v0.61)
    result.update({
        'ppl_perplexity': ppl.get('perplexity', 0.0),
        'ppl_determination': ppl.get('determination'),
        'ppl_confidence': ppl.get('confidence', 0.0),
    })

    # Layer 3.0 — NSSI fields (v0.60: expanded with new signal diagnostics)
    if l30:
        result.update({
            'l30_nssi_score': l30.get('nssi_score', 0.0),
            'l30_nssi_signals': l30.get('nssi_signals', 0),
            'l30_determination': l30.get('determination'),
            'l30_confidence': l30.get('confidence', 0.0),
            'l30_formulaic_density': l30.get('formulaic_density', 0.0),
            'l30_power_adj_density': l30.get('power_adj_density', 0.0),
            'l30_demonstrative_density': l30.get('demonstrative_density', 0.0),
            'l30_transition_density': l30.get('transition_density', 0.0),
            'l30_scare_quote_density': l30.get('scare_quote_density', 0.0),
            'l30_emdash_density': l30.get('emdash_density', 0.0),
            'l30_this_the_start_rate': l30.get('this_the_start_rate', 0.0),
            'l30_section_depth': l30.get('section_depth', 0),
            # v0.60: new diagnostic fields
            'l30_sent_length_cv': l30.get('sent_length_cv', 0.0),
            'l30_comp_ratio': l30.get('comp_ratio', 0.0),
            'l30_hapax_ratio': l30.get('hapax_ratio', 0.0),
            'l30_hapax_count': l30.get('hapax_count', 0),
            'l30_unique_words': l30.get('unique_words', 0),
        })
    else:
        result.update({
            'l30_nssi_score': 0.0, 'l30_nssi_signals': 0,
            'l30_determination': None, 'l30_confidence': 0.0,
            'l30_formulaic_density': 0.0, 'l30_power_adj_density': 0.0,
            'l30_demonstrative_density': 0.0, 'l30_transition_density': 0.0,
            'l30_scare_quote_density': 0.0, 'l30_emdash_density': 0.0,
            'l30_this_the_start_rate': 0.0, 'l30_section_depth': 0,
            'l30_sent_length_cv': 0.0, 'l30_comp_ratio': 0.0,
            'l30_hapax_ratio': 0.0, 'l30_hapax_count': 0,
            'l30_unique_words': 0,
        })

    # Layer 3.1 — DNA-GPT fields (API or Local proxy)
    if l31:
        proxy = l31.get('proxy_features', {})
        result.update({
            'l31_bscore': l31.get('bscore', 0.0),
            'l31_bscore_max': l31.get('bscore_max', 0.0),
            'l31_determination': l31.get('determination'),
            'l31_confidence': l31.get('confidence', 0.0),
            'l31_n_samples': l31.get('n_samples', 0),
            'l31_mode': 'local' if proxy else 'api',
            # v0.61: Local proxy features (zero when API mode)
            'l31_ncd': proxy.get('ncd', 0.0),
            'l31_internal_overlap': proxy.get('internal_overlap', 0.0),
            'l31_cond_surprisal': proxy.get('cond_surprisal', 0.0),
            'l31_repeat4': proxy.get('repeat4', 0.0),
            'l31_ttr': proxy.get('ttr', 0.0),
            'l31_composite': proxy.get('composite', 0.0),
        })
    else:
        result.update({
            'l31_bscore': 0.0, 'l31_bscore_max': 0.0,
            'l31_determination': None, 'l31_confidence': 0.0,
            'l31_n_samples': 0, 'l31_mode': None,
            'l31_ncd': 0.0, 'l31_internal_overlap': 0.0,
            'l31_cond_surprisal': 0.0, 'l31_repeat4': 0.0,
            'l31_ttr': 0.0, 'l31_composite': 0.0,
        })

    return result


def print_result(r, verbose=False):
    """Pretty-print a single result."""
    icons = {'RED': '🔴', 'AMBER': '🟠', 'YELLOW': '🟡', 'GREEN': '🟢', 'MIXED': '🔵', 'REVIEW': '⚪'}
    icon = icons.get(r['determination'], '?')

    print(f"\n  {icon} [{r['determination']}] {r['task_id'][:20]}  |  {r['occupation'][:45]}")
    print(f"     Attempter: {r['attempter'] or '(unknown)'} | Stage: {r['stage']} | Words: {r['word_count']} | Mode: {r.get('mode', '?')}")
    print(f"     Reason: {r['reason']}")

    # v0.60: Show calibrated confidence when available
    cal_conf = r.get('calibrated_confidence')
    p_val = r.get('p_value')
    if cal_conf is not None and cal_conf != r.get('confidence'):
        cal_str = f"     Calibrated: conf={cal_conf:.3f}"
        if p_val is not None:
            cal_str += f"  p={p_val:.3f}"
        cal_str += f"  [{r.get('calibration_stratum', '?')}]"
        print(cal_str)

    if verbose or r['determination'] in ('RED', 'AMBER'):
        # v0.60: Normalization and fairness gate
        delta = r.get('norm_obfuscation_delta', 0)
        lang = r.get('lang_support_level', 'SUPPORTED')
        if delta > 0 or lang != 'SUPPORTED':
            print(f"     NORM obfuscation: {delta:.1%}  invisible={r.get('norm_invisible_chars', 0)} homoglyphs={r.get('norm_homoglyphs', 0)}")
            print(f"     GATE support:     {lang} (fw_coverage={r.get('lang_fw_coverage', 0):.2f}, non_latin={r.get('lang_non_latin_ratio', 0):.2f})")
        print(f"     L0  Preamble:     {r['l0_score']:.2f} ({r['l0_severity']}, {r['l0_hits']} hits)")
        if r['l0_details']:
            for name, sev in r['l0_details']:
                print(f"         → [{sev}] {name}")
        print(f"     L2  Fingerprints: {r['l2_score']:.2f} ({r['l2_fingerprint_hits']} hits)")
        print(f"     L2.5 Composite:   {r['l25_composite']:.2f}")
        print(f"         CFD={r['l25_cfd']:.3f} frames={r['l25_distinct_frames']} MFSR={r['l25_mfsr']:.3f}")
        print(f"         meta={r['l25_meta_design']} FC={r['l25_framing']}/3 must={r['l25_must_rate']:.3f}/sent")
        print(f"         contractions={r['l25_contractions']} numbered_criteria={r['l25_numbered_criteria']}")
        print(f"     L2.7 IDI:          {r['l27_idi']:.1f}  (imp={r['l27_imperatives']} cond={r['l27_conditionals']} Y/N={r['l27_binary_specs']} MISS={r['l27_missing_refs']} flag={r['l27_flag_count']})")
        print(f"     L2.6 VSD:         {r['l26_vsd']:.1f}  (voice={r['l26_voice_score']:.1f} × spec={r['l26_spec_score']:.1f})")
        print(f"         gated={'YES' if r['l26_voice_gated'] else 'no'} casual={r['l26_casual_markers']} typos={r['l26_misspellings']}")
        print(f"         cols={r['l26_camel_cols']} calcs={r['l26_calcs']} hedges={r['l26_hedges']}")
        if r.get('ssi_triggered'):
            print(f"     SSI:  TRIGGERED  (spec={r['l26_spec_score']:.1f}, voice=0, hedges=0, {r['word_count']}w)")
        # Layer 3.0 — NSSI (v0.60)
        nssi_score = r.get('l30_nssi_score', 0.0)
        nssi_signals = r.get('l30_nssi_signals', 0)
        nssi_det = r.get('l30_determination')
        if nssi_score > 0 or nssi_det:
            det_str = nssi_det or 'n/a'
            print(f"     L3.0 NSSI:        {nssi_score:.3f}  ({nssi_signals} signals, det={det_str})")
            print(f"         formulaic={r.get('l30_formulaic_density', 0):.3f} power_adj={r.get('l30_power_adj_density', 0):.3f}"
                  f" demo={r.get('l30_demonstrative_density', 0):.3f} trans={r.get('l30_transition_density', 0):.3f}")
            # v0.60: show new signal diagnostics
            print(f"         sent_cv={r.get('l30_sent_length_cv', 0):.3f} comp_ratio={r.get('l30_comp_ratio', 0):.3f}"
                  f" hapax={r.get('l30_hapax_ratio', 0):.3f} (unique={r.get('l30_unique_words', 0)})")
        # Layer 3.1 — DNA-GPT
        bscore = r.get('l31_bscore', 0.0)
        dna_det = r.get('l31_determination')
        if bscore > 0 or dna_det:
            det_str = dna_det or 'n/a'
            print(f"     L3.1 DNA-GPT:     BScore={bscore:.4f}  (max={r.get('l31_bscore_max', 0):.4f}, "
                  f"samples={r.get('l31_n_samples', 0)}, det={det_str})")

        # v0.60: Channel summary
        cd = r.get('channel_details', {})
        if cd.get('channels'):
            print(f"     ── Channels ──")
            for ch_name, ch_info in cd['channels'].items():
                if ch_info['severity'] != 'GREEN':
                    eligible = '✓' if ch_info.get('mode_eligible') else '○'
                    print(f"     {eligible} {ch_name:18s} {ch_info['severity']:6s} score={ch_info['score']:.2f}  {ch_info['explanation'][:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-SUBMISSION SIMILARITY ANALYSIS (v0.53)
# ══════════════════════════════════════════════════════════════════════════════

def _word_shingles(text, k=3):
    words = re.findall(r'\w+', text.lower())
    if len(words) < k:
        return {tuple(words)} if words else set()
    return set(tuple(words[i:i+k]) for i in range(len(words) - k + 1))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_STRUCT_FEATURES = [
    'l25_composite', 'l25_cfd', 'l25_mfsr', 'l25_must_rate',
    'l27_idi', 'l26_spec_score', 'l26_voice_score',
]

def _structural_similarity(r1, r2):
    diff_sq = sum((r1.get(f, 0) - r2.get(f, 0)) ** 2 for f in _STRUCT_FEATURES)
    return 1.0 / (1.0 + math.sqrt(diff_sq))


def analyze_similarity(results, text_map, jaccard_threshold=0.40, struct_threshold=0.90):
    """Analyze cross-submission similarity within occupation groups."""
    by_occ = defaultdict(list)
    for r in results:
        occ = r.get('occupation', '(unknown)')
        by_occ[occ].append(r)

    shingle_cache = {}
    for tid, text in text_map.items():
        shingle_cache[tid] = _word_shingles(text)

    flagged_pairs = []

    for occ, group in by_occ.items():
        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                r_a, r_b = group[i], group[j]
                att_a = r_a.get('attempter', '').strip().lower()
                att_b = r_b.get('attempter', '').strip().lower()

                if att_a and att_b and att_a == att_b:
                    continue

                tid_a = r_a.get('task_id', '')
                tid_b = r_b.get('task_id', '')

                jac = _jaccard(
                    shingle_cache.get(tid_a, set()),
                    shingle_cache.get(tid_b, set())
                )
                struct = _structural_similarity(r_a, r_b)

                flags = []
                if jac >= jaccard_threshold:
                    flags.append('text')
                if struct >= struct_threshold:
                    flags.append('structural')

                if flags:
                    flagged_pairs.append({
                        'id_a': tid_a,
                        'id_b': tid_b,
                        'attempter_a': r_a.get('attempter', ''),
                        'attempter_b': r_b.get('attempter', ''),
                        'occupation': occ,
                        'jaccard': jac,
                        'structural': struct,
                        'flag_type': '+'.join(flags),
                        'det_a': r_a['determination'],
                        'det_b': r_b['determination'],
                    })

    flagged_pairs.sort(key=lambda p: p['jaccard'], reverse=True)
    return flagged_pairs


def print_similarity_report(pairs):
    """Print cross-submission similarity findings."""
    if not pairs:
        print("\n  No cross-attempter similarity clusters detected.")
        return

    print(f"\n{'='*90}")
    print(f"  SIMILARITY CLUSTERS: {len(pairs)} flagged pairs")
    print(f"{'='*90}")

    for p in pairs:
        icon = '🔴' if p['jaccard'] >= 0.70 else '🟠' if p['jaccard'] >= 0.50 else '🟡'
        print(f"\n  {icon} Jaccard={p['jaccard']:.2f}  Struct={p['structural']:.2f}  [{p['flag_type']}]")
        print(f"     {p['id_a'][:15]:15s} ({p['attempter_a'] or '?':20s}) [{p['det_a']}]")
        print(f"     {p['id_b'][:15]:15s} ({p['attempter_b'] or '?':20s}) [{p['det_b']}]")
        print(f"     Occupation: {p['occupation'][:50]}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATION (v0.60)
# Builds calibration tables from labeled baseline data.
# Uses split conformal prediction: nonconformity score = 1 - confidence.
# Ref: Vovk et al. (2005), Shafer & Vovk (2008)
# ══════════════════════════════════════════════════════════════════════════════

_CALIBRATION_ALPHAS = [0.01, 0.05, 0.10]  # significance levels


def calibrate_from_baselines(jsonl_path):
    """Build calibration tables from labeled baseline data.

    Reads a JSONL baseline file with ground_truth labels and computes
    per-stratum (domain × length_bin) nonconformity quantiles.

    Returns:
        dict: {
            'global': {alpha: threshold, ...},
            'strata': {(domain, length_bin): {alpha: threshold, ...}},
            'n_calibration': int,
            'strata_counts': {(domain, length_bin): int},
        }
        Returns None if insufficient labeled data (<20 records).
    """
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if rec.get('ground_truth') == 'human':
                        records.append(rec)
                except json.JSONDecodeError:
                    continue

    if len(records) < 20:
        return None

    # Nonconformity scores for human-labeled records
    # Higher score = more "abnormal" (more likely to be flagged as AI)
    nc_scores = [1.0 - float(r.get('confidence', 0)) for r in records]
    nc_scores.sort()

    # Global calibration quantiles
    global_cal = {}
    for alpha in _CALIBRATION_ALPHAS:
        # Quantile at (1 - alpha) gives the threshold
        idx = int(math.ceil((1 - alpha) * (len(nc_scores) + 1))) - 1
        idx = max(0, min(idx, len(nc_scores) - 1))
        global_cal[alpha] = nc_scores[idx]

    # Per-stratum calibration
    strata = defaultdict(list)
    for r in records:
        domain = r.get('domain', 'unknown') or 'unknown'
        length_bin = r.get('length_bin', 'unknown') or 'unknown'
        conf = float(r.get('confidence', 0))
        strata[(domain, length_bin)].append(1.0 - conf)

    strata_cal = {}
    strata_counts = {}
    for key, scores in strata.items():
        scores.sort()
        strata_counts[key] = len(scores)
        if len(scores) >= 10:  # minimum for per-stratum calibration
            strata_cal[key] = {}
            for alpha in _CALIBRATION_ALPHAS:
                idx = int(math.ceil((1 - alpha) * (len(scores) + 1))) - 1
                idx = max(0, min(idx, len(scores) - 1))
                strata_cal[key][alpha] = scores[idx]

    return {
        'global': global_cal,
        'strata': strata_cal,
        'n_calibration': len(records),
        'strata_counts': {f"{k[0]}_{k[1]}": v for k, v in strata_counts.items()},
    }


def apply_calibration(confidence, cal_table, domain=None, length_bin=None):
    """Apply conformal calibration to a raw confidence score.

    Returns:
        dict: {
            'raw_confidence': float,
            'calibrated_confidence': float,
            'p_value': float (conformal p-value against human baseline),
            'stratum_used': str ('global' or 'domain_length_bin'),
        }
    """
    if cal_table is None:
        return {
            'raw_confidence': confidence,
            'calibrated_confidence': confidence,
            'p_value': None,
            'stratum_used': 'uncalibrated',
        }

    nc_score = 1.0 - confidence

    # Try stratum-specific calibration first
    stratum_key = (domain or 'unknown', length_bin or 'unknown')
    if stratum_key in cal_table.get('strata', {}):
        cal = cal_table['strata'][stratum_key]
        stratum_label = f"{stratum_key[0]}_{stratum_key[1]}"
    else:
        cal = cal_table.get('global', {})
        stratum_label = 'global'

    # Compute conformal p-value
    # p-value = fraction of calibration scores >= this score
    # In practice, approximate from the quantiles
    if nc_score <= cal.get(0.01, 0):
        p_value = 1.0  # very typical (conforming)
    elif nc_score <= cal.get(0.05, 0):
        p_value = 0.05
    elif nc_score <= cal.get(0.10, 0):
        p_value = 0.10
    else:
        p_value = 0.01  # very atypical (nonconforming)

    # Calibrated confidence: scale raw confidence by stratum position
    # If this score is above the 95th percentile of human baseline,
    # boost confidence; if below, dampen it.
    alpha_05 = cal.get(0.05, 0.5)
    if nc_score > alpha_05:
        # Score is more extreme than 95% of human baseline → boost
        calibrated = min(confidence * 1.15, 0.99)
    elif nc_score < cal.get(0.10, 0.5):
        # Score is well within human baseline → dampen
        calibrated = confidence * 0.75
    else:
        calibrated = confidence

    return {
        'raw_confidence': round(confidence, 4),
        'calibrated_confidence': round(calibrated, 4),
        'p_value': round(p_value, 4) if p_value is not None else None,
        'stratum_used': stratum_label,
    }


def save_calibration(cal_table, path):
    """Save calibration table to JSON."""
    # Convert tuple keys to strings for JSON serialization
    serializable = {
        'global': cal_table['global'],
        'strata': {f"{k[0]}|{k[1]}": v for k, v in cal_table.get('strata', {}).items()}
                  if isinstance(list(cal_table.get('strata', {}).keys() or [('',)])[0], tuple)
                  else cal_table.get('strata', {}),
        'n_calibration': cal_table['n_calibration'],
        'strata_counts': cal_table.get('strata_counts', {}),
    }
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Calibration table saved to {path} ({cal_table['n_calibration']} records)")


def load_calibration(path):
    """Load calibration table from JSON."""
    with open(path, 'r') as f:
        raw = json.load(f)

    # Reconstruct tuple keys
    strata = {}
    for k, v in raw.get('strata', {}).items():
        parts = k.split('|')
        if len(parts) == 2:
            strata[(parts[0], parts[1])] = {float(ak): av for ak, av in v.items()}
        else:
            strata[k] = v

    return {
        'global': {float(k): v for k, v in raw.get('global', {}).items()},
        'strata': strata,
        'n_calibration': raw.get('n_calibration', 0),
        'strata_counts': raw.get('strata_counts', {}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LABELED DATA COLLECTION + BASELINE ANALYSIS (v0.53)
# ══════════════════════════════════════════════════════════════════════════════

_BASELINE_FIELDS = [
    'task_id', 'occupation', 'attempter', 'word_count', 'determination',
    'confidence', 'l0_score', 'l25_composite', 'l25_cfd', 'l25_mfsr',
    'l25_framing', 'l25_must_rate', 'l25_distinct_frames',
    'l27_idi', 'l27_imperatives', 'l27_conditionals',
    'l26_voice_score', 'l26_spec_score', 'l26_vsd', 'l26_voice_gated',
    'l26_hedges', 'l26_casual_markers', 'l26_misspellings',
    'ssi_triggered',
    # Layer 3 (v0.55)
    'l30_nssi_score', 'l30_nssi_signals', 'l30_determination',
    'l31_bscore', 'l31_determination',
    # v0.60: NSSI diagnostic fields
    'l30_sent_length_cv', 'l30_comp_ratio', 'l30_hapax_ratio',
    # v0.60: Normalization, fairness, metadata
    'norm_obfuscation_delta', 'norm_invisible_chars', 'norm_homoglyphs',
    'lang_support_level', 'lang_fw_coverage', 'lang_non_latin_ratio',
    'ground_truth',        # NEW: 'human' | 'ai' | 'mixed' | None
    'language',            # NEW: ISO 639-1 code or None
    'domain',              # v0.57
    'mode',                # v0.58: 'task_prompt' | 'generic_aigt'
    # v0.59: Windowing and stylometry
    'window_max_score', 'window_mean_score', 'window_variance',
    'window_hot_span', 'window_mixed_signal',
    'stylo_fw_ratio', 'stylo_sent_dispersion', 'stylo_ttr',
    # v0.60: Calibration
    'calibrated_confidence', 'p_value', 'calibration_stratum',
]


def collect_baselines(results, output_path):
    """Append scored results to JSONL file for baseline accumulation."""
    timestamp = datetime.now().isoformat()
    n_written = 0

    with open(output_path, 'a') as f:
        for r in results:
            record = {k: r.get(k) for k in _BASELINE_FIELDS}
            record['_timestamp'] = timestamp
            record['_version'] = 'v0.60'
            # v0.60: Compute length_bin from word_count
            wc = r.get('word_count', 0)
            if wc < 100:
                record['length_bin'] = 'short'
            elif wc < 300:
                record['length_bin'] = 'medium'
            elif wc < 800:
                record['length_bin'] = 'long'
            else:
                record['length_bin'] = 'very_long'
            f.write(json.dumps(record) + '\n')
            n_written += 1

    print(f"\n  Baseline data: {n_written} records appended to {output_path}")
    return n_written


def analyze_baselines(jsonl_path, output_csv=None):
    """Read accumulated baseline JSONL and compute per-occupation percentile tables."""
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print(f"No records found in {jsonl_path}")
        return

    df = pd.DataFrame(records)
    print(f"\n{'='*90}")
    print(f"  BASELINE ANALYSIS — {len(df)} records from {jsonl_path}")
    print(f"{'='*90}")

    det_counts = df['determination'].value_counts()
    print(f"\n  Overall distribution:")
    for det in ['RED', 'AMBER', 'YELLOW', 'GREEN']:
        ct = det_counts.get(det, 0)
        pct = ct / len(df) * 100
        print(f"    {det:>8}: {ct:>5} ({pct:.1f}%)")

    metrics = ['l25_composite', 'l25_cfd', 'l25_mfsr', 'l25_must_rate',
               'l27_idi', 'l26_spec_score', 'l26_voice_score', 'l26_vsd',
               'l30_nssi_score', 'l30_comp_ratio', 'l30_hapax_ratio',
               'l30_sent_length_cv',
               'norm_obfuscation_delta', 'lang_fw_coverage',
               'word_count']
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    occupations = sorted(df['occupation'].dropna().unique())
    if not occupations:
        occupations = ['(all)']
        df['occupation'] = '(all)'

    all_rows = []

    for occ in occupations:
        occ_df = df[df['occupation'] == occ]
        if len(occ_df) < 5:
            continue

        print(f"\n  ── {occ} (n={len(occ_df)}) ──")
        det_pcts = occ_df['determination'].value_counts()
        flags = det_pcts.get('RED', 0) + det_pcts.get('AMBER', 0)
        flag_rate = flags / len(occ_df) * 100
        print(f"     Flag rate: {flag_rate:.1f}% ({flags}/{len(occ_df)})")

        for m in metrics:
            if m not in occ_df.columns:
                continue
            vals = pd.to_numeric(occ_df[m], errors='coerce').dropna()
            if len(vals) < 3:
                continue

            pct_vals = vals.quantile(percentiles).to_dict()
            row = {'occupation': occ, 'metric': m, 'n': len(vals),
                   'mean': vals.mean(), 'std': vals.std()}
            row.update({f'p{int(k*100)}': v for k, v in pct_vals.items()})
            all_rows.append(row)

            p50 = pct_vals.get(0.50, 0)
            p90 = pct_vals.get(0.90, 0)
            p99 = pct_vals.get(0.99, 0)
            print(f"     {m:22s}  p50={p50:7.2f}  p90={p90:7.2f}  p99={p99:7.2f}  mean={vals.mean():7.2f}")

    if output_csv and all_rows:
        baseline_df = pd.DataFrame(all_rows)
        baseline_df.to_csv(output_csv, index=False)
        print(f"\n  Baseline percentiles written to {output_csv}")

    # ── v0.60: TPR@FPR calibration reporting ─────────────────────────────
    # Only available when ground_truth labels are present in the data.
    if 'ground_truth' in df.columns:
        labeled = df[df['ground_truth'].isin(['human', 'ai'])].copy()
        if len(labeled) >= 20:
            n_human = (labeled['ground_truth'] == 'human').sum()
            n_ai = (labeled['ground_truth'] == 'ai').sum()
            print(f"\n  ── TPR @ FPR (n={len(labeled)}: {n_human} human, {n_ai} AI) ──")

            if n_human >= 5 and n_ai >= 5:
                # Use confidence as the score (higher = more likely AI)
                scores = pd.to_numeric(labeled['confidence'], errors='coerce').fillna(0)
                labels = (labeled['ground_truth'] == 'ai').astype(int)

                # Compute TPR at target FPR levels
                thresholds = sorted(scores.unique(), reverse=True)
                for target_fpr, label in [(0.01, '1%'), (0.05, '5%'), (0.10, '10%')]:
                    best_tpr = 0.0
                    best_thresh = 1.0
                    for t in thresholds:
                        predicted_pos = (scores >= t)
                        fp = ((predicted_pos) & (labels == 0)).sum()
                        tp = ((predicted_pos) & (labels == 1)).sum()
                        fpr = fp / max(n_human, 1)
                        tpr = tp / max(n_ai, 1)
                        if fpr <= target_fpr and tpr > best_tpr:
                            best_tpr = tpr
                            best_thresh = t
                    print(f"     TPR @ {label:>3} FPR: {best_tpr:.1%}  (threshold={best_thresh:.3f})")

                # Also report overall flag rates by ground truth
                for gt_label in ['human', 'ai']:
                    subset = labeled[labeled['ground_truth'] == gt_label]
                    flagged = subset['determination'].isin(['RED', 'AMBER']).sum()
                    rate = flagged / max(len(subset), 1) * 100
                    print(f"     Flag rate ({gt_label:>5}): {rate:.1f}% ({flagged}/{len(subset)})")
            else:
                print(f"     Insufficient labeled data (need ≥5 each of human/ai)")
        else:
            print(f"\n  TPR@FPR: insufficient labeled records ({len(labeled)}/20 minimum)")

    # ── v0.60: Stratified reporting by domain × length_bin ───────────
    if 'domain' in df.columns and 'length_bin' in df.columns:
        df['_stratum'] = df['domain'].fillna('unknown').astype(str) + '×' + df['length_bin'].fillna('unknown').astype(str)
        strata = df['_stratum'].unique()
        if len(strata) > 1:
            print(f"\n  ── STRATIFIED FLAG RATES (domain × length_bin) ──")
            stratum_rates = {}
            for s in sorted(strata):
                s_df = df[df['_stratum'] == s]
                if len(s_df) < 3:
                    continue
                flagged = s_df['determination'].isin(['RED', 'AMBER']).sum()
                rate = flagged / len(s_df) * 100
                stratum_rates[s] = rate
                print(f"     {s:30s}  n={len(s_df):>4}  flag_rate={rate:5.1f}%")

            # Flag disparity warnings
            if stratum_rates:
                rates = list(stratum_rates.values())
                max_rate = max(rates)
                min_rate = min(rates)
                if max_rate - min_rate > 20:
                    print(f"\n  ⚠ CALIBRATION WARNING: Flag rate disparity across strata")
                    print(f"     Range: {min_rate:.1f}% — {max_rate:.1f}% (Δ={max_rate - min_rate:.1f}pp)")
                    print(f"     Consider per-stratum threshold calibration.")

    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# FILE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_xlsx(filepath, sheet=None, prompt_col='prompt', id_col='task_id',
              occ_col='occupation', attempter_col='attempter_name', stage_col='pipeline_stage_name'):
    """Load tasks from an xlsx file. Returns list of dicts."""
    import openpyxl
    wb = openpyxl.load_workbook(filepath, read_only=True)

    if sheet:
        ws = wb[sheet]
    else:
        for name in ['FullTaskX', 'Full Task Connected', 'Claim Sheet', 'Sample List']:
            if name in wb.sheetnames:
                ws = wb[name]
                break
        else:
            ws = wb[wb.sheetnames[0]]

    rows = list(ws.iter_rows(min_row=1, values_only=True))
    wb.close()

    if not rows:
        return []

    headers = [str(h).strip().lower() if h else '' for h in rows[0]]

    def find_col(candidates):
        for c in candidates:
            cl = c.lower()
            for i, h in enumerate(headers):
                if cl == h:
                    return i
        for c in candidates:
            cl = c.lower()
            if cl == 'id':
                continue
            for i, h in enumerate(headers):
                if cl in h:
                    return i
        return None

    prompt_idx = find_col([prompt_col, 'prompt', 'text', 'content'])
    id_idx = find_col([id_col, 'task_id', 'id'])
    occ_idx = find_col([occ_col, 'occupation', 'occ'])
    att_idx = find_col([attempter_col, 'attempter', 'claimed_by', 'claimed by'])
    stage_idx = find_col([stage_col, 'stage', 'pipeline_stage'])

    if prompt_idx is None:
        print(f"ERROR: Could not find prompt column. Headers: {headers}")
        return []

    tasks = []
    for row in rows[1:]:
        if not row or len(row) <= prompt_idx:
            continue
        prompt = str(row[prompt_idx]).strip() if row[prompt_idx] else ''
        if len(prompt) < 50:
            continue

        tasks.append({
            'prompt': prompt,
            'task_id': str(row[id_idx])[:20] if id_idx is not None and row[id_idx] else '',
            'occupation': str(row[occ_idx]) if occ_idx is not None and row[occ_idx] else '',
            'attempter': str(row[att_idx]) if att_idx is not None and row[att_idx] else '',
            'stage': str(row[stage_idx]) if stage_idx is not None and row[stage_idx] else '',
        })

    return tasks


def load_csv(filepath, prompt_col='prompt'):
    """Load tasks from CSV."""
    df = pd.read_csv(filepath)
    df = df.fillna('')

    col_map = {c.lower().strip(): c for c in df.columns}

    def resolve_col(*candidates):
        for c in candidates:
            key = c.lower().strip()
            if key in col_map:
                return col_map[key]
        for c in candidates:
            key = c.lower().strip()
            if key == 'id':
                continue
            for mapped_key, actual in col_map.items():
                if key in mapped_key:
                    return actual
        return None

    prompt_actual = resolve_col(prompt_col, 'prompt', 'text', 'content')
    id_actual = resolve_col('task_id', 'id')
    occ_actual = resolve_col('occupation', 'occ')
    att_actual = resolve_col('attempter_name', 'attempter', 'claimed_by')
    stage_actual = resolve_col('pipeline_stage_name', 'stage')

    if prompt_actual is None:
        print(f"ERROR: Could not find prompt column. Columns: {list(df.columns)}")
        return []

    tasks = []
    for _, row in df.iterrows():
        prompt = str(row.get(prompt_actual, ''))
        if len(prompt) < 50:
            continue
        tasks.append({
            'prompt': prompt,
            'task_id': str(row.get(id_actual, ''))[:20] if id_actual else '',
            'occupation': str(row.get(occ_actual, '')) if occ_actual else '',
            'attempter': str(row.get(att_actual, '')) if att_actual else '',
            'stage': str(row.get(stage_actual, '')) if stage_actual else '',
        })
    return tasks


def load_pdf(filepath):
    """Load text from PDF file. Each page becomes a separate task.

    Requires: pip install pypdf
    Returns list of dicts with 'prompt' and 'task_id' keys.
    """
    if not HAS_PYPDF:
        print("ERROR: pypdf not installed. Run: pip install pypdf")
        return []

    reader = PdfReader(filepath)
    tasks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and len(text.strip()) >= 50:
            tasks.append({
                'prompt': text.strip(),
                'task_id': f"page_{i+1}",
                'occupation': '',
                'attempter': '',
                'stage': '',
            })

    if not tasks:
        # If pages are too short individually, try combining all pages
        full_text = '\n'.join(
            page.extract_text() for page in reader.pages
            if page.extract_text()
        ).strip()
        if len(full_text) >= 50:
            tasks.append({
                'prompt': full_text,
                'task_id': 'full_document',
                'occupation': '',
                'attempter': '',
                'stage': '',
            })

    return tasks


class DetectorGUI:
    """Simple desktop GUI for v0.60 single-text and file analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("LLM Detector Pipeline v0.61")
        self.root.geometry("1040x760")

        self.file_var = tk.StringVar()
        self.prompt_col_var = tk.StringVar(value='prompt')
        self.sheet_var = tk.StringVar()
        self.attempter_var = tk.StringVar()
        self.provider_var = tk.StringVar(value='anthropic')
        self.api_key_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')

        self._build_layout()

    def _build_layout(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        file_row = ttk.Frame(frame)
        file_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(file_row, text='Input file (CSV/XLSX):').pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(file_row, text='Browse', command=self._browse_file).pack(side=tk.LEFT)

        opts = ttk.Frame(frame)
        opts.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(opts, text='Prompt column').grid(row=0, column=0, sticky='w')
        ttk.Entry(opts, textvariable=self.prompt_col_var, width=18).grid(row=0, column=1, sticky='w', padx=6)
        ttk.Label(opts, text='Sheet (xlsx)').grid(row=0, column=2, sticky='w')
        ttk.Entry(opts, textvariable=self.sheet_var, width=16).grid(row=0, column=3, sticky='w', padx=6)
        ttk.Label(opts, text='Attempter filter').grid(row=0, column=4, sticky='w')
        ttk.Entry(opts, textvariable=self.attempter_var, width=18).grid(row=0, column=5, sticky='w', padx=6)

        l3 = ttk.LabelFrame(frame, text='Layer 3.1 (DNA-GPT)')
        l3.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(l3, text='Provider').grid(row=0, column=0, sticky='w', padx=6, pady=6)
        ttk.Combobox(l3, textvariable=self.provider_var, values=['anthropic', 'openai'], width=12, state='readonly').grid(row=0, column=1, sticky='w', pady=6)
        ttk.Label(l3, text='API Key (optional)').grid(row=0, column=2, sticky='w', padx=(16, 6), pady=6)
        ttk.Entry(l3, textvariable=self.api_key_var, show='*').grid(row=0, column=3, sticky='ew', padx=(0, 6), pady=6)
        l3.columnconfigure(3, weight=1)

        ttk.Label(frame, text='Single text input (optional):').pack(anchor='w')
        self.text_input = tk.Text(frame, height=10, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, pady=(4, 8))

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text='Analyze Text', command=lambda: self._run_async(self._analyze_text)).pack(side=tk.LEFT)
        ttk.Button(actions, text='Analyze File', command=lambda: self._run_async(self._analyze_file)).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text='Clear Output', command=self._clear_output).pack(side=tk.LEFT)

        ttk.Label(frame, text='Results:').pack(anchor='w')
        self.output = tk.Text(frame, height=20, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, textvariable=self.status_var).pack(anchor='w', pady=(8, 0))

    def _browse_file(self):
        path = filedialog.askopenfilename(filetypes=[('Data files', '*.csv *.xlsx *.xlsm'), ('All files', '*.*')])
        if path:
            self.file_var.set(path)

    def _clear_output(self):
        self.output.delete('1.0', tk.END)
        self.status_var.set('Ready')

    def _run_async(self, fn):
        self.status_var.set('Running...')

        def runner():
            try:
                fn()
                self.root.after(0, lambda: self.status_var.set('Done'))
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set('Error'))
                self.root.after(0, lambda: messagebox.showerror('Analysis Error', str(exc)))

        threading.Thread(target=runner, daemon=True).start()

    def _append(self, text):
        self.root.after(0, lambda: (self.output.insert(tk.END, text), self.output.see(tk.END)))

    def _analyze_text(self):
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Enter text to analyze.'))
            return
        result = analyze_prompt(
            text,
            run_l3=True,
            api_key=self.api_key_var.get().strip() or None,
            dna_provider=self.provider_var.get(),
        )
        self._append(self._format_result(result) + '\n')

    def _analyze_file(self):
        path = self.file_var.get().strip()
        if not path:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Choose a CSV/XLSX file to analyze.'))
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.xlsx', '.xlsm'):
            tasks = load_xlsx(path, sheet=self.sheet_var.get().strip() or None, prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        elif ext == '.csv':
            tasks = load_csv(path, prompt_col=self.prompt_col_var.get().strip() or 'prompt')
        else:
            self.root.after(0, lambda: messagebox.showerror('Unsupported file', f'Unsupported extension: {ext}'))
            return
        if self.attempter_var.get().strip():
            needle = self.attempter_var.get().strip().lower()
            tasks = [t for t in tasks if needle in t.get('attempter', '').lower()]
        if not tasks:
            self.root.after(0, lambda: messagebox.showinfo('No tasks', 'No qualifying prompts found.'))
            return

        api_key = self.api_key_var.get().strip() or None
        counts = Counter()
        for i, task in enumerate(tasks, 1):
            r = analyze_prompt(
                task['prompt'],
                task_id=task.get('task_id', ''),
                occupation=task.get('occupation', ''),
                attempter=task.get('attempter', ''),
                stage=task.get('stage', ''),
                run_l3=True,
                api_key=api_key,
                dna_provider=self.provider_var.get(),
            )
            counts[r['determination']] += 1
            self._append(f"[{i}/{len(tasks)}] {self._format_result(r)}\n")

        summary = (
            f"\nSummary: RED={counts.get('RED', 0)} | AMBER={counts.get('AMBER', 0)} "
            f"| YELLOW={counts.get('YELLOW', 0)} | GREEN={counts.get('GREEN', 0)}\n"
        )
        self._append(summary)

    @staticmethod
    def _format_result(result):
        return (
            f"{result.get('determination')} | conf={result.get('confidence', 0):.2f} | "
            f"words={result.get('word_count', 0)} | reason={result.get('reason', '')}"
        )


def launch_gui():
    """Launch Tkinter GUI mode."""
    if not HAS_TK:
        print('ERROR: tkinter is not available in this Python environment.')
        return
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='LLM Detection Pipeline v0.61')
    parser.add_argument('input', nargs='?', help='Input file (.xlsx, .csv, or .pdf)')
    parser.add_argument('--gui', action='store_true', help='Launch desktop GUI mode')
    parser.add_argument('--text', help='Analyze a single text string')
    parser.add_argument('--sheet', help='Sheet name for xlsx files')
    parser.add_argument('--prompt-col', default='prompt', help='Column name containing prompts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all layer details')
    parser.add_argument('--output', '-o', help='Output CSV path')
    parser.add_argument('--attempter', help='Filter by attempter name (substring match)')
    parser.add_argument('--no-similarity', action='store_true',
                        help='Skip cross-submission similarity analysis')
    parser.add_argument('--similarity-threshold', type=float, default=0.40,
                        help='Jaccard threshold for text similarity (default: 0.40)')
    parser.add_argument('--collect', metavar='PATH',
                        help='Append scored results to JSONL file for baseline accumulation')
    parser.add_argument('--analyze-baselines', metavar='JSONL',
                        help='Read accumulated JSONL and print per-occupation percentile tables')
    parser.add_argument('--baselines-csv', metavar='PATH',
                        help='Write baseline percentile tables to CSV (use with --analyze-baselines)')
    parser.add_argument('--no-layer3', action='store_true',
                        help='Skip Layer 3 entirely (NSSI + DNA-GPT)')
    parser.add_argument('--api-key', metavar='KEY',
                        help='API key for Layer 3.1 (DNA-GPT). Falls back to '
                             'ANTHROPIC_API_KEY or OPENAI_API_KEY env var.')
    parser.add_argument('--provider', default='anthropic', choices=['anthropic', 'openai'],
                        help='LLM provider for DNA-GPT (default: anthropic)')
    parser.add_argument('--dna-model', metavar='MODEL',
                        help='Model name for DNA-GPT (default: auto per provider)')
    parser.add_argument('--dna-samples', type=int, default=3,
                        help='Number of regeneration samples for DNA-GPT (default: 3)')
    parser.add_argument('--mode', default='auto', choices=['task_prompt', 'generic_aigt', 'auto'],
                        help='Detection mode: task_prompt (prompt-structure primary), '
                             'generic_aigt (all channels), auto (heuristic). Default: auto')
    parser.add_argument('--calibrate', metavar='JSONL',
                        help='Build calibration table from labeled baseline JSONL and save to --cal-table')
    parser.add_argument('--cal-table', metavar='JSON',
                        help='Path to calibration table JSON (load for scoring, or save target for --calibrate)')
    args = parser.parse_args()

    if args.gui:
        launch_gui()
        return

    if not args.api_key:
        env_key = 'ANTHROPIC_API_KEY' if args.provider == 'anthropic' else 'OPENAI_API_KEY'
        args.api_key = os.environ.get(env_key)

    if args.analyze_baselines:
        if not os.path.exists(args.analyze_baselines):
            print(f"ERROR: File not found: {args.analyze_baselines}")
            return
        analyze_baselines(args.analyze_baselines, output_csv=args.baselines_csv)
        return

    # v0.60: Build calibration table from labeled baselines
    if args.calibrate:
        if not os.path.exists(args.calibrate):
            print(f"ERROR: File not found: {args.calibrate}")
            return
        cal = calibrate_from_baselines(args.calibrate)
        if cal is None:
            print("ERROR: Insufficient labeled human data for calibration (need ≥20)")
            return
        cal_path = args.cal_table or args.calibrate.replace('.jsonl', '_calibration.json')
        save_calibration(cal, cal_path)
        print(f"  Global quantiles: {cal['global']}")
        print(f"  Strata: {len(cal.get('strata', {}))} domain×length_bin tables")
        return

    # v0.60: Load calibration table if provided
    cal_table = None
    if args.cal_table and os.path.exists(args.cal_table):
        cal_table = load_calibration(args.cal_table)
        print(f"Loaded calibration table: {cal_table['n_calibration']} records, "
              f"{len(cal_table.get('strata', {}))} strata")

    run_l3 = not args.no_layer3

    if args.text:
        result = analyze_prompt(
            args.text, run_l3=run_l3,
            api_key=args.api_key, dna_provider=args.provider,
            dna_model=args.dna_model, dna_samples=args.dna_samples,
            mode=args.mode, cal_table=cal_table,
        )
        print_result(result, verbose=True)
        return

    if not args.input:
        parser.print_help()
        return

    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.xlsx', '.xlsm'):
        tasks = load_xlsx(args.input, sheet=args.sheet, prompt_col=args.prompt_col)
    elif ext == '.csv':
        tasks = load_csv(args.input, prompt_col=args.prompt_col)
    elif ext == '.pdf':
        tasks = load_pdf(args.input)
    else:
        print(f"ERROR: Unsupported file type: {ext}")
        return

    if not tasks:
        print("ERROR: No tasks found.")
        return

    if args.attempter:
        tasks = [t for t in tasks if args.attempter.lower() in t.get('attempter', '').lower()]
        print(f"Filtered to {len(tasks)} tasks matching attempter '{args.attempter}'")

    layer3_label = " + L3" if run_l3 else ""
    dna_label = " + DNA-GPT" if args.api_key else ""
    print(f"Processing {len(tasks)} tasks through pipeline v0.61{layer3_label}{dna_label}...")

    results = []
    text_map = {}
    for i, task in enumerate(tasks):
        r = analyze_prompt(
            task['prompt'],
            task_id=task.get('task_id', ''),
            occupation=task.get('occupation', ''),
            attempter=task.get('attempter', ''),
            stage=task.get('stage', ''),
            run_l3=run_l3,
            api_key=args.api_key,
            dna_provider=args.provider,
            dna_model=args.dna_model,
            dna_samples=args.dna_samples,
            mode=args.mode,
            cal_table=cal_table,
        )
        results.append(r)
        tid = task.get('task_id', f'_row{i}')
        text_map[tid] = task['prompt']
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(tasks)}...")

    det_counts = Counter(r['determination'] for r in results)
    print(f"\n{'='*90}")
    print(f"  PIPELINE v0.61 RESULTS (n={len(results)})")
    print(f"{'='*90}")
    for det in ['RED', 'AMBER', 'YELLOW', 'GREEN']:
        ct = det_counts.get(det, 0)
        pct = ct / len(results) * 100
        icons = {'RED': '🔴', 'AMBER': '🟠', 'YELLOW': '🟡', 'GREEN': '🟢', 'MIXED': '🔵', 'REVIEW': '⚪'}
        print(f"  {icons[det]} {det:>8}: {ct:>4} ({pct:.1f}%)")

    flagged = [r for r in results if r['determination'] in ('RED', 'AMBER')]
    if flagged:
        print(f"\n{'='*90}")
        print(f"  FLAGGED SUBMISSIONS: {len(flagged)}")
        print(f"{'='*90}")
        for r in sorted(flagged, key=lambda x: x['confidence'], reverse=True):
            print_result(r, verbose=args.verbose)

    yellow = [r for r in results if r['determination'] == 'YELLOW']
    if yellow:
        print(f"\n  YELLOW ({len(yellow)} minor signals):")
        for r in sorted(yellow, key=lambda x: x['confidence'], reverse=True)[:10]:
            print(f"    🟡 {r['task_id'][:12]:12} {r['occupation'][:40]:40} | {r['reason'][:50]}")

    if not args.no_similarity and len(results) >= 2:
        sim_pairs = analyze_similarity(
            results, text_map,
            jaccard_threshold=args.similarity_threshold,
        )
        print_similarity_report(sim_pairs)
    else:
        sim_pairs = []

    default_name = os.path.basename(args.input).rsplit('.', 1)[0] + '_pipeline_v056.csv'
    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_path = args.output or os.path.join(input_dir, default_name)

    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != 'l0_details'}
        row['l0_details'] = str(r.get('l0_details', []))
        flat.append(row)

    if sim_pairs:
        sim_lookup = defaultdict(list)
        for p in sim_pairs:
            sim_lookup[p['id_a']].append(f"{p['id_b']}(J={p['jaccard']:.2f})")
            sim_lookup[p['id_b']].append(f"{p['id_a']}(J={p['jaccard']:.2f})")
        for row in flat:
            tid = row.get('task_id', '')
            row['similarity_flags'] = '; '.join(sim_lookup.get(tid, []))

    pd.DataFrame(flat).to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")

    if args.collect:
        collect_baselines(results, args.collect)


if __name__ == '__main__':
    main()
