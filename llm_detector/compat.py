"""
Feature detection and optional dependency management.

Centralizes all try/except ImportError blocks so other modules can check
availability flags without repeating import logic.

Models are lazily loaded on first use via getter functions to avoid
slow imports when only checking flags or running --help.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# ── tkinter ──────────────────────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ── spaCy: lightweight sentencizer ──────────────────────────────────────────
try:
    import spacy  # noqa: F401
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logger.info("spacy not installed. Sentence segmentation will use regex fallback.")

_nlp = None

def get_nlp():
    """Return spaCy sentencizer, initializing on first call."""
    global _nlp
    if _nlp is None and HAS_SPACY:
        from spacy.lang.en import English
        _nlp = English()
        _nlp.add_pipe("sentencizer")
    return _nlp

# ── ftfy: robust text encoding repair ───────────────────────────────────────
try:
    import ftfy  # noqa: F401
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# ── sentence-transformers: semantic vector analysis ─────────────────────────
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
    import numpy  # noqa: F401
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
except Exception as e:
    HAS_SEMANTIC = False
    logger.info("sentence-transformers setup failed (%s). Semantic layer disabled.", e)

_EMBEDDER = None
_AI_CENTROIDS = None
_HUMAN_CENTROIDS = None
_EXTRA_CENTROID_PATHS = []


def register_centroid_path(directory):
    """Register an additional directory to search for centroid files.

    Call this before the first analysis so that get_semantic_models()
    picks up centroids from a custom --memory store directory.
    Resets cached centroids so the new path takes effect.
    """
    global _AI_CENTROIDS, _HUMAN_CENTROIDS
    centroid_file = os.path.join(str(directory), 'centroids', 'centroids_latest.npz')
    if centroid_file not in _EXTRA_CENTROID_PATHS:
        _EXTRA_CENTROID_PATHS.insert(0, centroid_file)
        # Reset cached centroids so they reload from the new path
        _AI_CENTROIDS = None
        _HUMAN_CENTROIDS = None


def get_semantic_models():
    """Return (embedder, ai_centroids, human_centroids), loading on first call.

    Checks registered memory-store paths, then .beet/centroids/centroids_latest.npz,
    before falling back to hardcoded archetypes.
    """
    global _EMBEDDER, _AI_CENTROIDS, _HUMAN_CENTROIDS
    if _EMBEDDER is None and HAS_SEMANTIC:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')

    if _AI_CENTROIDS is None and HAS_SEMANTIC:
        import numpy as np

        # Try data-derived centroids: registered paths first, then defaults
        centroid_paths = list(_EXTRA_CENTROID_PATHS) + [
            '.beet/centroids/centroids_latest.npz',
            os.path.expanduser('~/.beet/centroids/centroids_latest.npz'),
        ]

        loaded = False
        for cpath in centroid_paths:
            if os.path.exists(cpath):
                try:
                    data = np.load(cpath)
                    if 'ai_multi' in data and data['ai_multi'].shape[0] > 1:
                        _AI_CENTROIDS = data['ai_multi']
                        _HUMAN_CENTROIDS = data['human_multi']
                    else:
                        _AI_CENTROIDS = data['ai_centroid']
                        _HUMAN_CENTROIDS = data['human_centroid']
                    loaded = True
                    break
                except Exception:
                    continue

        if not loaded:
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
    return _EMBEDDER, _AI_CENTROIDS, _HUMAN_CENTROIDS

# ── transformers: local perplexity scoring ──────────────────────────────────
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: F401
    import torch  # noqa: F401
    HAS_PERPLEXITY = True
except ImportError:
    HAS_PERPLEXITY = False
except Exception as e:
    HAS_PERPLEXITY = False
    logger.info("transformers/torch setup failed (%s). Perplexity scoring disabled.", e)

_PPL_MODEL = None
_PPL_TOKENIZER = None

def get_perplexity_model():
    """Return (model, tokenizer), loading on first call."""
    global _PPL_MODEL, _PPL_TOKENIZER
    if _PPL_MODEL is None and HAS_PERPLEXITY:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _PPL_MODEL_ID = 'distilgpt2'
        _PPL_MODEL = GPT2LMHeadModel.from_pretrained(_PPL_MODEL_ID)
        _PPL_TOKENIZER = GPT2TokenizerFast.from_pretrained(_PPL_MODEL_ID)
        _PPL_MODEL.eval()
    return _PPL_MODEL, _PPL_TOKENIZER

# ── pypdf: PDF text extraction ──────────────────────────────────────────────
try:
    from pypdf import PdfReader  # noqa: F401
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
