"""Tests for individual analyzer modules."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.compat import HAS_SEMANTIC, HAS_PERPLEXITY
from llm_detector.analyzers.semantic_resonance import run_semantic_resonance
from llm_detector.analyzers.perplexity import run_perplexity
from llm_detector.analyzers.preamble import run_preamble
from tests.conftest import AI_TEXT, HUMAN_TEXT, CLINICAL_TEXT

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  -- {detail}")


def test_semantic_resonance():
    print("\n-- SEMANTIC RESONANCE --")

    short = "Hello world."
    r_short = run_semantic_resonance(short)
    check("Short text: no determination", r_short['determination'] is None)

    if HAS_SEMANTIC:
        r_ai = run_semantic_resonance(AI_TEXT)
        check("AI text: semantic_ai_score > 0", r_ai['semantic_ai_score'] > 0,
              f"got {r_ai['semantic_ai_score']}")
        check("AI text: semantic_delta > 0", r_ai['semantic_delta'] > 0,
              f"got {r_ai['semantic_delta']}")
        # determination may be None for short texts (needs ≥5 sentences)
        check("AI text: has determination or delta > 0",
              r_ai['determination'] is not None or r_ai['semantic_delta'] > 0,
              f"got det={r_ai['determination']}, delta={r_ai['semantic_delta']}")

        r_human = run_semantic_resonance(HUMAN_TEXT)
        check("Human text: lower ai_score", r_human['semantic_ai_score'] < r_ai['semantic_ai_score'],
              f"human={r_human['semantic_ai_score']}, ai={r_ai['semantic_ai_score']}")
    else:
        print("  (sentence-transformers not installed -- skipping model tests)")
        check("Unavailable: ai_score=0", r_short['semantic_ai_score'] == 0.0)


def test_perplexity():
    print("\n-- PERPLEXITY SCORING --")

    short = "Hello world."
    r_short = run_perplexity(short)
    check("Short text: no determination", r_short['determination'] is None)

    # Early return dicts should include variance fields
    check("Short text: surprisal_variance=0", r_short['surprisal_variance'] == 0.0)
    check("Short text: volatility_decay_ratio=1", r_short['volatility_decay_ratio'] == 1.0)

    if HAS_PERPLEXITY:
        r_normal = run_perplexity(CLINICAL_TEXT)
        check("Normal text: perplexity > 0", r_normal['perplexity'] > 0,
              f"got {r_normal['perplexity']}")
        check("Normal text: has reason", len(r_normal.get('reason', '')) > 0)
        # Surprisal diversity features
        check("Normal text: surprisal_variance > 0", r_normal['surprisal_variance'] > 0,
              f"got {r_normal['surprisal_variance']}")
        check("Normal text: volatility_decay_ratio > 0", r_normal['volatility_decay_ratio'] > 0,
              f"got {r_normal['volatility_decay_ratio']}")
        check("Normal text: has first_half_var", 'surprisal_first_half_var' in r_normal)
        check("Normal text: has second_half_var", 'surprisal_second_half_var' in r_normal)
    else:
        print("  (transformers/torch not installed -- skipping model tests)")
        check("Unavailable: perplexity=0", r_short['perplexity'] == 0.0)


def test_cot_leakage():
    print("\n-- COT LEAKAGE DETECTION --")

    # <think> tags — smoking gun for reasoning model artifacts
    text_think = "Here is the analysis.\n<think>\nLet me consider the options.\n</think>\nThe answer is 42."
    score, severity, hits = run_preamble(text_think)
    hit_names = [h[0] for h in hits]
    check("think tags: CRITICAL severity", severity == "CRITICAL")
    check("think tags: cot_leakage in hits", "cot_leakage" in hit_names)
    check("think tags: score == 0.99", score == 0.99)

    # Self-correction phrases
    text_correct = "The total revenue is $50M. Wait, actually let me recalculate that figure."
    score2, severity2, hits2 = run_preamble(text_correct)
    hit_names2 = [h[0] for h in hits2]
    check("self-correction: detected", len(hits2) > 0)
    check("self-correction: cot_self_correction in hits", "cot_self_correction" in hit_names2)

    # Reasoning model phrases
    text_reason = "Let me rethink the approach to this problem."
    score3, severity3, hits3 = run_preamble(text_reason)
    hit_names3 = [h[0] for h in hits3]
    check("cot reasoning: detected", "cot_reasoning" in hit_names3)

    # Clean text should not trigger
    clean = "The quarterly report shows steady growth across all divisions."
    score4, severity4, hits4 = run_preamble(clean)
    check("clean text: no CoT hits", all(h[0] not in ('cot_leakage', 'cot_reasoning', 'cot_self_correction') for h in hits4))


def test_feature_flags():
    print("\n-- FEATURE AVAILABILITY FLAGS --")
    check("HAS_SEMANTIC is bool", isinstance(HAS_SEMANTIC, bool))
    check("HAS_PERPLEXITY is bool", isinstance(HAS_PERPLEXITY, bool))
    print(f"    HAS_SEMANTIC={HAS_SEMANTIC}, HAS_PERPLEXITY={HAS_PERPLEXITY}")


if __name__ == '__main__':
    print("=" * 70)
    print("Analyzer Tests")
    print("=" * 70)

    test_feature_flags()
    test_semantic_resonance()
    test_perplexity()
    test_cot_leakage()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
