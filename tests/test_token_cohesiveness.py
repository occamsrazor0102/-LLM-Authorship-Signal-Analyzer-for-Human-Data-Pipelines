"""Tests for Token Cohesiveness (TOCSIN) analyzer."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.compat import HAS_SEMANTIC
from llm_detector.analyzers.token_cohesiveness import run_token_cohesiveness
from tests.conftest import AI_TEXT, HUMAN_TEXT

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


def test_short_text():
    print("\n-- TOCSIN: short text --")
    r = run_token_cohesiveness("Hello world.")
    check("Short text: no determination", r['determination'] is None)
    check("Short text: cohesiveness == 0", r['cohesiveness'] == 0.0)
    check("Short text: reason mentions short", 'short' in r['reason'].lower())


def test_has_semantic_guard():
    print("\n-- TOCSIN: HAS_SEMANTIC guard --")
    if not HAS_SEMANTIC:
        r = run_token_cohesiveness(AI_TEXT)
        check("No semantic: determination is None", r['determination'] is None)
        check("No semantic: reason mentions unavailable", 'unavailable' in r['reason'].lower())
    else:
        print("  (sentence-transformers installed -- guard test skipped)")
        check("HAS_SEMANTIC is True", True)


def test_result_structure():
    print("\n-- TOCSIN: result structure --")
    long_text = (
        "The comprehensive analysis provides a thorough examination of the key factors. "
        "Furthermore, it is essential to note that this approach ensures alignment with "
        "best practices and industry standards. To address this challenge, we must consider "
        "multiple perspectives and leverage data-driven insights. Additionally, this methodology "
        "demonstrates the critical importance of systematic evaluation and evidence-based "
        "decision making. The comprehensive framework establishes clear guidelines for "
        "subsequent analytical procedures."
    )
    r = run_token_cohesiveness(long_text)
    check("Has cohesiveness", 'cohesiveness' in r)
    check("Has cohesiveness_std", 'cohesiveness_std' in r)
    check("Has n_rounds", 'n_rounds' in r)
    check("Has determination", 'determination' in r)
    check("Has confidence", 'confidence' in r)
    check("Has reason", 'reason' in r)

    if HAS_SEMANTIC:
        check("cohesiveness >= 0", r['cohesiveness'] >= 0.0,
              f"got {r['cohesiveness']}")
        check("n_rounds > 0", r['n_rounds'] > 0,
              f"got {r['n_rounds']}")


if __name__ == '__main__':
    print("=" * 70)
    print("  TOKEN COHESIVENESS (TOCSIN) TESTS")
    print("=" * 70)

    test_short_text()
    test_has_semantic_guard()
    test_result_structure()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
