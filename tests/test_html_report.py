"""Tests for HTML report generator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.html_report import generate_html_report, _apply_highlights

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


def test_apply_highlights_basic():
    print("\n-- APPLY HIGHLIGHTS: basic --")
    text = "You must include all fields."
    spans = [{'start': 4, 'end': 8, 'text': 'must', 'pack': 'obligation', 'weight': 1.0}]
    html_out = _apply_highlights(text, spans)
    check("contains signal class", 'class="signal' in html_out, f"got: {html_out[:100]}")
    check("contains 'must'", 'must' in html_out)
    check("contains title attr", 'title="obligation"' in html_out, f"got: {html_out[:200]}")


def test_apply_highlights_overlapping():
    print("\n-- APPLY HIGHLIGHTS: overlapping spans (highest severity wins) --")
    text = "MUST include"
    spans = [
        {'start': 0, 'end': 4, 'text': 'MUST', 'pack': 'obligation', 'weight': 1.0, 'type': 'pattern'},
        {'start': 0, 'end': 4, 'text': 'MUST', 'pack': 'obligation', 'weight': 2.0, 'type': 'uppercase'},
    ]
    html_out = _apply_highlights(text, spans)
    # uppercase has higher priority than pattern, so it should win
    check("uppercase class wins", 'signal-uppercase' in html_out, f"got: {html_out}")


def test_apply_highlights_empty():
    print("\n-- APPLY HIGHLIGHTS: empty spans --")
    text = "Hello <world> & 'friends'"
    html_out = _apply_highlights(text, [])
    check("no spans -> escaped text", '&lt;world&gt;' in html_out, f"got: {html_out}")
    check("ampersand escaped", '&amp;' in html_out)


def test_generate_html_report_returns_string():
    print("\n-- GENERATE HTML REPORT: returns string --")
    text = "You must include all required fields."
    result = {
        'detection_spans': [
            {'start': 4, 'end': 8, 'text': 'must', 'pack': 'obligation', 'weight': 1.0},
        ],
        'determination': 'RED',
        'reason': 'High obligation density',
        'confidence': 0.85,
        'task_id': 'test_001',
        'word_count': 7,
        'mode': 'task_prompt',
        'channel_details': {
            'channels': {
                'prompt_structure': {'severity': 'RED', 'explanation': 'High pack scores'},
                'stylometry': {'severity': 'GREEN', 'explanation': 'Normal'},
            },
        },
    }
    html_out = generate_html_report(text, result)
    check("returns string", isinstance(html_out, str))
    check("contains DOCTYPE", '<!DOCTYPE html>' in html_out)
    check("contains determination", 'RED' in html_out)
    check("contains task_id", 'test_001' in html_out)
    check("contains signal span", 'class="signal' in html_out)


if __name__ == '__main__':
    print("=" * 70)
    print("  HTML REPORT GENERATOR TESTS")
    print("=" * 70)

    test_apply_highlights_basic()
    test_apply_highlights_overlapping()
    test_apply_highlights_empty()
    test_generate_html_report_returns_string()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
