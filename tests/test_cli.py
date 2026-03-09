"""Tests for CLI argument parsing and --disable-channel functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_cli_argparse_no_crash():
    """Ensure CLI parser can be constructed without errors (regression: duplicate --disable-channel)."""
    print("\n-- CLI ARGPARSE CONSTRUCTION --")
    import argparse
    # Importing main triggers the parser construction; a duplicate argument
    # definition would raise argparse.ArgumentError.
    from unittest.mock import patch
    try:
        from llm_detector.cli import main
        # Trigger parser construction by calling with --help, capturing SystemExit
        with patch('sys.argv', ['llm-detector', '--help']):
            try:
                main()
            except SystemExit:
                pass
        check("CLI parser constructed without errors", True)
    except argparse.ArgumentError as e:
        check("CLI parser constructed without errors", False, str(e))


def test_disable_channel_names_match_fusion():
    """Ensure --disable-channel valid names match actual channel names in fusion engine."""
    print("\n-- DISABLE-CHANNEL NAME CONSISTENCY --")
    from llm_detector.channels.prompt_structure import score_prompt_structure
    from llm_detector.channels.stylometric import score_stylometric
    from llm_detector.channels.continuation import score_continuation
    from llm_detector.channels.windowed import score_windowed

    # Get actual channel names from ChannelResult objects
    prompt_sig = {'composite': 0, 'framing_completeness': 0, 'cfd': 0,
                  'mfsr': 0, 'must_rate': 0, 'distinct_frames': 0,
                  'conditional_density': 0, 'meta_design_hits': 0,
                  'contractions': 0, 'numbered_criteria': 0,
                  'pack_constraint_score': 0, 'pack_exec_spec_score': 0,
                  'pack_boost': 0, 'pack_active_families': 0, 'pack_spans': []}
    voice_dis = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                 'spec_score': 0, 'contractions': 0, 'hedges': 0,
                 'casual_markers': 0, 'misspellings': 0, 'camel_cols': 0,
                 'calcs': 0, 'pack_schema_score': 0, 'pack_spans': []}

    ch_ps = score_prompt_structure(0, 'NONE', prompt_sig, voice_dis, None, 100)
    ch_st = score_stylometric(0, None)
    ch_co = score_continuation(None)
    ch_wi = score_windowed(None)

    actual_names = {ch_ps.channel, ch_st.channel, ch_co.channel, ch_wi.channel}
    expected_names = {'prompt_structure', 'stylometry', 'continuation', 'windowing'}

    check("Actual channel names match expected",
          actual_names == expected_names,
          f"actual={actual_names}, expected={expected_names}")


def test_version_consistency():
    """Ensure version strings are consistent across package."""
    print("\n-- VERSION CONSISTENCY --")
    import llm_detector
    check("Package version is 0.66.0",
          llm_detector.__version__ == '0.66.0',
          f"got {llm_detector.__version__}")


if __name__ == '__main__':
    print("=" * 70)
    print("CLI Tests")
    print("=" * 70)

    test_cli_argparse_no_crash()
    test_disable_channel_names_match_fusion()
    test_version_consistency()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
