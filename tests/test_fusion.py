"""Tests for the fusion/determine module and channel scoring."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.fusion import determine
from llm_detector.channels.stylometric import score_stylometric

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


def test_stylometry_integration():
    print("\n-- STYLOMETRY INTEGRATION (semantic + perplexity) --")

    ch_none = score_stylometric(0, None, semantic=None, ppl=None)
    check("No signals -> GREEN", ch_none.severity == 'GREEN')

    l30_r = {'determination': 'RED', 'nssi_score': 0.8, 'nssi_signals': 7, 'confidence': 0.85}
    ch_r = score_stylometric(0, l30_r, semantic=None, ppl=None)
    check("NSSI RED still works", ch_r.severity == 'RED')

    l28_amber = {
        'determination': 'AMBER', 'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20, 'confidence': 0.55,
    }
    ch_sem = score_stylometric(0, None, semantic=l28_amber, ppl=None)
    check("Semantic AMBER alone -> AMBER", ch_sem.severity == 'AMBER',
          f"got {ch_sem.severity}")
    check("Semantic in sub_signals", 'semantic_delta' in ch_sem.sub_signals)

    ppl_yellow = {
        'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30,
    }
    ch_ppl = score_stylometric(0, None, semantic=None, ppl=ppl_yellow)
    check("PPL YELLOW alone -> YELLOW", ch_ppl.severity == 'YELLOW',
          f"got {ch_ppl.severity}")
    check("Perplexity in sub_signals", 'perplexity' in ch_ppl.sub_signals)

    ch_boost = score_stylometric(0, l30_r, semantic=l28_amber, ppl=None)
    check("NSSI+Semantic boost > NSSI alone", ch_boost.score > ch_r.score,
          f"boost={ch_boost.score}, alone={ch_r.score}")


def test_determine_with_new_signals():
    print("\n-- DETERMINE WITH NEW SIGNALS --")

    l25_low = {'composite': 0.05, 'framing_completeness': 0}
    l26_none = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                'spec_score': 0, 'contractions': 5, 'hedges': 3}
    l27_none = {'idi': 2.0}

    det, _, _, _ = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                             mode='generic_aigt', semantic=None, ppl=None)
    check("No new signals -> GREEN", det == 'GREEN', f"got {det}")

    l28_amber = {
        'determination': 'AMBER', 'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20, 'confidence': 0.55,
    }
    det2, _, _, cd = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                                mode='generic_aigt', semantic=l28_amber, ppl=None)
    check("Semantic AMBER -> AMBER in generic_aigt",
          det2 in ('AMBER', 'RED'), f"got {det2}")

    check("4 channels in details", len(cd.get('channels', {})) == 4,
          f"got {len(cd.get('channels', {}))}")


def test_channel_ablation():
    print("\n-- CHANNEL ABLATION --")

    l25_sig = {'composite': 0.45, 'framing_completeness': 3, 'cfd': 0.4,
               'mfsr': 0.3, 'must_rate': 0.2, 'distinct_frames': 5,
               'conditional_density': 0.3, 'meta_design_hits': 2,
               'contractions': 0, 'numbered_criteria': 0}
    l26_high = {'voice_gated': False, 'vsd': 12, 'voice_score': 0,
                'spec_score': 12, 'contractions': 0, 'hedges': 0,
                'casual_markers': 0, 'misspellings': 0, 'camel_cols': 0,
                'calcs': 0}
    l27_high = {'idi': 10.0, 'imperatives': 5, 'conditionals': 3,
                'binary_specs': 2, 'missing_refs': 1, 'flag_count': 3}

    # Without ablation — should produce some signal
    det1, _, _, cd1 = determine(0.6, 'HIGH', l25_sig, l26_high, l27_high, 300,
                                 mode='task_prompt')
    check("No ablation produces signal", det1 != 'GREEN', f"got {det1}")
    check("disabled_channels empty", cd1.get('disabled_channels') == [])

    # Disable prompt_structure
    det2, _, _, cd2 = determine(0.6, 'HIGH', l25_sig, l26_high, l27_high, 300,
                                 mode='task_prompt',
                                 disabled_channels={'prompt_structure'})
    check("prompt_structure disabled in details",
          'prompt_structure' in cd2.get('disabled_channels', []))
    check("prompt_structure channel is GREEN",
          cd2['channels']['prompt_structure']['severity'] == 'GREEN')
    check("prompt_structure shows ablation label",
          'ablation' in cd2['channels']['prompt_structure']['explanation'])

    # Disable all channels — should produce GREEN or REVIEW
    det3, _, _, cd3 = determine(0, 'NONE', l25_sig, l26_high, l27_high, 300,
                                 mode='task_prompt',
                                 disabled_channels={'prompt_structure', 'stylometric',
                                                    'continuation', 'windowed'})
    check("All disabled -> GREEN/REVIEW", det3 in ('GREEN', 'REVIEW'), f"got {det3}")


def test_short_text_adjustment():
    print("\n-- SHORT-TEXT CHANNEL WEIGHT ADJUSTMENT --")

    l25_low = {'composite': 0.05, 'framing_completeness': 0, 'cfd': 0.01,
               'mfsr': 0, 'must_rate': 0, 'distinct_frames': 0,
               'conditional_density': 0, 'meta_design_hits': 0,
               'contractions': 0, 'numbered_criteria': 0}
    l26_none = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                'spec_score': 0, 'contractions': 5, 'hedges': 3,
                'casual_markers': 3, 'misspellings': 0, 'camel_cols': 0,
                'calcs': 0}
    l27_none = {'idi': 2.0, 'imperatives': 0, 'conditionals': 0,
                'binary_specs': 0, 'missing_refs': 0, 'flag_count': 0}

    # Normal text (300 words): no short-text adjustment
    _, _, _, cd_normal = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                                    mode='generic_aigt')
    check("Normal text: no short_text_adjustment",
          cd_normal.get('short_text_adjustment') == False)

    # Short text (50 words): short_text_adjustment should activate
    _, _, _, cd_short = determine(0, 'NONE', l25_low, l26_none, l27_none, 50,
                                   mode='generic_aigt')
    check("Short text: short_text_adjustment active",
          cd_short.get('short_text_adjustment') == True or
          cd_short.get('active_channels', 4) <= 2)
    check("active_channels tracked", 'active_channels' in cd_short)


def test_attack_type_derivation():
    print("\n-- ATTACK TYPE DERIVATION --")
    from llm_detector.baselines import derive_attack_type

    # No attack
    r_none = {'norm_homoglyphs': 0, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0}
    check("No attack -> 'none'", derive_attack_type(r_none) == 'none')

    # Homoglyph only
    r_homo = {'norm_homoglyphs': 5, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0.03}
    check("Homoglyph -> 'homoglyph'", derive_attack_type(r_homo) == 'homoglyph')

    # Zero-width only
    r_zw = {'norm_homoglyphs': 0, 'norm_invisible_chars': 10, 'norm_obfuscation_delta': 0.05}
    check("Zero-width -> 'zero_width'", derive_attack_type(r_zw) == 'zero_width')

    # Combined
    r_combined = {'norm_homoglyphs': 3, 'norm_invisible_chars': 5, 'norm_obfuscation_delta': 0.08}
    check("Combined -> 'combined'", derive_attack_type(r_combined) == 'combined')

    # Encoding (delta only, no specific type)
    r_enc = {'norm_homoglyphs': 0, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0.04}
    check("Encoding -> 'encoding'", derive_attack_type(r_enc) == 'encoding')

    # Below encoding threshold
    r_low = {'norm_homoglyphs': 0, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0.01}
    check("Low delta -> 'none'", derive_attack_type(r_low) == 'none')

    # Missing fields (graceful)
    check("Missing fields -> 'none'", derive_attack_type({}) == 'none')


if __name__ == '__main__':
    print("=" * 70)
    print("Fusion / Channel Scoring Tests")
    print("=" * 70)

    test_stylometry_integration()
    test_determine_with_new_signals()
    test_channel_ablation()
    test_short_text_adjustment()
    test_attack_type_derivation()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
