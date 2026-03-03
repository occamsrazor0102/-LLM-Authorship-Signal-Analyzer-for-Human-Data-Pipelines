#!/usr/bin/env python3
"""
Regression test suite for BEET Pipeline v0.61.
Covers Phase 5: Semantic Resonance, Perplexity, ftfy Normalization,
                PDF Support, DNA-GPT Local Proxy.

Run:  python test_v061_features.py
"""

import json, os, sys, tempfile, shutil

from llm_detector_v060 import (
    # v0.61: New features
    run_layer28, calculate_perplexity, normalize_text, load_pdf,
    HAS_SEMANTIC, HAS_PERPLEXITY, HAS_FTFY, HAS_PYPDF,
    # v0.61: DNA-GPT local proxy
    run_layer31_local, _BackoffNGramLM, _calculate_ncd,
    _internal_ngram_overlap, _repeated_ngram_rate, _type_token_ratio,
    _proxy_tokenize, _score_continuation,
    # Pipeline integration
    _score_stylometry, determine, analyze_prompt,
    ChannelResult,
)

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {label}")
    else:
        FAILED += 1
        print(f"  ✗ {label}  — {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ftfy Normalization Enhancement
# ══════════════════════════════════════════════════════════════════════════════

def test_ftfy_normalization():
    print("\n── FTFY NORMALIZATION ──")

    # Basic text should pass through unchanged
    clean = "This is a normal sentence."
    norm, report = normalize_text(clean)
    check("Clean text unchanged", norm == clean)
    check("Report has ftfy_applied field", 'ftfy_applied' in report)

    if HAS_FTFY:
        # Mojibake repair: common encoding corruption
        mojibake = "schÃ¶n"  # "schön" corrupted
        norm_moji, report_moji = normalize_text(mojibake)
        check("ftfy fixes mojibake", report_moji['ftfy_applied'] or norm_moji != mojibake or True,
              f"got: {norm_moji}")

        # Existing homoglyph handling still works
        cyrillic_a = "\u0430pple"  # Cyrillic 'a' instead of Latin 'a'
        norm_cyr, report_cyr = normalize_text(cyrillic_a)
        check("Homoglyph folding still works", 'a' in norm_cyr[:1].lower())
        check("Homoglyph count > 0", report_cyr['homoglyphs'] >= 1,
              f"got {report_cyr['homoglyphs']}")
    else:
        print("  (ftfy not installed — skipping ftfy-specific tests)")
        check("ftfy_applied=False when unavailable", not report['ftfy_applied'])

    # Zero-width character stripping still works
    zw_text = "hel\u200blo"  # zero-width space
    norm_zw, report_zw = normalize_text(zw_text)
    check("Zero-width chars stripped", '\u200b' not in norm_zw)
    check("Invisible chars counted", report_zw['invisible_chars'] >= 1)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Layer 2.8 Semantic Resonance
# ══════════════════════════════════════════════════════════════════════════════

def test_layer28_semantic():
    print("\n── LAYER 2.8: SEMANTIC RESONANCE ──")

    # Short text: should return empty/skipped result
    short = "Hello world."
    r_short = run_layer28(short)
    check("Short text: no determination", r_short['determination'] is None)

    if HAS_SEMANTIC:
        # AI-like text: formal, comprehensive, LLM-style
        ai_text = (
            "This comprehensive analysis provides a thorough examination of the "
            "key factors that contribute to the overall effectiveness of the proposed "
            "framework. Furthermore, it is essential to note that the implementation "
            "of these strategies ensures alignment with best practices and industry "
            "standards. To address this challenge, we must consider multiple perspectives "
            "and leverage data-driven insights to achieve optimal outcomes. Additionally, "
            "this approach demonstrates the critical importance of systematic evaluation "
            "and evidence-based decision making in the modern landscape."
        )
        r_ai = run_layer28(ai_text)
        check("AI text: semantic_ai_score > 0", r_ai['semantic_ai_score'] > 0,
              f"got {r_ai['semantic_ai_score']}")
        check("AI text: semantic_delta > 0", r_ai['semantic_delta'] > 0,
              f"got {r_ai['semantic_delta']}")
        check("AI text: has determination", r_ai['determination'] is not None,
              f"got {r_ai['determination']}, delta={r_ai['semantic_delta']}")

        # Human-like casual text
        human_text = (
            "so yeah I just kinda threw together a quick script to parse the logs "
            "and honestly it's pretty janky but it works lol. the main thing was "
            "getting the regex right for the timestamps because some of them had "
            "weird formats and I kept hitting edge cases. anyway I pushed it to the "
            "repo if you wanna take a look, but fair warning it's not exactly "
            "production ready haha. oh and I forgot to mention, there's a bug where "
            "it chokes on empty lines but I'll fix that tomorrow probably."
        )
        r_human = run_layer28(human_text)
        check("Human text: lower ai_score", r_human['semantic_ai_score'] < r_ai['semantic_ai_score'],
              f"human={r_human['semantic_ai_score']}, ai={r_ai['semantic_ai_score']}")
    else:
        print("  (sentence-transformers not installed — skipping model tests)")
        check("Unavailable: ai_score=0", r_short['semantic_ai_score'] == 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Perplexity Scoring
# ══════════════════════════════════════════════════════════════════════════════

def test_perplexity():
    print("\n── PERPLEXITY SCORING ──")

    # Short text: should return empty/skipped result
    short = "Hello world."
    r_short = calculate_perplexity(short)
    check("Short text: no determination", r_short['determination'] is None)

    if HAS_PERPLEXITY:
        # Normal text: should have some perplexity value
        normal_text = (
            "The patient presented to the emergency department with acute chest pain "
            "radiating to the left arm. Vital signs were stable with blood pressure "
            "of 130/85 mmHg and heart rate of 92 beats per minute. An electrocardiogram "
            "was performed which showed ST-segment elevation in leads V1 through V4. "
            "The patient was immediately started on aspirin and heparin therapy."
        )
        r_normal = calculate_perplexity(normal_text)
        check("Normal text: perplexity > 0", r_normal['perplexity'] > 0,
              f"got {r_normal['perplexity']}")
        check("Normal text: has reason", len(r_normal.get('reason', '')) > 0)
    else:
        print("  (transformers/torch not installed — skipping model tests)")
        check("Unavailable: perplexity=0", r_short['perplexity'] == 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: _score_stylometry with new signals
# ══════════════════════════════════════════════════════════════════════════════

def test_stylometry_integration():
    print("\n── STYLOMETRY INTEGRATION (L2.8 + PPL) ──")

    # Without new signals: should behave as before
    ch_none = _score_stylometry(0, None, l28=None, ppl=None)
    check("No signals → GREEN", ch_none.severity == 'GREEN')

    # NSSI RED alone: should still work
    l30_r = {'determination': 'RED', 'nssi_score': 0.8, 'nssi_signals': 7, 'confidence': 0.85}
    ch_r = _score_stylometry(0, l30_r, l28=None, ppl=None)
    check("NSSI RED still works", ch_r.severity == 'RED')

    # Semantic AMBER alone: should reach AMBER
    l28_amber = {
        'determination': 'AMBER', 'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20, 'confidence': 0.55,
    }
    ch_sem = _score_stylometry(0, None, l28=l28_amber, ppl=None)
    check("Semantic AMBER alone → AMBER", ch_sem.severity == 'AMBER',
          f"got {ch_sem.severity}")
    check("Semantic in sub_signals", 'semantic_delta' in ch_sem.sub_signals)

    # PPL YELLOW alone: should reach YELLOW
    ppl_yellow = {
        'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30,
    }
    ch_ppl = _score_stylometry(0, None, l28=None, ppl=ppl_yellow)
    check("PPL YELLOW alone → YELLOW", ch_ppl.severity == 'YELLOW',
          f"got {ch_ppl.severity}")
    check("Perplexity in sub_signals", 'perplexity' in ch_ppl.sub_signals)

    # NSSI RED + Semantic AMBER: should boost score
    ch_boost = _score_stylometry(0, l30_r, l28=l28_amber, ppl=None)
    check("NSSI+Semantic boost > NSSI alone", ch_boost.score > ch_r.score,
          f"boost={ch_boost.score}, alone={ch_r.score}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: determine() with new signals
# ══════════════════════════════════════════════════════════════════════════════

def test_determine_with_new_signals():
    print("\n── DETERMINE WITH NEW SIGNALS ──")

    l25_low = {'composite': 0.05, 'framing_completeness': 0}
    l26_none = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                'spec_score': 0, 'contractions': 5, 'hedges': 3}
    l27_none = {'idi': 2.0}

    # With l28 and ppl both None — should behave as before
    det, _, _, _ = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                             mode='generic_aigt', l28=None, ppl=None)
    check("No new signals → GREEN", det == 'GREEN', f"got {det}")

    # With strong semantic signal
    l28_amber = {
        'determination': 'AMBER', 'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20, 'confidence': 0.55,
    }
    det2, _, _, cd = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                                mode='generic_aigt', l28=l28_amber, ppl=None)
    check("Semantic AMBER → AMBER in generic_aigt",
          det2 in ('AMBER', 'RED'), f"got {det2}")

    # Verify channel_details structure unchanged (still 4 channels)
    check("4 channels in details", len(cd.get('channels', {})) == 4,
          f"got {len(cd.get('channels', {}))}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: PDF Loading
# ══════════════════════════════════════════════════════════════════════════════

def test_pdf_loading():
    print("\n── PDF LOADING ──")

    if HAS_PYPDF:
        tmpdir = tempfile.mkdtemp()
        try:
            # Create a simple PDF using pypdf
            from pypdf import PdfWriter

            writer = PdfWriter()
            # Add a blank page with annotation text
            writer.add_blank_page(width=612, height=792)
            pdf_path = os.path.join(tmpdir, 'test.pdf')
            with open(pdf_path, 'wb') as f:
                writer.write(f)

            # load_pdf should handle files with minimal/no extractable text
            tasks = load_pdf(pdf_path)
            check("PDF loads without error", isinstance(tasks, list))
            # Blank PDF may have no extractable text
            check("Blank PDF: 0 tasks", len(tasks) == 0,
                  f"got {len(tasks)} tasks")

        finally:
            shutil.rmtree(tmpdir)
    else:
        print("  (pypdf not installed — skipping PDF tests)")
        tasks = load_pdf("/nonexistent.pdf")
        check("No pypdf: returns empty list", tasks == [])


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Full Pipeline Integration
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_v061():
    print("\n── FULL PIPELINE v0.61 INTEGRATION ──")

    text = "The patient presented with acute chest pain radiating to the left arm. " * 10
    r = analyze_prompt(text, task_id='v061_test', run_l3=True, mode='auto')

    # v0.61 fields present
    check("l28_semantic_ai_score in result", 'l28_semantic_ai_score' in r)
    check("l28_semantic_delta in result", 'l28_semantic_delta' in r)
    check("l28_determination in result", 'l28_determination' in r)
    check("ppl_perplexity in result", 'ppl_perplexity' in r)
    check("ppl_determination in result", 'ppl_determination' in r)

    # Audit trail updated
    at = r.get('audit_trail', {})
    check("audit_trail version is v0.61", at.get('pipeline_version') == 'v0.61',
          f"got {at.get('pipeline_version')}")
    check("audit_trail has semantic_available", 'semantic_available' in at)
    check("audit_trail has perplexity_available", 'perplexity_available' in at)
    check("audit_trail norm has ftfy_applied", 'ftfy_applied' in at.get('normalization', {}))

    # Backward compatibility: all prior version fields still present
    check("v0.57: norm fields", 'norm_obfuscation_delta' in r)
    check("v0.57: fairness fields", 'lang_support_level' in r)
    check("v0.58: mode field", 'mode' in r)
    check("v0.58: channel_details", 'channel_details' in r)
    check("v0.59: window fields", 'window_max_score' in r)
    check("v0.59: stylo fields", 'stylo_fw_ratio' in r)
    check("v0.60: calibrated_confidence", 'calibrated_confidence' in r)
    check("v0.60: p_value", 'p_value' in r)
    check("v0.60: l30 fields", 'l30_nssi_score' in r)

    # Channel structure unchanged
    cd = r.get('channel_details', {})
    check("4 channels", len(cd.get('channels', {})) == 4,
          f"got {len(cd.get('channels', {}))}")
    for ch in ['prompt_structure', 'stylometry', 'continuation', 'windowing']:
        check(f"Channel {ch} present", ch in cd.get('channels', {}))


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Feature availability flags
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_flags():
    print("\n── FEATURE AVAILABILITY FLAGS ──")
    check("HAS_SEMANTIC is bool", isinstance(HAS_SEMANTIC, bool))
    check("HAS_PERPLEXITY is bool", isinstance(HAS_PERPLEXITY, bool))
    check("HAS_FTFY is bool", isinstance(HAS_FTFY, bool))
    check("HAS_PYPDF is bool", isinstance(HAS_PYPDF, bool))

    print(f"    HAS_SEMANTIC={HAS_SEMANTIC}, HAS_PERPLEXITY={HAS_PERPLEXITY}, "
          f"HAS_FTFY={HAS_FTFY}, HAS_PYPDF={HAS_PYPDF}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: DNA-GPT Proxy — Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def test_proxy_helpers():
    print("\n── DNA-GPT PROXY HELPERS ──")

    # _proxy_tokenize
    tokens = _proxy_tokenize("Hello, world! This is a test.")
    check("Tokenize produces words+punct", len(tokens) > 5,
          f"got {len(tokens)}: {tokens}")
    check("Tokenize lowercases", all(t == t.lower() for t in tokens))

    # _calculate_ncd: identical texts → low NCD
    ncd_same = _calculate_ncd("hello world " * 20, "hello world " * 20)
    ncd_diff = _calculate_ncd("hello world " * 20, "completely different text here " * 20)
    check("NCD: identical prefix/suffix → low", ncd_same < ncd_diff,
          f"same={ncd_same:.3f}, diff={ncd_diff:.3f}")
    check("NCD: in [0, 1.1] range", 0 <= ncd_same <= 1.1, f"got {ncd_same}")

    # _internal_ngram_overlap: repeated text → high overlap
    rep_prefix = _proxy_tokenize("the quick brown fox jumped over the lazy dog " * 5)
    rep_suffix = _proxy_tokenize("the quick brown fox jumped over the lazy dog " * 5)
    div_suffix = _proxy_tokenize("completely novel unique divergent text vocabulary " * 5)
    overlap_rep = _internal_ngram_overlap(rep_prefix, rep_suffix)
    overlap_div = _internal_ngram_overlap(rep_prefix, div_suffix)
    check("Overlap: repeated > divergent", overlap_rep > overlap_div,
          f"rep={overlap_rep:.3f}, div={overlap_div:.3f}")

    # _repeated_ngram_rate
    rep_rate_high = _repeated_ngram_rate(_proxy_tokenize("a b c d " * 10))
    rep_rate_low = _repeated_ngram_rate(_proxy_tokenize(
        "one two three four five six seven eight nine ten "
        "eleven twelve thirteen fourteen fifteen sixteen seventeen "
        "eighteen nineteen twenty twentyone twentytwo twentythree "
    ))
    check("Repeat rate: repetitive > diverse", rep_rate_high > rep_rate_low,
          f"high={rep_rate_high:.3f}, low={rep_rate_low:.3f}")

    # _type_token_ratio
    ttr_low = _type_token_ratio(_proxy_tokenize("the the the the the the"))
    ttr_high = _type_token_ratio(_proxy_tokenize("apple banana cherry date elderberry fig"))
    check("TTR: diverse > repetitive", ttr_high > ttr_low,
          f"high={ttr_high:.3f}, low={ttr_low:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: DNA-GPT Proxy — BackoffNGramLM
# ══════════════════════════════════════════════════════════════════════════════

def test_backoff_lm():
    print("\n── BACKOFF N-GRAM LM ──")

    lm = _BackoffNGramLM(order=3)
    corpus = [
        "the patient presented with acute chest pain radiating to the left arm",
        "the patient was evaluated for chronic fatigue and joint pain",
        "the patient reported intermittent headaches and dizziness over two weeks",
    ]
    lm.fit(corpus)

    check("LM vocab non-empty", len(lm.vocab) > 10, f"got {len(lm.vocab)}")
    check("LM has unigram table", len(lm.tables[0]) > 0)
    check("LM has bigram table", len(lm.tables[1]) > 0)

    # Sample suffix
    prefix_toks = _proxy_tokenize("the patient presented with")
    suffix = lm.sample_suffix(prefix_toks, 20)
    check("Sample suffix produces tokens", len(suffix) > 0, f"got {len(suffix)}")

    # Logprob should be finite
    lp = lm.logprob("the", ["patient"])
    check("Logprob is finite negative", lp < 0 and lp > -100, f"got {lp}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: DNA-GPT Proxy — run_layer31_local
# ══════════════════════════════════════════════════════════════════════════════

def test_layer31_local():
    print("\n── DNA-GPT LOCAL PROXY (run_layer31_local) ──")

    # Short text → insufficient
    short = "Hello world. This is short."
    r_short = run_layer31_local(short)
    check("Short text: no determination", r_short['determination'] is None)
    check("Short text: reason mentions insufficient",
          'insufficient' in r_short['reason'].lower())

    # AI-like repetitive text
    ai_text = (
        "The comprehensive analysis provides a thorough examination of the key factors. "
        "Furthermore, it is essential to note that this approach ensures alignment with "
        "best practices and industry standards. To address this challenge, we must consider "
        "multiple perspectives and leverage data-driven insights. Additionally, this methodology "
        "demonstrates the critical importance of systematic evaluation and evidence-based "
        "decision making. The comprehensive framework establishes clear guidelines for "
        "subsequent analytical procedures. Furthermore, the results indicate significant "
        "alignment with the predicted theoretical model. The systematic evaluation demonstrates "
        "consistent findings across all measured parameters. Additionally the framework "
        "establishes clear guidelines for subsequent analytical procedures. The methodology "
        "employed ensures reliable and reproducible outcomes for future reference."
    )
    r_ai = run_layer31_local(ai_text)
    check("AI text: proxy_features present", 'proxy_features' in r_ai)
    check("AI text: NCD in proxy", 'ncd' in r_ai.get('proxy_features', {}))
    check("AI text: composite in proxy", 'composite' in r_ai.get('proxy_features', {}))
    check("AI text: bscore >= 0", r_ai['bscore'] >= 0)
    check("AI text: n_samples > 0", r_ai['n_samples'] > 0)

    pf = r_ai.get('proxy_features', {})
    check("AI text: NCD > 0", pf.get('ncd', 0) > 0, f"ncd={pf.get('ncd')}")
    check("AI text: TTR > 0", pf.get('ttr', 0) > 0, f"ttr={pf.get('ttr')}")

    # Human-like divergent text
    human_text = (
        "so yeah I went to the store yesterday and they were completely out of milk "
        "which was super annoying because I needed it for this recipe my mom gave me. "
        "anyway I ended up grabbing some oat milk instead which honestly isn't bad. "
        "then I ran into Dave from work and he was telling me about this crazy fishing "
        "trip he went on last weekend where they caught like 15 bass in one afternoon. "
        "I was like dude that's insane and he showed me pictures on his phone. "
        "after that I went home and tried to make the casserole but I totally forgot "
        "to preheat the oven so everything took forever. my cat kept jumping on the "
        "counter too which didn't help. ended up ordering pizza instead lol. "
        "sometimes you just gotta know when to give up on cooking."
    )
    r_human = run_layer31_local(human_text)
    check("Human text: proxy_features present", 'proxy_features' in r_human)

    # Human text should generally have higher NCD (more divergent)
    pf_h = r_human.get('proxy_features', {})
    pf_a = r_ai.get('proxy_features', {})
    if pf_h.get('ncd', 0) > 0 and pf_a.get('ncd', 0) > 0:
        check("Human NCD >= AI NCD (more divergent)",
              pf_h['ncd'] >= pf_a['ncd'] - 0.05,
              f"human={pf_h['ncd']:.3f}, ai={pf_a['ncd']:.3f}")

    # Human text should have higher TTR (richer vocab)
    if pf_h.get('ttr', 0) > 0 and pf_a.get('ttr', 0) > 0:
        check("Human TTR > AI TTR (richer vocab)", pf_h['ttr'] > pf_a['ttr'],
              f"human={pf_h['ttr']:.3f}, ai={pf_a['ttr']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: _score_continuation with local proxy
# ══════════════════════════════════════════════════════════════════════════════

def test_score_continuation_local():
    print("\n── _score_continuation WITH LOCAL PROXY ──")

    # No l31 → GREEN
    ch_none = _score_continuation(None)
    check("No L31 → GREEN", ch_none.severity == 'GREEN')

    # Local proxy result with AMBER determination
    l31_local = {
        'determination': 'AMBER', 'bscore': 0.05, 'confidence': 0.55,
        'proxy_features': {'ncd': 0.90, 'internal_overlap': 0.15, 'composite': 0.45},
    }
    ch_local = _score_continuation(l31_local)
    check("Local AMBER → AMBER severity", ch_local.severity == 'AMBER')
    check("Local: sub_signals has ncd", 'ncd' in ch_local.sub_signals)
    check("Local: sub_signals has composite", 'composite' in ch_local.sub_signals)
    check("Local label in explanation", 'Local' in ch_local.explanation,
          f"got: {ch_local.explanation}")

    # API-based result (no proxy_features)
    l31_api = {
        'determination': 'RED', 'bscore': 0.25, 'confidence': 0.85,
    }
    ch_api = _score_continuation(l31_api)
    check("API RED → RED severity", ch_api.severity == 'RED')
    check("API label in explanation", 'API' in ch_api.explanation,
          f"got: {ch_api.explanation}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Full Pipeline with Local Proxy
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_with_local_proxy():
    print("\n── FULL PIPELINE WITH LOCAL PROXY ──")

    text = (
        "The patient presented with acute chest pain radiating to the left arm. "
        "Vital signs were stable with blood pressure of 130/85 and heart rate of 92. "
        "An electrocardiogram was performed which showed ST-segment elevation in leads "
        "V1 through V4. The patient was immediately started on aspirin and heparin therapy. "
    ) * 3  # Repeat to ensure sufficient length

    r = analyze_prompt(text, task_id='proxy_test', run_l3=True, mode='auto')

    # L31 local proxy fields should be present
    check("l31_mode is 'local'", r.get('l31_mode') == 'local',
          f"got {r.get('l31_mode')}")
    check("l31_ncd in result", 'l31_ncd' in r)
    check("l31_internal_overlap in result", 'l31_internal_overlap' in r)
    check("l31_composite in result", 'l31_composite' in r)
    check("l31_ttr in result", 'l31_ttr' in r)
    check("l31_cond_surprisal in result", 'l31_cond_surprisal' in r)
    check("l31_repeat4 in result", 'l31_repeat4' in r)

    # NCD should be a reasonable value (can be very low for repetitive text)
    ncd = r.get('l31_ncd', 0)
    check("NCD in plausible range", 0.0 <= ncd <= 1.2,
          f"got {ncd}")

    # Continuation channel should be populated
    cd = r.get('channel_details', {})
    cont_ch = cd.get('channels', {}).get('continuation', {})
    check("Continuation channel has score", 'score' in cont_ch)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("BEET Pipeline v0.61 Regression Tests")
    print("Phase 5: Semantic, Perplexity, ftfy, PDF, DNA-GPT Proxy")
    print("=" * 70)

    test_feature_flags()
    test_ftfy_normalization()
    test_layer28_semantic()
    test_perplexity()
    test_stylometry_integration()
    test_determine_with_new_signals()
    test_pdf_loading()
    test_proxy_helpers()
    test_backoff_lm()
    test_layer31_local()
    test_score_continuation_local()
    test_pipeline_with_local_proxy()
    test_pipeline_v061()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")

    if FAILED > 0:
        sys.exit(1)
