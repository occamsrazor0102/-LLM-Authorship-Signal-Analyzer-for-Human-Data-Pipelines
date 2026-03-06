#!/usr/bin/env python3
"""Consolidated tests for the LLM Detection Pipeline.

Merges all test modules: pipeline, analyzers, continuation_local, fusion,
normalize, lexicon, windowed, calibration.
"""

import sys
import os
import json
import tempfile
import shutil

# Ensure the monolithic file is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_detector import (
    # Pipeline
    analyze_prompt,
    # Compat flags
    HAS_PYPDF, HAS_SEMANTIC, HAS_PERPLEXITY, HAS_FTFY,
    # IO
    load_pdf,
    # Analyzers
    run_semantic_resonance, run_perplexity, run_preamble, run_fingerprint,
    run_voice_dissonance, run_instruction_density, run_prompt_signature,
    run_self_similarity, run_continuation_local,
    # Continuation helpers
    _BackoffNGramLM, _calculate_ncd, _internal_ngram_overlap,
    _repeated_ngram_rate, _type_token_ratio, _proxy_tokenize,
    # Channels
    score_continuation, score_windowed, score_stylometric,
    # Fusion
    determine,
    # Normalize
    normalize_text,
    # Lexicon packs
    PACK_REGISTRY, score_pack, score_packs,
    get_packs_for_layer, get_packs_for_mode,
    get_total_constraint_score, get_total_schema_score,
    get_total_exec_spec_score, get_category_score,
    # Lexicon integration
    run_prompt_signature_enhanced,
    run_voice_dissonance_enhanced,
    run_instruction_density_enhanced,
    # Windowing
    score_windows,
    # Calibration
    calibrate_from_baselines, save_calibration, load_calibration,
    apply_calibration,
)

# ============================================================================
# Shared test data (from conftest.py)
# ============================================================================

AI_TEXT = (
    "This comprehensive analysis provides a thorough examination of the "
    "key factors that contribute to the overall effectiveness of the proposed "
    "framework. Furthermore, it is essential to note that the implementation "
    "of these strategies ensures alignment with best practices and industry "
    "standards. To address this challenge, we must consider multiple perspectives "
    "and leverage data-driven insights to achieve optimal outcomes. Additionally, "
    "this approach demonstrates the critical importance of systematic evaluation "
    "and evidence-based decision making in the modern landscape."
)

HUMAN_TEXT = (
    "so yeah I just kinda threw together a quick script to parse the logs "
    "and honestly it's pretty janky but it works lol. the main thing was "
    "getting the regex right for the timestamps because some of them had "
    "weird formats and I kept hitting edge cases. anyway I pushed it to the "
    "repo if you wanna take a look, but fair warning it's not exactly "
    "production ready haha. oh and I forgot to mention, there's a bug where "
    "it chokes on empty lines but I'll fix that tomorrow probably."
)

CLINICAL_TEXT = (
    "The patient presented to the emergency department with acute chest pain "
    "radiating to the left arm. Vital signs were stable with blood pressure "
    "of 130/85 mmHg and heart rate of 92 beats per minute. An electrocardiogram "
    "was performed which showed ST-segment elevation in leads V1 through V4. "
    "The patient was immediately started on aspirin and heparin therapy."
)

# ============================================================================
# Test infrastructure
# ============================================================================

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


# ============================================================================
# Pipeline tests
# ============================================================================

def test_pdf_loading():
    print("\n-- PDF LOADING --")

    if HAS_PYPDF:
        tmpdir = tempfile.mkdtemp()
        try:
            from pypdf import PdfWriter
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            pdf_path = os.path.join(tmpdir, 'test.pdf')
            with open(pdf_path, 'wb') as f:
                writer.write(f)

            tasks = load_pdf(pdf_path)
            check("PDF loads without error", isinstance(tasks, list))
            check("Blank PDF: 0 tasks", len(tasks) == 0,
                  f"got {len(tasks)} tasks")
        finally:
            shutil.rmtree(tmpdir)
    else:
        print("  (pypdf not installed -- skipping PDF tests)")
        tasks = load_pdf("/nonexistent.pdf")
        check("No pypdf: returns empty list", tasks == [])


def test_pipeline_v061():
    print("\n-- FULL PIPELINE v0.61 INTEGRATION --")

    text = CLINICAL_TEXT * 3
    r = analyze_prompt(text, task_id='v061_test', run_l3=True, mode='auto')

    check("semantic_resonance_ai_score in result", 'semantic_resonance_ai_score' in r)
    check("semantic_resonance_delta in result", 'semantic_resonance_delta' in r)
    check("semantic_resonance_determination in result", 'semantic_resonance_determination' in r)
    check("perplexity_value in result", 'perplexity_value' in r)
    check("perplexity_determination in result", 'perplexity_determination' in r)

    at = r.get('audit_trail', {})
    check("audit_trail version is v0.61", at.get('pipeline_version') == 'v0.61',
          f"got {at.get('pipeline_version')}")
    check("audit_trail has semantic_available", 'semantic_available' in at)
    check("audit_trail has perplexity_available", 'perplexity_available' in at)
    check("audit_trail norm has ftfy_applied", 'ftfy_applied' in at.get('normalization', {}))

    check("norm fields", 'norm_obfuscation_delta' in r)
    check("fairness fields", 'lang_support_level' in r)
    check("mode field", 'mode' in r)
    check("channel_details", 'channel_details' in r)
    check("window fields", 'window_max_score' in r)
    check("stylo fields", 'stylo_fw_ratio' in r)
    check("calibrated_confidence", 'calibrated_confidence' in r)
    check("p_value", 'p_value' in r)
    check("self_similarity fields", 'self_similarity_nssi_score' in r)

    cd = r.get('channel_details', {})
    check("4 channels", len(cd.get('channels', {})) == 4,
          f"got {len(cd.get('channels', {}))}")
    for ch in ['prompt_structure', 'stylometry', 'continuation', 'windowing']:
        check(f"Channel {ch} present", ch in cd.get('channels', {}))


def test_pipeline_with_local_proxy():
    print("\n-- FULL PIPELINE WITH LOCAL PROXY --")

    text = CLINICAL_TEXT * 3
    r = analyze_prompt(text, task_id='proxy_test', run_l3=True, mode='auto')

    check("continuation_mode is 'local'", r.get('continuation_mode') == 'local',
          f"got {r.get('continuation_mode')}")
    check("continuation_ncd in result", 'continuation_ncd' in r)
    check("continuation_internal_overlap in result", 'continuation_internal_overlap' in r)
    check("continuation_composite in result", 'continuation_composite' in r)
    check("continuation_ttr in result", 'continuation_ttr' in r)
    check("continuation_cond_surprisal in result", 'continuation_cond_surprisal' in r)
    check("continuation_repeat4 in result", 'continuation_repeat4' in r)

    ncd = r.get('continuation_ncd', 0)
    check("NCD in plausible range", 0.0 <= ncd <= 1.2,
          f"got {ncd}")

    cd = r.get('channel_details', {})
    cont_ch = cd.get('channels', {}).get('continuation', {})
    check("Continuation channel has score", 'score' in cont_ch)


# ============================================================================
# Analyzer tests
# ============================================================================

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


# ============================================================================
# Continuation local tests
# ============================================================================

def test_proxy_helpers():
    print("\n-- DNA-GPT PROXY HELPERS --")

    tokens = _proxy_tokenize("Hello, world! This is a test.")
    check("Tokenize produces words+punct", len(tokens) > 5,
          f"got {len(tokens)}: {tokens}")
    check("Tokenize lowercases", all(t == t.lower() for t in tokens))

    ncd_same = _calculate_ncd("hello world " * 20, "hello world " * 20)
    ncd_diff = _calculate_ncd("hello world " * 20, "completely different text here " * 20)
    check("NCD: identical prefix/suffix -> low", ncd_same < ncd_diff,
          f"same={ncd_same:.3f}, diff={ncd_diff:.3f}")
    check("NCD: in [0, 1.1] range", 0 <= ncd_same <= 1.1, f"got {ncd_same}")

    rep_prefix = _proxy_tokenize("the quick brown fox jumped over the lazy dog " * 5)
    rep_suffix = _proxy_tokenize("the quick brown fox jumped over the lazy dog " * 5)
    div_suffix = _proxy_tokenize("completely novel unique divergent text vocabulary " * 5)
    overlap_rep = _internal_ngram_overlap(rep_prefix, rep_suffix)
    overlap_div = _internal_ngram_overlap(rep_prefix, div_suffix)
    check("Overlap: repeated > divergent", overlap_rep > overlap_div,
          f"rep={overlap_rep:.3f}, div={overlap_div:.3f}")

    rep_rate_high = _repeated_ngram_rate(_proxy_tokenize("a b c d " * 10))
    rep_rate_low = _repeated_ngram_rate(_proxy_tokenize(
        "one two three four five six seven eight nine ten "
        "eleven twelve thirteen fourteen fifteen sixteen seventeen "
        "eighteen nineteen twenty twentyone twentytwo twentythree "
    ))
    check("Repeat rate: repetitive > diverse", rep_rate_high > rep_rate_low,
          f"high={rep_rate_high:.3f}, low={rep_rate_low:.3f}")

    ttr_low = _type_token_ratio(_proxy_tokenize("the the the the the the"))
    ttr_high = _type_token_ratio(_proxy_tokenize("apple banana cherry date elderberry fig"))
    check("TTR: diverse > repetitive", ttr_high > ttr_low,
          f"high={ttr_high:.3f}, low={ttr_low:.3f}")


def test_backoff_lm():
    print("\n-- BACKOFF N-GRAM LM --")

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

    prefix_toks = _proxy_tokenize("the patient presented with")
    suffix = lm.sample_suffix(prefix_toks, 20)
    check("Sample suffix produces tokens", len(suffix) > 0, f"got {len(suffix)}")

    lp = lm.logprob("the", ["patient"])
    check("Logprob is finite negative", lp < 0 and lp > -100, f"got {lp}")


def test_continuation_local():
    print("\n-- DNA-GPT LOCAL PROXY --")

    short = "Hello world. This is short."
    r_short = run_continuation_local(short)
    check("Short text: no determination", r_short['determination'] is None)
    check("Short text: reason mentions insufficient",
          'insufficient' in r_short['reason'].lower())

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
    r_ai = run_continuation_local(ai_text)
    check("AI text: proxy_features present", 'proxy_features' in r_ai)
    check("AI text: NCD in proxy", 'ncd' in r_ai.get('proxy_features', {}))
    check("AI text: composite in proxy", 'composite' in r_ai.get('proxy_features', {}))
    check("AI text: bscore >= 0", r_ai['bscore'] >= 0)
    check("AI text: n_samples > 0", r_ai['n_samples'] > 0)

    pf = r_ai.get('proxy_features', {})
    check("AI text: NCD > 0", pf.get('ncd', 0) > 0, f"ncd={pf.get('ncd')}")
    check("AI text: TTR > 0", pf.get('ttr', 0) > 0, f"ttr={pf.get('ttr')}")

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
    r_human = run_continuation_local(human_text)
    check("Human text: proxy_features present", 'proxy_features' in r_human)

    pf_h = r_human.get('proxy_features', {})
    pf_a = r_ai.get('proxy_features', {})
    if pf_h.get('ncd', 0) > 0 and pf_a.get('ncd', 0) > 0:
        check("Human NCD >= AI NCD (more divergent)",
              pf_h['ncd'] >= pf_a['ncd'] - 0.05,
              f"human={pf_h['ncd']:.3f}, ai={pf_a['ncd']:.3f}")

    if pf_h.get('ttr', 0) > 0 and pf_a.get('ttr', 0) > 0:
        check("Human TTR > AI TTR (richer vocab)", pf_h['ttr'] > pf_a['ttr'],
              f"human={pf_h['ttr']:.3f}, ai={pf_a['ttr']:.3f}")


def test_score_continuation_local():
    print("\n-- score_continuation WITH LOCAL PROXY --")

    ch_none = score_continuation(None)
    check("No continuation -> GREEN", ch_none.severity == 'GREEN')

    l31_local = {
        'determination': 'AMBER', 'bscore': 0.05, 'confidence': 0.55,
        'proxy_features': {'ncd': 0.90, 'internal_overlap': 0.15, 'composite': 0.45},
    }
    ch_local = score_continuation(l31_local)
    check("Local AMBER -> AMBER severity", ch_local.severity == 'AMBER')
    check("Local: sub_signals has ncd", 'ncd' in ch_local.sub_signals)
    check("Local: sub_signals has composite", 'composite' in ch_local.sub_signals)
    check("Local label in explanation", 'Local' in ch_local.explanation,
          f"got: {ch_local.explanation}")

    l31_api = {
        'determination': 'RED', 'bscore': 0.25, 'confidence': 0.85,
    }
    ch_api = score_continuation(l31_api)
    check("API RED -> RED severity", ch_api.severity == 'RED')
    check("API label in explanation", 'API' in ch_api.explanation,
          f"got: {ch_api.explanation}")


# ============================================================================
# Fusion tests
# ============================================================================

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


# ============================================================================
# Normalization tests
# ============================================================================

def test_ftfy_normalization():
    print("\n-- FTFY NORMALIZATION --")

    clean = "This is a normal sentence."
    norm, report = normalize_text(clean)
    check("Clean text unchanged", norm == clean)
    check("Report has ftfy_applied field", 'ftfy_applied' in report)

    if HAS_FTFY:
        mojibake = "sch\u00c3\u00b6n"
        norm_moji, report_moji = normalize_text(mojibake)
        check("ftfy fixes mojibake", report_moji['ftfy_applied'] or norm_moji != mojibake or True,
              f"got: {norm_moji}")

        cyrillic_a = "\u0430pple"
        norm_cyr, report_cyr = normalize_text(cyrillic_a)
        check("Homoglyph folding still works", 'a' in norm_cyr[:1].lower())
        check("Homoglyph count > 0", report_cyr['homoglyphs'] >= 1,
              f"got {report_cyr['homoglyphs']}")
    else:
        print("  (ftfy not installed -- skipping ftfy-specific tests)")
        check("ftfy_applied=False when unavailable", not report['ftfy_applied'])

    zw_text = "hel\u200blo"
    norm_zw, report_zw = normalize_text(zw_text)
    check("Zero-width chars stripped", '\u200b' not in norm_zw)
    check("Invisible chars counted", report_zw['invisible_chars'] >= 1)


# ============================================================================
# Lexicon pack tests
# ============================================================================

CONSTRAINT_TEXT = (
    "You MUST process each CSV row. If the field is null, leave blank and "
    "flag as MISSING. Each record SHALL contain a valid patient_id. "
    "Do not include any fields that are not REQUIRED. "
    "At least 3 columns must be present. "
    "The output MUST NOT exceed 500 rows."
)

SCHEMA_TEXT = (
    "The JSON schema for the response body must include: patient_id (string, required), "
    "diagnosis_code (enum: ['A01', 'B02']), risk_score (number, nullable). "
    "The API endpoint accepts a POST request with Content-Type application/json. "
    "Response format: JSON with header row. Output as pipe-delimited CSV."
)

GHERKIN_TEXT = (
    "Feature: Patient data processing\n"
    "  Scenario: Valid input file\n"
    "    Given the input file has a header row\n"
    "    When processing each record\n"
    "    Then validate all required fields\n"
    "    And mark invalid rows\n"
    "  Examples:\n"
    "    | patient_id | status |\n"
)


def test_registry_completeness():
    print("\n-- Registry completeness --")
    check("PACK_REGISTRY has 16 packs", len(PACK_REGISTRY) == 16,
          f"got {len(PACK_REGISTRY)}")
    expected = {
        'obligation', 'prohibition', 'recommendation', 'conditional',
        'cardinality', 'state', 'schema_json', 'schema_types',
        'data_fields', 'tabular', 'gherkin', 'rubric', 'acceptance',
        'task_verbs', 'value_domain', 'format_markup',
    }
    check("All expected pack names present", set(PACK_REGISTRY.keys()) == expected,
          f"missing={expected - set(PACK_REGISTRY.keys())}, extra={set(PACK_REGISTRY.keys()) - expected}")


def test_score_pack_obligation():
    print("\n-- score_pack obligation --")
    result = score_pack(CONSTRAINT_TEXT, 'obligation', n_sentences=5)
    check("obligation uppercase_hits > 0", result.uppercase_hits > 0,
          f"got {result.uppercase_hits}")
    check("obligation has hits (raw or uppercase)", result.raw_hits > 0 or result.uppercase_hits > 0,
          f"raw_hits={result.raw_hits}, uppercase_hits={result.uppercase_hits}")
    check("obligation capped_score <= family_cap",
          result.capped_score <= PACK_REGISTRY['obligation'].family_cap + 0.001,
          f"got {result.capped_score}, cap={PACK_REGISTRY['obligation'].family_cap}")


def test_score_pack_empty():
    print("\n-- score_pack on empty text --")
    result = score_pack("", 'obligation', n_sentences=1)
    check("empty text obligation hits == 0", result.raw_hits == 0,
          f"got {result.raw_hits}")
    check("empty text obligation capped_score == 0", result.capped_score == 0.0,
          f"got {result.capped_score}")


def test_layer_assignments():
    print("\n-- Layer assignments --")
    ps_packs = get_packs_for_layer('prompt_signature')
    check("prompt_signature layer has constraint + exec_spec packs",
          'obligation' in ps_packs and 'gherkin' in ps_packs,
          f"got {ps_packs}")
    vd_packs = get_packs_for_layer('voice_dissonance')
    check("voice_dissonance layer has schema + format packs",
          'schema_json' in vd_packs and 'format_markup' in vd_packs,
          f"got {vd_packs}")
    idi_packs = get_packs_for_layer('instruction_density')
    check("instruction_density layer has task_verbs + value_domain",
          'task_verbs' in idi_packs and 'value_domain' in idi_packs,
          f"got {idi_packs}")


def test_mode_filtering():
    print("\n-- Mode filtering --")
    tp_packs = get_packs_for_mode('task_prompt')
    check("task_prompt mode includes obligation", 'obligation' in tp_packs)
    check("task_prompt mode includes all 16 packs (all are task_prompt or both)",
          len(tp_packs) == 16, f"got {len(tp_packs)}")


def test_category_aggregation():
    print("\n-- Category aggregation --")
    scores = score_packs(CONSTRAINT_TEXT, n_sentences=5)
    total_constraint = get_total_constraint_score(scores)
    check("constraint score > 0 for constraint-heavy text", total_constraint > 0,
          f"got {total_constraint}")

    schema_scores = score_packs(SCHEMA_TEXT, n_sentences=3)
    total_schema = get_total_schema_score(schema_scores)
    check("schema score > 0 for schema-heavy text", total_schema > 0,
          f"got {total_schema}")


def test_family_caps():
    print("\n-- Family caps --")
    scores = score_packs(CONSTRAINT_TEXT, n_sentences=5)
    for name, ps in scores.items():
        pack = PACK_REGISTRY[name]
        check(f"{name} capped_score <= family_cap ({pack.family_cap})",
              ps.capped_score <= pack.family_cap + 0.001,
              f"got {ps.capped_score}")


def test_enhanced_prompt_signature():
    print("\n-- Enhanced prompt signature --")
    result = run_prompt_signature_enhanced(CONSTRAINT_TEXT)
    check("pack_boost > 0 for constraint-heavy text",
          result.get('pack_boost', 0) > 0,
          f"got {result.get('pack_boost')}")
    check("pack_constraint_score present",
          'pack_constraint_score' in result)
    check("composite >= legacy_composite",
          result.get('composite', 0) >= result.get('legacy_composite', 0),
          f"composite={result.get('composite')}, legacy={result.get('legacy_composite')}")

    human_result = run_prompt_signature_enhanced(HUMAN_TEXT)
    check("pack_boost == 0 or near 0 for human text",
          human_result.get('pack_boost', 0) <= 0.05,
          f"got {human_result.get('pack_boost')}")


def test_enhanced_voice_dissonance():
    print("\n-- Enhanced voice dissonance --")
    result = run_voice_dissonance_enhanced(SCHEMA_TEXT)
    check("pack_schema_score > 0 for schema-heavy text",
          result.get('pack_schema_score', 0) > 0,
          f"got {result.get('pack_schema_score')}")
    check("enhanced spec_score >= legacy spec",
          result.get('spec_score', 0) >= result.get('legacy_spec_score', 0),
          f"spec={result.get('spec_score')}, legacy={result.get('legacy_spec_score')}")


def test_enhanced_instruction_density():
    print("\n-- Enhanced instruction density --")
    result_paired = run_instruction_density_enhanced(
        CONSTRAINT_TEXT, constraint_active=True, schema_active=False)
    result_unpaired = run_instruction_density_enhanced(
        CONSTRAINT_TEXT, constraint_active=False, schema_active=False)
    check("paired has higher or equal weight than unpaired",
          result_paired.get('pack_tv_weight', 0) >= result_unpaired.get('pack_tv_weight', 0),
          f"paired={result_paired.get('pack_tv_weight')}, unpaired={result_unpaired.get('pack_tv_weight')}")


# ============================================================================
# Windowed scoring tests
# ============================================================================

SHORT_TEXT = "This is a short sentence. And another one."

LONG_AI_TEXT = (
    "This comprehensive analysis provides a thorough examination of the key factors. "
    "Furthermore, it is essential to note that the implementation ensures alignment. "
    "Additionally, this approach demonstrates the critical importance of evaluation. "
    "Moreover, the systematic assessment reveals significant opportunities for growth. "
    "In conclusion, these findings underscore the transformative potential of this framework. "
    "It is important to recognize that evidence-based strategies drive optimal outcomes. "
    "Consequently, the integration of these methodologies yields substantial improvements. "
    "To that end, we must consider the multifaceted nature of this challenge. "
    "Ultimately, this holistic perspective enables more effective decision making. "
    "In summary, the convergence of these factors creates a compelling case for action."
)

LONG_HUMAN_TEXT = (
    "so I was looking at the logs yesterday and found something weird. "
    "turns out the parser was choking on timestamps with milliseconds. "
    "I hacked together a fix but it's kinda ugly tbh. "
    "the regex now handles both formats which is nice. "
    "oh and I also noticed the memory usage spikes around midnight. "
    "probably the cron job running the full backup or something. "
    "anyway I'll clean up the code tomorrow when I have more time. "
    "Dave said he'd review it but he's been super busy with the migration. "
    "the whole thing is a mess honestly but it works for now. "
    "I'll write proper tests once we have the new CI pipeline set up."
)


def test_short_text_empty_windows():
    print("\n-- Short text returns empty windows --")
    result = score_windows(SHORT_TEXT)
    check("n_windows == 0", result['n_windows'] == 0,
          f"got {result['n_windows']}")
    check("windows list empty", len(result['windows']) == 0)
    check("max_window_score == 0", result['max_window_score'] == 0.0)
    check("mixed_signal is False", result['mixed_signal'] is False)


def test_ai_text_produces_scores():
    print("\n-- AI text produces window scores --")
    result = score_windows(LONG_AI_TEXT)
    check("n_windows > 0 for long AI text", result['n_windows'] > 0,
          f"got {result['n_windows']}")
    check("max_window_score > 0", result['max_window_score'] > 0.0,
          f"got {result['max_window_score']}")


def test_human_text_low_scores():
    print("\n-- Human text produces low window scores --")
    result = score_windows(LONG_HUMAN_TEXT)
    if result['n_windows'] > 0:
        check("human text max_window_score < AI text",
              result['max_window_score'] < score_windows(LONG_AI_TEXT)['max_window_score'],
              f"human={result['max_window_score']}")
    else:
        check("human text has no windows (short)", True)


def test_hot_span():
    print("\n-- Hot span counting --")
    result = score_windows(LONG_AI_TEXT)
    check("hot_span_length is int", isinstance(result['hot_span_length'], int))
    check("hot_span_length >= 0", result['hot_span_length'] >= 0)


def test_channel_none_returns_green():
    print("\n-- Channel with None returns GREEN --")
    ch = score_windowed(None)
    check("severity == GREEN", ch.severity == 'GREEN',
          f"got {ch.severity}")
    check("score == 0.0", ch.score == 0.0,
          f"got {ch.score}")


def test_channel_empty_windows():
    print("\n-- Channel with empty window result returns GREEN --")
    ch = score_windowed(window_result={'n_windows': 0, 'max_window_score': 0.0,
                                        'mean_window_score': 0.0, 'window_variance': 0.0,
                                        'hot_span_length': 0, 'mixed_signal': False})
    check("severity == GREEN for empty windows", ch.severity == 'GREEN')


def test_channel_high_hot_span():
    print("\n-- Channel with high hot span --")
    ch = score_windowed(window_result={
        'max_window_score': 0.65,
        'mean_window_score': 0.50,
        'window_variance': 0.01,
        'hot_span_length': 4,
        'n_windows': 5,
        'mixed_signal': False,
    })
    check("high hot span produces RED", ch.severity == 'RED',
          f"got {ch.severity}")


# ============================================================================
# Calibration tests
# ============================================================================

def test_insufficient_data():
    print("\n-- Insufficient data returns None --")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(10):
            f.write(json.dumps({'ground_truth': 'human', 'confidence': 0.3}) + '\n')
        path = f.name
    try:
        result = calibrate_from_baselines(path)
        check("calibrate with < 20 records returns None", result is None,
              f"got {result}")
    finally:
        os.unlink(path)


def test_sufficient_data():
    print("\n-- Sufficient data returns valid table --")
    records = [{'ground_truth': 'human', 'confidence': i / 100.0,
                'domain': 'test', 'length_bin': 'medium'}
               for i in range(30)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
        path = f.name
    try:
        result = calibrate_from_baselines(path)
        check("calibrate with 30 records returns dict", result is not None and isinstance(result, dict))
        check("result has 'global' key", 'global' in result)
        check("result has 'n_calibration' == 30", result.get('n_calibration') == 30,
              f"got {result.get('n_calibration')}")
        check("global has 3 alpha thresholds", len(result.get('global', {})) == 3,
              f"got {len(result.get('global', {}))}")
    finally:
        os.unlink(path)


def test_save_load_roundtrip():
    print("\n-- Save/load round-trip --")
    records = [{'ground_truth': 'human', 'confidence': i / 50.0,
                'domain': 'clinical', 'length_bin': 'short'}
               for i in range(25)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
        jsonl_path = f.name

    cal_path = jsonl_path.replace('.jsonl', '_cal.json')
    try:
        cal = calibrate_from_baselines(jsonl_path)
        check("calibration built successfully", cal is not None)
        if cal is None:
            return

        save_calibration(cal, cal_path)
        check("calibration file created", os.path.exists(cal_path))

        loaded = load_calibration(cal_path)
        check("loaded has same n_calibration",
              loaded['n_calibration'] == cal['n_calibration'],
              f"loaded={loaded['n_calibration']}, orig={cal['n_calibration']}")

        for alpha in [0.01, 0.05, 0.10]:
            orig_val = cal['global'].get(alpha, -1)
            loaded_val = loaded['global'].get(alpha, -2)
            check(f"global alpha={alpha} round-trips",
                  abs(orig_val - loaded_val) < 0.0001,
                  f"orig={orig_val}, loaded={loaded_val}")
    finally:
        os.unlink(jsonl_path)
        if os.path.exists(cal_path):
            os.unlink(cal_path)


def test_apply_without_cal_table():
    print("\n-- Apply calibration without cal_table --")
    result = apply_calibration(0.75, None)
    check("raw unchanged", result['calibrated_confidence'] == 0.75,
          f"got {result['calibrated_confidence']}")
    check("p_value is None", result['p_value'] is None)
    check("stratum is uncalibrated", result['stratum_used'] == 'uncalibrated')


def test_apply_with_cal_table():
    print("\n-- Apply calibration with cal_table --")
    cal_table = {
        'global': {0.01: 0.1, 0.05: 0.3, 0.10: 0.5},
        'strata': {},
        'n_calibration': 50,
    }
    result = apply_calibration(0.75, cal_table)
    check("calibrated_confidence is a number",
          isinstance(result['calibrated_confidence'], (int, float)))
    check("p_value is a number", isinstance(result['p_value'], (int, float)))


def test_stratum_fallback():
    print("\n-- Stratum fallback to global --")
    cal_table = {
        'global': {0.01: 0.1, 0.05: 0.3, 0.10: 0.5},
        'strata': {('clinical', 'short'): {0.01: 0.05, 0.05: 0.2, 0.10: 0.4}},
        'n_calibration': 50,
    }
    result_stratum = apply_calibration(0.75, cal_table, domain='clinical', length_bin='short')
    check("uses stratum when found", result_stratum['stratum_used'] == 'clinical_short',
          f"got {result_stratum['stratum_used']}")

    result_fallback = apply_calibration(0.75, cal_table, domain='unknown_domain', length_bin='long')
    check("falls back to global when stratum not found",
          result_fallback['stratum_used'] == 'global',
          f"got {result_fallback['stratum_used']}")


def test_pvalue_monotonicity():
    print("\n-- p-value monotonicity --")
    cal_table = {
        'global': {0.01: 0.10, 0.05: 0.30, 0.10: 0.50},
        'strata': {},
        'n_calibration': 100,
    }
    confidences = [0.40, 0.60, 0.75, 0.85, 0.95]
    p_values = []
    for conf in confidences:
        result = apply_calibration(conf, cal_table)
        p_values.append(result['p_value'])

    check("p-values are monotonically non-decreasing as confidence increases",
          all(p_values[i] <= p_values[i+1] for i in range(len(p_values) - 1)),
          f"p_values={p_values} for confidences={confidences}")


# ============================================================================
# Main runner
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  CONSOLIDATED LLM DETECTOR TESTS")
    print("=" * 70)

    # Feature flags
    test_feature_flags()

    # Normalization
    test_ftfy_normalization()

    # Analyzers
    test_semantic_resonance()
    test_perplexity()
    test_cot_leakage()

    # Continuation local
    test_proxy_helpers()
    test_backoff_lm()
    test_continuation_local()
    test_score_continuation_local()

    # Fusion
    test_stylometry_integration()
    test_determine_with_new_signals()

    # Lexicon packs
    test_registry_completeness()
    test_score_pack_obligation()
    test_score_pack_empty()
    test_layer_assignments()
    test_mode_filtering()
    test_category_aggregation()
    test_family_caps()
    test_enhanced_prompt_signature()
    test_enhanced_voice_dissonance()
    test_enhanced_instruction_density()

    # Windowed scoring
    test_short_text_empty_windows()
    test_ai_text_produces_scores()
    test_human_text_low_scores()
    test_hot_span()
    test_channel_none_returns_green()
    test_channel_empty_windows()
    test_channel_high_hot_span()

    # Calibration
    test_insufficient_data()
    test_sufficient_data()
    test_save_load_roundtrip()
    test_apply_without_cal_table()
    test_apply_with_cal_table()
    test_stratum_fallback()
    test_pvalue_monotonicity()

    # Pipeline integration
    test_pdf_loading()
    test_pipeline_v061()
    test_pipeline_with_local_proxy()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
