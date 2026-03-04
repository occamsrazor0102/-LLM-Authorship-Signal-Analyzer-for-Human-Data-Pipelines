#!/usr/bin/env python3
"""
BEET Detector ↔ Lexicon Packs Integration v1.0
═══════════════════════════════════════════════
Drop-in integration that enhances Layers 2.5, 2.6, and 2.7 with the
externalized versioned lexicon packs.

Usage:
    # Option A: Monkey-patch existing detector (simplest migration)
    import llm_detector
    import lexicon_packs_integration as lpi
    lpi.patch_detector(llm_detector)

    # Option B: Call enhanced layer functions directly
    from lexicon_packs_integration import (
        run_layer25_enhanced,
        run_layer26_enhanced,
        run_layer27_enhanced,
    )
    result = run_layer25_enhanced(text)

    # Option C: Use in analyze_prompt_enhanced() as full pipeline wrapper
    from lexicon_packs_integration import analyze_prompt_enhanced
    result = analyze_prompt_enhanced(text)
"""

import re
from typing import Dict, Optional

import lexicon_packs as lp

# Import the base detector — adjust the module name to match your file
try:
    import llm_detector as _det
except ImportError:
    _det = None


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED LAYER 2.5: PROMPT-ENGINEERING SIGNATURES + PACKS
# ══════════════════════════════════════════════════════════════════════════════

def run_layer25_enhanced(text: str, base_result: Optional[dict] = None) -> dict:
    """Enhanced Layer 2.5 with lexicon pack integration.

    Runs the legacy L2.5 first (or accepts pre-computed result),
    then augments with Priority 1 (constraint families) and Priority 3
    (exec-spec/rubric/Gherkin) packs.

    The enhanced composite score blends:
      - Legacy composite (existing CONSTRAINT_FRAMES + META_DESIGN_PATTERNS)
      - Pack constraint score (obligation + prohibition + recommendation +
        conditional + cardinality + state)
      - Pack exec-spec score (Gherkin + rubric + acceptance)

    New distinct_families count replaces the flat distinct_pats count,
    giving better resolution on the "how many different constraint types
    were used" question.
    """
    # Run legacy L2.5 if not provided
    if base_result is None:
        if _det is None:
            raise ImportError("llm_detector module required for base L2.5")
        base_result = _det.run_layer25(text)

    sents = base_result.get('_sentences', None)
    if sents is None:
        if _det and hasattr(_det, 'get_sentences'):
            sents = _det.get_sentences(text)
        else:
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
            sents = [s for s in sents if s.strip()]
    n_sents = max(len(sents), 1)

    # Score Priority 1 packs (constraint families)
    constraint_names = [n for n in lp.get_packs_for_layer('L2.5')
                        if lp.PACK_REGISTRY[n].category == 'constraint']
    constraint_scores = lp.score_packs(text, constraint_names, n_sents)
    total_constraint = lp.get_total_constraint_score(constraint_scores)

    # Score Priority 3 packs (exec-spec families)
    exec_spec_names = [n for n in lp.get_packs_for_layer('L2.5')
                       if lp.PACK_REGISTRY[n].category == 'exec_spec']
    exec_spec_scores = lp.score_packs(text, exec_spec_names, n_sents)
    total_exec_spec = lp.get_total_exec_spec_score(exec_spec_scores)

    # Count active pack families (replaces flat distinct_pats)
    all_pack_scores = {**constraint_scores, **exec_spec_scores}
    active_families = sum(1 for s in all_pack_scores.values() if s.raw_hits > 0)

    # ── Enhanced composite ──────────────────────────────────────────────
    # Legacy composite is already 0–1. Pack scores are capped per family.
    # Blend: legacy gets full weight; packs add boost with diminishing returns.
    legacy_composite = base_result.get('composite', 0.0)

    # Pack contribution: constraint families are high-yield, exec-spec adds precision
    pack_boost = 0.0

    # Constraint pack boost (Priority 1)
    if total_constraint >= 0.40:
        pack_boost += 0.20
    elif total_constraint >= 0.20:
        pack_boost += 0.12
    elif total_constraint >= 0.08:
        pack_boost += 0.05

    # Exec-spec pack boost (Priority 3)
    if total_exec_spec >= 0.30:
        pack_boost += 0.15
    elif total_exec_spec >= 0.15:
        pack_boost += 0.08
    elif total_exec_spec >= 0.05:
        pack_boost += 0.03

    # Family diversity bonus (analogous to old distinct_pats bonus)
    if active_families >= 6:
        pack_boost += 0.15
    elif active_families >= 4:
        pack_boost += 0.08
    elif active_families >= 2:
        pack_boost += 0.03

    # RFC 2119 uppercase normative keyword bonus
    uc_total = sum(s.uppercase_hits for s in all_pack_scores.values())
    if uc_total >= 3:
        pack_boost += 0.10
    elif uc_total >= 1:
        pack_boost += 0.04

    enhanced_composite = min(legacy_composite + pack_boost, 1.0)

    # ── Build enhanced result ───────────────────────────────────────────
    result = dict(base_result)
    result.update({
        # Enhanced scores
        'composite': enhanced_composite,
        'legacy_composite': legacy_composite,
        'pack_boost': round(pack_boost, 4),
        # Pack diagnostics
        'pack_constraint_score': round(total_constraint, 4),
        'pack_exec_spec_score': round(total_exec_spec, 4),
        'pack_active_families': active_families,
        'pack_uc_hits': uc_total,
        # Per-pack breakdown (for diagnostics/debugging)
        'pack_details': {
            name: {
                'hits': s.raw_hits,
                'capped': round(s.capped_score, 4),
                'uc': s.uppercase_hits,
            }
            for name, s in all_pack_scores.items()
            if s.raw_hits > 0
        },
    })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED LAYER 2.6: VSD + SCHEMA/FORMAT PACKS
# ══════════════════════════════════════════════════════════════════════════════

def run_layer26_enhanced(text: str, base_result: Optional[dict] = None) -> dict:
    """Enhanced Layer 2.6 with schema/structured-output vocabulary.

    The legacy spec_score only catches spreadsheet/file cues (camelCase columns,
    filenames, calculations, tab refs, column listings). This enhancement adds
    JSON Schema, OpenAPI, data serialization, and format/markup vocabulary.

    The enhanced spec_score blends legacy + pack scores, improving recall
    on API tasks, data engineering prompts, and code generation specs.
    """
    if base_result is None:
        if _det is None:
            raise ImportError("llm_detector module required for base L2.6")
        base_result = _det.run_layer26(text)

    words = text.split()
    n_words = len(words)
    per100 = max(n_words / 100, 1)

    # Score Priority 2 packs (schema families)
    schema_names = [n for n in lp.get_packs_for_layer('L2.6')
                    if lp.PACK_REGISTRY[n].category == 'schema']
    schema_scores = lp.score_packs(text, schema_names, n_sentences=1)
    total_schema = lp.get_total_schema_score(schema_scores)

    # Score Priority 5 packs (format/markup)
    format_names = [n for n in lp.get_packs_for_layer('L2.6')
                    if lp.PACK_REGISTRY[n].category == 'format']
    format_scores = lp.score_packs(text, format_names, n_sentences=1)
    total_format = lp.get_category_score(format_scores, 'format')

    all_pack_scores = {**schema_scores, **format_scores}

    # ── Enhanced spec_score ─────────────────────────────────────────────
    legacy_spec = base_result.get('spec_score', 0.0)

    # Schema vocabulary adds to spec_score (per-100-words scale to match legacy)
    schema_per100 = sum(s.weighted_hits for s in schema_scores.values()) / per100
    format_per100 = sum(s.weighted_hits for s in format_scores.values()) / per100

    # Weight schema higher than format (schema is primary signal per roadmap)
    pack_spec_boost = schema_per100 * 2.0 + format_per100 * 1.0
    enhanced_spec = legacy_spec + pack_spec_boost

    # ── Recalculate VSD with enhanced spec ──────────────────────────────
    voice_score = base_result.get('voice_score', 0.0)
    enhanced_vsd = voice_score * enhanced_spec

    # ── SSI re-check with enhanced spec ─────────────────────────────────
    ssi_spec_threshold = 5.0 if base_result.get('contractions', 0) == 0 else 7.0
    enhanced_ssi = (
        enhanced_spec >= ssi_spec_threshold
        and voice_score < 0.5
        and base_result.get('hedges', 0) == 0
        and n_words >= 150
    )

    result = dict(base_result)
    result.update({
        'spec_score': round(enhanced_spec, 2),
        'legacy_spec_score': legacy_spec,
        'vsd': round(enhanced_vsd, 1),
        'legacy_vsd': base_result.get('vsd', 0.0),
        'ssi_enhanced': enhanced_ssi,
        # Pack diagnostics
        'pack_schema_score': round(total_schema, 4),
        'pack_format_score': round(total_format, 4),
        'pack_schema_per100': round(schema_per100, 3),
        'pack_format_per100': round(format_per100, 3),
        'pack_spec_boost': round(pack_spec_boost, 3),
        'pack_details': {
            name: {
                'hits': s.raw_hits,
                'capped': round(s.capped_score, 4),
            }
            for name, s in all_pack_scores.items()
            if s.raw_hits > 0
        },
    })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED LAYER 2.7: IDI + TYPED INSTRUCTION OPERATORS
# ══════════════════════════════════════════════════════════════════════════════

def run_layer27_enhanced(text: str, base_result: Optional[dict] = None,
                         constraint_active: bool = False,
                         schema_active: bool = False) -> dict:
    """Enhanced Layer 2.7 with typed task-verb and value-domain operators.

    Key insight from roadmap: action verbs alone are noisy, but action verbs
    PLUS constraint or schema operators are strong. The `constraint_active`
    and `schema_active` flags enable pairing bonuses when other layers
    have already detected structural signal.

    Args:
        text: Normalized text.
        base_result: Pre-computed legacy L2.7 result (optional).
        constraint_active: True if L2.5 pack constraint score > threshold.
        schema_active: True if L2.6 pack schema score > threshold.
    """
    if base_result is None:
        if _det is None:
            raise ImportError("llm_detector module required for base L2.7")
        base_result = _det.run_layer27(text)

    words = text.split()
    n_words = len(words)
    per100 = max(n_words / 100, 1)

    # Score Priority 4 packs (instruction families)
    idi_names = lp.get_packs_for_layer('L2.7')
    idi_scores = lp.score_packs(text, idi_names, n_sentences=1)

    task_verb_score = idi_scores.get('task_verbs', lp.PackScore('task_verbs', 'instruction'))
    value_domain_score = idi_scores.get('value_domain', lp.PackScore('value_domain', 'instruction'))

    # Density per 100 words (to match legacy IDI scale)
    tv_per100 = task_verb_score.weighted_hits / per100
    vd_per100 = value_domain_score.weighted_hits / per100

    # ── Pairing bonus ───────────────────────────────────────────────────
    # Task verbs gain signal when constraint or schema packs also fired.
    # Without pairing, task_verbs contribute at reduced weight (0.5x).
    # With pairing, they contribute at full weight (1.0x).
    if constraint_active or schema_active:
        tv_weight = 1.0
        pairing_label = 'paired'
    else:
        tv_weight = 0.5
        pairing_label = 'unpaired'

    # Value-domain operators always contribute at full weight
    # (they are strong task signal regardless of pairing)
    pack_idi_boost = (tv_per100 * tv_weight * 1.0) + (vd_per100 * 2.0)

    legacy_idi = base_result.get('idi', 0.0)
    enhanced_idi = legacy_idi + pack_idi_boost

    result = dict(base_result)
    result.update({
        'idi': round(enhanced_idi, 1),
        'legacy_idi': legacy_idi,
        'pack_idi_boost': round(pack_idi_boost, 2),
        'pack_tv_per100': round(tv_per100, 3),
        'pack_vd_per100': round(vd_per100, 3),
        'pack_tv_pairing': pairing_label,
        'pack_tv_weight': tv_weight,
        'pack_details': {
            name: {
                'hits': s.raw_hits,
                'capped': round(s.capped_score, 4),
            }
            for name, s in idi_scores.items()
            if s.raw_hits > 0
        },
    })

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FULL ENHANCED PIPELINE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def analyze_prompt_enhanced(text: str, **kwargs) -> dict:
    """Drop-in replacement for analyze_prompt() with pack enhancement.

    Runs the base pipeline, then re-scores L2.5/L2.6/L2.7 with packs,
    and re-runs determine() with enhanced layer results.

    All kwargs are forwarded to the base analyze_prompt().
    """
    if _det is None:
        raise ImportError("llm_detector module required")

    # Run base pipeline
    result = _det.analyze_prompt(text, **kwargs)

    # Extract base layer results for re-enhancement
    normalized_text = text  # Already normalized inside analyze_prompt
    # (In production, you'd extract the normalized text from the pipeline)

    # Re-run layers with pack enhancement
    base_l25 = {
        'composite': result.get('l25_composite', 0.0),
        'cfd': result.get('l25_cfd', 0.0),
        'distinct_frames': result.get('l25_distinct_frames', 0),
        'mfsr': result.get('l25_mfsr', 0.0),
        'framing_completeness': result.get('l25_framing', 0),
        'conditional_density': result.get('l25_conditional_density', 0.0),
        'meta_design_hits': result.get('l25_meta_design', 0),
        'meta_design_details': [],
        'contractions': result.get('l25_contractions', 0),
        'must_count': 0,
        'must_rate': result.get('l25_must_rate', 0.0),
        'numbered_criteria': result.get('l25_numbered_criteria', 0),
    }

    base_l26 = {
        'voice_score': result.get('l26_voice_score', 0.0),
        'spec_score': result.get('l26_spec_score', 0.0),
        'vsd': result.get('l26_vsd', 0.0),
        'voice_gated': result.get('l26_voice_gated', False),
        'casual_markers': result.get('l26_casual_markers', 0),
        'misspellings': result.get('l26_misspellings', 0),
        'contractions': result.get('l25_contractions', 0),
        'em_dashes': 0,
        'camel_cols': result.get('l26_camel_cols', 0),
        'filenames': 0,
        'calcs': result.get('l26_calcs', 0),
        'tabs': 0,
        'col_listings': 0,
        'hedges': result.get('l26_hedges', 0),
    }

    base_l27 = {
        'idi': result.get('l27_idi', 0.0),
        'imperatives': result.get('l27_imperatives', 0),
        'imp_rate': 0.0,
        'conditionals': result.get('l27_conditionals', 0),
        'cond_rate': 0.0,
        'binary_specs': result.get('l27_binary_specs', 0),
        'missing_refs': result.get('l27_missing_refs', 0),
        'flag_count': result.get('l27_flag_count', 0),
    }

    # Enhanced layers
    enh_l25 = run_layer25_enhanced(text, base_result=base_l25)
    enh_l26 = run_layer26_enhanced(text, base_result=base_l26)

    # Determine pairing context for L2.7
    constraint_active = enh_l25.get('pack_constraint_score', 0) >= 0.08
    schema_active = enh_l26.get('pack_schema_score', 0) >= 0.05

    enh_l27 = run_layer27_enhanced(
        text,
        base_result=base_l27,
        constraint_active=constraint_active,
        schema_active=schema_active,
    )

    # Update result with enhanced values
    result.update({
        # Enhanced composites
        'l25_composite': enh_l25['composite'],
        'l25_legacy_composite': enh_l25.get('legacy_composite', 0.0),
        'l25_pack_boost': enh_l25.get('pack_boost', 0.0),
        'l25_pack_constraint': enh_l25.get('pack_constraint_score', 0.0),
        'l25_pack_exec_spec': enh_l25.get('pack_exec_spec_score', 0.0),
        'l25_pack_families': enh_l25.get('pack_active_families', 0),
        'l25_pack_uc_hits': enh_l25.get('pack_uc_hits', 0),

        'l26_spec_score': enh_l26['spec_score'],
        'l26_legacy_spec': enh_l26.get('legacy_spec_score', 0.0),
        'l26_vsd': enh_l26['vsd'],
        'l26_legacy_vsd': enh_l26.get('legacy_vsd', 0.0),
        'l26_pack_schema': enh_l26.get('pack_schema_score', 0.0),
        'l26_pack_format': enh_l26.get('pack_format_score', 0.0),

        'l27_idi': enh_l27['idi'],
        'l27_legacy_idi': enh_l27.get('legacy_idi', 0.0),
        'l27_pack_boost': enh_l27.get('pack_idi_boost', 0.0),
        'l27_pack_tv_pairing': enh_l27.get('pack_tv_pairing', 'unknown'),
    })

    # Re-run determination with enhanced layer results
    det, reason, confidence, channel_details = _det.determine(
        result.get('l0_score', 0.0),
        result.get('l0_severity', 'NONE'),
        enh_l25, enh_l26, enh_l27,
        result.get('word_count', 0),
        l30=None,  # Would need to pass through from base result
        l31=None,
        lang_gate={'support_level': result.get('lang_support_level', 'SUPPORTED')},
        norm_report={'obfuscation_delta': result.get('norm_obfuscation_delta', 0.0)},
        mode=result.get('mode', 'auto'),
        l2_score=result.get('l2_score', 0.0),
    )

    result['determination'] = det
    result['reason'] = reason
    result['confidence'] = confidence
    result['channel_details'] = channel_details
    result['_pack_enhanced'] = True

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MONKEY-PATCH HELPER
# ══════════════════════════════════════════════════════════════════════════════

def patch_detector(detector_module):
    """Monkey-patch the detector module to use enhanced layer functions.

    Usage:
        import llm_detector
        import lexicon_packs_integration as lpi
        lpi.patch_detector(llm_detector)
        # Now llm_detector.run_layer25() returns enhanced results
    """
    original_l25 = detector_module.run_layer25
    original_l26 = detector_module.run_layer26
    original_l27 = detector_module.run_layer27

    def patched_l25(text):
        base = original_l25(text)
        return run_layer25_enhanced(text, base_result=base)

    def patched_l26(text):
        base = original_l26(text)
        return run_layer26_enhanced(text, base_result=base)

    def patched_l27(text):
        base = original_l27(text)
        return run_layer27_enhanced(text, base_result=base)

    detector_module.run_layer25 = patched_l25
    detector_module.run_layer26 = patched_l26
    detector_module.run_layer27 = patched_l27

    print(f"[lexicon_packs] Patched {detector_module.__name__} with "
          f"{len(lp.PACK_REGISTRY)} lexicon packs (v{lp.__version__})")


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Test enhanced layers directly (without base detector)
    test_prompt = """
    You are a senior pharmacovigilance analyst at a mid-size biopharmaceutical company.
    Your task is to review the attached adverse event dataset (AE_jokinib_Q3.xlsx)
    and produce a signal detection report.

    REQUIREMENTS:
    1. You MUST classify each AE by System Organ Class using MedDRA v27.0 terminology.
    2. Each row MUST have a valid case_id. If absent, flag as MISSING.
    3. You SHALL NOT modify the original reporter narratives.
    4. Calculate the PRR (Proportional Reporting Ratio) for each preferred term.
       If the PRR exceeds 2.0, mark the signal as "disproportionate."
    5. The output schema must include: case_id (string, required),
       preferred_term (string), soc_code (enum), prr_value (number),
       signal_flag (boolean).

    ACCEPTANCE CRITERIA:
    - All required fields populated with no more than 3% MISSING values
    - PRR calculations verified against the reference dataset (golden_prr.csv)
    - Edge cases: duplicate case_ids, missing onset dates, unknown causality

    Given the dataset contains at least 500 records,
    When processing each adverse event,
    Then validate against the MedDRA hierarchy before classification.

    Format output as pipe-delimited CSV with header row.
    Expected output: exactly 8 columns per the schema definition.

    Rubric: Pass/fail on completeness, accuracy of PRR calculation, and
    correct MedDRA mapping. Source of truth: WHO-UMC signal detection guidelines.
    """

    print("=" * 70)
    print("ENHANCED LAYER TEST (standalone, no base detector)")
    print("=" * 70)

    # Simulate base results (what legacy layers would return)
    mock_l25 = {
        'composite': 0.45,
        'cfd': 0.35,
        'distinct_frames': 6,
        'mfsr': 0.12,
        'framing_completeness': 2,
        'conditional_density': 0.08,
        'meta_design_hits': 2,
        'meta_design_details': [],
        'contractions': 0,
        'must_count': 3,
        'must_rate': 0.18,
        'numbered_criteria': 5,
    }

    mock_l26 = {
        'voice_score': 0.3,
        'spec_score': 4.2,
        'vsd': 1.26,
        'voice_gated': False,
        'casual_markers': 0,
        'misspellings': 0,
        'contractions': 0,
        'em_dashes': 0,
        'camel_cols': 2,
        'filenames': 3,
        'calcs': 2,
        'tabs': 0,
        'col_listings': 1,
        'hedges': 0,
    }

    mock_l27 = {
        'idi': 8.5,
        'imperatives': 12,
        'imp_rate': 4.8,
        'conditionals': 5,
        'cond_rate': 2.0,
        'binary_specs': 2,
        'missing_refs': 1,
        'flag_count': 2,
    }

    enh_l25 = run_layer25_enhanced(test_prompt, base_result=mock_l25)
    enh_l26 = run_layer26_enhanced(test_prompt, base_result=mock_l26)
    constraint_active = enh_l25.get('pack_constraint_score', 0) >= 0.08
    schema_active = enh_l26.get('pack_schema_score', 0) >= 0.05
    enh_l27 = run_layer27_enhanced(test_prompt, base_result=mock_l27,
                                    constraint_active=constraint_active,
                                    schema_active=schema_active)

    print(f"\n── Layer 2.5 ──")
    print(f"  Legacy composite:    {mock_l25['composite']:.3f}")
    print(f"  Enhanced composite:  {enh_l25['composite']:.3f}  (+{enh_l25.get('pack_boost', 0):.3f})")
    print(f"  Constraint score:    {enh_l25.get('pack_constraint_score', 0):.3f}")
    print(f"  Exec-spec score:     {enh_l25.get('pack_exec_spec_score', 0):.3f}")
    print(f"  Active families:     {enh_l25.get('pack_active_families', 0)}")
    print(f"  UC normative hits:   {enh_l25.get('pack_uc_hits', 0)}")
    if enh_l25.get('pack_details'):
        for name, detail in enh_l25['pack_details'].items():
            print(f"    {name:20s} hits={detail['hits']} capped={detail['capped']:.3f}")

    print(f"\n── Layer 2.6 ──")
    print(f"  Legacy spec_score:   {mock_l26['spec_score']:.2f}")
    print(f"  Enhanced spec_score: {enh_l26['spec_score']:.2f}  (+{enh_l26.get('pack_spec_boost', 0):.3f})")
    print(f"  Schema score:        {enh_l26.get('pack_schema_score', 0):.3f}")
    print(f"  Format score:        {enh_l26.get('pack_format_score', 0):.3f}")
    print(f"  Legacy VSD:          {mock_l26['vsd']:.1f}")
    print(f"  Enhanced VSD:        {enh_l26['vsd']:.1f}")
    if enh_l26.get('pack_details'):
        for name, detail in enh_l26['pack_details'].items():
            print(f"    {name:20s} hits={detail['hits']} capped={detail['capped']:.3f}")

    print(f"\n── Layer 2.7 ──")
    print(f"  Legacy IDI:          {mock_l27['idi']:.1f}")
    print(f"  Enhanced IDI:        {enh_l27['idi']:.1f}  (+{enh_l27.get('pack_idi_boost', 0):.2f})")
    print(f"  Task verb density:   {enh_l27.get('pack_tv_per100', 0):.3f}/100w")
    print(f"  Value domain density:{enh_l27.get('pack_vd_per100', 0):.3f}/100w")
    print(f"  Pairing mode:        {enh_l27.get('pack_tv_pairing', '?')}")
    if enh_l27.get('pack_details'):
        for name, detail in enh_l27['pack_details'].items():
            print(f"    {name:20s} hits={detail['hits']} capped={detail['capped']:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"IMPACT SUMMARY")
    print(f"  L2.5: {mock_l25['composite']:.3f} → {enh_l25['composite']:.3f}  "
          f"(+{enh_l25['composite'] - mock_l25['composite']:.3f})")
    print(f"  L2.6: spec {mock_l26['spec_score']:.2f} → {enh_l26['spec_score']:.2f}  "
          f"VSD {mock_l26['vsd']:.1f} → {enh_l26['vsd']:.1f}")
    print(f"  L2.7: IDI {mock_l27['idi']:.1f} → {enh_l27['idi']:.1f}")

    # Would this change the determination?
    if enh_l25['composite'] >= 0.60:
        print(f"\n  L2.5 enhanced composite {enh_l25['composite']:.3f} ≥ 0.60 → RED eligible")
    elif enh_l25['composite'] >= 0.40:
        print(f"\n  L2.5 enhanced composite {enh_l25['composite']:.3f} ≥ 0.40 → AMBER eligible")
