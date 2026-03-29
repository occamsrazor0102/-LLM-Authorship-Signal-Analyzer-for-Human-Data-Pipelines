"""Full analysis pipeline orchestration."""

from llm_detector._constants import PIPELINE_VERSION, get_length_bin, is_ssi_triggered
from llm_detector.compat import HAS_SEMANTIC, HAS_PERPLEXITY
from llm_detector.normalize import normalize_text
from llm_detector.language_gate import check_language_support
from llm_detector.analyzers.preamble import run_preamble
from llm_detector.analyzers.fingerprint import run_fingerprint_full
from llm_detector.lexicon.integration import (
    run_prompt_signature_enhanced,
    run_voice_dissonance_enhanced,
    run_instruction_density_enhanced,
)
from llm_detector.analyzers.semantic_resonance import run_semantic_resonance
from llm_detector.analyzers.self_similarity import run_self_similarity
from llm_detector.analyzers.continuation_api import run_continuation_api_multi
from llm_detector.analyzers.continuation_local import run_continuation_local_multi
from llm_detector.analyzers.perplexity import run_perplexity
from llm_detector.analyzers.token_cohesiveness import run_token_cohesiveness
from llm_detector.analyzers.stylometry import mask_topical_content, extract_stylometric_features
from llm_detector.analyzers.semantic_flow import run_semantic_flow
from llm_detector.analyzers.windowing import score_windows, score_surprisal_windows, get_hot_window_spans
from llm_detector.fusion import determine
from llm_detector.calibration import apply_calibration


def analyze_prompt(text, task_id='', occupation='', attempter='', stage='',
                   run_l3=True, api_key=None, dna_provider='anthropic',
                   dna_model=None, dna_samples=3,
                   ground_truth=None, language=None, domain=None,
                   mode='auto', cal_table=None, memory_store=None,
                   disabled_channels=None, precomputed_continuation=None,
                   ppl_model=None):
    """Run full v0.68 pipeline on a single prompt. Returns result dict."""
    normalized_text, norm_report = normalize_text(text)
    word_count_raw = len(text.split())
    word_count = len(normalized_text.split())
    lang_gate = check_language_support(normalized_text, word_count)
    text_for_analysis = normalized_text

    preamble_score, preamble_severity, preamble_hits, preamble_spans = run_preamble(text_for_analysis)
    fingerprint_score, fingerprint_hits, fingerprint_rate, fingerprint_spans = run_fingerprint_full(text_for_analysis)
    prompt_sig = run_prompt_signature_enhanced(text_for_analysis)
    voice_dis = run_voice_dissonance_enhanced(text_for_analysis)
    instr_density = run_instruction_density_enhanced(
        text_for_analysis,
        constraint_active=(prompt_sig.get('pack_constraint_score', 0) > 0.08),
        schema_active=(voice_dis.get('pack_schema_score', 0) > 0.05),
    )

    self_sim = None
    if run_l3:
        self_sim = run_self_similarity(text_for_analysis)

    cont_result = None
    if precomputed_continuation is not None:
        cont_result = precomputed_continuation
    elif run_l3 and api_key:
        cont_result = run_continuation_api_multi(
            text_for_analysis, api_key=api_key, provider=dna_provider,
            model=dna_model, n_samples=dna_samples,
        )
    elif run_l3:
        cont_result = run_continuation_local_multi(text_for_analysis)

    semantic = run_semantic_resonance(text_for_analysis)
    ppl = run_perplexity(text_for_analysis, model_id=ppl_model)
    tocsin = run_token_cohesiveness(text_for_analysis)

    semantic_flow = run_semantic_flow(text_for_analysis)

    surprisal_traj = {}
    token_losses = ppl.get('token_losses')
    if token_losses:
        surprisal_traj = score_surprisal_windows(token_losses)

    masked_text, mask_count = mask_topical_content(text_for_analysis)
    stylo_features = extract_stylometric_features(text_for_analysis, masked_text)
    window_result = score_windows(text_for_analysis)

    # Detection spans merged from all annotation sources
    detection_spans = list(preamble_spans)
    detection_spans.extend(
        {'start': s, 'end': e, 'text': t, 'source': 'fingerprint', 'label': w, 'type': 'fingerprint'}
        for s, e, t, _, w in fingerprint_spans
    )
    detection_spans.extend(prompt_sig.get('pack_spans', []))
    detection_spans.extend(voice_dis.get('pack_spans', []))
    detection_spans.extend(instr_density.get('pack_spans', []))
    for hw in get_hot_window_spans(text_for_analysis, precomputed_result=window_result):
        detection_spans.append({
            'start': hw[0], 'end': hw[1], 'text': '', 'source': 'hot_window',
            'label': f'score={hw[2]:.2f}', 'type': 'window',
        })
    detection_spans.sort(key=lambda x: x.get('start', 0))

    det, reason, confidence, channel_details = determine(
        preamble_score, preamble_severity, prompt_sig, voice_dis, instr_density, word_count,
        self_sim=self_sim, cont_result=cont_result,
        lang_gate=lang_gate, norm_report=norm_report,
        mode=mode, fingerprint_score=fingerprint_score,
        semantic=semantic, ppl=ppl,
        tocsin=tocsin,
        semantic_flow=semantic_flow,
        window_result=window_result,
        disabled_channels=disabled_channels,
    )

    length_bin = get_length_bin(word_count)
    cal_result = apply_calibration(confidence, cal_table, domain=domain, length_bin=length_bin)

    audit_trail = {
        'pipeline_version': PIPELINE_VERSION,
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
        'semantic_available': HAS_SEMANTIC,
        'perplexity_available': HAS_PERPLEXITY,
    }

    result = {
        'task_id': task_id,
        'occupation': occupation,
        'attempter': attempter,
        'stage': stage,
        'word_count': word_count,
        'word_count_raw': word_count_raw,
        'determination': det,
        'reason': reason,
        'confidence': confidence,
        'calibrated_confidence': cal_result['calibrated_confidence'],
        'conformity_level': cal_result['conformity_level'],
        'calibration_stratum': cal_result['stratum_used'],
        'mode': channel_details.get('mode', mode),
        'channel_details': channel_details,
        'audit_trail': audit_trail,
        'pipeline_version': PIPELINE_VERSION,
        # Detection spans
        'detection_spans': detection_spans,
        # Normalization
        'norm_obfuscation_delta': norm_report.get('obfuscation_delta', 0.0),
        'norm_invisible_chars': norm_report.get('invisible_chars', 0),
        'norm_homoglyphs': norm_report.get('homoglyphs', 0),
        'norm_attack_types': norm_report.get('attack_types', []),
        # Fairness gate
        'lang_support_level': lang_gate.get('support_level', 'SUPPORTED'),
        'lang_fw_coverage': lang_gate.get('function_word_coverage', 0.0),
        'lang_non_latin_ratio': lang_gate.get('non_latin_ratio', 0.0),
        # Preamble
        'preamble_score': preamble_score,
        'preamble_severity': preamble_severity,
        'preamble_hits': len(preamble_hits),
        'preamble_details': preamble_hits,
        # Fingerprint (diagnostic-only)
        'fingerprint_score': fingerprint_score,
        'fingerprint_hits': fingerprint_hits,
        # Prompt signature
        'prompt_signature_composite': prompt_sig['composite'],
        'prompt_signature_cfd': prompt_sig['cfd'],
        'prompt_signature_distinct_frames': prompt_sig['distinct_frames'],
        'prompt_signature_mfsr': prompt_sig['mfsr'],
        'prompt_signature_framing': prompt_sig['framing_completeness'],
        'prompt_signature_conditional_density': prompt_sig['conditional_density'],
        'prompt_signature_meta_design': prompt_sig['meta_design_hits'],
        'prompt_signature_contractions': prompt_sig['contractions'],
        'prompt_signature_must_rate': prompt_sig['must_rate'],
        'prompt_signature_numbered_criteria': prompt_sig['numbered_criteria'],
        # Instruction density
        'instruction_density_idi': instr_density['idi'],
        'instruction_density_imperatives': instr_density['imperatives'],
        'instruction_density_conditionals': instr_density['conditionals'],
        'instruction_density_binary_specs': instr_density['binary_specs'],
        'instruction_density_missing_refs': instr_density['missing_refs'],
        'instruction_density_flag_count': instr_density['flag_count'],
        # Voice dissonance
        'voice_dissonance_voice_score': voice_dis['voice_score'],
        'voice_dissonance_spec_score': voice_dis['spec_score'],
        'voice_dissonance_vsd': voice_dis['vsd'],
        'voice_dissonance_voice_gated': voice_dis['voice_gated'],
        'voice_dissonance_casual_markers': voice_dis['casual_markers'],
        'voice_dissonance_misspellings': voice_dis['misspellings'],
        'voice_dissonance_camel_cols': voice_dis['camel_cols'],
        'voice_dissonance_calcs': voice_dis['calcs'],
        'voice_dissonance_hedges': voice_dis['hedges'],
        # SSI
        'ssi_triggered': is_ssi_triggered(voice_dis, word_count),
        # Metadata
        'ground_truth': ground_truth,
        'language': language,
        'domain': domain,
        # Windowed scoring
        'window_max_score': window_result.get('max_window_score', 0.0),
        'window_mean_score': window_result.get('mean_window_score', 0.0),
        'window_variance': window_result.get('window_variance', 0.0),
        'window_hot_span': window_result.get('hot_span_length', 0),
        'window_n_windows': window_result.get('n_windows', 0),
        'window_mixed_signal': window_result.get('mixed_signal', False),
        'window_fw_trajectory_cv': window_result.get('fw_trajectory_cv', 0.0),
        'window_comp_trajectory_mean': window_result.get('comp_trajectory_mean', 0.0),
        'window_comp_trajectory_cv': window_result.get('comp_trajectory_cv', 0.0),
        'window_changepoint': window_result.get('changepoint'),
        # Pack diagnostics
        'pack_constraint_score': prompt_sig.get('pack_constraint_score', 0.0),
        'pack_exec_spec_score': prompt_sig.get('pack_exec_spec_score', 0.0),
        'pack_schema_score': voice_dis.get('pack_schema_score', 0.0),
        'pack_active_families': prompt_sig.get('pack_active_families', 0),
        'pack_prompt_boost': prompt_sig.get('pack_boost', 0.0),
        'pack_idi_boost': instr_density.get('pack_idi_boost', 0.0),
        # Stylometric features
        'stylo_fw_ratio': stylo_features.get('function_word_ratio', 0.0),
        'stylo_sent_dispersion': stylo_features.get('sent_length_dispersion', 0.0),
        'stylo_ttr': stylo_features.get('type_token_ratio', 0.0),
        'stylo_avg_word_len': stylo_features.get('avg_word_length', 0.0),
        'stylo_short_word_ratio': stylo_features.get('short_word_ratio', 0.0),
        'stylo_mask_count': mask_count,
        'stylo_mattr': stylo_features.get('mattr', 0.0),
    }

    result.update({
        'semantic_resonance_ai_score': semantic.get('semantic_ai_score', 0.0),
        'semantic_resonance_human_score': semantic.get('semantic_human_score', 0.0),
        'semantic_resonance_ai_mean': semantic.get('semantic_ai_mean', 0.0),
        'semantic_resonance_human_mean': semantic.get('semantic_human_mean', 0.0),
        'semantic_resonance_delta': semantic.get('semantic_delta', 0.0),
        'semantic_resonance_determination': semantic.get('determination'),
        'semantic_resonance_confidence': semantic.get('confidence', 0.0),
    })

    result.update({
        'perplexity_value': ppl.get('perplexity', 0.0),
        'perplexity_determination': ppl.get('determination'),
        'perplexity_confidence': ppl.get('confidence', 0.0),
        'surprisal_variance': ppl.get('surprisal_variance', 0.0),
        'surprisal_first_half_var': ppl.get('surprisal_first_half_var', 0.0),
        'surprisal_second_half_var': ppl.get('surprisal_second_half_var', 0.0),
        'volatility_decay_ratio': ppl.get('volatility_decay_ratio', 1.0),
        'binoculars_score': ppl.get('binoculars_score', 0.0),
        'binoculars_determination': ppl.get('binoculars_determination'),
    })

    _ss = self_sim or {}
    result.update({
        'self_similarity_nssi_score': _ss.get('nssi_score', 0.0),
        'self_similarity_nssi_signals': _ss.get('nssi_signals', 0),
        'self_similarity_determination': _ss.get('determination'),
        'self_similarity_confidence': _ss.get('confidence', 0.0),
        'self_similarity_formulaic_density': _ss.get('formulaic_density', 0.0),
        'self_similarity_power_adj_density': _ss.get('power_adj_density', 0.0),
        'self_similarity_demonstrative_density': _ss.get('demonstrative_density', 0.0),
        'self_similarity_transition_density': _ss.get('transition_density', 0.0),
        'self_similarity_scare_quote_density': _ss.get('scare_quote_density', 0.0),
        'self_similarity_emdash_density': _ss.get('emdash_density', 0.0),
        'self_similarity_this_the_start_rate': _ss.get('this_the_start_rate', 0.0),
        'self_similarity_section_depth': _ss.get('section_depth', 0),
        'self_similarity_sent_length_cv': _ss.get('sent_length_cv', 0.0),
        'self_similarity_comp_ratio': _ss.get('comp_ratio', 0.0),
        'self_similarity_hapax_ratio': _ss.get('hapax_ratio', 0.0),
        'self_similarity_hapax_count': _ss.get('hapax_count', 0),
        'self_similarity_unique_words': _ss.get('unique_words', 0),
        'self_similarity_shuffled_comp_ratio': _ss.get('shuffled_comp_ratio', 0.0),
        'self_similarity_structural_compression_delta': _ss.get('structural_compression_delta', 0.0),
    })

    _cr = cont_result or {}
    _proxy = _cr.get('proxy_features', {})
    result.update({
        'continuation_bscore': _cr.get('bscore', 0.0),
        'continuation_bscore_max': _cr.get('bscore_max', 0.0),
        'continuation_determination': _cr.get('determination'),
        'continuation_confidence': _cr.get('confidence', 0.0),
        'continuation_n_samples': _cr.get('n_samples', 0),
        'continuation_mode': ('local' if _proxy else 'api') if cont_result else None,
        'continuation_ncd': _proxy.get('ncd', 0.0),
        'continuation_internal_overlap': _proxy.get('internal_overlap', 0.0),
        'continuation_cond_surprisal': _proxy.get('cond_surprisal', 0.0),
        'continuation_repeat4': _proxy.get('repeat4', 0.0),
        'continuation_ttr': _proxy.get('ttr', 0.0),
        'continuation_composite': _proxy.get('composite', 0.0),
        'continuation_composite_variance': _proxy.get('composite_variance', 0.0),
        'continuation_composite_stability': _proxy.get('composite_stability', 0.0),
        'continuation_improvement_rate': _proxy.get('improvement_rate', 0.0),
        'continuation_ncd_matrix_mean': _proxy.get('ncd_matrix_mean', 0.0),
        'continuation_ncd_matrix_variance': _proxy.get('ncd_matrix_variance', 0.0),
        'continuation_ncd_matrix_min': _proxy.get('ncd_matrix_min', 0.0),
    })

    result.update({
        'tocsin_cohesiveness': tocsin.get('cohesiveness', 0.0),
        'tocsin_cohesiveness_std': tocsin.get('cohesiveness_std', 0.0),
        'tocsin_determination': tocsin.get('determination'),
        'tocsin_confidence': tocsin.get('confidence', 0.0),
        'perplexity_comp_ratio': ppl.get('comp_ratio', 0.0),
        'perplexity_zlib_normalized_ppl': ppl.get('zlib_normalized_ppl', 0.0),
        'perplexity_comp_ppl_ratio': ppl.get('comp_ppl_ratio', 0.0),
        'semantic_flow_variance': semantic_flow.get('flow_variance', 0.0),
        'semantic_flow_mean': semantic_flow.get('flow_mean', 0.0),
        'semantic_flow_std': semantic_flow.get('flow_std', 0.0),
        'semantic_flow_determination': semantic_flow.get('determination'),
        'semantic_flow_confidence': semantic_flow.get('confidence', 0.0),
        'ppl_burstiness': ppl.get('ppl_burstiness', 0.0),
        'sentence_ppl_count': ppl.get('sentence_ppl_count', 0),
        'surprisal_trajectory_cv': surprisal_traj.get('surprisal_trajectory_cv', 0.0),
        'surprisal_var_of_var': surprisal_traj.get('surprisal_var_of_var', 0.0),
        'surprisal_stationarity': surprisal_traj.get('surprisal_stationarity', 0.0),
    })

    shadow_disagreement = None
    if memory_store is not None:
        shadow_disagreement = memory_store.check_shadow_disagreement(result)

    result['shadow_disagreement'] = shadow_disagreement
    result['shadow_ai_prob'] = (shadow_disagreement or {}).get('shadow_ai_prob')

    return result
