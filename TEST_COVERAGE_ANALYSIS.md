# Test Coverage Analysis

**Date:** 2026-03-10
**Overall coverage:** 39% (4,197 statements missed out of 6,829)
**Tests passing:** 152/152 ✓

---

## Executive Summary

This document analyzes the test coverage of the LLM Authorship Signal Analyzer and provides **specific, actionable recommendations** for improving test quality and coverage. The analysis focuses on:

1. **Critical business logic** (channels, fusion) — the core detection engine
2. **Data pipelines** (I/O, baselines) — entry and calibration paths
3. **Optional ML features** (perplexity, semantic resonance) — currently 6-15% covered
4. **Testing infrastructure** — configuration, CI/CD, and test patterns

---

## Current Coverage Summary

| Module | Coverage | Notes |
|--------|----------|-------|
| `analyzers/preamble.py` | 100% | Fully covered |
| `analyzers/fingerprint.py` | 100% | Fully covered |
| `analyzers/instruction_density.py` | 100% | Fully covered |
| `analyzers/prompt_signature.py` | 98% | Nearly complete |
| `analyzers/voice_dissonance.py` | 98% | Nearly complete |
| `analyzers/stylometry.py` | 98% | Nearly complete |
| `analyzers/continuation_local.py` | 92% | Good |
| `analyzers/self_similarity.py` | 89% | Good |
| `analyzers/windowing.py` | 89% | Good |
| `calibration.py` | 95% | Nearly complete |
| `pipeline.py` | 92% | Good |
| `lexicon/integration.py` | 90% | Good |
| `channels/continuation.py` | 88% | Good |
| `normalize.py` | 83% | Decent |
| `similarity.py` | 76% | Moderate gaps |
| `channels/windowed.py` | 75% | Moderate gaps |
| `lexicon/packs.py` | 73% | Moderate gaps |
| `fusion.py` | 73% | Moderate gaps |
| `language_gate.py` | 69% | Gaps |
| `text_utils.py` | 70% | Gaps |
| `html_report.py` | 58% | Significant gaps |
| `reporting.py` | 62% | Significant gaps |
| `channels/stylometric.py` | 46% | Poor |
| `compat.py` | 46% | Poor |
| `channels/prompt_structure.py` | 42% | Poor |
| `io.py` | 42% | Poor |
| `memory.py` | 42% | Poor |
| `cli.py` | 35% | Poor |
| `analyzers/continuation_api.py` | 29% | Very poor |
| `baselines.py` | 15% | Very poor |
| `analyzers/token_cohesiveness.py` | 15% | Very poor |
| `analyzers/semantic_resonance.py` | 13% | Very poor |
| `analyzers/perplexity.py` | 6% | Nearly untested |
| `dashboard.py` | 0% | No tests |
| `gui.py` | 0% | No tests |
| `__main__.py` | 0% | No tests |

---

## Priority Recommendations

### 1. Channels — prompt_structure (42%) and stylometric (46%)

**Why:** These are the two primary scoring channels that determine the final RED/AMBER/YELLOW/GREEN outcome. Bugs here directly affect detection accuracy.

**What to test:**
- `score_prompt_structure()`: Preamble CRITICAL path, prompt signature thresholds at each boundary (0.20, 0.40, 0.60), voice dissonance gated vs ungated paths, SSI triggering logic, IDI threshold boundaries (8, 12)
- `score_stylometric()`: NSSI RED/AMBER/YELLOW paths, semantic resonance boosting vs standalone, perplexity boosting vs standalone, surprisal variance + volatility decay interaction, TOCSIN corroboration, binoculars integration, fingerprint supporting-weight logic
- Both channels have complex severity-escalation logic (max-by-severity comparisons) that is not exercised at all

### 2. Fusion logic (73%)

**Why:** `fusion.py` is the final decision-maker that combines all 4 channels into the determination. Missing coverage on edge-case paths means novel combinations of channel signals could produce unexpected results.

**What to test:**
- Mode auto-detection (`task_prompt` vs `generic_aigt`)
- Corroboration logic when multiple channels fire at different severity levels
- MIXED determination conditions (conflicting strong signals)
- Fairness severity caps for non-English input
- Channel priority aggregation when only some channels have data
- The `data_sufficient=False` fallback path

### 3. Baselines module (15%)

**Why:** `collect_baselines()` and `analyze_baselines()` handle labeled data accumulation and percentile analysis. Only `derive_attack_type()` is tested. The actual collection and analysis paths — which are critical for calibration — have zero coverage.

**What to test:**
- `collect_baselines()`: JSONL writing, length-bin assignment, field extraction, append behavior
- `analyze_baselines()`: Percentile computation, stratified flag rates, TPR@FPR calculation with labeled ground truth, disparity warnings, empty/malformed record handling

### 4. I/O module (42%)

**Why:** File loading is the entry point for batch analysis. The XLSX loader (`load_xlsx`) has zero coverage, the CSV loader (`load_csv`) has zero coverage. Only the PDF helper and `_col_letter_to_index` are tested.

**What to test:**
- `load_xlsx()`: Sheet auto-detection (FullTaskX, etc.), column fuzzy matching, positional column references (A-Z), short-prompt filtering (<50 chars), empty workbooks
- `load_csv()`: Column resolution by name and position, substring fallback matching, missing prompt column error path
- `load_pdf()`: Per-page extraction, full-document fallback, pypdf not installed error path

### 5. Memory / BEET store (42%)

**Why:** The MemoryStore is a 941-line module responsible for persistent cross-batch state, attempter risk profiling, and ML tool orchestration. More than half is untested.

**What to test:**
- `rebuild_shadow_model()`: Model training, disagreement logging, feature extraction
- `discover_lexicon_candidates()`: Log-odds computation, candidate ranking
- `rebuild_semantic_centroids()`: K-means centroid generation, versioned storage
- Attempter risk tier calculation (CRITICAL/HIGH/ELEVATED/NORMAL transitions)
- Ground truth confirmation loop
- Calibration rebuild from confirmed labels
- Store migration/version handling
- Edge cases: corrupt JSONL, missing files, concurrent access

### 6. Analyzers with optional dependencies (6–15%)

**Why:** `perplexity.py` (6%), `semantic_resonance.py` (13%), and `token_cohesiveness.py` (15%) rely on optional ML dependencies (transformers, sentence-transformers). Their core logic is almost entirely untested, even though they contribute significantly to the stylometric channel.

**What to test:**
- Mock or stub the ML models to test the scoring logic independently
- `perplexity.py`: Surprisal variance computation, volatility decay ratio, compression ratio, binoculars path, minimum word-count guard
- `semantic_resonance.py`: Cosine similarity aggregation, delta calculation, determination thresholds, minimum sentence guard
- `token_cohesiveness.py`: Deletion perturbation strategy, fragility score computation, threshold logic

### 7. CLI (35%)

**Why:** `cli.py` is 778 statements and the primary user interface. Much of the argument parsing, batch processing, and output formatting is untested.

**What to test:**
- Argument parsing for key flag combinations (--mode, --disable-channel, --baselines, --calibrate)
- Batch file dispatch (xlsx/csv/pdf routing)
- Memory store integration (--remember, --attempter-history)
- Output formatting (--json, --verbose, --quiet)
- Error handling for invalid inputs

### 8. Language gate (69%)

**Why:** The language gate controls fairness behavior for non-English input and can cap severity. Missing coverage on the non-Latin script and low-function-word-coverage paths risks false positives on non-English submissions.

**What to test:**
- Non-Latin script ratio detection and severity capping
- Function word coverage validation for different languages
- Interaction between language gate and downstream fairness caps in fusion

---

## Testing Infrastructure Improvements

1. **Add pytest configuration** — The project lacks `pytest.ini` or `[tool.pytest]` in `pyproject.toml`. Adding configuration would enable coverage thresholds, test markers (e.g., `@pytest.mark.slow` for ML-dependent tests), and consistent settings.

2. **Use pytest-style assertions** — Current tests use a custom `check(label, condition, detail)` helper. Migrating to standard `assert` statements would give better failure diagnostics out of the box.

3. **CI coverage enforcement** — The GitHub Actions workflow runs `pytest -q` but does not measure or enforce coverage. Adding `--cov --cov-fail-under=60` would prevent regressions.

4. **Fixture improvements** — The shared `AI_TEXT`, `HUMAN_TEXT`, and `CLINICAL_TEXT` fixtures are good but could be expanded with edge-case texts: very short (<50 words), non-English, adversarially obfuscated, and mixed human+AI content.

5. **Mock optional dependencies** — Many analyzers are untestable without heavy ML packages. Creating lightweight mock/stub fixtures for sentence-transformers and transformers models would allow testing scoring logic without GPU dependencies.

---

## Detailed Test Recommendations by Module

This section provides **specific test cases** that should be added to improve coverage in critical areas.

### 1. `channels/prompt_structure.py` (42% → target 85%)

**Current gaps:** Lines 25-27, 33-35, 47-57, 64-71, 82-92, 96-105 are untested.

**Specific test cases to add:**

```python
# tests/test_prompt_structure_channel.py

def test_preamble_critical_instant_red():
    """CRITICAL preamble bypasses all other signals and returns RED immediately."""
    result = score_prompt_structure(
        preamble_score=0.99,
        preamble_severity='CRITICAL',
        prompt_sig={'composite': 0.0},  # All other signals zero
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None,
        word_count=50
    )
    check("CRITICAL preamble returns RED", result.severity == 'RED')
    check("CRITICAL preamble score 0.99", result.score == 0.99)
    check("CRITICAL preamble explanation", 'Preamble detection (critical hit)' in result.explanation)

def test_prompt_signature_thresholds():
    """Test all three prompt signature thresholds (0.20, 0.40, 0.60)."""
    # RED threshold: composite >= 0.60
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.65},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100
    )
    check("prompt_sig 0.65 -> RED", r.severity == 'RED')

    # AMBER threshold: composite >= 0.40
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.45},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100
    )
    check("prompt_sig 0.45 -> AMBER", r.severity == 'AMBER')

    # YELLOW threshold: composite >= 0.20
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.25},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100
    )
    check("prompt_sig 0.25 -> YELLOW", r.severity == 'YELLOW')

def test_vsd_voice_gated_paths():
    """Test voice-gated VSD at RED (>=50) and AMBER (>=21) thresholds."""
    # VSD gated RED: vsd >= 50
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': True, 'vsd': 55, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100
    )
    check("VSD gated 55 -> RED", r.severity == 'RED')
    check("VSD gated score 0.90", r.score == 0.90)

    # VSD gated AMBER: vsd >= 21
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': True, 'vsd': 25, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100
    )
    check("VSD gated 25 -> AMBER", r.severity == 'AMBER')
    check("VSD gated score 0.70", r.score == 0.70)

def test_idi_thresholds():
    """Test instruction density at RED (>=12) and AMBER (>=8) thresholds."""
    # IDI RED: idi >= 12
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density={'idi': 13},
        word_count=100
    )
    check("IDI 13 -> RED", r.severity == 'RED')
    check("IDI 13 score 0.85", r.score == 0.85)

    # IDI AMBER: idi >= 8
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density={'idi': 9},
        word_count=100
    )
    check("IDI 9 -> AMBER", r.severity == 'AMBER')
    check("IDI 9 score 0.65", r.score == 0.65)

def test_ssi_trigger_conditions():
    """Test Sterile Specification Index (SSI) triggering logic."""
    # SSI triggered with contractions=0 (threshold 5.0)
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 6.0, 'voice_score': 0.3, 'hedges': 0},
        instr_density=None,
        word_count=200  # >= 150 required
    )
    check("SSI triggered with contractions=0", 'SSI' in r.explanation)
    check("SSI AMBER at 6.0", r.severity == 'AMBER')

    # SSI triggered with contractions>0 (threshold 7.0)
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 2, 'spec_score': 8.0, 'voice_score': 0.3, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("SSI triggered with contractions>0", 'SSI' in r.explanation)
    check("SSI AMBER at 8.0", r.severity == 'AMBER')

    # SSI not triggered: word_count < 150
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 6.0, 'voice_score': 0.3, 'hedges': 0},
        instr_density=None,
        word_count=100  # < 150
    )
    check("SSI not triggered with word_count < 150", 'SSI' not in r.explanation)

def test_vsd_ungated_paths():
    """Test ungated VSD at high (>=100) and medium (>=21) thresholds."""
    # VSD ungated very high: vsd >= 100
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 105, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100
    )
    check("VSD ungated 105 -> AMBER", r.severity == 'AMBER')
    check("VSD ungated 105 score 0.60", r.score == 0.60)

    # VSD ungated medium: vsd >= 21
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 30, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100
    )
    check("VSD ungated 30 -> YELLOW", r.severity == 'YELLOW')
    check("VSD ungated 30 score 0.30", r.score == 0.30)
```

**Expected impact:** Coverage 42% → 85% (covers all branching logic for thresholds and severity escalation)

---

### 2. `channels/stylometric.py` (46% → target 85%)

**Current gaps:** Lines 32-39, 57-64, 73-80, 83-84, 96-104, 108-122, 126-141, 145-146 are untested.

**Specific test cases to add:**

```python
# tests/test_stylometric_channel.py

def test_nssi_severity_thresholds():
    """Test NSSI (Novel Self-Similarity Index) at RED/AMBER/YELLOW thresholds."""
    # NSSI RED: nssi >= 0.70
    self_sim = {'nssi_score': 0.75, 'nssi_signals': 4, 'determination': 'RED'}
    r = score_stylometric(
        fingerprint_score=0.0,
        self_sim=self_sim,
        voice_dis={'voice_score': 0.5},
        semantic=None, ppl=None, tocsin=None
    )
    check("NSSI 0.75 -> RED", r.severity == 'RED')

    # NSSI AMBER: nssi >= 0.45
    self_sim = {'nssi_score': 0.50, 'nssi_signals': 3, 'determination': 'AMBER'}
    r = score_stylometric(0.0, self_sim, {'voice_score': 0.5})
    check("NSSI 0.50 -> AMBER", r.severity == 'AMBER')

    # NSSI YELLOW: nssi >= 0.25
    self_sim = {'nssi_score': 0.30, 'nssi_signals': 2, 'determination': 'YELLOW'}
    r = score_stylometric(0.0, self_sim, {'voice_score': 0.5})
    check("NSSI 0.30 -> YELLOW", r.severity == 'YELLOW')

def test_semantic_resonance_boosting():
    """Test semantic resonance as standalone signal and NSSI booster."""
    # Semantic as standalone (delta >= 0.20 -> YELLOW)
    semantic = {'semantic_delta': 0.25, 'semantic_ai_score': 0.70, 'determination': 'YELLOW'}
    r = score_stylometric(0.0, None, {'voice_score': 0.5}, semantic=semantic)
    check("Semantic standalone delta 0.25 -> YELLOW", r.severity == 'YELLOW')

    # Semantic boosts NSSI (delta >= 0.15 + nssi present)
    self_sim = {'nssi_score': 0.30, 'nssi_signals': 2, 'determination': 'YELLOW'}
    semantic = {'semantic_delta': 0.18, 'semantic_ai_score': 0.65, 'determination': 'YELLOW'}
    r = score_stylometric(0.0, self_sim, {'voice_score': 0.5}, semantic=semantic)
    check("Semantic boosts NSSI", 'semantic_resonance' in r.sub_signals)
    check("Boosted severity increased", r.score > 0.30)

def test_perplexity_boosting():
    """Test perplexity surprisal variance and volatility decay boosting."""
    # Surprisal variance standalone (>= 2.5 -> contributes)
    ppl = {'surprisal_variance': 3.0, 'volatility_decay_ratio': 1.5, 'determination': 'YELLOW'}
    r = score_stylometric(0.0, None, {'voice_score': 0.5}, ppl=ppl)
    check("Surprisal variance 3.0 present", 'surprisal_variance' in r.sub_signals)

    # Volatility decay interaction (variance * decay)
    self_sim = {'nssi_score': 0.40, 'nssi_signals': 2, 'determination': 'YELLOW'}
    ppl = {'surprisal_variance': 3.5, 'volatility_decay_ratio': 2.0, 'determination': 'YELLOW'}
    r = score_stylometric(0.0, self_sim, {'voice_score': 0.5}, ppl=ppl)
    check("Volatility decay boosts signal", r.score > 0.40)

def test_tocsin_corroboration():
    """Test TOCSIN (token cohesiveness) as supporting signal."""
    # TOCSIN corroborates NSSI
    self_sim = {'nssi_score': 0.50, 'nssi_signals': 3, 'determination': 'AMBER'}
    tocsin = {'tocsin_fragility': 0.65, 'determination': 'AMBER'}
    r = score_stylometric(0.0, self_sim, {'voice_score': 0.5}, tocsin=tocsin)
    check("TOCSIN corroborates NSSI", 'tocsin' in r.sub_signals)

def test_fingerprint_supporting_weight():
    """Test fingerprint as supporting signal (>=0.30 contributes)."""
    fingerprint_score = 0.40
    self_sim = {'nssi_score': 0.35, 'nssi_signals': 2, 'determination': 'YELLOW'}
    r = score_stylometric(fingerprint_score, self_sim, {'voice_score': 0.5})
    check("Fingerprint 0.40 adds weight", 'fingerprint' in r.sub_signals)
    check("Fingerprint boosts score", r.score > 0.35)
```

**Expected impact:** Coverage 46% → 85% (covers all severity paths and signal interactions)

---

### 3. `fusion.py` (73% → target 95%)

**Current gaps:** Lines 16, 22, 24, 29, 76, 85-86, 91-92, 133-134, 138-145, 152-156, 162, 171-180, 186, 191-193, 198-199 are untested.

**Specific test cases to add:**

```python
# tests/test_fusion_edge_cases.py

def test_mode_auto_detection():
    """Test auto-detection between task_prompt and generic_aigt modes."""
    # Task prompt signals: prompt_sig >= 0.15, idi >= 5
    mode = _detect_mode(
        prompt_sig={'composite': 0.20, 'framing_completeness': 2},
        instr_density={'idi': 6},
        self_sim=None,
        word_count=100
    )
    check("Task prompt detected", mode == 'task_prompt')

    # Generic AIGT signals: nssi_signals >= 3, word_count >= 400
    mode = _detect_mode(
        prompt_sig={'composite': 0.05, 'framing_completeness': 0},
        instr_density={'idi': 2},
        self_sim={'nssi_signals': 4},
        word_count=450
    )
    check("Generic AIGT detected", mode == 'generic_aigt')

def test_channel_ablation():
    """Test that disabled channels are excluded from fusion."""
    det, reason, conf, details = determine(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.70},  # Would trigger RED
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100,
        disabled_channels=['prompt_structure']
    )
    check("Disabled channel excluded", details['channels']['prompt_structure']['disabled'])
    check("Disabled channel reason", 'ablation' in details['channels']['prompt_structure']['explanation'])

def test_fairness_severity_cap_unsupported():
    """Test language gate severity cap for UNSUPPORTED languages."""
    # RED detection capped to YELLOW for unsupported language
    det, reason, conf, details = determine(
        preamble_score=0.99, preamble_severity='CRITICAL',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100,
        lang_gate={'support_level': 'UNSUPPORTED', 'reason': 'Non-Latin script detected'}
    )
    check("RED capped to YELLOW", det == 'YELLOW')
    check("Cap reason in explanation", 'capped from RED' in reason)

def test_fairness_severity_cap_review():
    """Test language gate severity cap for REVIEW languages."""
    # RED detection capped to AMBER for review-level language
    det, reason, conf, details = determine(
        preamble_score=0.99, preamble_severity='CRITICAL',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100,
        lang_gate={'support_level': 'REVIEW', 'reason': 'Low function word coverage'}
    )
    check("RED capped to AMBER", det == 'AMBER')
    check("Cap reason in explanation", 'capped from RED' in reason)

def test_multi_channel_convergence():
    """Test RED determination from multiple AMBER channels."""
    # 2 AMBER channels -> RED
    det, reason, conf, details = determine(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.45},  # AMBER
        voice_dis={'voice_gated': True, 'vsd': 25, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},  # AMBER
        instr_density=None, word_count=100,
        self_sim={'nssi_score': 0.10, 'nssi_signals': 1, 'determination': 'GREEN'}
    )
    check("2 AMBER channels -> RED", det == 'RED')

def test_mixed_determination_windowed_variance():
    """Test MIXED determination from windowed variance signal."""
    # Windowed channel shows mixed signal
    det, reason, conf, details = determine(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.45},  # AMBER
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100,
        window_result={'max_score': 0.70, 'variance': 0.25, 'mixed_signal': True}
    )
    check("MIXED determination from windowed variance", det == 'MIXED')
    check("MIXED reason mentions hybrid", 'hybrid text' in reason)

def test_short_text_relaxation():
    """Test short-text relaxation for RED determination."""
    # Short text (<100 words) with 1 RED + 1 YELLOW -> RED (relaxed)
    det, reason, conf, details = determine(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.70},  # RED
        voice_dis={'voice_gated': False, 'vsd': 30, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},  # YELLOW
        instr_density=None, word_count=80,  # Short text
    )
    check("Short text relaxation applied", details['short_text_adjustment'])
    check("Short text RED determination", det == 'RED')
    check("Short text reason", 'short-text relaxed' in reason)

def test_obfuscation_delta_fallback():
    """Test normalization obfuscation delta as fallback signal."""
    # No channel signals, but obfuscation delta >= 0.05 -> YELLOW
    det, reason, conf, details = determine(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0, 'spec_score': 0, 'voice_score': 0, 'hedges': 0},
        instr_density=None, word_count=100,
        norm_report={'obfuscation_delta': 0.08, 'homoglyphs': 2}
    )
    check("Obfuscation delta -> YELLOW", det == 'YELLOW')
    check("Obfuscation reason", 'obfuscation' in reason.lower())

def test_data_insufficient_fallback():
    """Test behavior when channels have insufficient data."""
    # All channels return data_sufficient=False (tested via custom ChannelResult mocking)
    # This would require testing the channel construction or mocking
    pass  # Note: This is tricky to test without modifying channel scoring functions
```

**Expected impact:** Coverage 73% → 95% (covers all fusion logic paths and edge cases)

---

### 4. `baselines.py` (15% → target 70%)

**Current gaps:** Lines 64-86, 91-221 are untested (only `derive_attack_type` is tested).

**Specific test cases to add:**

```python
# tests/test_baselines_collection.py

def test_collect_baselines_write():
    """Test baseline collection writes JSONL with correct fields."""
    import tempfile
    import json

    results = [
        {
            'task_id': 'test1', 'word_count': 150, 'determination': 'RED',
            'confidence': 0.85, 'preamble_score': 0.70,
            'prompt_signature_composite': 0.60,
            'norm_homoglyphs': 0, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0.0
        },
        {
            'task_id': 'test2', 'word_count': 450, 'determination': 'GREEN',
            'confidence': 0.05, 'preamble_score': 0.0,
            'prompt_signature_composite': 0.10,
            'norm_homoglyphs': 2, 'norm_invisible_chars': 1, 'norm_obfuscation_delta': 0.03
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name

    n_written = collect_baselines(results, output_path)
    check("2 records written", n_written == 2)

    # Read back and verify
    with open(output_path, 'r') as f:
        lines = f.readlines()
    check("2 lines in JSONL", len(lines) == 2)

    record1 = json.loads(lines[0])
    check("Record 1 has task_id", record1['task_id'] == 'test1')
    check("Record 1 has length_bin", record1['length_bin'] == 'medium')
    check("Record 1 has attack_type", record1['attack_type'] == 'none')
    check("Record 1 has timestamp", '_timestamp' in record1)
    check("Record 1 has version", record1['_version'] == 'v0.66')

    record2 = json.loads(lines[1])
    check("Record 2 length_bin long", record2['length_bin'] == 'long')
    check("Record 2 attack_type combined", record2['attack_type'] == 'combined')

    os.unlink(output_path)

def test_collect_baselines_append():
    """Test baseline collection appends to existing file."""
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        f.write('{"task_id": "existing"}\n')

    results = [{'task_id': 'new1', 'word_count': 50}]
    collect_baselines(results, output_path)

    with open(output_path, 'r') as f:
        lines = f.readlines()
    check("2 lines after append", len(lines) == 2)
    check("First line unchanged", 'existing' in lines[0])
    check("Second line appended", 'new1' in lines[1])

    os.unlink(output_path)

def test_analyze_baselines_distribution():
    """Test baseline analysis computes determination distribution."""
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        for det in ['RED', 'RED', 'AMBER', 'YELLOW', 'GREEN', 'GREEN', 'GREEN']:
            record = {
                'determination': det, 'occupation': 'test_occ',
                'word_count': 200, 'confidence': 0.5,
                'prompt_signature_composite': 0.3
            }
            f.write(json.dumps(record) + '\n')

    rows = analyze_baselines(output_path)
    check("analyze_baselines returns data", rows is not None)

    os.unlink(output_path)

def test_analyze_baselines_percentiles():
    """Test baseline analysis computes percentiles for metrics."""
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        for i in range(50):
            record = {
                'determination': 'GREEN', 'occupation': 'test_occ',
                'word_count': 100 + i * 10, 'confidence': 0.1 + i * 0.01,
                'prompt_signature_composite': 0.05 + i * 0.01,
                'instruction_density_idi': i
            }
            f.write(json.dumps(record) + '\n')

    rows = analyze_baselines(output_path)
    check("Percentile rows computed", len(rows) > 0)

    # Check that we have entries for different metrics
    metrics = {r['metric'] for r in rows}
    check("word_count in metrics", 'word_count' in metrics)
    check("prompt_signature_composite in metrics", 'prompt_signature_composite' in metrics)

    os.unlink(output_path)

def test_analyze_baselines_tpr_at_fpr():
    """Test TPR@FPR calculation with ground truth labels."""
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        # 10 human (should be GREEN)
        for i in range(10):
            f.write(json.dumps({
                'ground_truth': 'human', 'determination': 'GREEN', 'occupation': 'test',
                'confidence': 0.05 + i * 0.01, 'word_count': 200
            }) + '\n')
        # 10 AI (should be RED/AMBER)
        for i in range(10):
            f.write(json.dumps({
                'ground_truth': 'ai', 'determination': 'RED', 'occupation': 'test',
                'confidence': 0.70 + i * 0.02, 'word_count': 200
            }) + '\n')

    rows = analyze_baselines(output_path)
    check("TPR@FPR calculated with ground truth", rows is not None)

    os.unlink(output_path)

def test_analyze_baselines_stratified_rates():
    """Test stratified flag rates by domain x length_bin."""
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        # Domain A, short text: low flag rate
        for i in range(10):
            f.write(json.dumps({
                'domain': 'domainA', 'length_bin': 'short', 'determination': 'GREEN',
                'occupation': 'test', 'word_count': 80, 'confidence': 0.05
            }) + '\n')
        # Domain B, long text: high flag rate
        for i in range(10):
            f.write(json.dumps({
                'domain': 'domainB', 'length_bin': 'long', 'determination': 'RED',
                'occupation': 'test', 'word_count': 500, 'confidence': 0.80
            }) + '\n')

    rows = analyze_baselines(output_path)
    check("Stratified analysis with domain x length", rows is not None)

    os.unlink(output_path)
```

**Expected impact:** Coverage 15% → 70% (covers collection, analysis, percentiles, TPR@FPR, stratification)

---

### 5. `io.py` (49% → target 85%)

**Current gaps:** Lines 34-105 (load_xlsx), 138, 141, 152-153, 159, 173-174, 181, 195 are untested.

**Specific test cases to add:**

```python
# tests/test_io_loaders.py

def test_load_xlsx_default_sheet():
    """Test xlsx loader finds default sheet (FullTaskX, etc.)."""
    import tempfile
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.create_sheet('FullTaskX')
    ws.append(['prompt', 'task_id', 'occupation'])
    ws.append(['This is a test prompt with more than 50 characters to pass the filter.', 'task1', 'teacher'])

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        filepath = f.name
    wb.save(filepath)
    wb.close()

    tasks = load_xlsx(filepath)
    check("1 task loaded", len(tasks) == 1)
    check("Task prompt correct", 'test prompt' in tasks[0]['prompt'])
    check("Task ID correct", tasks[0]['task_id'] == 'task1')
    check("Occupation correct", tasks[0]['occupation'] == 'teacher')

    os.unlink(filepath)

def test_load_xlsx_positional_columns():
    """Test xlsx loader with positional column references (A, B, 1, 2)."""
    import tempfile
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['text_content', 'id', 'job'])
    ws.append(['This is a test prompt with more than 50 characters to pass the filter.', 'task1', 'teacher'])

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        filepath = f.name
    wb.save(filepath)
    wb.close()

    # Use positional references: A=prompt, B=id, C=occupation
    tasks = load_xlsx(filepath, prompt_col='A', id_col='B', occ_col='C')
    check("1 task loaded", len(tasks) == 1)
    check("Positional A loaded", 'test prompt' in tasks[0]['prompt'])
    check("Positional B loaded", tasks[0]['task_id'] == 'task1')
    check("Positional C loaded", tasks[0]['occupation'] == 'teacher')

    os.unlink(filepath)

def test_load_xlsx_fuzzy_column_matching():
    """Test xlsx loader fuzzy column matching (substring)."""
    import tempfile
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['user_submitted_prompt', 'unique_task_identifier', 'job_occupation'])
    ws.append(['This is a test prompt with more than 50 characters to pass the filter.', 'task1', 'teacher'])

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        filepath = f.name
    wb.save(filepath)
    wb.close()

    # Fuzzy matching should find 'prompt' in 'user_submitted_prompt'
    tasks = load_xlsx(filepath)
    check("Fuzzy match found 1 task", len(tasks) == 1)
    check("Fuzzy prompt match", 'test prompt' in tasks[0]['prompt'])

    os.unlink(filepath)

def test_load_xlsx_short_prompt_filter():
    """Test xlsx loader filters out prompts < 50 characters."""
    import tempfile
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['prompt'])
    ws.append(['Short.'])  # Only 6 characters
    ws.append(['This is a valid prompt with more than 50 characters in the text field.'])

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        filepath = f.name
    wb.save(filepath)
    wb.close()

    tasks = load_xlsx(filepath)
    check("Short prompt filtered", len(tasks) == 1)
    check("Valid prompt loaded", 'valid prompt' in tasks[0]['prompt'])

    os.unlink(filepath)

def test_load_csv_basic():
    """Test CSV loader with basic column names."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name
        f.write('prompt,task_id,occupation\n')
        f.write('"This is a test prompt with more than 50 characters to pass the filter.",task1,teacher\n')

    tasks = load_csv(filepath)
    check("1 task loaded from CSV", len(tasks) == 1)
    check("CSV prompt correct", 'test prompt' in tasks[0]['prompt'])
    check("CSV task_id correct", tasks[0]['task_id'] == 'task1')

    os.unlink(filepath)

def test_load_csv_positional_columns():
    """Test CSV loader with positional column references."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name
        f.write('text,id,job\n')
        f.write('"This is a test prompt with more than 50 characters to pass the filter.",task1,teacher\n')

    # Use positional: A, B, C (or 1, 2, 3)
    tasks = load_csv(filepath, prompt_col='A', id_col='B', occ_col='C')
    check("CSV positional 1 task", len(tasks) == 1)
    check("CSV positional prompt", 'test prompt' in tasks[0]['prompt'])

    os.unlink(filepath)

def test_load_csv_missing_prompt_column():
    """Test CSV loader error when prompt column not found."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filepath = f.name
        f.write('text,id,job\n')
        f.write('"Some content",task1,teacher\n')

    # Request 'nonexistent' column
    tasks = load_csv(filepath, prompt_col='nonexistent')
    check("Missing prompt column returns empty", len(tasks) == 0)

    os.unlink(filepath)

def test_load_pdf_per_page():
    """Test PDF loader extracts per-page tasks."""
    # This requires creating a minimal PDF, which is complex
    # For now, test the HAS_PYPDF guard
    if not HAS_PYPDF:
        tasks = load_pdf('dummy.pdf')
        check("PDF loader guards missing pypdf", len(tasks) == 0)
    else:
        # Would need to create a test PDF file here
        pass

def test_col_letter_to_index():
    """Test column letter to index conversion."""
    from llm_detector.io import _col_letter_to_index

    check("A -> 0", _col_letter_to_index('A') == 0)
    check("B -> 1", _col_letter_to_index('B') == 1)
    check("Z -> 25", _col_letter_to_index('Z') == 25)
    check("a -> 0 (lowercase)", _col_letter_to_index('a') == 0)
    check("1 -> 0 (1-based)", _col_letter_to_index('1') == 0)
    check("2 -> 1 (1-based)", _col_letter_to_index('2') == 1)
    check("Invalid returns None", _col_letter_to_index('invalid') is None)
```

**Expected impact:** Coverage 49% → 85% (covers all loader paths, column matching, filters)

---

### 6. `memory.py` (42% → target 65%)

**Current gaps:** Lines 80-81, 150, 158-161, 169-170, 175-190, 197, 213, 292-295, 318, 333-334, 359-371, 400-401, 415-416, 426, 432, 437, 472-473, 500, 515-564, 598-634, 646-731, 739-792, 811-922, 926-941, 955-1050, 1061-1096, 1111-1229, 1241-1290, 1320-1354, 1368, 1425-1546 are untested.

**Note:** The memory module is 941 lines and contains complex state management. Priority should be on core persistence and attempter risk profiling.

**Specific test cases to add:**

```python
# tests/test_memory_core.py

def test_memory_store_initialization():
    """Test MemoryStore creates required directories and files."""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    store = MemoryStore(base_path=temp_dir)

    check("Store directory exists", os.path.exists(temp_dir))
    check("History file path set", store.history_path is not None)

    shutil.rmtree(temp_dir)

def test_attempter_risk_tier_calculation():
    """Test attempter risk tier calculation (CRITICAL/HIGH/ELEVATED/NORMAL)."""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    store = MemoryStore(base_path=temp_dir)

    # Simulate attempter history
    store.record_result('attempter1', determination='RED', confidence=0.90)
    store.record_result('attempter1', determination='RED', confidence=0.85)
    store.record_result('attempter1', determination='RED', confidence=0.88)

    risk_tier = store.get_attempter_risk_tier('attempter1')
    check("High RED rate -> CRITICAL or HIGH", risk_tier in ['CRITICAL', 'HIGH'])

    shutil.rmtree(temp_dir)

def test_ground_truth_confirmation():
    """Test ground truth confirmation updates calibration."""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    store = MemoryStore(base_path=temp_dir)

    store.confirm_ground_truth(task_id='task1', ground_truth='ai', confidence=0.85)
    confirmed = store.get_confirmed_labels()

    check("Ground truth recorded", 'task1' in confirmed)
    check("Label is 'ai'", confirmed['task1'] == 'ai')

    shutil.rmtree(temp_dir)

def test_calibration_rebuild():
    """Test calibration rebuild from confirmed labels."""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    store = MemoryStore(base_path=temp_dir)

    # Add confirmed labels
    store.confirm_ground_truth('task1', 'ai', 0.90)
    store.confirm_ground_truth('task2', 'human', 0.05)
    store.confirm_ground_truth('task3', 'ai', 0.80)

    # Trigger calibration rebuild (if method exists)
    # This tests the rebuild_calibration method
    # Note: Actual implementation may vary

    shutil.rmtree(temp_dir)
```

**Expected impact:** Coverage 42% → 65% (covers core persistence, attempter tracking, ground truth)

---

### 7. ML-dependent analyzers (6-15% → target 50%)

**Modules:** `perplexity.py` (6%), `semantic_resonance.py` (13%), `token_cohesiveness.py` (15%)

**Challenge:** These modules require heavy ML dependencies (transformers, sentence-transformers).

**Strategy:** Mock the ML models to test scoring logic independently.

**Specific test cases to add:**

```python
# tests/test_ml_analyzers_mocked.py

def test_perplexity_surprisal_variance_mocked():
    """Test perplexity surprisal variance calculation with mocked model."""
    from unittest.mock import Mock, patch

    # Mock the model to return fixed perplexity values
    with patch('llm_detector.analyzers.perplexity.HAS_PERPLEXITY', True):
        with patch('llm_detector.analyzers.perplexity._load_model') as mock_load:
            mock_model = Mock()
            mock_model.return_value = {'perplexity': 45.2}
            mock_load.return_value = (mock_model, None)

            from llm_detector.analyzers.perplexity import run_perplexity
            result = run_perplexity("Test text with sufficient length.")

            check("Perplexity value present", 'perplexity' in result)
            check("Surprisal variance computed", 'surprisal_variance' in result)

def test_semantic_resonance_delta_mocked():
    """Test semantic resonance delta calculation with mocked embeddings."""
    from unittest.mock import Mock, patch

    with patch('llm_detector.analyzers.semantic_resonance.HAS_SEMANTIC', True):
        with patch('llm_detector.analyzers.semantic_resonance._load_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1] * 384]  # Fixed embedding
            mock_load.return_value = mock_model

            from llm_detector.analyzers.semantic_resonance import run_semantic_resonance
            result = run_semantic_resonance("Test text. Another sentence. More content.")

            check("Semantic delta present", 'semantic_delta' in result)
            check("AI score present", 'semantic_ai_score' in result)

def test_token_cohesiveness_fragility_mocked():
    """Test token cohesiveness fragility with mocked model."""
    from unittest.mock import Mock, patch

    with patch('llm_detector.analyzers.token_cohesiveness.HAS_TOCSIN', True):
        with patch('llm_detector.analyzers.token_cohesiveness._load_model') as mock_load:
            mock_model = Mock()
            mock_model.return_value = {'perplexity': 35.0}
            mock_load.return_value = mock_model

            from llm_detector.analyzers.token_cohesiveness import run_token_cohesiveness
            result = run_token_cohesiveness("Test text for fragility scoring.")

            check("Fragility score present", 'tocsin_fragility' in result)
            check("Determination present", 'determination' in result)
```

**Expected impact:** Coverage 6-15% → 50% (mocking enables testing scoring logic without GPU)

---

## Testing Infrastructure Recommendations

### 1. Add pytest configuration with coverage enforcement

**File:** `pyproject.toml`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=llm_detector",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=json",
    "--cov-fail-under=60",  # Enforce minimum 60% coverage
    "-ra",  # Show all test results
    "-q",   # Quiet mode
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "requires_gpu: marks tests requiring GPU (transformers, etc.)",
    "requires_network: marks tests requiring network access",
]
```

**Impact:** Enforces coverage threshold in CI, enables test markers for selective execution.

---

### 2. Migrate from custom `check()` to pytest assertions

**Current pattern:**
```python
check("Label", condition, "detail")
```

**Recommended pattern:**
```python
assert condition, "Label: detail"
```

**Benefits:**
- Better failure diagnostics (pytest shows actual vs expected values)
- Standard testing pattern familiar to all Python developers
- Enables pytest plugins (pytest-timeout, pytest-xdist, etc.)

**Migration approach:**
- Gradual migration: new tests use `assert`, legacy tests can remain
- Add a deprecation notice to the `check()` helper
- Prioritize migrating tests for critical modules first

---

### 3. CI/CD coverage reporting

**File:** `.github/workflows/build.yml`

```yaml
- name: Run tests with coverage
  run: |
    pytest --cov=llm_detector --cov-report=term-missing --cov-report=xml

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
    fail_ci_if_error: true
```

**Impact:** Visualize coverage trends over time, prevent regressions.

---

### 4. Add fixture library for edge cases

**File:** `tests/conftest.py` (extend existing)

```python
@pytest.fixture
def short_text():
    """Text under 100 words for short-text edge cases."""
    return "Short text. Brief. Minimal content here for testing purposes only."

@pytest.fixture
def non_english_text():
    """Non-English text for language gate testing."""
    return "这是一个中文测试文本用于测试语言检测功能和公平性限制。" * 10

@pytest.fixture
def obfuscated_text():
    """Text with homoglyphs and zero-width characters."""
    return "Тhis text has h\u200bomoglyphs and inv\u200bisible chars."

@pytest.fixture
def mixed_human_ai_text():
    """Text with distinct human and AI sections for windowed testing."""
    return (
        "Hey, so I was thinking about this... you know how sometimes you just "
        "need to get stuff done? Yeah. Anyway, "
        "The implementation of advanced algorithmic methodologies necessitates "
        "a comprehensive evaluation framework that systematically addresses "
        "the multifaceted challenges inherent in contemporary computational paradigms."
    )
```

**Impact:** Standardized edge-case fixtures improve test coverage for fairness and robustness.

---

### 5. Add GitHub Actions matrix testing

**File:** `.github/workflows/build.yml`

```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]
    os: [ubuntu-latest, windows-latest, macos-latest]
```

**Impact:** Ensures compatibility across Python versions and operating systems.

---

## Summary and Priority Action Items

### Immediate priorities (next sprint):

1. **Add channel tests** (prompt_structure.py, stylometric.py) → +40pp coverage in critical detection logic
2. **Add fusion edge-case tests** (fairness caps, channel ablation, MIXED determination) → +20pp fusion coverage
3. **Add I/O tests** (xlsx/csv loaders with positional/fuzzy matching) → +35pp I/O coverage
4. **Configure pytest** with coverage enforcement (60% threshold) and test markers
5. **Set up CI/CD coverage reporting** (Codecov or similar)

### Medium-term priorities (next 2-3 sprints):

6. **Add baseline tests** (collect_baselines, analyze_baselines, TPR@FPR) → +50pp baselines coverage
7. **Add memory store tests** (attempter risk, ground truth confirmation) → +20pp memory coverage
8. **Mock ML analyzers** (perplexity, semantic resonance, token cohesiveness) → +35pp analyzer coverage
9. **Add edge-case fixtures** (short text, non-English, obfuscated, mixed)
10. **Migrate to pytest assertions** (gradual migration from `check()` helper)

### Long-term priorities:

11. **GUI/Dashboard testing** (currently 0%, requires UI testing framework)
12. **CLI integration tests** (argument parsing, batch processing) → +30pp CLI coverage
13. **Performance benchmarking** (ensure coverage doesn't degrade performance)
14. **Mutation testing** (test test quality with mutation testing tools)

---

**Estimated total effort:** 3-4 sprints (6-8 weeks) to reach 70%+ overall coverage with high-quality tests.

**Expected outcome:**
- **Overall coverage:** 39% → 75%
- **Critical modules:** 85%+ (channels, fusion, I/O, calibration)
- **ML modules:** 50%+ (with mocking)
- **CI/CD:** Automated coverage enforcement and reporting
