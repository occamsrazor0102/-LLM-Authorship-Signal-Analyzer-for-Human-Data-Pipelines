# Test Coverage Analysis

**Date:** 2026-03-10
**Overall coverage:** 38% (4,204 statements missed out of 6,822)
**Tests passing:** 152/152

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
