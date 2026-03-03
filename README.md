# LLM Authorship Signal Analyzer for Human Data Pipelines

A stylometric detection pipeline that identifies LLM-generated or LLM-assisted task prompts submitted through human data collection workflows. Designed for quality assurance in benchmark construction (GDPval-style evaluation tasks), clinical education assessments, and any pipeline where humans are expected to author original prompts but may submit LLM-generated content instead.

## The Problem

Human data pipelines — where workers author task prompts, evaluation scenarios, or assessment items — are vulnerable to a specific failure mode: a contributor uses an LLM to generate their submission rather than writing it themselves. This degrades data quality because LLM-generated prompts exhibit systematic biases in structure, vocabulary, and specification patterns that contaminate the resulting benchmark.

Standard AI-text detectors (GPTZero, Originality.ai, etc.) are trained on prose and perform poorly on task prompts, which are inherently instructional and specification-heavy. This tool is purpose-built for that domain.

## How It Works

The pipeline analyzes text across multiple independent layers, each targeting a different authorship signal. No single layer is definitive — the system combines evidence across layers using priority-based aggregation and multi-layer convergence logic.

### Detection Layers

**Layer 0 — Preamble Detection**
Catches LLM output artifacts that weren't cleaned before submission: assistant acknowledgments ("Sure, here's..."), artifact delivery frames ("Below is a rewritten..."), first-person creation claims ("I've drafted..."), and meta-design language ("failure-inducing", "designed to test").

**Layer 2 — Intrinsic Fingerprints** *(diagnostic only)*
A 120-word tiered lexicon of LLM-preferred vocabulary drawn from Kobak et al. (2025, 379 excess style words from 15M PubMed abstracts), Gray (2024), Liang et al. (2024), and GPTZero. Words like "delve," "utilize," "comprehensive," "facilitate." Used for convergence and cross-submission similarity analysis, not as a standalone trigger — too many legitimate formal texts use these words individually.

**Layer 2.5 — Prompt-Engineering Signatures**
Structural patterns characteristic of LLM-generated task prompts: Constraint Frame Density (CFD), Must-Frame Saturation Rate (MFSR), meta-evaluation design language, numbered criteria lists, and framing completeness scoring. Detects the systematic over-specification that LLMs produce when asked to write evaluation prompts.

**Layer 2.6 — Voice-Specification Dissonance (VSD)**
Measures the contradiction between casual voice markers (contractions, hedges, colloquialisms, misspellings) and technical specification density (CamelCase columns, file references, formulas, tab references). High specification with zero casual voice is the profile of sterile LLM output.

**Layer 2.6 — Prompt Specification Index (PSI)** *(v0.53.2)*
Measures the density of prompt-engineering specification language: role assignment ("you are a pharmacist"), agent directives ("you must," "your response"), output format specs ("format as," "structure your"), completeness demands ("ensure that," "be comprehensive"), and response meta-framing ("your response should demonstrate"). Individual patterns are normal in human-authored prompts — a human writing a GDPval prompt *should* include role-setting. The signal is the formulaic *stacking* of multiple categories in a short span that characterizes LLM-generated prompts.

**Layer 2.6 — Sterile Specification Index (SSI)**
Catches high combined specification (tech + prompt) with suppressed voice markers, low contractions, and zero hedging — the profile of clean formal LLM output that lacks the natural variation of human writing. Two tiers: SSI-STRICT (high confidence, AMBER) and SSI-RELAXED (moderate confidence, YELLOW).

**Layer 2.7 — Instruction Density Index (IDI)**
Counts formal-exhaustive specification patterns: imperative verbs, conditional constraints, binary specification markers, missing-reference flags, and flag-count modifiers. High IDI indicates the systematic exhaustive coverage that LLMs produce when generating task specifications.

### Aggregation

Signals from all layers are collected and resolved using priority-based ordering. The highest-severity, highest-priority signal becomes the primary determination. Lower-priority signals are preserved as supporting evidence in the audit trail.

A **multi-layer convergence floor** triggers YELLOW when 3+ layers show sub-threshold but non-zero signal simultaneously — catching the "nothing screams but everything whispers" profile where moderately sophisticated LLM prompts evade every individual threshold.

### Determination Levels

| Level | Meaning | Action |
|-------|---------|--------|
| 🔴 RED | Strong evidence of LLM generation | Flag for review, likely reject |
| 🟠 AMBER | Substantial evidence, high confidence | Flag for manual review |
| 🟡 YELLOW | Minor signals or convergence pattern | Note for awareness, may be legitimate |
| 🟢 GREEN | No significant signals detected | Pass |

## Installation

```bash
pip install openpyxl pandas
# Optional (improves sentence segmentation):
pip install spacy
```

No model downloads required. The spaCy sentencizer is a rule-based component that ships with the package.

## Usage

### Single Text Analysis

```bash
python llm_detector_v0532.py --text "Your prompt text here"
```

### Desktop GUI (v0.55)

```bash
# Standalone executable launcher
./llm_detector_v055_gui.py

# Alternative: launch directly from the pipeline script
python "llm_detector_v055-OAI API.py" --gui
```

GUI mode supports:
- Single text analysis with full pipeline scoring.
- Batch CSV/XLSX analysis with prompt column and sheet selection.
- Optional attempter filtering.
- Optional DNA-GPT provider/API key settings.

### File Mode (XLSX/CSV)

```bash
python llm_detector_v0532.py input.xlsx --sheet "Sheet1" --prompt-col "prompt"
python llm_detector_v0532.py input.csv --prompt-col "content"
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--text` | Analyze a single text string |
| `--sheet` | Sheet name for XLSX files |
| `--prompt-col` | Column name containing prompts (default: "prompt") |
| `--verbose`, `-v` | Show all layer details for every result |
| `--output`, `-o` | Output CSV path |
| `--attempter` | Filter by attempter name (substring match) |
| `--no-similarity` | Skip cross-submission similarity analysis |
| `--similarity-threshold` | Jaccard threshold for similarity flagging (default: 0.40) |
| `--collect PATH` | Append scored results to JSONL for baseline accumulation |
| `--analyze-baselines JSONL` | Compute percentile distributions from accumulated data |
| `--baselines-csv PATH` | Output path for baseline analysis CSV |

### Python API

```python
import llm_detector_v0532 as detector

result = detector.analyze_prompt(
    text="You are a board-certified pharmacist. Analyze the following...",
    task_id="task_001",
    occupation="pharmacist",
    attempter="worker_42",
)

print(result['determination'])   # RED / AMBER / YELLOW / GREEN
print(result['reason'])          # Primary signal description
print(result['confidence'])      # 0.0 - 1.0
print(result['supporting_signals'])  # List of other signals that fired

# Layer-level diagnostics
print(result['l26_prompt_spec'])  # PSI score
print(result['l26_total_spec'])   # Combined tech + prompt spec
print(result['l27_idi'])          # Instruction Density Index
print(result['l25_composite'])    # Prompt-engineering composite

# Cross-submission similarity (batch mode)
results = [detector.analyze_prompt(t['prompt'], ...) for t in tasks]
text_map = {r['task_id']: t['prompt'] for r, t in zip(results, tasks)}
flags = detector.analyze_similarity(results, text_map)
```

## Cross-Submission Similarity Analysis

When processing multiple submissions (batch mode), the pipeline runs pairwise similarity analysis across different attempters to detect copied or templated submissions. Uses both Jaccard text similarity and structural feature similarity (layer scores as vectors). Flags pairs where the same text or structural profile appears across different contributors.

## Baseline Collection

The pipeline supports longitudinal threshold calibration through baseline accumulation:

```bash
# Accumulate data across runs
python llm_detector_v0532.py batch1.xlsx --collect baselines.jsonl
python llm_detector_v0532.py batch2.xlsx --collect baselines.jsonl

# Analyze accumulated distributions
python llm_detector_v0532.py --analyze-baselines baselines.jsonl --baselines-csv report.csv
```

This exports per-occupation percentile distributions for all layer scores, enabling data-driven threshold tuning as more labeled data accumulates.

## Testing

The regression test harness validates that changes don't break existing behavior:

```bash
python test_regression_v0532.py
```

12 tests covering smoke tests (known LLM/human texts), aggregation priority ordering, and cross-submission similarity logic.

## Design Principles

**Density over presence.** Individual prompt-engineering patterns (role-setting, format directives) are expected in human-authored prompts. The signal is the *density* and formulaic stacking of multiple categories — not any single pattern.

**No single-layer vetoes.** Every layer can be defeated individually. The convergence floor ensures that when multiple layers whisper, the system still listens.

**Voice gate preserves specificity.** VSD requires actual casual voice absence, not just specification presence. A human writing formal text naturally varies more than an LLM generating sterile specifications.

**Diagnostic layers inform but don't trigger.** Layer 2 fingerprints participate in convergence and similarity analysis but don't fire standalone signals — the false positive rate on individual vocabulary items is too high.

**Audit trail by default.** Every determination includes the primary signal, all supporting signals, and full layer-level diagnostics. Nothing is hidden from the reviewer.

## Version History

| Version | Key Changes |
|---------|-------------|
| v0.53.2 | Prompt Specification Score (PSI), SSI relaxation, multi-layer convergence floor |
| v0.53.1 | Layer 2 fingerprint expansion (27→120 words), tech_parens whitelist, reason bundling |
| v0.53 | Cross-submission similarity analysis, baseline collection framework, regression harness |
| v0.52.2 | Specification Convergence bonus, SSI supplementary check |
| v0.52.1 | IDI priority fix, constraint pattern boundary fix, CSV fillna |
| v0.52 | Expanded constraint frames, IDI layer, SSI |
| v0.51 | Bug fixes (casual markers, preamble anchors, em-dash regex, spaCy sentencizer) |
| v0.5 | Initial release (L0, L2, L2.5, L2.6) |

## License

MIT
