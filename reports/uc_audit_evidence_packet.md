# UC Audit Evidence Packet — Pipeline Authorship Review

**Prompt Under Review:** "Senior NP IBD Specialist" UC Audit-Ready Evidence Packet Prompt
**Pipeline Version:** v0.68 (LLM Authorship Signal Analyzer)
**Analysis Mode:** `task_prompt`
**Date of Analysis:** 2026-03-19
**Word Count:** 732

---

## 1. Executive Summary

| Field | Value |
|-------|-------|
| **Determination** | **AMBER** |
| **Confidence** | 0.700 |
| **Triggering Rule** | `primary_amber_single_channel` |
| **Primary Reason** | Prompt-structure: prompt_sig=0.30 (YELLOW), IDI=8 (AMBER), SSI=9 (AMBER) + Stylometry: PPL=22 (YELLOW), Bino=0.32 (AMBER) + Continuation: BScore=0.000 (Local, YELLOW) |

**Interpretation:** The AMBER determination indicates *substantial evidence* of LLM-assisted prompt generation. This prompt should be flagged for manual review. Multiple independent channels detected convergent signals of machine-generated specification language, hyper-precise constraint framing, and low perplexity consistent with LLM output.

---

## 2. Channel-by-Channel Breakdown

### Channel 1: Prompt Structure (PRIMARY) — AMBER (0.70)

This is the dominant signal channel and triggered the AMBER determination.

| Metric | Value | Severity | Notes |
|--------|-------|----------|-------|
| Prompt Signature Composite | 0.300 | YELLOW | Elevated constraint-frame density |
| Constraint Frame Density (CFD) | 0.225 | — | 2 distinct frames detected |
| Must-Frame Saturation Rate (MFSR) | 0.000 | — | Must-rate = 0.375 (high but below MFSR threshold) |
| Instruction Density Index (IDI) | 8.0 | AMBER | 30 imperatives, 8 conditionals, 1 binary spec |
| SSI (Specification Saturation) | 9 | AMBER | Triggered by convergence of IDI + lexicon packs |
| Conditional Density | 0.200 | — | High if/unless/when branching logic |
| Contractions | 1 | — | Near-zero informal register |

**Key Finding:** 30 imperative constructions in 732 words is extremely high density (4.1 imperatives per 100 words). Combined with 8 conditional clauses and a must-rate of 37.5%, this prompt reads as machine-generated specification rather than organic clinical instruction.

### Channel 2: Stylometry (SUPPORTING) — AMBER (0.278)

| Metric | Value | Severity | Notes |
|--------|-------|----------|-------|
| Perplexity (distilgpt2) | 21.65 | YELLOW | Low surprise = predictable token sequences |
| Binoculars Score | 0.3196 | AMBER | Cross-model divergence suggests generation |
| Zlib-Normalized PPL | 9.24 | — | Below typical human-authored thresholds |
| Comp-PPL Ratio | 1.97 | — | Near-equilibrium between compression and PPL |
| Surprisal Variance | 9.63 | — | Moderate burstiness (1.77) |

**Key Finding:** The Binoculars score of 0.32 falls in the AMBER range, indicating the text's statistical properties align more closely with LLM output than human authorship. The low perplexity of 21.65 reinforces this — human-authored clinical prompts typically exhibit higher surprise.

### Channel 3: Continuation (PRIMARY) — YELLOW (0.289)

| Metric | Value | Notes |
|--------|-------|-------|
| BScore | 0.000 | Local proxy (no API key) |
| NCD | 0.825 | Moderate normalized compression distance |
| Internal Overlap | 0.023 | Low self-repetition in continuations |
| Composite | 0.497 | Borderline; stability = 0.96 |
| Composite Variance | 0.003 | Very stable across samples |

**Key Finding:** The continuation channel used the local n-gram proxy (no API key available). Results are borderline YELLOW — the text can be regenerated with moderate fidelity, suggesting template-like structure, but the signal is not strong enough alone to escalate.

### Channel 4: Windowing (SUPPORTING) — GREEN (0.00)

| Metric | Value | Notes |
|--------|-------|-------|
| Max Window Score | 0.15 | No hot windows |
| Mean Window Score | 0.025 | Uniformly low |
| N Windows | 18 | Full coverage |
| FW Trajectory CV | 0.273 | Moderate variation across windows |
| Compression CV | 0.050 | Very uniform compression profile |
| Changepoint | None | No style shifts detected |

**Key Finding:** No windowing signals detected. The prompt maintains consistent style throughout — no spliced sections or style changes. This is consistent with either single-author human writing OR single-pass LLM generation.

---

## 3. Lexicon Pack Analysis

| Pack Family | Score | Signals |
|-------------|-------|---------|
| **Cardinality** | High | "exactly 6", "exactly 3", "exactly 10", "at least 2", "at least 8", "each of the", "every row" |
| **Obligation** | High | 14 instances of "must" across the prompt |
| **Prohibition** | Present | "Do not include" |
| **Conditional** | Present | "only when", "If a statement is", "unless" |
| **Task Verbs** | Present | "Label each", "map to", "format" |
| **Value Domain** | Present | "Missing" (2 instances), "No" |
| **Active Families** | 4 of 16 | Constraint, task_verbs, cardinality, obligation |
| **Prompt Boost** | 0.20 | Applied due to constraint-heavy vocabulary |
| **IDI Boost** | 1.48 | Significant amplification from pack convergence |

**Key Finding:** The prompt is saturated with cardinality constraints ("exactly N" appears 8 times) and obligation markers ("must" appears 14 times). This pattern — hyper-precise numeric constraints combined with obligation framing — is a strong LLM authorship signal in the `task_prompt` mode, as human clinicians rarely specify deliverable structures with this level of mechanical precision.

---

## 4. Detection Spans (46 Total)

The pipeline identified 46 character-level spans across the prompt text. High-weight spans (most diagnostic):

| Span Text | Pack | Weight | Location |
|-----------|------|--------|----------|
| "exactly 6" | cardinality | 1.5 | Multiple locations |
| "exactly 3" | cardinality | 1.5 | Psychosocial section |
| "exactly 10" | cardinality | 1.5 | Therapy_Gate spec |
| "must contain" | obligation | 1.5 | Workbook tab spec |
| "Do not include" | prohibition | 1.2 | Protocol section |
| "at least 2" | cardinality | 1.2 | Therapy_Gate row |
| "at least 8" | cardinality | 1.2 | Final_Status spec |
| "If a statement is" | conditional | 1.2 | Citation format |
| "each of the" | cardinality | 1.0 | Audit trail section |
| "every row" | cardinality | 1.0 | Word count constraint |
| "Label each" | task_verbs | 1.0 | Fact labeling |
| "only when" | cardinality | 0.8 | Tier 2 usage rule |
| "map to" | task_verbs | 0.8 | Action_Map spec |
| "Missing" | value_domain | 2.0 | Final_Status values (2x) |

---

## 5. Voice Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Voice Score | 0.342 | Low informal register |
| Specification Score | 8.95 | Very high specification density |
| Voice-Specification Dissonance (VSD) | 3.10 | Moderate dissonance |
| Casual Markers | 0 | Zero casual language |
| Misspellings | 0 | Zero errors |
| Hedges | 0 | Zero hedging language |
| CamelCase Columns | 29 | Schema-specification vocabulary |

**Key Finding:** The VSD of 3.1 indicates the prompt has virtually no informal voice markers but extremely high specification density. Zero casual markers, zero misspellings, and zero hedges in a 732-word prompt from a clinician composing for an internal huddle is unusual. The 29 CamelCase column references (Fact_ID, Tier_1_Status, etc.) suggest the prompt was generated with database-schema-level precision atypical of clinical NP writing.

---

## 6. Stylometric Profile

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Function Word Ratio | 0.267 | Slightly below average |
| TTR (Type-Token Ratio) | 0.425 | Moderate vocabulary diversity |
| MATTR | 0.819 | High moving-average TTR |
| Avg Word Length | 5.44 | Above average (specification terms) |
| Short Word Ratio | 0.335 | Below average |
| Sentence Length CV | 0.616 | Moderate variation |
| Hapax Ratio | 0.580 | High (many unique terms) |
| Compression Ratio | 0.427 | Moderate |
| Structural Compression Delta | -0.002 | No structural ordering effect |

---

## 7. Semantic Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| AI Centroid Similarity | 0.227 | Moderate alignment with AI archetype |
| Human Centroid Similarity | 0.075 | Low alignment with human archetype |
| AI-Human Delta | 0.171 | Text leans toward AI centroid |
| Semantic Flow Variance | 0.040 | Moderate topic coherence |
| Semantic Flow Mean | 0.324 | Moderate inter-sentence similarity |
| TOCSIN Cohesiveness | 0.010 | Very low semantic fragility |

**Key Finding:** The semantic resonance delta of 0.171 shows the prompt's embedding profile aligns 3x more closely with the AI archetype centroid than the human centroid. While this alone is not diagnostic, it corroborates the structural findings.

---

## 8. Normalization & Fairness Gates

| Check | Result |
|-------|--------|
| Obfuscation Delta | 0.0% (clean) |
| Invisible Characters | 0 |
| Homoglyphs | 0 |
| Language Support | SUPPORTED |
| FW Coverage | 0.271 |
| Non-Latin Ratio | 0.0% |
| ftfy Applied | No |

No obfuscation or adversarial text manipulation detected. The fairness gate confirms the text is in a fully supported language with adequate function-word coverage for reliable analysis.

---

## 9. Reviewer Guidance

### What AMBER Means for This Prompt

The AMBER determination means the pipeline has found **substantial convergent evidence across multiple independent channels** that this prompt was likely authored with significant LLM assistance. This does NOT mean the clinical content is wrong or inappropriate — it means the *authorship signal* is inconsistent with unassisted human writing.

### Specific Concerns for a Human Data Pipeline

1. **Hyper-specification:** 30 imperatives and 14 "must" constraints in 732 words suggest the prompt was iteratively refined with an LLM to achieve mechanical precision
2. **Schema-level detail:** CamelCase column names, exact row counts, and enumerated allowed values mirror database specification language, not clinical communication
3. **Zero informality:** A complete absence of hedges, casual markers, contractions, and misspellings in a prompt ostensibly from a busy NP for an internal huddle is statistically unusual
4. **Cardinality saturation:** The pattern of "exactly N" (8 occurrences) is a hallmark of LLM-generated task prompts in the pipeline's training data

### Recommended Actions

- **If this prompt is part of a human-authored benchmark:** Flag for manual review; request the author confirm unassisted authorship
- **If LLM assistance is permitted:** No action needed; the clinical content can be evaluated on its merits
- **If provenance matters:** Request the author's drafting history or version control artifacts

---

## 10. Raw Pipeline Output Reference

Full JSON result saved to: `reports/uc_audit_full_result.json`

Pipeline configuration:
- Mode: `task_prompt`
- Continuation: Local proxy (no API key)
- Calibration: Uncalibrated (no baseline corpus loaded)
- Shadow Model: Not available
- Semantic/Perplexity: Available and active
