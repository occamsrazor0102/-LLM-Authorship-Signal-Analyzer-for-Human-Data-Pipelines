# BEET v0.61 → v0.62 Lexicon Gap Analysis

## Methodology
Compared every pattern/keyword in the current codebase against the 6-priority
roadmap. Cells marked ✅ = covered, ⚠️ = partial, ❌ = absent.

---

## Priority 1: CONSTRAINT_FRAMES (BCP 14 / EARS / Cardinality)

### BCP 14 / RFC 2119 Normative Keywords (uppercase forms)
| Keyword | Current Coverage | Notes |
|---------|-----------------|-------|
| MUST | ⚠️ `\bmust\b` (case-insensitive) | Not tracking uppercase normative form separately |
| MUST NOT | ❌ | Absent |
| REQUIRED | ❌ | Absent |
| SHALL | ❌ | Absent (in `_ENGLISH_FUNCTION_WORDS` but not constraint-tracked) |
| SHALL NOT | ❌ | Absent |
| SHOULD | ❌ | Only `should be visible`, `should address`, `should be delivered` |
| SHOULD NOT | ❌ | Absent |
| RECOMMENDED | ❌ | Absent |
| MAY | ❌ | Only `may not` variants |
| OPTIONAL | ❌ | Absent |

### EARS Templates (Event/Action/Response/State)
| Pattern | Current Coverage | Notes |
|---------|-----------------|-------|
| When [event], [system] shall... | ❌ | Only `\bwhen\b` counted in IDI conditionals |
| While [state], [system] shall... | ❌ | Absent |
| If [condition] then [result] | ⚠️ | `\bif\b[^.]*?,` in L2.5 conditional_density |
| Where [feature] is included... | ❌ | Absent |

### Obligation / Prohibition / Conditional / State / Cardinality Subfamilies
| Operator | Current Coverage | Notes |
|---------|-----------------|-------|
| exactly | ❌ | Only `with exactly \d+` |
| at least | ⚠️ | `at least \d+[%$]?` present |
| at most | ❌ | Absent |
| no more than | ✅ | Present |
| between | ❌ | Absent as constraint operator |
| one of | ❌ | Absent |
| each | ❌ | Only `every \w+ must\b` |
| every | ⚠️ | `every \w+ must\b` present |
| only | ❌ | Absent |
| if present | ❌ | Absent |
| if absent | ❌ | Absent |
| without exceeding | ✅ | Present |
| or higher / or lower | ✅ | Present |
| within \d+% | ✅ | Present |

**Gap Summary**: Current CONSTRAINT_FRAMES covers ~12 of ~35 needed operators.
Heavy on numeric thresholds, missing entire obligation/prohibition/state subfamilies.

---

## Priority 2: run_layer26 Schema / Structured-Output Lexicon

### JSON Schema Keywords
| Keyword | Current Coverage | Notes |
|---------|-----------------|-------|
| schema | ❌ | Absent |
| object / array / string / integer / number / boolean | ❌ | Absent |
| enum | ❌ | Absent |
| required (JSON Schema) | ❌ | Absent |
| properties / additionalProperties / patternProperties | ❌ | Absent |
| $ref / if/then/else (conditional schema) | ❌ | Absent |
| field / key / value | ❌ | Absent (only `col_listings` regex) |
| header / query / path / parameter (OpenAPI) | ❌ | Absent |
| request / response | ❌ | Absent |

### Tabular / CSV Specs (RFC 4180 / Frictionless)
| Term | Current Coverage | Notes |
|---------|-----------------|-------|
| row / column | ⚠️ | `col_listings` regex in L2.6 |
| field / header | ⚠️ | `col_listings` captures some |
| delimiter / dialect | ❌ | Absent |
| worksheet / sheet | ⚠️ | `tabs` regex: `tab \d|sheet \d` |
| text/csv | ❌ | Absent |

### Current spec_score Components (L2.6)
| Component | Status | Notes |
|-----------|--------|-------|
| camel_cols (CamelCase identifiers) | ✅ | Working |
| filenames (.csv, .xlsx, etc.) | ✅ | Working |
| calcs (formula language) | ✅ | Working |
| tabs (tab/sheet refs) | ✅ | Working |
| col_listings (column: X) | ✅ | Working |
| tech_parens | ✅ | Working |

**Gap Summary**: spec_score is entirely spreadsheet/file-oriented. Zero coverage
of JSON Schema, OpenAPI, API contract, or data serialization vocabulary.
For prompts involving API tasks, data engineering, or code generation, recall is near zero.

---

## Priority 3: META_DESIGN_PATTERNS (Gherkin / Executable-Spec / Rubric)

### Gherkin Keywords
| Keyword | Current Coverage | Notes |
|---------|-----------------|-------|
| Feature | ❌ | Absent |
| Scenario | ❌ | Absent |
| Given | ❌ | Absent |
| When (Gherkin) | ❌ | Not tracked as spec keyword |
| Then | ❌ | Absent |
| And / But (Gherkin) | ❌ | Absent |
| Examples | ❌ | Absent |
| Background | ❌ | Absent |

### Rubric / Evaluation Language
| Term | Current Coverage | Notes |
|---------|-----------------|-------|
| acceptance criteria | ⚠️ | `acceptance (checklist|criteria)` present |
| pass/fail | ❌ | Absent |
| rubric | ❌ | Absent |
| grader | ❌ | Only `(used for|for) grading` |
| checklist | ⚠️ | `acceptance (checklist|criteria)` only |
| verification | ❌ | Absent |
| test case | ❌ | Absent |
| edge case | ❌ | Absent |
| expected output | ❌ | Absent |
| source of truth | ⚠️ | `authoritative source of truth` present |
| grounded / cite source / evidence | ⚠️ | `grounded in\b` present |
| scenario (evaluation) | ⚠️ | `scenario anchor date` only |

### Current META_DESIGN_PATTERNS
| Pattern | Status | Notes |
|---------|--------|-------|
| workflows? tested | ✅ | Working |
| acceptance (checklist\|criteria) | ✅ | Working |
| (used for\|for) grading | ✅ | Working |
| SOC \d{2}-?\d{4} | ✅ | Working |
| expected effort | ✅ | Working |
| deliberate (anomalies\|errors\|issues) | ✅ | Working |
| checkable artifacts | ✅ | Working |
| authoritative source of truth | ✅ | Working |
| scenario anchor date | ✅ | Working |
| avoid vague language | ✅ | Working |
| explicit non-functional | ✅ | Working |
| grounded in | ✅ | Working |

**Gap Summary**: 12 patterns present, all domain-specific to GDPval-style tasks.
Zero Gherkin coverage. Rubric language is thin — only "acceptance criteria" and
"grading" caught. Missing the entire executable-specification vocabulary.

---

## Priority 4: run_layer27 IDI Typed Instruction Operators

### Current IDI Keywords
| Category | Keywords | Count |
|----------|----------|-------|
| Imperatives | must, include, create, load, set, show, use, derive, treat, mark | 10 |
| Conditionals | if, otherwise, when, unless | 4 |
| Binary specs | Yes/No | 2 patterns |
| Missingness | MISSING | 1 |
| Flags | flag | 1 |

### Bloom-Style Task Verbs (roadmap)
Missing: classify, identify, extract, label, compare, evaluate, rewrite,
translate, summarize, justify, rank, design, generate, format, populate,
validate, convert, normalize, parse, map

### Value-Domain Operators (roadmap)
Missing: true/false, 0/1, null, none, unknown, leave blank, default,
fallback, allowed values, valid values, one of, return as

**Gap Summary**: IDI covers 18 total tokens across 5 categories.
Roadmap calls for ~40+ tokens across task-verbs, value-domain, and formatting.
Action verbs alone are noisy; the roadmap specifically recommends pairing
action verbs + constraint/schema operators for signal strength.

---

## Overall Gap Score

| Priority | Current Coverage | Estimated Recall | Target Recall |
|----------|-----------------|------------------|---------------|
| P1: CONSTRAINT_FRAMES | ~35% | Low | High |
| P2: Schema/Output Lexicon | ~5% | Very Low | Medium-High |
| P3: META_DESIGN_PATTERNS | ~25% | Low-Medium | High |
| P4: IDI Typed Operators | ~30% | Low-Medium | Medium-High |
| P5: Format/Markup | ~10% | Very Low | Medium |
| P6: Function/Discourse | ~60% | Medium | Medium |

**Conclusion**: The biggest gaps are in P1 (obligation/prohibition operators)
and P2 (schema/API vocabulary). These are where the highest-yield expansion
work should focus. P3 adds precision via executable-spec patterns.
The roadmap's observation that "flat word bags are where detectors go to die"
points directly to the need for the versioned pack architecture with per-family
weights and caps.
