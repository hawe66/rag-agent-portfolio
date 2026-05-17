# Week 4 — 데이터 전처리 진단 + Chunking 비교 실험

## Target Limitations
- **LIM-001**: Text order mismatch (멀티컬럼)
- **LIM-005**: No category filtering
- **LIM-006**: Common term contamination
- **LIM-008**: Retrieval gap for specific facts

## Task Checklist

### §1.3 RAGAS Environment Check (First)
- [x] Check RAGAS installation: v0.4.3
- [x] Test RAGAS import — works with deprecation warnings
- [x] OpenAI evaluator LLM — works

**API Notes (v0.4.3):**
```python
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
from ragas.llms import llm_factory
```

### §2 Baseline Retrospective
- [x] Capture actual chunk text for LIM-005 instance (category cross-contamination)
  - "정수기 필터 교체" query: 2/5 wrong category (airpurifier)
  - Common terms "필터", "교체" cause contamination
- [x] Capture actual chunk text for LIM-008 instance (배터리 충전 시간 failure)
  - Query retrieves vacuum chunks but no explicit "충전 시간" (duration) found
  - Info may not exist in docs or buried in interleaved text
- [x] Document improvement hypotheses
  - Hypothesis A: metadata filtering (LIM-005, LIM-006)
  - Hypothesis B: layout-aware parsing (LIM-001, LIM-008)

### §3 Data Quality Diagnosis
- [x] Document structure analysis (tables, headers, step patterns)
  - Tables: 16-43 per PDF
  - Images: 3-98 per PDF (vacuum_complex highest)
  - Step patterns: numeric ("1 ", "2 "), not circled
- [x] Quantify LIM-001: multi-column page percentage
  - **79.4% multi-column (200/252 pages)**
  - Strongly supports C3 (layout-aware parsing)
- [x] Noise identification (headers, footers, repeated text)
  - Repeated safety warnings, icon instructions
- [x] Visualization: token distribution by manual
  - Total: 163,213 tokens
  - Range: 13,406 (vacuum_simple) to 37,519 (vacuum_complex)

### §4 Chunking Experiments
- [ ] **A (baseline)**: Recursive 1000/200 — RAGAS measurement
- [ ] **B1 (small)**: Recursive 500/100
- [ ] **B2 (large)**: Recursive 1500/300
- [ ] **C3 (layout-aware)**: unstructured/docling/marker — 2hr time box
- [ ] **C2 (custom sep)**: fallback if C3 fails
- [ ] Comparison table with RAGAS scores
- [ ] Qualitative analysis on LIM-005/008 queries

### §5 Metadata Expansion
- [ ] Add: section, has_table, has_step_number, token_count
- [ ] Verify in vector store

### §6 Retrospective Document
- [ ] `docs/week4_retrospective.md`

### §10 Presentation Materials
- [ ] `docs/week4_presentation_materials.md`

## Outputs
- `w4/data_analysis.ipynb`
- `w4/chunking_experiments.ipynb`
- `docs/week4_retrospective.md`
- `docs/week4_presentation_materials.md`
