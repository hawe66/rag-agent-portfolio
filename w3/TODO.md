# Week 3 - Task 1: PDF Loader Comparison

## Goal
Compare PyPDFLoader vs PDFPlumberLoader for Korean text extraction quality.

## Test Setup
- **File:** `data/raw_pdfs/waterpurifier_complex.pdf` (1.7 MB, smallest)
- **Pages:** 9-12 (actual content starts here)
- **Output:** `w3/pdf_loader_comparison.ipynb`

## Evaluation Criteria

### Text Quality
- [x] Korean text renders correctly (no garbled characters)
- [x] Section headers preserved
- [x] Bullet points / numbered lists intact
- [x] Tables structure readable (partial - complex tables lose structure)
- [x] No missing text

### Image Detection (lightweight)
- [x] Count images per page
- [x] Log bounding box positions (for Phase 2 text-image proximity)

### Performance
- [x] Note extraction speed for 40 pages

## Deliverables
- [x] Notebook with side-by-side comparison
- [x] Decision: which loader to use (with reasoning)

## Decision
**Selected:** PDFPlumberLoader
- Slightly faster (~20%)
- Equivalent Korean text quality
- Built-in image/table APIs for future use

## Known Limitations (to revisit later)

**LIM-001: Text Order Mismatch (SEVERE)**
- Both loaders fail on multi-column layouts — text gets interleaved, **breaking sentences mid-way**
- Example (Page 11): Left column has steps, right column has table. Extracted text mixes them:
  ```
  2
  제품의 3 m 거리 이내에서 'Hi LG'(하이 엘지)를
  기능 명령어 (예시)        ← table header injected mid-sentence
  말하세요.                 ← orphaned sentence continuation
  ```
- Impact: Incoherent chunks → degraded retrieval → bad generation
- Potential solutions: `unstructured`, `docling`, `marker-pdf`, vision LLM

**LIM-002: Incomplete Image Detection**
- Both pdfplumber and PyMuPDF miss some images (vector graphics, nested XObjects)
- Potential solutions: pdf2image (rasterize), vision LLM

## Notes
For task 4, consider using cohere embedding model for visual parser quality. (Future TODO)

---

# Week 3 - Task 2: Metadata Schema & Chunking

## Goal
Implement chunking with metadata schema per §3.2, ready for Phase 2 extension.

## Decisions

### Chunk-page relationship: Option B
- Multiple chunks per page if page > chunk_size
- **Never cross page boundaries** (preserves citation accuracy)
- Note: Option C (cross page boundaries) may be preferable for Advanced RAG (Week 4+) — revisit later

### Metadata schema
```python
{
    "source": "waterpurifier_complex.pdf",
    "category": "waterpurifier",
    "complexity": "complex",
    "model_name": "WD520AWB",
    "page": 12,
    "chunk_id": "WD520AWB_p012_c003",
    "chunk_index": 3,               # position within page
    "char_count": 987,              # for debug/analysis
    # Phase 2 placeholders
    "section": None,
    "step_number": None,
    "image_ids": [],
}
```

## Checklist
- [x] Parse filename → category, complexity
- [x] Extract model_name from filename (with limitation, see LIM-003)
- [x] Implement chunking (chunk_size=1000, overlap=200, no page crossing)
- [x] Generate chunk_id: `{model}_{page}_{chunk_index}`
- [x] Test on all 6 PDFs

## Results
- **Total chunks:** 320 across 6 PDFs
- **Avg chunk size:** 538-672 chars

## Known Limitation

**LIM-003: Model Name Extraction (MINOR)**
- Filename parser cannot reliably extract actual LG model numbers
- Falls back to `{category}_{complexity}` or picks up MFL document IDs
- Workaround: Manual metadata description required for baseline RAG
- Future: Parse model name from PDF first page content

## Output
- `w3/chunking_metadata.ipynb`

---

# Week 3 - Task 3: Embedding

## Goal
Embed chunks using a consistent embedding model.

## Decision

**Selected:** OpenAI `text-embedding-3-small`

**Reasoning:**
- Dependency conflict on macOS 13: torch < 2.2 (no wheels) vs sentence-transformers requires torch >= 2.4
- Local Korean models (ko-sroberta-multitask) blocked by this
- OpenAI embeddings: no local dependencies, negligible cost (~$0.02 for 320 chunks)

**LIM-004: Local Korean Embedding Blocked (macOS 13)**
- sentence-transformers requires torch >= 2.4
- macOS 13 ARM only has wheels for torch < 2.2
- Workaround: Use OpenAI API embeddings
- Future: Upgrade to macOS 14+ or use cloud/Linux environment

## Checklist
- [x] Embed test chunks with OpenAI
- [x] Verify embedding dimensions (1536)
- [x] Test on all 320 chunks

## Results
- **Model:** text-embedding-3-small
- **Dimensions:** 1536
- **Chunks embedded:** 320
- Korean text works fine

## Output
- `w3/embedding_test.ipynb`

---

# Week 3 - Task 4 & 5: Vector DB + Retriever

## macOS 13 Dependency Fix
Working combination for Apple Silicon + macOS 13:
```bash
uv add "chromadb==0.4.22" "onnxruntime==1.16.3" "protobuf>=3.20,<5" langchain-chroma
```

## Goal
Store embeddings in Chroma and implement similarity search retrieval.

## Spec (from WEEK3_TASKS.md)
- **Vector DB:** Chroma (langchain-chroma)
- **Retriever:** similarity search, k=5

## Checklist
- [x] Store all 320 chunks + embeddings in Chroma
- [x] Implement retriever with k=5
- [x] Test retrieval with sample Korean queries
- [x] Verify retrieved chunks are relevant

## Results
- **Vector store:** 320 documents in Chroma (persistent at `data/chroma_db/`)

### Retrieval Quality Assessment

| Query | Correct Category | Notes |
|-------|------------------|-------|
| 정수기 필터 교체 | 3/5 | 2 results from airpurifier (wrong) |
| 공기청정기 필터 청소 | 3/5 | 1 vacuumcleaner, 1 waterpurifier (wrong) |
| 청소기 배터리 충전 | 4/5 | 1 airpurifier filter content (garbage) |
| Wi-Fi 연결 안됨 | 5/5 | Cross-category OK (common feature) |

### Critical Issues (Baseline Limitations)

**LIM-005: No Category Filtering**
- Semantic similarity alone mixes products
- Query for "정수기" returns airpurifier results
- Impact: User asks about specific product but gets mixed answers

**LIM-006: Common Term Contamination**
- "필터" exists in all product categories
- Causes cross-category false matches
- Same issue with "청소", "교체", etc.

**Improvement candidates for Week 4:**
- Metadata filtering by category
- Hybrid search (semantic + keyword)
- Re-ranking with category boost

## Output
- `w3/vectordb_retriever.ipynb`

---

# Week 3 - Task 6: Generator (LLM)

## Goal
Complete baseline RAG pipeline with LLM generation.

## Spec (from WEEK3_TASKS.md §3.3)
- **LLM:** gpt-4o-mini
- **Prompt:** Context-grounded QA with source citation

## Checklist
- [x] Load existing Chroma vector store
- [x] Create RAG chain with prompt template
- [x] Test end-to-end with sample queries
- [x] Evaluate response quality (grounded? cites source?)

## Results

### RAG Response Evaluation

| Query | Grounded | Complete | Source Cited | Correct Product |
|-------|----------|----------|--------------|-----------------|
| 정수기 필터 교체 | ✅ | ✅ | ✅ | ✅ |
| 공기청정기 필터 청소 | ✅ | ✅ | ✅ | ✅ |
| 청소기 배터리 충전 | ⚠️ | ❌ | ❌ WRONG | ❌ |
| Wi-Fi 연결 | ✅ | ✅ | ✅ | N/A |

**Score: 3/4 queries answered correctly**

### Critical Issues

**LIM-007: Wrong Source Citation on Failure**
- When LLM says "문서에서 확인할 수 없습니다", it still cites a source
- Cited source is WRONG (airpurifier for vacuum question)
- Impact: Misleading user with incorrect reference

**LIM-008: Retrieval Gap for Specific Facts**
- Battery charging time query failed
- Either info not in chunks, or retrieval missed relevant docs
- Need to verify if this info exists in source PDFs

## Output
- `w3/rag_pipeline.ipynb`
