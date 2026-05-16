# Week 3 Summary: Baseline RAG 구현

## 상황
Phase 1 Advanced RAG의 첫 단계로 Naive RAG baseline 구현. 이후 개선의 기준점이 되는 파이프라인 구축.

## 구현 결과

### Pipeline Components
| Component | 선택 | 비고 |
|-----------|------|------|
| PDF Loader | PDFPlumberLoader | PyPDF 대비 20% 빠름, 품질 동등 |
| Chunking | RecursiveCharacterTextSplitter | chunk_size=1000, overlap=200, 페이지 경계 유지 |
| Embedding | OpenAI text-embedding-3-small | 1536 dims |
| Vector DB | Chroma 0.4.22 | 320 documents |
| Retriever | Similarity search | k=5 |
| Generator | gpt-4o-mini | temperature=0.1 |

### Baseline Metrics
- **Total chunks:** 320 (6 PDFs)
- **Avg chunk size:** 538-672 chars
- **Retrieval accuracy:** 15/20 correct category (75%)
- **RAG response score:** 3/4 queries correct (75%)

## 발견된 한계 (Problems to Address)

### LIM-001: Text Order Mismatch (SEVERE)
- **문제:** 멀티컬럼 레이아웃에서 텍스트 순서가 섞임. 문장이 중간에 끊기고 다른 컬럼 내용 삽입.
- **영향:** 청크 내 문맥 파괴 → 검색/생성 품질 저하
- **후보 해결책:** layout-aware parsing (unstructured, docling, marker-pdf), Vision LLM

### LIM-002: Incomplete Image Detection
- **문제:** pdfplumber, PyMuPDF 모두 일부 이미지 감지 실패 (vector graphics, nested XObjects)
- **영향:** Phase 2 cross-modal alignment 시 image-text pair 누락 가능
- **후보 해결책:** pdf2image (rasterize), Vision LLM

### LIM-003: Model Name Extraction (MINOR)
- **문제:** 파일명에서 실제 LG 모델번호 추출 불가. MFL 문서번호 또는 fallback 사용.
- **영향:** chunk_id, model_name 메타데이터 불일치
- **후보 해결책:** PDF 첫 페이지에서 파싱, 또는 수동 매핑

### LIM-004: Local Korean Embedding Blocked (macOS 13)
- **문제:** sentence-transformers requires torch >= 2.4, macOS 13 ARM은 torch < 2.2만 지원
- **영향:** ko-sroberta-multitask 등 로컬 한국어 임베딩 사용 불가
- **후보 해결책:** macOS 14 업그레이드, 또는 OpenAI embeddings 유지

### LIM-005: No Category Filtering
- **문제:** Semantic similarity만으로 검색 → 제품 카테고리 혼합
- **영향:** "정수기" 질문에 airpurifier 결과 반환 (3/5 correct)
- **후보 해결책:** 메타데이터 필터링, hybrid search, re-ranking with category boost

### LIM-006: Common Term Contamination
- **문제:** "필터", "청소", "교체" 등 ���통 용어가 cross-category false match 유발
- **영향:** 검색 정밀도 저하
- **후보 해결책:** Query expansion with category, metadata filtering

### LIM-007: Wrong Source Citation on Failure
- **문제:** LLM이 "문서에서 확인할 수 없습니다" 응답 시에도 잘못된 출처 인용
- **영향:** 사용자에게 잘못된 참조 제공
- **후보 해결책:** Prompt 수정 (답변 불가 시 출처 생략)

### LIM-008: Retrieval Gap for Specific Facts
- **문제:** "청소기 배터리 충전 시간" 질문 실패
- **영향:** 특정 정보 검색 누락
- **후보 해결책:** 청크 커버리지 검증, 질문-청크 매핑 분석

## macOS 13 Dependency Fix
```bash
uv add "chromadb==0.4.22" "onnxruntime==1.16.3" "protobuf>=3.20,<5" langchain-chroma
```

## Week 4 개선 후보
1. 메타데이터 필터�� (category 기반)
2. Hybrid search (semantic + keyword)
3. Re-ranking
4. Prompt 개선 (실패 시 출처 생략)
5. 청크 커버리지 분석

## 산출물
- `w3/pdf_loader_comparison.ipynb`
- `w3/chunking_metadata.ipynb`
- `w3/embedding_test.ipynb`
- `w3/vectordb_retriever.ipynb`
- `w3/rag_pipeline.ipynb`
- `w3/TODO.md`
- `data/chroma_db/` (vector store)
