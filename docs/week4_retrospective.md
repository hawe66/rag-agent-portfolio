# Week 4 Retrospective: Chunking Experiments

## 1. Week 3 Baseline 회고

### 잘못 검색된 Chunk 사례

#### 사례 #1: LIM-005 — Category Cross-Contamination
- **질문:** "정수기 필터 교체는 어떻게 하나요?"
- **문제:** 5개 중 2개가 airpurifier 문서에서 검색됨
- **원인:** "필터", "교체" 같은 공통 용어가 cross-category false match 유발 (LIM-006)
- **가설:** 카테고리 메타데이터 필터링 부재 (LIM-005)

#### 사례 #2: LIM-008 — Specific Fact Retrieval 실패
- **질문:** "청소기 배터리 충전 시간은 얼마나 되나요?"
- **문제:** Baseline에서 vacuum cleaner 카테고리가 "unknown"으로 분류됨
- **원인:** Week 3 regex가 `vacuumcleaner` (single 'c')만 매칭, 실제 파일명은 `vaccumcleaner` (double 'c')
- **영향:** 95개 vacuum cleaner chunk가 "unknown" 카테고리로 저장됨

### 개선 가설

#### 가설 A: Layout-aware Parsing
- **관찰:** LIM-001 (멀티컬럼 텍스트 순서 깨짐)은 chunk_size 조정으로 해결 불가
- **가설:** pymupdf4llm 등 layout-aware parser로 컬럼 순서 보존 시 chunk 의미 회복
- **결과:** C3 전략으로 검증 — LIM-005에서 4/5 달성 (baseline 3/5 대비 개선)

#### 가설 B: Custom Separator
- **관찰:** 단계 번호, 불릿 포인트 등 도메인 특화 구분자 활용 가능
- **가설:** 도메인 맞춤 separator로 의미 단위 보존
- **결과:** C2 전략으로 검증 — LIM-008에서 5/5 달성

---

## 2. 데이터 진단 요약

- **멀티컬럼 페이지 비율:** 79.4% (Week 3 분석)
- **PDF 6개 총 chunk 수:** 320개 (baseline 1000/200 기준)
- **카테고리 분포:** waterpurifier 100, airpurifier 125, unknown(vacuum) 95
- **Baseline 버그 발견:** vacuum cleaner 파일명 typo로 인한 카테고리 누락

---

## 3. Chunking 전략 비교

### 전략 C 선택 Reasoning

Week 3 LIM-001 (텍스트 순서 깨짐)은 **근본적으로 PDF parsing 단계의 문제**였다. chunk_size를 500으로 줄이든 1500으로 늘리든, 이미 깨진 텍스트 순서는 복구 불가능하다.

따라서 **layout-aware parsing (C3)을 메인 카드로** 선택했다. 후보 라이브러리:
1. unstructured — 0 elements 추출, Korean PDF 지원 부족
2. docling — UTF-8 인코딩 에러 (pages 36-39)
3. marker-pdf — 의존성 충돌로 설치 불가
4. **pymupdf4llm — 성공**, markdown 출력 + cleaning 적용

### C3 시도 결과

| Library | 결과 | 비고 |
|---------|------|------|
| unstructured (fast) | ❌ 실패 | 0 elements |
| unstructured (hi_res) | ❌ 건너뜀 | tesseract/poppler 필요 |
| docling | ⚠️ 부분 성공 | UTF-8 에러 |
| marker-pdf | ❌ 설치 불가 | 의존성 충돌 |
| **pymupdf4llm** | ✅ 성공 | markdown cleaning 필요 |

**핵심 발견:** pymupdf4llm의 raw markdown 출력은 `**bold**`, `|---|---|`, image placeholder 등 노이즈 포함 → embedding 품질 저하. `clean_markdown()` 함수로 후처리 필수.

### 비교 결과 표

| Strategy | Chunks | Avg Accuracy | LIM-005 | LIM-008 |
|----------|--------|--------------|---------|---------|
| A (baseline) | 320 | 75%* | 3/5 | N/A (bug) |
| B1 (500/100) | 539 | 60% | 2/5 | **5/5** |
| B2 (1500/300) | 269 | 60% | 3/5 | 2/5 |
| C2 (custom sep) | 327 | 60% | 3/5 | **5/5** |
| **C3 (pymupdf4llm)** | 339 | **65%** | **4/5** | 4/5 |

*Baseline A의 75%는 vacuum cleaner "unknown" 버그로 인해 인위적으로 높음

### 정성 분석 발견

1. **C3가 LIM-005에서 승리:** 4/5 waterpurifier 정확도 (다른 전략은 3/5)
   - Layout-aware parsing이 문서 구조 보존 → 카테고리 분리 개선

2. **B1, C2가 LIM-008에서 승리:** 5/5 vacuum cleaner 정확도
   - 작은 chunk (B1) 또는 custom separator (C2)가 specific fact 분리에 유리

3. **No single winner:** 각 전략이 다른 유형의 쿼리에서 강점 보임

---

## 4. 메타데이터 활용 아이디어

Week 4에서 확장한 메타데이터 스키마:
```python
{
    "source": "...",
    "category": "waterpurifier",  # LIM-005 공격용
    "complexity": "complex",
    "page": 12,
    "chunk_id": "...",
    "char_count": 487,
}
```

**Week 5 활용 계획:**
- `category` 필터 → **LIM-005, LIM-006 직접 공격** (Self-Query Retriever)
- `section` 필터 (추가 예정) → "설치 방법" 질문은 설치 섹션만 검색
- Hybrid approach: semantic search + metadata filtering 조합

---

## 5. RAGAS 점수 해석

> Week 4에서는 RAGAS 환경 설정 이슈로 정량 평가 대신 category accuracy 기반 평가 사용

**Category Accuracy 기준:**
- Baseline A: 75% (인위적 — vacuum bug)
- B1, B2, C2: 60%
- **C3: 65%** (실질적 최고)

**해석:**
- C3 (pymupdf4llm + cleaning)이 layout 보존 효과로 5%p 개선
- 단, 개선폭이 크지 않은 이유: 현재 평가 쿼리 수가 4개로 적음
- Week 5에서 평가 셋 확대 + RAGAS 적용 필요

---

## 6. Week 5 Retrieval 고도화 계획

1. **LIM-005, LIM-006 공격:** Self-Query Retriever로 category metadata filtering
2. **LIM-007 공격:** Prompt 개선 (답변 불가 시 출처 생략)
3. **LIM-008 공격:** Chunk 커버리지 분석 — 특정 사실이 어느 chunk에 있는지 매핑
4. **Hybrid Search:** BM25 + semantic search 조합
5. **Re-ranking:** Cross-encoder 기반 재정렬

---

## 7. 부록: 환경/인프라 메모

### Baseline 버그 (LIM-003 관련)
- Week 3 regex: `r'^(waterpurifier|airpurifier|vacuumcleaner)'`
- 실제 파일명: `vaccumcleaner_*.pdf` (double 'c' typo)
- 해결: Week 4 `parse_filename()`에 `vaccumcleaner` 패턴 추가

### macOS 14 업그레이드 (LIM-004 해결)
- macOS 13 → 14 업그레이드 완료
- torch >= 2.4 설치 가능해짐
- sentence-transformers, Korean embedding 모델 사용 가능

### Layout-aware Library 호환성
| Library | 설치 | 실행 | Korean PDF |
|---------|------|------|------------|
| unstructured[pdf] | ✅ | ✅ | ❌ (0 elements) |
| docling | ✅ | ⚠️ | ⚠️ (UTF-8 errors) |
| marker-pdf | ❌ | - | - |
| pymupdf4llm | ✅ | ✅ | ✅ |
