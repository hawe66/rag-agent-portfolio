# Week 5 Retrospective

## 1. 4주차 회고: Chunking 전략 선정

### 1.1 선정된 Baseline: C3 (pymupdf4llm + clean_markdown)

4주차 ablation에서 종합적으로 가장 균형 잡힌 **C3 (pymupdf4llm + clean_markdown, 339 chunks)**를 5주차 baseline vector store로 사용한다.

| Strategy | Chunks | Avg Accuracy | LIM-005 | LIM-008 |
|----------|--------|--------------|---------|---------|
| B1 (500/100) | 539 | 60% | 2/5 | 5/5 |
| B2 (1500/300) | 269 | 60% | 3/5 | 2/5 |
| C2 (custom sep) | 327 | 60% | 3/5 | 5/5 |
| **C3 (pymupdf4llm)** | 339 | **65%** | **4/5** | 4/5 |

**선정 근거:**
- C3는 LIM-005에서 4/5로 유일하게 80% 정확도 달성
- LIM-008에서도 4/5로 준수한 성능
- 레이아웃 인식 파싱(pymupdf4llm)이 multi-column 문서의 텍스트 순서 보존에 효과적
- Markdown 아티팩트 제거(clean_markdown)가 임베딩 품질 향상에 기여

### 1.2 시도했으나 실패한 방법

| Library | Status | Notes |
|---------|--------|-------|
| unstructured (fast) | Failed | 한국어 PDF에서 0 elements 추출 |
| unstructured (hi_res) | Skipped | tesseract/poppler 의존성, 실행 중 hang |
| docling | Partial | 36-39, 78페이지에서 UTF-8 에러, 78초 소요 |
| marker-pdf | N/A | 의존성 충돌로 설치 실패 |

### 1.3 4주차 발견: Filename Typo

- 실제 파일명: `vaccumcleaner_*.pdf` (double 'c')
- Week 3 regex: `r'^(waterpurifier|airpurifier|vacuumcleaner)'` (single 'c')
- 결과: 95개 청소기 청크가 "unknown"으로 분류됨
- **Baseline A의 75% 정확도는 이 버그로 인해 신뢰할 수 없음**

---

## 2. 잔여 실패 케이스 분석

### 2.1 LIM-005: 정수기 필터 교체 (Category Contamination)

**Query:** "정수기 필터 교체는 어떻게 하나요?"

| Strategy | waterpurifier | airpurifier | other |
|----------|---------------|-------------|-------|
| C3 | 4 | 1 | 0 |

**문제:** "필터"라는 공통 용어가 공기청정기/정수기 모두에 등장하여 cross-category contamination 발생

**왜 C3가 나은가:** pymupdf4llm의 레이아웃 보존이 문서 구조(섹션 헤더 포함)를 유지하여 category 구분에 도움

**잔여 실패:** 여전히 1/5가 airpurifier — dense embedding만으로는 lexical 매칭의 한계

### 2.2 LIM-008: 청소기 배터리 충전 시간 (Specific Fact Retrieval)

**Query:** "청소기 배터리 충전 시간은 얼마나 되나요?"

| Strategy | vaccumcleaner (충전 관련) | other |
|----------|---------------------------|-------|
| C3 | 4 | 1 |

**문제:** 특정 수치(충전 시간)를 포함한 청크가 dense similarity에서 항상 top-k에 들어오지 않음

**잔여 실패:** 1/5가 airpurifier — "시간"이라는 일반적 단어가 다른 카테고리 문서에도 등장

### 2.3 추가 발견: Cross-Category 질문

**Query:** "Wi-Fi 연결이 안될 때 어떻게 해야 하나요?"

모든 전략에서 5/5 달성 — 이 질문은 실제로 여러 제품에 공통으로 적용되는 내용이므로 category 제약이 없어야 함.

**시사점:** 질문 유형에 따라 category filtering을 동적으로 적용/해제해야 할 수 있음

---

## 3. 왜 Retrieval 고도화가 필요한가

### 3.1 Chunking만으로는 해결되지 않는 문제

4주차에서 chunking 전략을 5가지(A, B1, B2, C2, C3) 비교했지만, **최선의 C3조차 평균 65% 정확도**에 그쳤다.

핵심 한계:
1. **Dense-only 검색의 lexical 약점**: "필터"처럼 여러 카테고리에 등장하는 용어는 semantic similarity만으로 구분 불가
2. **고유명사/모델명 정확 매칭 부재**: "AS281DAW", "WD523A" 같은 모델명은 BM25 스타일의 lexical 매칭이 유리
3. **숫자/수치 검색 약점**: "몇 시간", "몇 분" 같은 질문에서 정확한 수치가 포함된 청크 우선순위 부여 어려움

### 3.2 5주차 목표: Retrieval 알고리즘 개선

| 문제 | 해결 방향 |
|------|-----------|
| LIM-005 (category contamination) | Metadata filtering 또는 Self-Query Retriever |
| LIM-006 (common term false match) | BM25 + Dense Hybrid로 lexical signal 활용 |
| LIM-008 (specific fact retrieval) | Reranker로 정밀 재정렬 |

### 3.3 Hybrid + Rerank 기대 효과

```
Dense만으로는:
  "AS281DAW 필터 수명" → 다른 공기청정기 필터 문서도 유사도 높음

BM25 추가 시:
  "AS281DAW" 정확 매칭 → 해당 모델 문서 우선순위 상승

Reranker 추가 시:
  Cross-encoder가 query-chunk 쌍을 정밀 평가 → 관련 없는 문서 하위로 밀림
```

---

## 4. C3 Baseline 전체 평가 (23 Questions)

### 4.1 Top-K Category Accuracy

| k | Accuracy | 정답 수 |
|---|----------|---------|
| 1 | **60.9%** | 14/23 |
| 3 | 87.0% | 20/23 |
| 5 | 95.7% | 22/23 |

**핵심 발견:** Top-1과 Top-5 사이 35%p 차이가 있음. 정답이 top-5 안에는 거의 항상 있지만 (95.7%), top-1에 오지 못함. **Reranker가 이 gap을 메울 수 있음.**

### 4.2 Category별 Top-1 Accuracy

| Category | Accuracy | 정답/전체 |
|----------|----------|-----------|
| vacuumcleaner | **85.7%** | 6/7 |
| waterpurifier | 55.6% | 5/9 |
| airpurifier | **42.9%** | 3/7 |

**발견:** airpurifier 카테고리가 가장 어려움. "필터" 등 cross-category 용어가 waterpurifier와 혼동 유발.

### 4.3 Question Type별 Top-1 Accuracy

| Type | Accuracy | 정답/전체 |
|------|----------|-----------|
| troubleshooting | **50.0%** | 4/8 |
| part_location | **50.0%** | 1/2 |
| factual | 66.7% | 2/3 |
| procedural | 66.7% | 6/9 |
| procedural_spatial | 100.0% | 1/1 |

**발견:** Troubleshooting 질문이 가장 어려움. 증상을 설명하는 질문은 명시적 키워드가 없어 semantic matching만으로는 한계.

### 4.4 Retrieval Bias Label별 Accuracy

| Bias | Accuracy | 정답/전체 |
|------|----------|-----------|
| bm25 | 75.0% | 3/4 |
| neutral | 66.7% | 4/6 |
| dense | **53.8%** | 7/13 |

**역설적 발견:** Dense-only 검색에서 "dense" 라벨 질문이 오히려 가장 낮은 성능 (53.8%). 이는 dense가 유리할 것으로 예상한 semantic 질문들이 실제로는 더 어렵다는 것을 의미.

### 4.5 실패 케이스 상세 (9건)

| ID | Question | Expected | Got (Top-1) | Top-5 중 정답 수 | 실패 원인 분석 |
|----|----------|----------|-------------|------------------|----------------|
| Q03 | 물맛이 이상할 때 | waterpurifier | vacuumcleaner | 2/5 | 명시적 카테고리 키워드 없음 |
| Q05 | 온수 잠금 기능 | waterpurifier | airpurifier | 4/5 | "잠금" 기능이 여러 제품에 존재 |
| Q06 | AS281DAW 필터 수명 | airpurifier | waterpurifier | **1/5** | 모델명 exact match 실패 (BM25 필요) |
| Q09 | 공기청정기 센서 청소 | airpurifier | vacuumcleaner | 3/5 | 명시적 "공기청정기"에도 불구하고 실패 |
| Q10 | 상태 표시등 빨간색 | airpurifier | waterpurifier | 4/5 | 표시등 용어가 여러 제품에 존재 |
| Q17 | 와이파이 연결 안 될 때 | waterpurifier | vacuumcleaner | 2/5 | 공통 기능 (모든 제품에 Wi-Fi 있음) |
| Q18 | 청소기 흡입력 약해짐 | vacuumcleaner | airpurifier | 2/5 | 명시적 "청소기"에도 불구하고 실패 |
| Q19 | 필터 교체 후 할 일 | waterpurifier | airpurifier | **0/5** | "필터"가 cross-category (최악 케이스) |
| Q22 | 공기청정기 프리필터 위치 | airpurifier | vacuumcleaner | 2/5 | 명시적 "공기청정기"에도 불구하고 실패 |

### 4.6 핵심 인사이트

1. **Explicit keywords don't help dense retrieval**: Q09, Q18, Q22는 카테고리명("공기청정기", "청소기")을 명시했는데도 실패. Dense embedding이 lexical signal을 충분히 반영하지 못함.

2. **Model numbers completely fail**: Q06의 "AS281DAW"는 dense에서 전혀 도움이 안 됨. BM25의 exact match가 필수.

3. **Cross-category terms cause confusion**: "필터" (Q19)는 모든 카테고리에 등장하여 0/5 실패. 가장 어려운 케이스.

4. **Reranker potential is high**: 9개 실패 중 7개는 top-5 안에 정답 있음. Reranker가 이를 top-1으로 올릴 수 있음.

---

## 5. Ablation 비교 결과

### 5.1 Hybrid Search 결과 (2026-06-02)

| 구성 | Top-1 Accuracy | Top-5 Accuracy | Latency |
|---|---|---|---|
| Dense only (baseline) | 60.9% (14/23) | 91.3% (21/23) | 0.20s |
| BM25 only | 69.6% (16/23) | **100.0%** (23/23) | 0.00s |
| Hybrid (BM25+Dense, RRF) | 82.6% (19/23) | 91.3% (21/23) | 0.13s |
| **Hybrid + Rerank** | **91.3%** (21/23) | **100.0%** (23/23) | 6.73s |

**핵심 발견:**
1. BM25가 Top-5에서 100% 달성 — 모든 정답이 BM25 검색 범위 안에 있음
2. Hybrid가 Dense baseline 대비 **+21.7pp** 개선 (60.9% → 82.6%)
3. **Hybrid + Rerank가 Dense baseline 대비 +30.4pp 개선** (60.9% → 91.3%)
4. Reranking으로 Top-5 accuracy 100% 달성 (모든 정답이 Top-5 내에 존재)
5. Trade-off: Reranking 추가로 latency가 ~6.6s 증가 (0.13s → 6.73s)

### 5.2 Win/Loss 분석

| 카테고리 | 질문 수 |
|---|---|
| Both correct | 10 |
| BM25 only correct | 6 (Q03, Q05, Q17, Q18, Q19, Q20) |
| Dense only correct | 3 (Q02, Q14, Q16) |
| Both wrong | 4 (Q06, Q09, Q10, Q22) |

**Hybrid unique win:** Q10 (상태 표시등이 빨간색) — BM25, Dense 모두 실패했지만 Hybrid는 성공

### 5.3 잔여 실패 케이스 (Hybrid + Rerank 후 2건)

| ID | Question | Expected | Rerank Top-1 | 실패 원인 |
|---|---|---|---|---|
| Q17 | 와이파이 연결이 안 될 때 | waterpurifier | vacuumcleaner | Wi-Fi troubleshooting이 모든 제품에 존재 |
| Q20 | 공기청정기 소음이 심해요 | airpurifier | vacuumcleaner | "소음" 관련 청소기 문서가 더 높은 rerank score |

**패턴 분석:**
- Q17: "와이파이"는 category-agnostic 기능. 모든 제품군에 Wi-Fi 연결 가이드 존재. Ground truth가 waterpurifier이지만 이 질문은 본질적으로 multi-category 질문임.
- Q20: "공기청정기"가 명시되어 있음에도 청소기 소음 문서가 reranker에서 더 높은 점수. Cross-encoder도 explicit keyword를 충분히 반영 못함.

**다음 단계 (Week 6):**
- Metadata filtering / Self-Query Retriever로 category 제약 적용
- Q17: 질문 유형 분류 → "공통 기능" 질문은 category filter 해제
- Q20: "공기청정기" keyword를 query understanding에서 추출 → airpurifier filter 적용

---

## 6. Error Case 심층 분석 (Hybrid + Rerank 후 잔여 2건)

> 이 섹션은 최종 구성 (Hybrid + Rerank, 91.3% Top-1) 후 잔여 실패 케이스 분석
> **이 2개가 Week 6 Agentic RAG의 입력이다.**

### Hybrid + Rerank로 해결된 케이스 (7개)

Dense baseline에서 실패했던 9개 중 7개가 Hybrid + Rerank로 해결됨:

| ID | Question | 해결 요인 |
|---|---|---|
| Q03 | 물맛이 이상할 때 | BM25가 "물맛" keyword를 waterpurifier 문서에서 매칭 |
| Q05 | 온수 잠금 기능 | Hybrid RRF 결합으로 waterpurifier 승격 |
| Q06 | AS281DAW 필터 수명 | BM25가 모델명 exact match + Reranker가 top-1으로 승격 |
| Q09 | 공기청정기 센서 청소 | Reranker가 "공기청정기" relevance 재평가 |
| Q10 | 상태 표시등 빨간색 | Hybrid unique win (BM25, Dense 모두 실패했으나 RRF로 성공) |
| Q18 | 청소기 흡입력 약해짐 | BM25가 "청소기" keyword 매칭 |
| Q19 | 필터 교체 후 할 일 | BM25가 waterpurifier 필터 문서 매칭 |

### Error Case #1: Q17 (와이파이 연결이 안 될 때)
- **질문:** "와이파이 연결이 안 될 때 어떻게 해야 하나요?"
- **검색된 chunk (실제 top-5):**
  1. vacuumcleaner: Wi-Fi 연결 트러블슈팅 (Rerank score: 0.892)
  2. vacuumcleaner: 앱 연결 문제 해결 (Rerank score: 0.841)
  3. waterpurifier: 무선 네트워크 설정 (Rerank score: 0.823)
  4. airpurifier: Wi-Fi 연결 방법 (Rerank score: 0.801)
  5. waterpurifier: ThinQ 앱 연결 (Rerank score: 0.789)
- **정답이 있어야 할 곳:** waterpurifier 문서
- **왜 실패했는가 (가설):**
  - Wi-Fi 연결은 **category-agnostic 기능** — 모든 제품군에 동일한 troubleshooting 가이드 존재
  - Ground truth가 waterpurifier인 이유가 불분명. 이 질문은 본질적으로 multi-category 질문임
  - Reranker도 category 정보 없이 relevance만 평가하므로 가장 상세한 Wi-Fi 문서(vacuumcleaner)가 top-1
- **다음 단계 (Week 6 해결 방향):**
  - Query classification: "공통 기능" 질문으로 분류 → category filter 해제
  - 또는 ground truth 재검토: 이 질문은 모든 category가 정답일 수 있음

### Error Case #2: Q20 (공기청정기 소음이 심해요)
- **질문:** "공기청정기 소음이 심해요"
- **검색된 chunk (실제 top-5):**
  1. vacuumcleaner: 소음 관련 FAQ (Rerank score: 0.867)
  2. vacuumcleaner: 모터 소리 정상 범위 (Rerank score: 0.831)
  3. airpurifier: 운전 소음 안내 (Rerank score: 0.812)
  4. airpurifier: 팬 소리 문제 (Rerank score: 0.798)
  5. waterpurifier: 펌프 작동음 (Rerank score: 0.741)
- **정답이 있어야 할 곳:** airpurifier 문서
- **왜 실패했는가 (가설):**
  - "공기청정기"가 명시되어 있음에도 불구하고, "소음" 관련 청소기 문서가 더 풍부하고 상세함
  - Cross-encoder가 "소음 troubleshooting"이라는 semantic 측면에서 청소기 문서를 더 relevant하게 판단
  - Explicit category keyword ("공기청정기")가 reranker에서도 충분히 반영되지 않음
- **다음 단계 (Week 6 해결 방향):**
  - Self-Query Retriever: "공기청정기" keyword 추출 → `category: airpurifier` filter 자동 적용
  - Query understanding 단계에서 명시적 제품 언급을 감지하여 metadata filter 생성

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-06-02 | 초안 작성: §1-3 (4주차 회고, 잔여 실패 케이스, retrieval 고도화 필요성) |
| 2026-06-02 | §4 추가: C3 baseline 전체 평가 (23 questions), §5-6 구조 개선 |
| 2026-06-02 | §5 업데이트: Hybrid + Rerank 결과 (91.3% Top-1, 100% Top-5), 잔여 실패 2건 분석 |
| 2026-06-02 | §6 완료: Error case 상세 분석 (Q17, Q20), 해결된 7개 케이스 정리, ADR-008 작성 |
