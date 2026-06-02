# ADR-008: Hybrid + Rerank 검색 전략 선택

## 상황

4주차 C3 baseline (pymupdf4llm + clean_markdown, 339 chunks)의 Dense-only 검색이 60.9% Top-1 정확도에 그쳤다. 주요 한계:

1. **모델명 exact match 실패**: "AS281DAW 필터 수명" 같은 질문에서 모델명이 dense embedding에서 의미 있는 signal로 작용하지 않음
2. **명시적 카테고리 키워드 무시**: "공기청정기 센서 청소"에서 "공기청정기"가 명시되어 있음에도 다른 카테고리 문서 반환
3. **Cross-category term confusion**: "필터"가 모든 제품군에 등장하여 정수기/공기청정기 구분 실패

## 고려한 선택지

| 구성 | Top-1 Accuracy | Top-5 Accuracy | Latency |
|---|---|---|---|
| Dense only (baseline) | 60.9% (14/23) | 95.7% (22/23) | 0.20s |
| BM25 only | 69.6% (16/23) | 100.0% (23/23) | 0.01s |
| Hybrid (BM25+Dense, RRF) | 82.6% (19/23) | 91.3% (21/23) | 0.13s |
| **Hybrid + Rerank** | **91.3%** (21/23) | **100.0%** (23/23) | 6.73s |

**각 구성의 특성:**

- **Dense only**: 의미 기반 검색에 강하지만 lexical signal (모델명, 고유명사) 약함
- **BM25 only**: 정확한 키워드 매칭에 강하지만, 동의어/유사 표현에서 miss
- **Hybrid (RRF)**: 두 retriever의 rank를 결합. BM25 unique win 6개 + Dense unique win 3개를 모두 커버
- **Hybrid + Rerank**: 1차 검색 k=20 확보 후 Cross-encoder로 top-5 재정렬. Top-5에 있던 정답을 Top-1으로 승격

## 최종 결정

**Hybrid + Rerank** (BM25+Dense RRF → bge-reranker-v2-m3-ko)

## 이유

1. **정확도 최대화**: Top-1 91.3%, Top-5 100.0%로 baseline 대비 +30.4pp 향상
2. **BM25의 lexical signal 활용**: 모델명 "AS281DAW" 같은 exact match 문제 해결
3. **Reranker의 정밀 재정렬**: Cross-encoder가 query-document pair를 직접 평가하여 Top-5에서 Top-1으로 정답 승격
4. **한국어 도메인 최적화**: `dragonkue/bge-reranker-v2-m3-ko`가 한국어 문서에서 강한 성능

**Trade-off 수용:**
- Latency 6.73s는 실시간 서비스에 부담이지만, 본 프로젝트는 정확도 우선의 실험 환경
- Production 배포 시 reranker 경량화 또는 캐싱으로 완화 가능

## 잔여 한계 (Week 6 입력)

Hybrid + Rerank로도 2개 질문 실패:

| ID | Question | Expected | 실제 | 원인 |
|---|---|---|---|---|
| Q17 | 와이파이 연결이 안 될 때 | waterpurifier | vacuumcleaner | Wi-Fi는 category-agnostic 기능 |
| Q20 | 공기청정기 소음이 심해요 | airpurifier | vacuumcleaner | "소음" 관련 청소기 문서가 rerank score 높음 |

이 케이스들은 **metadata filtering** (Self-Query Retriever) 또는 **query understanding**으로 해결 예정.

## 향후 계획

1. **Week 6**: Self-Query Retriever로 category metadata filtering 적용
2. **Production 전환 시**: Reranker latency 최적화 (distillation, caching, GPU 활용)
3. **모니터링**: Production에서 BM25/Dense 각각의 기여도 tracking
