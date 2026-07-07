# Week 6 Retrospective

## 1. 5주차 회고 정리

### 1.1 5주차 최종 Retrieval 전략

**Hybrid (BM25 + Dense, RRF) + bge-reranker-v2-m3-ko**

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 91.3% (21/23) |
| Top-5 Accuracy | 100.0% (23/23) |
| Latency | ~6.73s |

이 전략은 6주차 Agentic RAG의 `retrieve` 노드에 그대로 사용된다.

### 1.2 5주차 이후 잔여 실패 케이스 (2건)

| ID | Question | Expected | Actual | 실패 원인 |
|----|----------|----------|--------|-----------|
| Q17 | 와이파이 연결이 안 될 때 | waterpurifier | vacuumcleaner | Wi-Fi는 category-agnostic 기능 |
| Q20 | 공기청정기 소음이 심해요 | airpurifier | vacuumcleaner | "소음" 관련 청소기 문서가 rerank score 높음 |

**처리 방침:**
- Q17: Ground truth 의심 — **분석에서 제외, 코드 실행에는 포함**
- Q20: Explicit category keyword("공기청정기")가 reranker에서도 작동 안 함 — **Self-Query/metadata filter 후보**

### 1.3 왜 Agentic 구조가 필요한가

5주차 Hybrid+Rerank는 retrieval 자체는 100% Top-5에 정답을 회수한다. 하지만 Top-1 정확도가 91.3%에서 막혀 있고, 잔여 케이스의 본질이 "검색 알고리즘의 한계"가 아니라 **"질문 유형에 맞는 다른 처리가 필요"**한 케이스(category-agnostic, explicit keyword 무시 등)다.

즉 문제가 **단일 알고리즘 튜닝이 아니라 시스템 구조의 문제**로 옮겨졌다. Agentic 구조는 검색 결과를 보고 다음 행동을 조건부로 선택할 수 있게 해서, 같은 retriever로도 질문 유형별 다른 전략을 적용할 수 있다.

---

## 2. Agentic RAG vs Baseline 정량 비교

> §7 실험 후 작성 예정

| 구성 | Top-1 (22q) | Top-1 (23q) | Faithfulness | Answer Relevancy | Context Precision | 평균 Latency(s) |
|---|---|---|---|---|---|---|
| Baseline: 5주차 Hybrid+Rerank | ? | 91.3% | ? | ? | ? | 6.73 |
| Agentic RAG | ? | ? | ? | ? | ? | ? |

---

## 3. 운영 관점 회고

> §8 실험 후 작성 예정

### 3.1 재검색이 도움이 된 사례

(TBD)

### 3.2 재검색이 불필요했거나 악영향을 준 사례

(TBD)

### 3.3 Baseline 대비 평균 latency 변화

(TBD)

### 3.4 Agentic RAG가 본 도메인에 필요한가

(TBD)

---

## 4. 본 도메인 특화 분석

> §8 실험 후 작성 예정

### 4.1 Q20 (공기청정기 소음) 해결 여부

(TBD: Self-Query metadata filter 작동 여부)

### 4.2 Q17 (와이파이) Routing 동작

(TBD: 분석에서 제외하지만 동작 기록)

### 4.3 Regression 여부

(TBD: 5주차에서 풀린 21문항이 Agentic에서도 풀리는지)

### 4.4 불필요한 retry 통계

(TBD: grade가 not_relevant로 잘못 판단해서 retry 들어간 비율)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-06-02 | 초안 작성: §1 5주차 회고 정리 완료 |
