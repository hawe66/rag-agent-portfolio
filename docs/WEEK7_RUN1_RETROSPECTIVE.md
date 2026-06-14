# Week 7 Retrospective — Evaluation Framework (Phase 1 Pre-Freeze)

> **작성일**: 2026-06-08
> **이 주차의 핵심**: "만드는 능력보다 측정하는 능력"

---

## 1. 6주차 회고 마무리

### 1.1 Agentic RAG 결과 요약 (from Week 6)

| 지표 | 값 |
|------|-----|
| Top-1 Accuracy (23q) | 95.7% (22/23) |
| Top-1 Accuracy (22q, Q17 제외) | 100.0% (22/22) |
| Average Latency | 34.46s |
| Retry 분포 | retry=0: 19건, retry=1: 1건, retry=2: 3건 |

**유일한 실패 케이스**: Q17 (와이파이 연결이 안 될 때) — category-agnostic 질문으로 metadata filter가 도움 안 됨.

### 1.2 Agentic이 효과적인 질문 유형 (가설)

- **out_of_scope**: cannot_answer 로직이 명시적으로 거절 처리 → Agentic 필수
- **multi_hop**: Self-Query로 카테고리 추출 + retry로 누락 정보 보완 → Agentic 우위 예상
- **factual**: Baseline (Hybrid+Rerank)로 충분 → Agentic 오버헤드 불필요

---

## 2. Golden Set v1 구축 결과

### 2.1 데이터셋 통계

| 항목 | 값 |
|------|-----|
| 총 문항 수 | 35 |
| factual | 20 (including 3 image-required) |
| comparison | 4 |
| multi_hop | 4 |
| out_of_scope | 4 |
| safety | 2 |

### 2.2 라벨링 방법

- 기존 v2 (23문항)에 q_type 라벨 부착
- 신규 12문항 추가 (comparison 4, multi_hop 3, out_of_scope 4, safety 2)
- reference_context: PDF 문서명 + 페이지 번호 수동 라벨링
- ground_truth: 1-2문장 정답 요지 수동 작성

### 2.3 도메인 특성

- **safety 카테고리 약함**: 가전 매뉴얼은 의료/법률만큼 safety가 강하지 않음
- "직접 분해/수리 금지", "감전 위험" 정도의 안내만 존재
- 이는 도메인 한계로 정직하게 기록

---

## 3. Baseline 구성

- **Retriever**: Hybrid+Rerank (5주차 최종)
  - BM25 + Dense (RRF fusion, weight 0.5:0.5)
  - Cross-encoder reranking (dragonkue/bge-reranker-v2-m3-ko)
  - First stage k=20, final k=5
- **Generator**: gpt-4o-mini (temperature=0)
- **Judge**: gpt-4o-mini (비용 절감, self-preference 한계 인지)

---

## 4. RAGAS 4지표 측정 결과

> 실행 후 채우기

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---------------|--------------|------------------|-------------------|----------------|
| Baseline (Hybrid+Rerank) | ? | ? | ? | ? |
| Agentic (LangGraph) | ? | ? | ? | ? |

### 4.1 Baseline vs Agentic 비교

(실행 후 분석)

---

## 5. RAGAS 한계 직접 확인

### 5.1 사례 1: 점수 높은데 답변 나쁜 경우

**질문**: (실행 후 채우기)

**응답**: (실행 후 채우기)

**RAGAS 점수**: Faithfulness X.XX

**분석**:
- 어떤 지표가 어긋났는가:
- 왜 어긋났는가 (가설):
- 필요한 추가 지표: Evidence Coverage (답변이 질문의 모든 측면을 커버하는지)

### 5.2 사례 2: 점수 낮은데 답변 괜찮은 경우

**질문**: (실행 후 채우기)

**응답**: (실행 후 채우기)

**RAGAS 점수**: Faithfulness X.XX

**분석**:
- 어떤 지표가 어긋났는가:
- 왜 어긋났는가 (가설):
- 필요한 추가 지표:

### 5.3 RAGAS 한계 요약

| 한계 유형 | 발생 빈도 | 원인 가설 | 필요 보완 지표 |
|-----------|-----------|-----------|----------------|
| High score, bad answer | ?건 | ? | Evidence Coverage |
| Low score, good answer | ?건 | ? | ? |

---

## 6. 도메인 특화 메트릭

### 6.1 Refusal Accuracy

| 지표 | 값 |
|------|-----|
| Overall Accuracy | ?% |
| True Positive Rate (올바른 거절) | ?% |
| False Positive Rate (오거절) | ?% |

**정의**:
- TP: out_of_scope 질문에 올바르게 "확인할 수 없습니다" 답변
- FP: 답해야 하는 질문에 잘못 거절

**거절 패턴**: "제공된 문서에서 확인할 수 없습니다."

### 6.2 Citation Accuracy

| 지표 | 값 |
|------|-----|
| Citation Accuracy | ?% |
| Correct Citations | ?/? |

**정의**:
- 답변에 명시한 (model_name, page) 출처가 reference_context와 일치하는가
- 페이지 ±2 범위 허용 (도메인 휴리스틱)

**본 도메인 특성**: 가전 매뉴얼은 "어느 모델 어느 페이지" 출처가 본질. 4주차 메타데이터 활용의 핵심 검증 지점.

---

## 7. Pairwise LLM-as-judge (P1)

> Agentic 완료 시 채우기

### 7.1 비교 방법

- 같은 질문에 대해 Baseline/Agentic 답변을 judge가 비교
- 평가 기준: 직접 답변 여부, context 근거, 출처 적절성, 불확실 시 거절, safety 안내, 간결성

### 7.2 결과

| 비교 | Baseline 승 | Agentic 승 | 동점 |
|------|-------------|------------|------|
| factual (5q) | ? | ? | ? |
| comparison (4q) | ? | ? | ? |
| multi_hop (4q) | ? | ? | ? |
| out_of_scope (4q) | ? | ? | ? |
| safety (2q) | ? | ? | ? |

---

## 8. 질문 유형별 분석 (P1)

> 실행 후 채우기

| q_type | n | Faithfulness | Context Recall | Refusal Acc | Citation Acc | 관찰 |
|--------|---|--------------|----------------|-------------|--------------|------|
| factual | 20 | ? | ? | N/A | ? | ? |
| comparison | 4 | ? | ? | N/A | ? | ? |
| multi_hop | 4 | ? | ? | N/A | ? | ? |
| out_of_scope | 4 | N/A | N/A | ? | N/A | ? |
| safety | 2 | ? | N/A | ? | ? | ? |

### 8.1 분석 질문 답변

1. **Baseline 충분한 유형**: (예상) factual - 단순 정보 검색은 Hybrid+Rerank로 충분
2. **Agentic 효과 큰 유형**: (예상) out_of_scope, multi_hop
3. **Agentic 적용해도 이득 없는 유형**: (예상) factual - 오히려 latency 증가
4. **여전히 약한 유형**: (예상) safety - 도메인 자체의 한계
5. **8주차 freeze 전 가장 먼저 고쳐야 할 약점**: (실행 후 결정)

---

## 9. Known Limitations (8주차 README 연결)

1. **Q17 유형 (category-agnostic)**: Self-Query가 도움 안 됨. 2/23 실패.
2. **safety 카테고리 약함**: 가전 도메인 특성. 의료/법률 수준의 안전 안내 부재.
3. **image-required 질문**: Phase 1에서는 텍스트만으로 불완전한 답변. Phase 2 motivation.
4. **Latency**: Agentic RAG 34.46s (Baseline 대비 5x). 실시간 서비스에는 부적합.

---

## 10. Phase 2 의제

1. **cross-modal alignment**: 텍스트 + 이미지 통합 검색
2. **image-required 질문 해결**: Q21-Q23 (화살표 방향, 부품 위치 등)
3. **latency 최적화**: 캐싱, 병렬화, 모델 경량화 검토

---

## 11. 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-06-08 | 초안 작성 (P0 완료 기준 템플릿) |
