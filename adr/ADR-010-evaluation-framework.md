# ADR-010: Week 7 평가 프레임워크 선정

## 상황

Phase 1 (Advanced RAG) 마무리 전 체계적인 평가 프레임워크가 필요함.

- 6주차 Agentic RAG 완성 (Top-1 95.7%, Latency 34.46s)
- 8주차 코드 freeze 전 "무엇을 측정해야 하나" 결정 필요
- 기존 평가 셋 (v2, 23문항)은 q_type 라벨 없이 단순 정확도만 측정
- RAGAS 4지표만으로는 도메인 특성 (out_of_scope 거절, 출처 정확성) 평가 불가

## 고려한 선택지

### 1. RAGAS 4지표 only
- Faithfulness, Answer Relevancy, Context Precision, Context Recall
- 장점: 업계 표준, 자동화 용이
- 단점: out_of_scope 거절 평가 불가, citation 검증 불가

### 2. RAGAS + 도메인 메트릭
- RAGAS 4지표 + Refusal Accuracy + Citation Accuracy
- 장점: 가전 매뉴얼 도메인 특성 반영
- 단점: 수동 라벨링 공수 (ground_truth, reference_context)

### 3. Pairwise LLM-as-judge only
- Baseline vs Agentic 직접 비교
- 장점: 상대적 우열 판단 명확
- 단점: 절대적 품질 수치화 어려움, self-preference bias

### 4. Human evaluation only
- 장점: 가장 정확
- 단점: 개인 프로젝트로 N=1 평가자, 시간 비용 높음

## 최종 결정

**RAGAS 4지표 + 도메인 메트릭 + Pairwise judge 조합**

| 평가 유형 | 지표 | 용도 |
|-----------|------|------|
| 자동 평가 | RAGAS 4지표 | 전체 품질 정량화 |
| 도메인 평가 | Refusal Accuracy | out_of_scope 질문 거절 정확도 |
| 도메인 평가 | Citation Accuracy | 출처 (모델명+페이지) 정확도 |
| 비교 평가 | Pairwise LLM judge | Baseline vs Agentic 상대 비교 |

## 이유

### RAGAS만으로 부족한 이유

1. **out_of_scope 처리 미반영**: RAGAS는 "답변이 context에 충실한가"만 평가. "답하지 말아야 할 질문에 올바르게 거절했는가"는 평가 불가.

2. **Citation 검증 불가**: 가전 매뉴얼 RAG에서 "어느 모델 몇 페이지"라는 출처가 핵심. RAGAS는 이를 측정하지 않음.

3. **질문 유형별 분석 부재**: factual vs multi_hop vs comparison 등 q_type별 성능 차이 분석 필요.

### 도메인 메트릭 정의

**Refusal Accuracy**
- TP: out_of_scope 질문에 올바르게 "확인할 수 없습니다" 답변
- FP: 답해야 하는 질문에 잘못 거절
- 거절 패턴: "제공된 문서에서 확인할 수 없습니다."

**Citation Accuracy**
- 답변에 명시한 (model_name, page) 출처가 reference_context와 일치하는가
- 페이지 ±2 범위 허용 (도메인 휴리스틱: 관련 내용이 인접 페이지에 걸쳐 있을 수 있음)

### Pairwise judge 필요성

- RAGAS 점수만으로는 "어느 시스템이 더 나은가" 직관적 판단 어려움
- Baseline (Hybrid+Rerank) vs Agentic (LangGraph) 직접 비교로 Agentic의 가치 입증
- q_type별로 승/패/동점 집계하여 "Agentic이 효과적인 질문 유형" 가설 검증

## Trade-off

| 항목 | 비용 | 이득 |
|------|------|------|
| Golden Set 라벨링 | 35문항 수동 라벨링 (ground_truth, reference_context) | q_type별 분석 가능, Citation Accuracy 측정 가능 |
| Judge LLM 비용 | gpt-4o-mini 사용 (RAGAS + Pairwise) | 비용 절감, 단 self-preference 한계 인지 |
| 복합 평가 체계 | 해석 복잡성 증가 | 도메인 특성 반영한 정확한 평가 |

## 향후 계획

### 8주차 연결 (Freeze 전 완료 사항)

1. **Golden Set v1 확정**: 35문항, q_type 라벨 완료
2. **Baseline/Agentic 모두 평가 실행**: RAGAS + Refusal + Citation
3. **Pairwise 비교 결과 문서화**: q_type별 승패 기록
4. **RAGAS 한계 사례 2건 발굴**: "점수 높은데 답변 나쁜 경우", "점수 낮은데 답변 괜찮은 경우"
5. **Known Limitations 정리**: 8주차 README에 Phase 1 한계로 명시

### Phase 2 연결

- modality_label (text-only, image-required) 라벨 활용
- Phase 1: text-only만으로 평가
- Phase 2: image-required 질문까지 평가 범위 확장
- "텍스트만으로는 부족" → "cross-modal로 확장" 서사 구체화

## 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-06-08 | 초안 작성 |
