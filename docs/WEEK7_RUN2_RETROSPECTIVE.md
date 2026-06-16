# Week 7 Retrospective — Run 2 (Rework: Cross-modal 측정 프레임 + Citation 인프라 수정)

> **작성일**: 2026-06-16
> **관계**: RUN1(첫 스프린트, `WEEK7_RUN1_RETROSPECTIVE.md`)의 미완(§4~9 TBD)과 한계를 받아, 평가를 정직하게 닫고 **cross-modal을 측정할 수 있는 형태로 재설계**한 2차 스프린트의 회고.
> **이 회고의 핵심**: (1) Citation 0%·AR 0.29의 진짜 원인을 디버그로 규명·수정, (2) image-required로 cross-modal 기준선을 정량화.
> **측정 분모 규약(공통):** Golden Set v2 = **41문항**. RAGAS·Citation은 **answerable 37**(out_of_scope 4 제외). Refusal은 별도. image-required 헤드라인은 **8** 기준. 표마다 n 명기.

---

## 1. RUN1 대비 무엇이 바뀌었나

- **Golden Set v1(35) → v2(41).** image-required를 "그림 없이는 본문 텍스트로 답이 안 되는" 질문으로 재정의·신설(8건), 타입 균등화(위치2/배치2/방향2/아이콘2). 구 Q21~23(텍스트로도 답 가능)은 폐기/흡수.
- **평가 코드:** 골든셋 CSV 로더 + `modality_label` 스키마 통일, Refusal/Citation 구현, modality별 분해.
- **Citation 인프라 수정:** RUN1에서 Citation이 0%였던 원인을 디버그로 규명(§3) 후 수정.
- **보고 방식:** refusal이 평균을 왜곡하므로 RAGAS를 overall/answered-only/refused-only로 분할 보고.

---

## 2. Golden Set v2 통계

| 항목 | 값 |
|------|-----|
| 총 문항 | 41 |
| q_type | factual 27 / comparison 4 / multi_hop 4 / out_of_scope 4 / safety 2 |
| modality | text-only 27 / image-helpful 6 / **image-required 8** |

image-required 8건은 도면 렌더 대조로 "본문에 답 없음" 검증 완료. `reference_context = "model p.N (도면/섹션명)"`, `figure_ref = "model p.N fig:figname"` 규약.

---

## 3. 디버그 — 두 이상치의 진짜 원인

### 3.1 Citation 0% — regex 정상, 원인은 page 정보 부재

- `_CITATION_RE`는 실제 답변 출처 포맷 3종을 모두 매칭 → regex 정상.
- 원인: (가) 답변 프롬프트에 retrieved chunk의 page가 안 흘러감 + (나) C3 스토어가 `by_page=False`로 빌드돼 청크 page가 사실상 무력화.
- **수정:** C3를 `by_page=True`로 재빌드(page 메타 복원) + 답변 프롬프트에 chunk별 `[source p.N]` 태그 주입(baseline `run_rag` 및 agentic generate 노드 둘 다).

### 3.2 AR 0.29 — RAGAS 결함 아님, refusal 평균 효과

- answerable 37 중 refused 8 / answered 29. refused는 AR=0이라 산술 평균을 끌어내림.
- 분리하면 answered-only Faithfulness 0.87·CP 0.94로 retrieval·grounding 양호.

---

## 4. RAGAS 4지표 (Baseline, n=37) — 3분할 보고

| 지표 | overall (37) | answered-only (29) | refused-only (8) |
|------|------|------|------|
| Faithfulness | 0.737 | **0.872** | 0.250 |
| Answer Relevancy | 0.290 | 0.370 | 0.000 |
| Context Precision | 0.847 | **0.939** | 0.515 |
| Context Recall | 0.658 | 0.741 | 0.354 |

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---------------|--------------|------------------|-------------------|----------------|
| Baseline (Hybrid+Rerank) | 0.737 / **0.872** | 0.290 / 0.370 | 0.847 / **0.939** | 0.658 / 0.741 |
| Agentic (LangGraph) | 측정 대기 (P1) | 측정 대기 | 측정 대기 | 측정 대기 |

**해석:** overall이 낮아 보이는 주원인은 refused 8건의 0점(특히 AR) — RAGAS 자체 결함이 아닌 평균 효과. baseline은 답한 케이스에선 정직하게 잘 답함(F 0.87·CP 0.94). Agentic은 미측정(답변/contexts는 pkl에 있어 동일 하네스로 비교 가능).

---

## 5. RAGAS 한계 직접 확인

### 5.1 사례 1 — 점수 높은데 결함 (Faithfulness가 출처 페이지를 못 봄)

answered-only Faithfulness 0.87로 높지만, 같은 답변들의 Citation page 정확도는 **18.9%**(§6.2). 답이 주어진 context엔 충실하나(=Faithfulness 만족) 출처 페이지가 틀려도 Faithfulness는 감지 못 함. → 필요 보완: Citation/Source-page Accuracy(도입함).

### 5.2 사례 2 — 점수 낮은데 행동은 합리적 (refusal·매뉴얼식 답변 페널티)

(a) refused 8건 AR=0 — 근거 부족/답 불가에 거절한 합리적 행동이 0점. (b) answered-only AR 0.37 — 매뉴얼식 다중 안내/길이에 RAGAS judge 보수적(judge×도메인 mismatch). → 필요 보완: refusal-aware 평가 + Evidence Coverage.

### 5.3 요약

| 한계 유형 | 발생 | 원인 가설 | 필요 보완 |
|-----------|------|-----------|-----------|
| High score, 결함 | F 0.87 ↔ Citation page 18.9% | Faithfulness가 페이지 안 봄 | Citation/Source-page Accuracy |
| Low score, 합리적 | refused 8 + 매뉴얼식 답변 | 거절=0 평균효과 + judge 보수성 | refusal-aware + Evidence Coverage |

---

## 6. 도메인 특화 메트릭

### 6.1 Refusal Accuracy

| 지표 | 값 |
|------|-----|
| Over-refusal (FP, answerable 37 중) | **8/37 = 21.6%** |
| 올바른 거절 (TP, out_of_scope 4) | 측정 대기 (out_of_scope 전용 집계 필요) |

FP(오거절) 사례: comparison "유선/무선 장단점", "AS181DAW vs AS281DAW 차이" — 답 가능한데 거절. image-required 일부도 거절(텍스트만으론 답 못 함 → Phase 2 동기). 거절 패턴: "제공된 문서에서 확인할 수 없습니다."

### 6.2 Citation Accuracy (인프라 수정 후)

| 지표 | 값 | 의미 |
|------|-----|------|
| Citation Accuracy (model AND page±2) | **0% → 18.9%** (7/37) | retrieval 정확도와 결합된 lower-bound |
| Model match (카테고리) | **34/37 = 91.9%** | ✅ 인프라 정상 — 올바른 제품 청크 받아 정직히 인용 |
| Page match (±2) | 7/37 = 18.9% | ⚠ retrieval이 정답 페이지 근방을 못 가져옴 |
| No citation | 1/37 | refusal 추정 |

- **핵심:** 인프라(by_page=True + prompt tag)는 model match 91.9%로 정상. 잔여 격차는 **retrieval 정확도 한계.**
- q_type별 Citation: factual 18.5%(27) / comparison 50.0%(4) / multi_hop 0.0%(4) / safety 0.0%(2).
  - multi_hop 0%: 다중 페이지 참조 필요한데 단일 청크로 부족. safety 0%: 안전주의 텍스트가 앞쪽에 분산. comparison 50%: 단순 카테고리 비교는 잘 잡음.
- 4주차 메타데이터(model, page)가 회수되는 검증 지점. "eval 인프라 버그를 측정으로 발견·수정"한 사례.

---

## 7. Pairwise LLM-as-judge (P1)

> Agentic RAGAS 측정 후 진행. 현재 **미측정**.

| 비교 | Baseline 승 | Agentic 승 | 동점 |
|------|-------------|------------|------|
| (전 q_type) | 측정 대기 | 측정 대기 | 측정 대기 |

---

## 8. 질문 유형별 분석 (Baseline)

| q_type | n | answered-only AR | Citation (model+page) | 관찰 |
|--------|---|------|------|------|
| factual | 27 | 0.360 | 18.5% | 검색 양호하나 page 정밀도 낮음 |
| comparison | 4 | 0.737* | 50.0% | *answered n=1, 무시. 단순 비교는 인용 잘 됨 |
| multi_hop | 4 | 0.400 | 0.0% | 다중 페이지 참조 실패 — 단일 청크 한계 |
| safety | 2 | 0.236 | 0.0% | 안전 텍스트 분산 → retrieval 어려움 |
| out_of_scope | 4 | N/A | N/A | RAGAS 분모 제외, Refusal로 평가 |

**freeze 전 최우선 약점:** retrieval page 정밀도(multi-page/분산 텍스트). Citation 18.9%의 병목. Agentic 효과(out_of_scope·multi_hop 우위 가설)는 P1 측정으로 검증 필요.

---

## 9. Modality별 분석 — Cross-modal Baseline (헤드라인)

Phase 2 cross-modal 효과를 잴 **기준선.** text-only baseline이 image-required에서 얼마나 실패하는가.

| modality | n | Model match | Page match (Citation) | 관찰 |
|----------|---|------|------|------|
| image-required | 8 | **4/8 = 50%** | 2/8 = 25% (IR-W1, IR-V3) | overall model 91.9%보다 현저히 낮음 |

- **핵심 발견:** 도면이 핵심인 질문에서 text-only retrieval은 제품(simple/complex)조차 절반만 맞춤(50% vs 91.9%). image-anchor 질문에서 simple/complex 혼동이 심함.
- → **Phase 2 cross-modal motivation의 정량 근거.** cross-modal이 이 gap(model 50%·page 25%)을 좁히는지가 Phase 2 측정 목표.

---

## 10. Known Limitations (8주차 README 연결)

1. **Retrieval page 정밀도**: Citation 18.9% 병목. multi_hop·safety 0%. 인프라 정상(model 91.9%), 병목은 retrieval.
2. **image-required**: text-only retrieval이 model 50%·page 25%로 붕괴 → Phase 2 동기.
3. **LIM-002(도면 추출)**: 6개 매뉴얼 도면이 거의 전부 vector graphics(raster XObject 추출 불가). 추출 경로는 페이지/영역 rasterize → vision으로 확정.
4. **safety 약함**: 가전 도메인 특성.
5. **over-refusal**: 답 가능 질문 8/37(21.6%) 오거절.
6. **Latency**: Agentic 34.46s/q (Baseline 6.73s/q 대비 ≈5.1×). Citation 재측정 런 30.03s/q는 MPS warmup 변동으로 헤드라인 제외.

---

## 11. Phase 2 의제

1. cross-modal alignment — §9 기준선(model 50%/page 25%) 대비 효과 측정.
2. LIM-002 해결 — 페이지 rasterize → vision, `image_ids`·`figure_ref` 결합.
3. retrieval page 정밀도 — multi_hop multi-chunk, 분산 텍스트(safety) 대응.
4. latency 최적화 — 캐싱·병렬화·reranker int8.

---

## 12. 측정 대기 (P1, 다음 세션)

- Agentic RAGAS 4지표 + Pairwise judge.
- Refusal Accuracy의 out_of_scope TP(올바른 거절률).
- (참고) ADR-010 보강 + ADR-011 등재 + PROJECT_CONTEXT 갱신.

---

## 13. 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-06-16 | Run 2 회고 신규. v2(41) 통계, 디버그(Citation/AR) 원인·수정, Baseline RAGAS 3분할, RAGAS 한계 2사례, Refusal(FP 21.6%)+Citation(0→18.9%, model 91.9%), q_type·Modality 헤드라인(IR model 4/8·page 2/8), LIM-002 vector 확정. Agentic RAGAS·Pairwise·out_of_scope TP는 측정 대기. |
