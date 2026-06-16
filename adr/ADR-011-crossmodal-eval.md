# ADR-011: 크로스모달 평가 프레임 및 Run-2 평가 보정

> 관련: ADR-010(평가 프레임워크 — 기본 RAGAS+도메인 메트릭). 본 ADR은 7주차 2차 스프린트(rework)에서 평가를 cross-modal 측정 가능 형태로 확장하고, Run-1에서 드러난 측정 결함을 보정한 결정을 기록한다.
> 근거 데이터: `docs/WEEK7_RUN2_RETROSPECTIVE.md`.

## 상황

- Run-1 평가는 Citation 0%, AnswerRelevancy 0.29로 나왔고, 원인 미규명 상태였다.
- Golden Set의 image-required 질문(구 Q21~23)이 텍스트로도 답이 되어 cross-modal 변별력이 없었다.
- 평가 코드가 단일 평균만 보고해 refusal이 평균을 왜곡했고, RAGAS만으로는 출처 페이지 정확성·거절 품질을 측정하지 못했다.
- Phase 2(cross-modal)로 무게를 옮기려면, 평가가 "텍스트만으로는 못 푸는 지점"을 수치로 보여줘야 한다.

## 고려한 선택지

1. **Run-1 수치를 그대로 freeze하고 Phase 2에서 평가를 새로 짠다.** — 빠르지만 cross-modal 효과를 잴 기준선이 없고, Citation 0%가 영구 오류로 남는다.
2. **평가 인프라를 cross-modal 확장 가능하게 보정하면서 닫는다.** — modality 라벨·분해 메트릭·기준선·전방호환 훅을 지금 만들고, Citation/AR 결함을 디버그로 규명·수정.
3. (Citation 한정) 0%를 "측정 불가"로 기록하고 fix를 Phase 2로 이월. — P0 도메인 메트릭을 미달로 남김.

## 최종 결정

선택지 **2** 채택. 세부:

1. **modality 3분류**(text-only / image-helpful / image-required)로 골든셋 라벨링. image-required 8건은 "본문 텍스트에 답 없음(렌더 대조 검증)"으로 신설하고 **타입 균등화**(위치2 / 배치2 / 방향2 / 아이콘2).
2. **모든 메트릭을 modality·q_type별로 분해**하고, **text-only baseline의 image-required 성적(model match 4/8=50%, page match 2/8=25%)을 Phase 2 cross-modal 기준선**으로 채택.
3. **cross-modal 전방호환만 열어둠(구현 X):** 평가 하네스에 `context_provider` 주입점 + 골든셋 `figure_ref` 컬럼. Phase 2에서 retrieve만 교체하면 동일 메트릭으로 텍스트-only vs cross-modal 비교.
4. **Citation 인프라 수정 채택:** C3 스토어를 `by_page=True`로 재빌드(페이지 메타 복원) + 답변 프롬프트에 chunk별 `[source p.N]` 태그 주입(baseline·agentic 동일). Citation은 retrieval 정확도와 결합된 **lower-bound 메트릭**으로 해석.
5. **RAGAS 3분할 보고**(overall / answered-only / refused-only)로 refusal 평균 왜곡 방지.
6. **LIM-002 근거로 Phase 2 도면 추출 경로 확정:** 매뉴얼 도면이 거의 전부 vector graphics → raster XObject 추출 불가 → **페이지/영역 rasterize → vision 처리**.

## 이유

- **Citation 0%는 측정 인프라 버그였다.** 수정 후 model match 91.9%로 인프라 정상 확인, 잔여 18.9%는 retrieval page 정밀도 한계로 분리됨 → "0% 실패"가 아니라 "인프라 정상 + retrieval 과제"로 정확히 진단. (eval 인프라 자체를 검증한 사례.)
- **AR 0.29는 RAGAS 결함이 아니라 refusal 평균 효과.** 분리 시 Faithfulness 0.87·CP 0.94 → 답한 케이스 품질은 양호. 단일 평균은 오해를 부르므로 분할 보고가 정직.
- **image-required model 50%** 발견이 Phase 2 동기의 정량 근거. 텍스트만으로는 도면 질문의 retrieval 자체가 무너진다.
- 전방호환 훅만 두고 구현은 미루는 것은 단순성 원칙(불필요한 미사용 코드 금지)과 "측정 가능성 먼저" 전략에 부합.

## 향후 계획

- **Phase 2:** LIM-002 해결(rasterize→vision), 이미지 임베딩 방식 결정(CLIP류 vs Vision LLM), `context_provider`를 cross-modal로 교체 → 본 ADR의 기준선(model 50%/page 25%) 대비 효과 측정.
- retrieval page 정밀도(multi_hop multi-chunk, 분산 safety 텍스트)는 Citation 병목 해소 과제로 이월.
