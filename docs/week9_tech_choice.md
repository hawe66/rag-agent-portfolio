# 9주차 기술 선택 — 멀티모달 RAG (Phase 2 착수)

> 본 프로젝트의 본 과제(cross-modal alignment)가 시작되는 주차. 앞선 7주는 전부 텍스트 인프라였고, ADR-011이 Phase 2 진입로(추출 경로·기준선·전방호환 훅)를 이미 깔아뒀다.

## 선택한 기술

**멀티모달 RAG / 멀티모달 에이전트** (옵션 A). 멀티 에이전트(옵션 B)는 6주차 Agentic RAG(LangGraph)로 이미 다뤘고 본 과제(그림-텍스트 정렬)와 무관해 기각.

**구현 형태:** 단일 방식을 고르지 않고, **두 cross-modal 파이프라인을 비교하는 ablation 실험**으로 간다.
- **C1 (메인) — Vision-LLM 캡션 → 텍스트 RAG 병합**
- **C2 (비교군) — CLIP류 이미지 임베딩 검색**
- **C0 (기준선) — 기존 text-only RAG** (ADR-011 측정치: image-required model match 50% / page match 25%)

## 선택 이유

- **도메인 연결:** LG 가전 매뉴얼의 지시 정보(부품 위치, 끼움/회전 방향, 제어판 아이콘)는 도면에만 있고 본문 텍스트엔 없다(7주차 image-required 8건으로 정량 확인). text-only RAG는 이 질문에서 model 50%·page 25%로 무너진다 → cross-modal이 필요한 명확한 근거가 이미 측정돼 있음.
- **기존 RAG repo에 더하는 가치:** retrieve/context 단계만 cross-modal로 교체하면 기존 Chroma·Hybrid·reranker·agent·평가 하네스를 그대로 쓰면서 "텍스트가 놓친 시각 정보"를 회수. ADR-011의 `context_provider` 훅과 `figure_ref` 컬럼이 이를 위해 미리 열려 있음.
- **포트폴리오 관점:** 단순히 "멀티모달 붙였다"가 아니라, **두 접근(임베딩 검색 vs VLM 추론)을 통제 실험으로 비교해 도메인에 맞는 선택을 데이터로 정당화**하는 서사. "측정하는 능력"을 한 번 더 보여줌.

## 현재 RAG와의 차이

- **기존 RAG가 못하던 것:** 도면에만 있는 위치/방향/아이콘 정보를 답하지 못함. text-only retrieval은 image-required에서 제품(simple/complex)조차 절반만 맞춤(model 50%).
- **이 기술을 붙이면:** 페이지 도면의 시각 정보를 (C1) 캡션 텍스트로 변환해 검색·인용하거나, (C2) 이미지 임베딩으로 직접 검색하고 VLM이 읽어 답할 수 있게 됨. image-required 질문의 retrieval·answer 정확도를 baseline 대비 끌어올리는 것이 목표.

## 선택 기술 핵심 개념 정리 (옵션 A)

1. **Vision-Language Model의 역할:** 이미지를 입력으로 받아 텍스트로 설명·추론. 본 프로젝트에선 (C1) 페이지 도면을 구조화된 캡션(위치/방향/아이콘/표)으로 변환, 또는 (late-fusion) 답변 시점에 페이지를 직접 읽어 답.
2. **이미지+텍스트 동시 처리:** 질문(텍스트) + 페이지(이미지)를 함께 프롬프트에 넣어, "이미지에서 확인 가능한 정보만" 답하고 보지 못한 건 추측하지 않게 제약(거절 로직과 연결).
3. **이미지 설명을 기존 RAG와 결합:** 캡션을 page·figure_ref 메타데이터를 단 "image-derived 청크"로 만들어 기존 텍스트 청크와 같은 인덱스에 넣음 → Hybrid+Rerank가 함께 검색, Citation 유지.
4. **text-only vs multimodal RAG 차이:** 입력(텍스트 only ↔ 텍스트+이미지), 처리(파싱 시 이미지 폐기 ↔ VLM/임베딩으로 시각정보 회수), 비용·latency(낮음 ↔ VLM 호출로 증가), 적합 도메인(텍스트 충분 ↔ 도면/표/스캔 정보가 본질).
5. **내 도메인에서 이미지 정보가 필요한 이유:** 가전 매뉴얼은 "그림=절차/공간배치, 텍스트=주의/조건"으로 분업. 조작 방향·부품 위치·제어판 아이콘은 그림에만 존재 → 텍스트 파싱 시 손실(LIM-002: 도면이 vector라 raster 추출도 불가, rasterize→vision 필요).

## MVP 구현 계획

**이번 주 최소 범위 (상세는 `WEEK9_TASKS.md`):**
- 6개 매뉴얼의 도면 보유 페이지를 **전체 페이지 rasterize**(95dpi 기준, audit서 가독성 확인).
- **C1**: gpt-4o-mini로 페이지 캡션 생성 → image-derived 청크로 cross-modal 스토어에 인덱싱 → Hybrid+Rerank 검색 → gpt-4o-mini 답변.
- **C2**: 멀티링궐 CLIP(`clip-ViT-B-32-multilingual-v1`)로 페이지 임베딩 → 이미지 인덱스 → 텍스트 질문으로 검색 → (late-fusion) 검색된 페이지를 gpt-4o-mini가 읽어 답변.
- **평가**: image-required 8건 + text-only 대조 일부에 대해 C0/C1/C2 비교(아래 실험 설계). 테스트 케이스 ≥3.

**사용 모델/프레임워크/도구:** gpt-4o-mini(vision, caption+answer), `clip-ViT-B-32-multilingual-v1`(sentence-transformers, C2 임베딩), PyMuPDF(rasterize), Chroma(인덱스), 기존 `src/agent.py`·`src/evaluation.py` 하네스(`context_provider` 훅).

**10주차 확장 방향:** 도면 영역 crop별 캡션, 표→markdown 구조화, late-fusion 정교화, (CLIP이 retrieval에서 가치를 보이면) 캡션+이미지 fusion, citation 연결 강화.

## 실험 설계 — "추론이 필요하면 VLM" 가설 검증

**가설:** 멀티모달 파싱에서 *세부를 읽고 추론*해야 하는 질문(우리 image-required: 위치/방향/아이콘)은 VLM(캡션 or 직접 읽기)이 필요하다. CLIP 임베딩은 "비슷한 그림 찾기"엔 쓸 수 있으나 디테일을 *읽지는* 못한다.

**공정성 원칙:** 결과는 정직하게. CLIP이 retrieval에서 선전하면 "CLIP=찾기 / VLM=읽기"의 분업 결론으로 기록(이게 더 정교한 결론). 가설을 강제하지 않는다.

**두 축으로 측정 (대상: image-required 8 + text-only 대조 일부):**

| 구성 | Retrieval(page-hit) | Answer(추론 필요) | 기대 |
|---|---|---|---|
| C0 text-only | 기준선(page 25%) | 낮음 | 도면 정보 부재 |
| C1 캡션→텍스트RAG | 캡션-텍스트 검색 | gpt-4o-mini(text) | **answer↑** (시각정보가 텍스트화됨) |
| C2 CLIP 검색 | 이미지 임베딩 검색 | (a) 검색만 / (b) +VLM 읽기 | retrieval은 가능, **(a) answer 불가 → (b) VLM 붙여야 답** |

- **결론 도출 경로:** C2(a)(CLIP 검색만)는 추론 질문의 answer를 못 맞춘다 → C2(b)(CLIP+VLM 읽기)로 올라가야 답이 된다. 즉 **검색 방식과 무관하게 "추론 단계엔 VLM이 필수"**가 드러남. C1은 그 VLM 추론을 인덱싱 시점에 끝내둔 형태 → 기존 텍스트 스택 재사용 + citation 유지라는 추가 이점.

## MVP 평가 (정성)

### 구현한 것
- [ ] 선택 기술(C1)의 핵심 패턴(rasterize→caption→index→retrieve→answer)을 구현했는가?
- [ ] 비교군(C2 CLIP)을 구현했는가?
- [ ] 기존 RAG repo(평가 하네스·agent)와 연결했는가?
- [ ] 테스트 케이스(≥3) + image-required 8 비교를 실행했는가?

### 정성 평가 (실행 후 작성)
- 이 기술이 기존 RAG에 더한 가치 (image-required answer/page-hit가 baseline 대비 얼마나?)
- 가장 어려웠던 점 (캡션 품질? CLIP 한국어·도면 OOD? late-fusion 비용?)
- 복잡도 대비 이득 (C1 vs C2의 ROI)
- 10주차 개선 방향

## 연결

- 근거/기준선: `adr/ADR-011-crossmodal-eval.md`, `docs/WEEK7_RUN2_RETROSPECTIVE.md` §9.
- 구현 명세: `docs/WEEK9_TASKS.md`.
- 실행 결과·정량 비교는 `docs/week9_evaluation.md`(또는 본 문서 평가 절)에 채운다.
