# WEEK 11 TASKS — 멀티모달 Agent 구현 (Option A) + 캐리된 크로스모달 마무리

> **트랙 = Option A (Multimodal Agent).** 6주차에서 이미 단일 Agentic RAG(LangGraph)를 했으므로, 11주차는 **그 위에 tool 조합 + fallback을 얹어 '실패에 대응하며 끝까지 답하는' 멀티모달 Agent**로 확장한다.
> **핵심 원칙(과제):** tool 하나를 완벽히 만드는 게 아니라 **모듈화·입출력 스키마·실패 감지/대응·흐름 기록**이 목적. OCR/Vision 정확도는 완벽하지 않아도 된다.
> **선행/재사용:** Week 9 결과(`docs/week9_evaluation.md`, `WEEK9_RETROSPECTIVE.md`). RAG Tool의 검색 매체 = **캡션→텍스트 RAG(`src/mm_retrieval.py`)** — Week 9에서 CLIP을 기각하고 채택한 방식. 경로: `w9/`·`w11/`·`src/`·`docs/`.
> 역할 분담 유지(전략=명세, Claude Code=구현).

---

## 0. 이번 주 구성 (두 워크스트림)

| 워크스트림 | 내용 | 우선순위 |
|---|---|---|
| **W1 — 11주차 과제(멀티모달 Agent)** | tools ≥3 + LangGraph + fallback + 시나리오 ≥3 + 아키텍처 문서 | **P0** (이번 주 마감 산출물) |
| **W2 — 캐리된 Phase-2 마무리** | region-crop 캡션 · golden set v3 · 잡무 | **P1** (프로젝트 본선, 시간 되는 만큼) |

**연결 고리(중요):** W1의 fallback을 "캡션 RAG 실패 → 이미지를 직접 읽는 Vision tool로 에스컬레이션"으로 설계하면, 이것이 곧 Week 9가 지목한 **cross-modal 병목(아이콘 모양·공간 위치·방향)을 질의 시점에 우회**하는 경로다. 즉 과제의 fallback 요구와 우리 프로젝트의 다음 단계가 같은 지점에서 만난다. W2의 region-crop은 이 Vision tool의 **인덱스측(사전) 강화**이고, W1의 Vision tool은 **질의측(즉시) 읽기**다 — 서로 보완.

> **스코프 스왑 옵션:** region-crop(W2)을 프로젝트 본선으로 먼저 밀고 싶으면 P0/P1을 바꿔도 된다. 단 11주차 과제(W1)가 이번 주 마감 산출물이라 기본은 W1=P0.

---

## 1. 우선순위 트랙

| 순위 | 작업 | 산출물 | 절 |
|---|---|---|---|
| **P0** | Tool 3개+ 설계·구현(책임분리·스키마·에러핸들링) | `src/agent_tools.py`, `w11/week11_multimodal_agent.ipynb` | §2 |
| **P0** | LangGraph Agent 구성 + fallback ≥1 | `src/mm_agent.py` | §3 |
| **P0** | 멀티스텝 시나리오 ≥3 실행 + 흐름 기록 | `data/week11_scenarios.json` | §4 |
| **P0** | 아키텍처 문서 | `docs/week11_architecture.md` | §5 |
| **P1** | region-crop 캡션(구조화 JSON) | `src/region_caption.py`, `data/week11_captions_crop.json` | §6 |
| **P1** | golden set v3 (GT 결함 정비) | `data/eval/golden_set_v3.csv` | §7 |
| **P2** | 잡무: w9 노트북 재작성, 커밋 확인 | — | §8 |

> P0가 이번 주 필수 최소 기준(§9). P1은 프로젝트 본선이나 과제 마감엔 선택.

---

## 2. Tool 설계·구현 (P0)

> 산출물: `src/agent_tools.py` (재사용 가능하게 모듈로) + 노트북에서 시연. 각 tool = **단일 책임 · 명확한 입출력 스키마 · 실패 시 명시적 신호**.

| tool | 입력 | 출력(스키마) | 구현 매핑 |
|---|---|---|---|
| `ocr_tool` | `image_path` | `{text: str, confidence: float, ok: bool}` | pytesseract 등. **주의(도메인 진실):** 우리 매뉴얼 도면은 vector line-art라 OCR이 약함(LIM-002와 정합) → 낮은 confidence는 §3 Vision 에스컬레이션의 트리거가 된다 |
| `image_analysis_tool` | `image_path, question` | `{image_summary: str, confidence: float, ok: bool}` | **gpt-4o vision**(질의측 읽기). `bbox` 인자 옵션으로 두어 §6 crop과 결합 가능 |
| `rag_search_tool` | `query` | `{docs: [...], scores: [...], caption_hit: bool, ok: bool}` | **`src/mm_retrieval.py`**(캡션→텍스트, 모달리티-aware). Week 9 채택 매체 |
| `answer_generation_tool` | `context, question` | `{answer: str, is_grounded: bool}` | 근거 부족 시 `is_grounded=False` + 거절 문구(기존 REFUSAL 패턴 재사용) |

원칙: OCR tool은 "텍스트 추출"만, 그 안에서 검색·답변 금지. 각 tool은 실패를 **예외 또는 `ok=False`**로 밖에 알린다(삼키지 않음).

---

## 3. LangGraph Agent + Fallback (P0)

> 산출물: `src/mm_agent.py` (`AgentState`, `build_mm_agent`, `run_mm_agent`).

### 3.1 State (공유 상태)
```
AgentState = {
  question, input_type,            # "text" | "image" | "pdf"
  image_path,
  ocr_text, image_summary,         # tool 중간 결과
  docs, scores, caption_hit,
  evidence, is_grounded, confidence,
  answer, refused,
  route_history, fallback_history  # 흐름·실패 기록(문서화의 근거)
}
```

### 3.2 라우팅 (입력/질문 유형별)
- `input_type=image/pdf` → `ocr_tool` → (실패 감지) → `rag_search_tool` → `answer_generation_tool`
- `input_type=text` → `rag_search_tool` → (caption_hit·근거 판단) → `answer_generation_tool`

### 3.3 Fallback (≥1 필수 — **권장 3개**, 프로젝트 의미와 연결)
1. **OCR 실패/저신뢰 → `image_analysis_tool`(Vision) 우회.** line-art에서 OCR이 약하다는 도메인 진실을 그대로 시연.
2. **캡션 RAG 근거 부족(caption_hit=F 또는 저score) → `image_analysis_tool`로 페이지 이미지 직접 읽기.** ← **Week 9 병목(아이콘·위치·방향)의 질의측 우회**. 이게 이번 주 fallback의 하이라이트.
3. **근거 부족(is_grounded=F) → 거절**("제공된 문서에서 확인할 수 없습니다"). "실패 시 아무 답이나 생성하지 않는다"가 핵심.
- (선택) 검색 결과 부족 → query 재작성 후 1회 재검색.

모든 fallback 발동은 `fallback_history`에 기록(문서 §5의 근거).

---

## 4. 멀티스텝 시나리오 (P0, ≥3, 엣지 1개 이상)

> 산출물: 실행 로그 `data/week11_scenarios.json` + 표는 `docs/week11_architecture.md`.

1. **이미지 페이지 → OCR → 검색 → 답변** (happy path).
2. **텍스트 질문 → 캡션 RAG(저신뢰) → Vision 에스컬레이션 → 답변** — 예: IR-A3(공기제균 아이콘) 류. 캡션이 못 담은 정보를 질의측 Vision이 읽는 케이스. **이번 주의 핵심 시나리오.**
3. **스캔/이미지-only 페이지 → OCR 실패 → Vision 우회 → 검색 → 근거 부족 → 거절** (엣지: 실패 누적 상황에서 안전 거절).

각 시나리오는 아래 표로:

| 시나리오 | 기대 tool 흐름 | 실제 tool 흐름 | 성공 | fallback 발동 | 비고 |
|---|---|---|---|---|---|

---

## 5. 아키텍처 문서 (P0)

> 산출물: `docs/week11_architecture.md`. 포함: **Tool 구성(책임·입출력 스키마) / 라우팅 로직 / State 흐름 / fallback 전략 / 시나리오 결과 표 / 회고(잘 된 것·가장 불안정했던 지점·12주차 개선점).**
> (선택) 실행 trace를 mermaid로 1장.

---

## 6. [P1] region-crop 캡션 — 캐리된 병목 돌파

> 산출물: `src/region_caption.py`, `data/week11_captions_crop.json`. Week 9 §6/§retrospective의 **유일한 돌파구**.

- 페이지에서 **영역(bbox)을 잘라 아이콘/부품이 프레임을 채우게** 한 뒤 gpt-4o로 캡션 → **구조화 JSON**(`{region_bbox, position(상/하/좌/우/앞/뒤), label(원문 전사), shape(빗금/사선/화살표 방향 등)}`).
- 영역 검출: vector라 XObject bbox가 없으므로 **레이아웃 휴리스틱 또는 pymupdf drawings 클러스터링**으로 후보 영역 → (MVP는 IR8 관련 페이지 우선).
- 직접 겨냥: **IR-A1/A2(위치), IR-A3/V3(아이콘 모양), IR-V2(방향)** — 6개 잔여 실패 중 5개.
- 결합: §2 `image_analysis_tool`의 `bbox` 인자로 crop 재사용. 완성 시 mm 스토어에 region-caption 청크로 추가(전방호환).

---

## 7. [P1] golden set v3 — 대조군 GT 정비

> 산출물: `data/eval/golden_set_v3.csv`. Week 9 엄격 judge가 노출한 결함 수리(§week9_evaluation §6).

- Q01: GT가 모델답변보다 덜 정확 → 매뉴얼 기준 재작성.
- Q11: GT(3.5시간) ↔ 매뉴얼(4시간) 모순 → 매뉴얼값으로 수정.
- Q19: 모델 미특정 → simple/complex 정답이 갈림 → 질문에 모델/카테고리 명시.
- Q06: ref(air_C p.18 '필터 교체하기')가 실제 p18(각 부 명칭)과 불일치 의심 → 원본 대조 후 수정.
- (Q08 ref p10→p24는 Week 9에서 수리 완료.)
- 용도: §4 시나리오 채점의 기준선. 수정 이력은 notes에 남김.

---

## 8. [P2] 잡무

- **w9 노트북 재작성:** 옛 단일콜 흐름 반영 → 현행 `run_comparison()`(스테이지·체크포인트) 흐름으로.
- **커밋 확인:** Week 9~10 산출물이 `3891615`(week9-10 커밋)에 실제로 들어갔는지 `git log --stat`로 검증(현재 `git status` clean).

---

## 9. 최소 기준 + 검증

**최소 기준:** tool 3개+ · Agent 1개 · 시나리오 3개+ (엣지 1개+) · fallback 1개+ 동작 · 흐름 기록 · 아키텍처 문서.

**검증(P0):**
- 각 tool을 단독 호출해 입출력 스키마·실패 신호(`ok=False`/예외)가 스키마대로 나오는지.
- fallback이 **의도한 트리거에서 실제로 발동**하는지(로그로 확인) — 특히 시나리오 2의 캡션RAG→Vision 에스컬레이션.
- 실패 시 답을 지어내지 않고 거절/우회로 가는지(hallucination 억제).
- 시나리오 표의 "실제 흐름"이 `route_history`/`fallback_history`와 일치하는지.
- **다음(12주차):** region-caption 완성 → Week 9 IR8 재평가(캡션 병목 해소 여부) → 그 결과로 **ADR-012(크로스모달 MVP 결론) 작성**(현재까지 의도적 연기). agent도 12주차 개선점 반영.
- 이론을 우리 코드에 매핑 — Chain↔Agent = 6주차 agentic RAG의 조건부 라우팅 / Tool Use = §2 tool 4종의 책임분리·스키마 / 추론 패턴(ReAct·Reflection) = §3 라우팅·근거판단 / 실패·Fallback = §3 에스컬레이션·거절(오차 전파 대비).

---

## 10. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-07-13 | 11주차 신규. Option A 멀티모달 Agent(tools+LangGraph+fallback+시나리오+아키텍처 문서)를 P0로, 캐리된 Phase-2(region-crop 캡션·golden v3·잡무)를 P1로 통합. fallback을 '캡션RAG→Vision 에스컬레이션'으로 설계해 과제 요구와 Week 9 병목 우회를 연결. RAG tool = `src/mm_retrieval.py`(캡션→텍스트). |
