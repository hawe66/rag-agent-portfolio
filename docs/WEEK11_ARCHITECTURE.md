# Week 11 아키텍처 — 멀티모달 Agent (Option A)

> 명세: `docs/WEEK11_TASKS.md`. 구현: `src/agent_tools.py` · `src/mm_agent.py` · `src/week11_scenarios.py`.
> 실행 로그(원자료): `data/week11_scenarios.json`. 오프라인 검증: `tests/test_week11_agent.py` (14 tests).
> 시연: `w11/week11_multimodal_agent.ipynb`.

## 1. Tool 구성 (§2 — 단일 책임 · 스키마 · 실패 신호)

| tool | 입력 | 출력 스키마 | 실패 신호 | 구현 |
|---|---|---|---|---|
| `ocr_tool` | `image_path` | `{text, confidence(0~1), ok}` | `ok=False`+`error` (파일/엔진 오류). **빈 페이지는 `ok=True, conf=0.0`** — "못 읽음"은 결과이지 오류가 아님 | pytesseract kor+eng, word-level conf 평균 |
| `image_analysis_tool` | `image_path, question, bbox?` | `{image_summary, confidence, ok}` | `ok=False`+`error`. JSON 계약 위반 시 원문 보존+conf=0.5 (지불한 내용은 버리지 않음) | gpt-4o vision, detail=high. `bbox` 인자 = 10주차 region-crop 결합 지점 |
| `rag_search_tool` | `query` | `{docs, scores, caption_hit, ok}` | `ok=False`+`error`, 결과 0건도 `ok=False` | `ModalityAwareRetriever`(텍스트3+캡션2, Week 9 채택 매체) + cross-encoder 점수 |
| `answer_generation_tool` | `context, question` | `{answer, is_grounded, ok}` | `is_grounded=False` = 거절 계약 발동("제공된 문서에서 확인할 수 없습니다") | Week 7/9와 동일한 출처 태그·인용 계약 |

원칙 준수: OCR은 추출만, 검색·답변 금지. 무거운 리소스(리트리버·리랭커·LLM)는 `make_*` 팩토리로 주입(기존 `make_text_retriever` 스타일). 모든 실패는 `ok=False`로 표면화 — 삼키지 않음(오프라인 테스트로 고정).

## 2. State와 라우팅 (§3.1~3.2)

`MMAgentState`(TypedDict): `question, input_type, image_path` / 중간 결과 `ocr_text, image_summary, docs, scores, caption_hit` / 판정 `evidence, is_grounded, confidence, grade_result, vision_escalated` / 출력 `answer, refused` / 기록 `route_history, fallback_history`.

```
entry ─(input_type)─┬─ image/pdf → ocr ──┬─(사용 가능)────→ rag
                    │                    └─(F1)──────────→ vision_input → rag
                    └─ text ─────────────────────────────→ rag
rag ──(ok)→ grade ──┬─(충분)─────────────────────────────→ generate
    └─(실패)→ refuse └─(F2: 부족, 최초 1회)───────────────→ vision_escalate → generate
                     └─(부족, 소진)───────────────────────→ refuse (F3)
generate ──┬─(grounded)──────────────────────────────────→ END
           ├─(F2: ungrounded, 미에스컬레이션)─────────────→ vision_escalate → generate
           └─(ungrounded, 소진)──────────────────────────→ refuse (F3)
```

- OCR "사용 가능" = `ok ∧ conf ≥ 0.5 ∧ 인식 글자 ≥ 30`. 글자 수 조건은 실측에서 나옴: **252페이지 스캔 결과 거의-빈 도면 페이지가 conf 0.92에 글자 2자로 통과하는 함정** 확인 → conf 단독 트리거 기각.
- `vision_escalated` 플래그가 에스컬레이션을 1회로 제한 — 루프 없음(구조적으로 최대 노드 7회 방문).

## 3. Fallback 전략 (§3.3 — 3개 구현, 전부 실측 발동)

| # | 트리거 | 대응 | 프로젝트 의미 |
|---|---|---|---|
| **F1** | OCR 실패/저신뢰/저텍스트 | 입력 이미지를 gpt-4o가 직접 읽음 | LIM-002(vector line-art → OCR 무력)의 시연 |
| **F2** | `grade(not_relevant)` **또는** `generate(ungrounded)` | **검색된 페이지 이미지**를 질의 시점에 gpt-4o가 읽고 컨텍스트에 추가 | Week 9 병목(캡션이 아이콘·위치·방향을 못 담음)의 질의측 우회 시도 |
| **F3** | 에스컬레이션 후에도 근거 부족 | 명시적 거절 | "실패 시 아무 답이나 생성하지 않는다" — Week 6부터의 안전 계약 유지 |

**F2 트리거가 2중인 이유(1차 실행의 발견):** 최초 설계는 `grade(not_relevant)` 단독 트리거였다. 1차 실행에서 S2/S4 모두 grader가 캡션 컨텍스트를 "충분"으로 오판 → 에스컬레이션이 한 번도 발동하지 않고 generate가 거절함. 이는 Week 9 §6이 기록한 grader 관대함의 재현이다. 수정: **generate의 거절(`is_grounded=False`)을 두 번째 트리거로 추가** — 생성기가 실제로 답을 못 만들었다는 신호가 grader 판단보다 강건하다. 수정 후 전 시나리오에서 의도대로 발동.

## 4. 시나리오 결과 (§4 — 실행 로그 `data/week11_scenarios.json`)

| 시나리오 | 기대 흐름 | 실제 흐름 (route_history) | 성공 | fallback | 비고 |
|---|---|---|---|---|---|
| S1 이미지→OCR→검색→답변 | ocr→rag→grade→generate | 동일 | ✅ 정답+출처 | 0 | OCR conf 0.89, 필터 교체주기(6/12개월) 정확 |
| S2 IR-A1 위치 질문→F2 | rag→grade→…→vision_escalate→generate | rag→grade→generate→**F2**→generate→refuse | ✅ 흐름 / ❌ 답변(정직 거절) | F2, F3 | 아래 §5 분석 |
| S3 엣지: OCR-dead+매뉴얼 밖 질문 | ocr→vision_input→rag→grade→(…) | ocr→**F1**→rag→grade→**F2**→generate→**F3** refuse | ✅ 안전 거절 | 3개 전부 | 실패 누적 상황에서 지어내지 않음 |
| S4 IR-A3 아이콘 모양→F2 | rag→grade→…→vision_escalate→generate | rag→grade→generate→**F2**→generate→refuse | ✅ 흐름 / ❌ 답변(정직 거절) | F2, F3 | 아래 §5 분석 |

latency 27~39s (Week 6 agentic 27s와 동급 — vision 에스컬레이션 1회당 약 +8~10s).

## 5. 결과 해석 — 질의측 vision도 같은 상한에 부딪힌다 (정직 기록)

과제 기준(모듈화·스키마·실패 감지/대응·흐름 기록)은 전부 충족했고 fallback도 의도한 트리거에서 발동했다. 그러나 **답변 품질 관점에서 F2(질의측 full-page vision)는 Week 9 병목을 뚫지 못했다**:

- **S4 (아이콘 모양):** gpt-4o가 검색된 페이지를 직접 읽고도 "세 번째 아이콘"까지만 서술, **모양은 못 읽음**. Week 9의 발견("150dpi 전체 페이지에서는 gpt-4o도 표 안 미세 아이콘을 못 읽는다")이 **질의 시점 읽기에서도 재현**됐다. 인덱스측(캡션)이든 질의측(escalation)이든, full-page가 문제다.
- **S2 (조작부 위치):** vision이 "제품 상단"이라는 실제 위치 정보를 추출했으나 두 가지가 겹쳐 거절로 끝남: (a) **에스컬레이션 대상 선정이 모델을 무시** — 질문은 AS281DAW(complex)인데 검색 랭킹 1위인 simple 매뉴얼 p.18을 읽음(rerank 점수: simple 0.024 > complex 0.004), (b) "앞면 어느 위치"에 대해 "상단"은 부분 정보라 답변 모델이 거절. 부분 정보로 답을 지어내지 않은 것 자체는 계약 준수.

**따라서 "VLM이 필요하다 → 영역 단위로 VLM을 써야 한다"(Week 9 결론)가 질의측에서도 확인됐다.** `image_analysis_tool`의 `bbox` 인자가 그 결합 지점이다 — region-crop(P1, §6)이 완성되면 F2가 페이지 대신 **영역 crop**을 읽는다.

## 6. 회고

- **잘 된 것:** tool 계약(스키마+`ok`)이 그래프 로직을 단순하게 유지시킴 — 노드는 dict 키만 본다. 실측 기반 트리거 보정 2건(OCR 글자수 조건, F2 이중 트리거)이 이번 주의 실질 엔지니어링. 3개 fallback 전부 로그로 발동 입증. 안전 계약(거절) 유지.
- **가장 불안정했던 지점:** (1) **grader** — 단독으로는 F2 트리거로 못 쓴다(1차 실행에서 실증). (2) **에스컬레이션 대상 선정** — 검색 랭킹을 그대로 따르므로 질문 속 모델(simple/complex)을 무시한다.
- **12주차 개선점:**
  1. **F2에 region-crop 결합** (P1 `src/region_caption.py` 완성 후 `bbox` 인자로) — S2/S4류의 실질 정답 전환이 목표.
  2. **에스컬레이션 대상의 모델-인지 선정** — Week 6 Self-Query(카테고리 추출)를 확장해 모델명→complexity 매핑, 후보 필터링.
  3. grader를 신호 중 하나로 강등하고 rerank 점수·caption_hit를 함께 쓰는 근거 판정.
  4. region-caption 완성 → Week 9 IR8 재평가 → **ADR-012 작성**(연기 유지).

## 7. P1 결과 — region-crop 캡션 MVP + golden set v3

> 구현: `src/region_caption.py` (검출→300dpi crop→구조화 JSON 캡션, 캐시 `data/week11_captions_crop.json`).
> 구조 워크스루: `w11/week11_region_caption.ipynb` (실행 완료본, 오프라인).

- **파이프라인:** vector drawings 점유 그리드(4pt cell + dilate) → 연결 성분 → 면적 ≥5% bbox → PDF에서 300dpi clip 렌더 → gpt-4o가 crop+주변 인쇄 텍스트를 읽고 `{summary, elements:[{label, shape, position_in_figure, direction}]}` JSON. 페이지 내 위치는 bbox 기하로 계산(VLM에 묻지 않음). IR8 관련 5페이지 → 12개 영역.
- **잡힌 것 (Week 9 실패 6건 대비):** IR-V3 **"연결 끊김 = 원형 + 사선"**(Week 9 내내 못 잡던 빗금), IR-A3 공기제균 아이콘 모양, IR-V2 거치 방향("아래로 끼움"). §5의 상한이 crop 단위에서 실제로 뚫림 — 정량 확인은 Week 12 IR8 재평가.
- **부분적인 것:** IR-A1/V1(부품 위치) — p18 crop이 콜아웃 9개 중 5개만 전사(라벨 리스트가 crop 밖 본문에 있는 구조). margin 확대/콜아웃-라벨 후처리가 Week 12 후보.
- **golden set v3** (`data/eval/golden_set_v3.csv`, 41문항 중 4개 수리, 전부 원본 PDF 대조): Q01(필터별 6/12개월 + ref p.15→p.18), Q06(**1년** — p.35 표를 300dpi crop으로 판독 확정, 기존 "6~12개월"은 매뉴얼에 근거 없음, ref p.18→p.35), Q11(3.5→**4시간**, ref p.12→p.16), Q19(모델 명시 + GT를 매뉴얼 실제 절차로 교체). 수리 내역은 각 행 notes에 기록.
- **미실행(의도):** mm 스토어 리빌드 + IR8 재평가는 Week 12 — 그 결과로 ADR-012.

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-07-13 | 작성 — tool 4종 + LangGraph agent + fallback 3종 + 시나리오 4건 실행. F2 이중 트리거·OCR 글자수 조건은 실측 후 보정. |
| 2026-07-14 | §7 추가 — P1 완료: region-crop 캡션 MVP(12영역, 사선·모양·방향 첫 포착) + golden set v3(4문항 수리). 노트북 2본 실행 완료본으로 갱신. |
