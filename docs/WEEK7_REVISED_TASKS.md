# WEEK 7 REVISED — 평가 정직 마무리 + Cross-modal 측정 프레임 (Phase 1 Closeout, 1+2단계)

> **이 문서는 `WEEK7_TASKS.md`를 대체(supersede)한다.** 기존 명세로 1회 sprint를 돌려 **Golden Set v1(35문항)까지는 만들었으나, RAGAS·Refusal·Citation 측정값이 전부 TBD**이고, 평가 코드는 RAGAS 4지표만 있고 골든셋 CSV 로더·도메인 메트릭이 없다.
> 이 문서의 목표는 두 가지다. **(1단계)** Phase 1 평가를 실측치로 정직하게 닫는다. **(2단계)** 골든셋·로더·메트릭을 **cross-modal을 측정할 수 있는 형태로 재설계**한다 — 지금은 텍스트 baseline만 측정하지만, Phase 2에서 이미지를 붙이면 같은 하네스로 텍스트-only vs cross-modal을 비교할 수 있게.
> 선결: `WEEK6_REVISED_TASKS.md` 완료(특히 `src/agent.py`). 시작 전 `PROJECT_CONTEXT.md`, `docs/week7_retrospective.md`(§4~8 TBD)를 읽는다.
> **경로 주의:** 실제 repo는 `w7/`·`adr/ADR-010-evaluation-framework.md`·`data/eval/golden_set_v1.csv`·`src/evaluation.py` 구조다.

---

## 0. 현재 상태 진단

이미 된 것:
- `data/eval/golden_set_v1.csv` — 35문항, q_type 5종 + `modality_label` 컬럼(text-only/image-required/N/A).
- `src/evaluation.py` — RAGAS 4지표(Faithfulness/ResponseRelevancy/ContextPrecision/ContextRecall) 측정 함수.
- `adr/ADR-010-evaluation-framework.md` — 파일 존재.

안 된 것 (이번에 닫는다):
1. `week7_retrospective.md` §4~8 전부 **TBD** — RAGAS 측정값, RAGAS 한계 사례 2건, Refusal/Citation Accuracy, q_type별 분석 미작성. 결과 json 없음.
2. `src/evaluation.py`에 **Refusal Accuracy·Citation Accuracy 미구현**. RAGAS 4지표만.
3. evaluation.py가 **골든셋 CSV가 아니라 옛 `eval_questions_v2.json`(다른 스키마)을 로드.** 35문항 골든셋용 로더 없음. 스키마 불일치(`requires_image: bool` ↔ `modality_label`).
4. **cross-modal 측정 불가** — `image_required` 질문 3건(Q21~23)이 텍스트로도 답이 되어 변별력 없음. modality별 메트릭 분해 없음. (audit 결과는 §6.)

---

## 1. 이 sprint의 핵심 원칙

- **평가가 cross-modal 효과를 "볼 수 있어야" 닫힌 것이다.** 텍스트 baseline 숫자만 채우는 건 절반이다. image-required에서 텍스트 baseline이 **실제로 실패하는 지점**을 수치로 고정해야, Phase 2에서 cross-modal이 그 gap을 좁히는 걸 측정할 수 있다.
- **단일 진실 공급원(SSOT)은 `golden_set_v1.csv`.** 옛 `eval_questions_v2.json` 경로는 폐기하고 CSV 로더로 통일한다.
- **하네스는 modality-agnostic하게.** 메트릭 계산이 retrieval context의 출처(텍스트 only / 텍스트+이미지)를 몰라도 되게 만들고, "어떤 context를 줄지"만 플래그로 갈아끼운다. → Phase 2에서 코드 재작성 없이 cross-modal 비교.

---

## 2. 우선순위 트랙

| 우선순위 | 작업 | 산출물 | 비고 |
|---|---|---|---|
| **P0** | 골든셋 CSV 로더 + 스키마 통일(modality_label) | `src/evaluation.py` | §3 |
| **P0** | Refusal Accuracy · Citation Accuracy 구현 | `src/evaluation.py` | §4 |
| **P0** | RAGAS 4지표 + 도메인 메트릭 실행(Baseline & Agentic) | `w7/week7_evaluation.ipynb` | §5 |
| **P0** | `week7_retrospective.md` §4~8 실측치로 채움 | retrospective | §5 |
| **P0** | Golden Set v2: image-required 질문 추가 + modality 3분류 | `data/eval/golden_set_v2.csv` | §6 |
| **P0** | modality별 메트릭 분해 + 텍스트-only baseline의 image-required 실패율 | retrospective §9 | §7 |
| **P1** | RAGAS 한계 사례 2건 분석 | retrospective §6 | §5.4 |
| **P1** | cross-modal 전방호환 훅(context 주입 플래그, image_ids 매핑) | `src/evaluation.py` | §8 |
| **P1** | ADR-010 보강 + ADR-011(cross-modal eval) 초안 | `adr/` | §10 |
| **P2** | `PROJECT_CONTEXT.md` Phase 1 종료·ADR 008~010 등재 | PROJECT_CONTEXT | §11 |

---

## 3. 골든셋 로더 + 스키마 통일 (P0)

> 산출물: `src/evaluation.py` 수정.

- `load_golden_set(path: Path) -> list[EvalQuestion]` 추가 — `data/eval/golden_set_v1.csv`(이후 v2)를 읽는다.
- `EvalQuestion` 스키마 통일: 기존 `requires_image: bool`을 **`modality_label: str`**("text-only"|"image-helpful"|"image-required")로 교체. `requires_image`는 `modality_label == "image-required"`로 파생 property만 남김(하위호환).
- 옛 `eval_questions_v2.json` 경로(현 `__main__`/노트북)는 골든셋 CSV로 교체. JSON은 더 이상 SSOT가 아님(원하면 `data/legacy/`로 이동).
- CSV 컬럼: `question, ground_truth, reference_context, q_type, modality_label, notes`. **§8에서 `figure_ref` 컬럼 추가**(전방호환).

검증: 로더가 35(이후 ~43)문항을 누락 없이 파싱, `modality_label` 분포 출력.

---

## 4. 도메인 메트릭 구현 (P0)

> 산출물: `src/evaluation.py`. (현재 미구현 — 신규 함수.)

### 4.1 Refusal Accuracy
```
TP: out_of_scope/safety(거절 대상) 질문에 "제공된 문서에서 확인할 수 없습니다" 류로 올바르게 거절
FP: 답해야 하는 질문(factual/comparison/multi_hop)에 잘못 거절
Refusal Accuracy = (올바른 거절 + 올바른 응답) / 전체
FP rate = 오거절 수 / 답해야 하는 질문 수
```
- 거절 탐지: 답변 텍스트의 거절 패턴 매칭("확인할 수 없습니다" 등) — 패턴 목록을 상수로.

### 4.2 Citation Accuracy
```
답변에 명시된 (model/category, page) 출처가 reference_context와 일치하는가
모델/카테고리 일치 + 페이지 ±2 → 정답 (도메인 휴리스틱)
out_of_scope는 분모에서 제외(출처 없어야 정답)
```
- 출처 추출 regex(예: "AS281DAW p.15", "공기청정기 18쪽") → reference_context와 매칭.
- 4주차 메타데이터(category/page)가 회수되는 지점 — ADR에 명시.

검증: 두 메트릭이 q_type별로 분해되어 나오는지(out_of_scope/safety는 Refusal, factual류는 Citation 중심).

---

## 5. 측정 실행 & 회고 채우기 (P0)

> 산출물: `w7/week7_evaluation.ipynb`, `data/eval/week7_results.json`, `week7_retrospective.md` §4~8.

- **Baseline(Hybrid+Rerank)와 Agentic(`src/agent.run_agent`) 둘 다** 골든셋 전체에 대해 RAGAS 4지표 + Refusal + Citation 측정.
- Judge: gpt-4o-mini(비용) 사용 시 self-preference 한계를 retrospective에 명시. 예산 되면 다른 계열.
- §4 표(RAGAS), §6 Refusal/Citation, §8 q_type별 표를 **실측치**로 채움. §1.1 Agentic 수치(95.7%)와 정합 확인.

### 5.4 RAGAS 한계 사례 2건 (P1)
- 점수 높은데 답 나쁜 사례 / 점수 낮은데 답 괜찮은 사례 각 1건 + 어긋난 지표·원인 가설·필요 보완지표(Evidence Coverage 등). retrospective §5.

---

## 6. Golden Set v2 — Cross-modal 측정 가능하게 (P0, 핵심)

> 산출물: `data/eval/golden_set_v2.csv`. v1(35) → v2(~43).

### 6.1 도면 audit 결과 (이 설계의 근거 — retrospective §10에도 기록)

6개 PDF 도면은 **거의 전부 vector graphics**다(raster는 표지/제품사진 2~3장뿐, vector drawings 4천~28만/매뉴얼). 따라서 **raster XObject 추출 방식(현 `parsing.py` image_markers, pymupdf4llm "picture omitted" 마커)은 지시용 도면을 한 장도 못 잡는다.** 페이지를 rasterize하면 도면이 선명히 읽힌다. → **LIM-002 갱신:** "미해결, Phase 2 첫 장애물" → "정량 확인됨; 추출 경로는 raster XObject가 아니라 **페이지/영역 rasterize → vision 처리**로 확정." (3단계 = 다음 명세 입력.)

### 6.2 modality 3분류 (operational 정의)

- **text-only**: 답이 해당 페이지 본문 텍스트에 존재.
- **image-helpful**: 텍스트로 답 가능하나 그림이 이해를 보조.
- **image-required**: 답(위치/방향/콜아웃 식별)이 **본문 텍스트에 없고 도면에만** 존재. ← cross-modal 측정의 핵심.

### 6.3 신규 image-required 질문 8건 (승인됨)

| ID | 카테고리 | 질문 | 그림이 답하는 것(텍스트 부재) | reference (model · page · figure) | 검증 |
|---|---|---|---|---|---|
| IR-A1 | 공기청정기 | 조작부는 제품 앞면의 어느 위치에 있나요? | 위치(우측 상단), 텍스트는 기능만 | airpurifier_complex p.18 (각 부 명칭 도면) | ✅ 렌더 확인 |
| IR-A2 | 공기청정기 | 상태 표시부는 제품의 어느 쪽에 있나요? | 위치, 텍스트는 기능만 | airpurifier_complex p.18 (각 부 명칭 도면) | ✅ |
| IR-A3 | 공기청정기 | 프리필터(하부 흡입구 커버)는 앞/뒤 어디에 있나요? | 위치(후면 추정) | airpurifier_complex p.14 (필터 청소) | ⚠ 본문 위치서술 부재 최종확인 |
| IR-W1 | 정수기 | 필터 두 개 중 왼쪽 자리(①)에는 어떤 필터를 끼우나요? | 콜아웃 ①② 좌/우 배치, 도면에만 | waterpurifier_complex p.29 (필터 장착) | ✅ |
| IR-W2 | 정수기 | 필터 체결부 화살표는 필터의 어느 면에 맞춰야 하나요? | 화살표/돌출부 면(앞·위), inset에만 | waterpurifier_complex p.29 (▼ inset) | ✅ |
| IR-V1 | 무선청소기 | 흡입구는 길이조절 파이프의 어느 끝에 끼우나요? | 콜아웃 ①③④ 조립 위치, 도면에만 | vacuumcleaner_complex p.15 (제품 조립) | ✅ |
| IR-V2 | 무선청소기 | 흡입구 거치대 걸이는 어느 방향으로 돌리나요? | 곡선 화살표 방향, 텍스트는 방향 생략 | vacuumcleaner_complex p.15 (거치대 inset) | ✅ |
| IR-V3 | 무선청소기 | 먼지통 분리 버튼은 본체의 어느 위치에 있나요? | 위치(상단 손잡이 부근 추정) | vacuumcleaner_complex p.16 (먼지 분리기) | ⚠ 본문 위치서술 부재 최종확인 |

- **⚠ 2건(IR-A3, IR-V3):** 최종 확정 시 해당 페이지 본문에 위치 서술이 정말 없는지 1회 대조 후 ground_truth를 **도면 기준으로** 작성. 있으면 image-helpful로 강등.
- **기존 Q21(필터 회전방향):** image-required가 아니라 "방향이 도면에서 와야 하는데 텍스트로 위장된" 케이스 → IR-W2로 흡수/재도출. Q22/Q23도 IR-A3/IR-V3로 재도출(ground_truth를 도면 기준으로 다시 씀).
- 각 image-required 질문의 `reference_context`는 **모델 + 페이지 + 도면 식별자**를 포함(예: `airpurifier_complex p.18 fig:각부명칭`).

### 6.4 대조군 유지
- text-only/​image-helpful 문항을 함께 두어 modality별 비교가 의미를 갖게 한다. v1의 35문항은 modality_label 재검수(특히 현 image-required 3건 재분류) 후 유지.

---

## 7. modality별 메트릭 분해 + 텍스트-only baseline 실패율 (P0)

> 산출물: retrospective §9(신규 섹션).

- 모든 메트릭(RAGAS 4 + Refusal + Citation + Top-1)을 **modality_label별로 분해**한 표.
- **헤드라인 지표:** "text-only baseline의 image-required 정확도/실패율." 이 값이 **낮게(=실패) 나오는 것이 정상이자 목표** — 그림 없이는 못 푸는 질문이므로. 이 숫자가 Phase 2 cross-modal 효과를 재는 기준선.
- 판정 기준: image-required 질문에서 baseline 답이 (a) 틀림 (b) "확인할 수 없음"으로 회피 (c) 위치/방향을 환각 — 셋 다 실패로 집계.

| modality | n | Top-1 | Faithfulness | Citation | 비고 |
|---|---|---|---|---|---|
| text-only | ? | ? | ? | ? | |
| image-helpful | ? | ? | ? | ? | |
| **image-required** | ~8 | **(낮을 것)** | ? | ? | **cross-modal 기준선** |

---

## 8. Cross-modal 전방호환 훅 (P1)

> 산출물: `src/evaluation.py` 인터페이스 + `golden_set_v2.csv` 컬럼. **구현은 안 함, 자리만.**

- 평가 하네스의 context 생성부를 **주입 가능**하게: `run_eval(..., context_provider=...)`. 기본은 텍스트 retrieval. Phase 2에서 `context_provider`만 cross-modal(텍스트+이미지 캡션/임베딩)로 교체하면 동일 메트릭으로 비교.
- 골든셋에 **`figure_ref` 컬럼** 추가(image-required 질문의 도면 페이지/식별자). Phase 2의 `image_ids` 메타데이터(3주차부터 스키마에 비워둔 필드)와 매핑될 키.
- CLAUDE.md 단순성 원칙: 지금 cross-modal 코드를 쓰지 않는다. 시그니처와 컬럼만 열어둔다.

---

## 9. 산출물 체크리스트

### P0
- [ ] `src/evaluation.py`: 골든셋 로더 + modality_label 스키마 통일 (§3)
- [ ] `src/evaluation.py`: Refusal Accuracy + Citation Accuracy (§4)
- [ ] `w7/week7_evaluation.ipynb` + `data/eval/week7_results.json`: Baseline & Agentic 측정 (§5)
- [ ] `docs/week7_retrospective.md` §4~8 실측치 (§5)
- [ ] `data/eval/golden_set_v2.csv`: image-required 8건 + modality 3분류 (§6)
- [ ] retrospective §9: modality별 분해 + image-required baseline 실패율 (§7)

### P1
- [ ] retrospective §5: RAGAS 한계 사례 2건 (§5.4)
- [ ] `src/evaluation.py`: context_provider 훅 + `figure_ref` 컬럼 (§8)
- [ ] `adr/ADR-010` 보강 + `adr/ADR-011-crossmodal-eval.md` 초안 (§10)

### P2
- [ ] `PROJECT_CONTEXT.md`: Phase 1 종료 반영 + ADR 008~010(+011 초안) 등재 (§11)

---

## 10. ADR (P1)

- **ADR-010 보강:** modality-aware 평가, image-required 정의, 텍스트-only baseline 실패율을 cross-modal 기준선으로 삼는 결정 추가.
- **ADR-011(초안) — Cross-modal 평가 프레임:** Decision(modality 3분류 + 분해 메트릭 + context_provider 훅), Context(LIM-002 vector 확인 → rasterize+vision 경로), Consequence(3단계=이미지 추출·임베딩 구현이 다음 명세). 파일 `adr/ADR-011-crossmodal-eval.md`.

---

## 11. PROJECT_CONTEXT 갱신 (P2)

- §2 ADR 목록에 ADR-008/009/010(+011 초안) 등재.
- §3 sanity check 표 갱신(LIM-002 정량 결과).
- §7 변경이력 + Phase 1 종료 상태 기록. Phase 2 본과제(이미지 추출·임베딩·cross-modal·공간추론)가 다음임을 명시.

---

## 12. 작업 순서

1. §3 로더·스키마 통일 → §4 도메인 메트릭 구현(코드 먼저, SSOT 확정).
2. §6 Golden Set v2(image-required 8건 추가, ⚠ 2건 본문 대조, Q21~23 재도출).
3. §5 Baseline & Agentic 측정 실행 → retrospective §4~8 채움.
4. §7 modality별 분해 + image-required baseline 실패율(헤드라인).
5. (P1) §5.4 RAGAS 한계, §8 전방호환 훅, §10 ADR.
6. (P2) §11 PROJECT_CONTEXT 갱신.

---

## 13. 다음(3단계 = 별도 명세)으로 연결

이 명세가 닫히면 **텍스트-only baseline + image-required 실패율(기준선) + 재사용 가능한 modality-aware 하네스**가 확보된다. 다음 명세(WEEK8+ / Phase 2)는:
- LIM-002 해결: 페이지/영역 **rasterize → vision 처리**로 도면 추출(vector라 XObject 추출 불가 확정).
- 이미지 임베딩 방식 결정(CLIP류 vs Vision LLM), `image_ids`·`figure_ref` 결합.
- `context_provider`를 cross-modal로 교체 → 본 명세의 하네스로 텍스트-only 대비 효과 측정.
- (8주차 guardrail은 별도 경량 작업 — README 보안 단락 + agent 그래프에 guardrail 부착 위치 다이어그램. 발표자 아님.)

---

## 14. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-14 | `WEEK7_TASKS.md` supersede. (1단계) 평가 정직 마무리: 골든셋 로더·스키마 통일·Refusal/Citation 구현·실측 + 회고 채움. (2단계) cross-modal 측정 프레임: image-required 8건·modality 3분류·분해 메트릭·텍스트-only 실패율 기준선·전방호환 훅. 도면 audit(vector 확인)으로 LIM-002 정량화 및 추출경로 확정. 실제 repo 경로 반영. |
