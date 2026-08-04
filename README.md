# 매뉴얼 도면을 읽는 RAG — Advanced RAG에서 Cross-modal Agent까지

**LG 가전 매뉴얼(한국어 PDF 6종)을 대상으로, 텍스트 RAG → 크로스모달 RAG → Agent로
단계적으로 진화시킨 12주 포트폴리오 프로젝트.**

## 무엇이 문제인가

가전 매뉴얼은 정보를 두 곳에 나눠 담는다. **텍스트에는 조건·주의사항**이, **그림에는 모양·위치·방향**이 있다.

> "AS181DAW 공기청정기에서 조작부는 제품의 어느 면에 있나요?"

본문은 "제품의 기능을 설정할 수 있습니다"라는 **기능만** 적고, 조작부가 윗면에 있다는 사실은
**도면의 지시선에만** 있다. 텍스트 전용 RAG는 정답 페이지를 정확히 찾아내고도 답하지 못한다.

실제로 이런 질문 27개를 직접 만들어 측정한 결과, **텍스트 전용 RAG의 정답은 0/27**이었다
(그중 23건은 지어내지 않고 정직하게 거절했다).

## 어떻게 접근했나

```mermaid
graph LR
    subgraph P1["Phase 1 · 텍스트 RAG (3~8주)"]
        A[PDF 파싱<br/>layout-aware] --> B[Hybrid 검색<br/>BM25+Dense RRF] --> C[Cross-encoder<br/>Rerank] --> D[Agentic RAG<br/>LangGraph 4노드]
    end
    subgraph P2["Phase 2 · 크로스모달 (9~12주)"]
        E[페이지 rasterize] --> F[영역 검출<br/>vector drawing 그리드] --> G[영역 크롭 300dpi<br/>→ VLM 구조화 캡션] --> H[modality-aware 검색<br/>텍스트3 + 캡션2]
    end
    D --> H
    H --> I[멀티모달 Agent<br/>tool 4종 + 폴백]
```

**핵심 설계 판단**: 도면을 이미지 임베딩(CLIP)으로 검색하지 않고, **VLM으로 구조화된 텍스트 캡션을
만들어 기존 텍스트 인덱스에 합류**시켰다. 선화(line-art) 도면 + 한국어 라벨에서 CLIP은
캡션 대비 75% vs 12%로 완패했다 (ADR-012).

## Benchmark

| 주차 | 무엇을 했나 | 핵심 수치 |
|---|---|---|
| 3주 | Baseline (Dense-only) | Top-1 **60.9%** |
| 5주 | Hybrid(BM25+Dense, RRF) + Reranker | Top-1 **91.3%** (21/23), Top-5 100%, 6.73s |
| 6주 | Agentic RAG (LangGraph 4노드 + Self-Query) | Top-1 **95.7%** (23q), 27.01s (4.01x) |
| 7주 | 평가 프레임 (RAGAS + 도메인 메트릭) | Citation model **91.9%**, Faithfulness 0.87 |
| 9주 | 크로스모달 1차 (페이지 캡션) | caption-hit 2/8 → **6/8**, 캡션 75% vs CLIP 12% |
| 11주 | 멀티모달 Agent (tool 4종 + 폴백) | full-page vision으로는 아이콘·위치 **못 뚫음** (정직한 거절) |
| **12주** | **영역 캡션 3-way 비교 (n=27)** | **정답 C0 0/27 → C1page 1/27 → C1region 8/27** |

12주차 상세 (image-required 27문항, 수동 채점):

| 구성 | 정답 | 정답+부분 | 거절 | 지연 |
|---|---|---|---|---|
| C0 텍스트만 | 0/27 | 0/27 | 23 (전부 적절) | 24.3s |
| C1page 페이지 캡션 | 1/27 | 5/27 | 17 (전부 적절) | 25.1s |
| **C1region 영역 캡션** | **8/27** | **12/27** | 13 (5건 부적절) | 25.7s |

유형별로 보면 이득은 **아이콘에 집중**됐다(8문항 중 6문항 정답). **방향은 세 구성 모두 0/4.**
→ `docs/WEEK12_EVALUATION.md`, `adr/ADR-012-crossmodal-conclusion.md`

## 알려진 한계 (정직하게)

- **표본이 작다.** image-required 27문항, 유형별 n=4~8. 1문항이 12~25%p를 움직인다.
- **방향 유형은 실패했다** (0/4). 영역 캡션이 화살표를 기술한 경우에도 생성 단계가
  "화살표 방향으로 열어야 합니다"라는 동어반복을 냈다.
- **근거가 있는데 거절한 사례 6건**(F-GEN). 현재 수치는 캡션 정보량 한계와 생성 실패가 섞인 값이다.
- **영역 캡션은 15페이지에만 존재한다.** 골든셋이 지목한 페이지만 크롭했으므로, 252페이지 전면
  확장 시에도 같은 이득이 나오는지는 **측정하지 않았다**.
- **채점자 = 작성자.** RAGAS·LLM judge를 쓰지 않고 수동 채점했으나, 그 대가로 채점자 편향이 있다.
- **모델 라우팅**: simple/complex 페어에서 잘못된 매뉴얼을 검색하는 사례가 남아 있다(IR-W6).

## 실행 방법

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # OPENAI_API_KEY 입력

# 텍스트 인덱스 → 크로스모달 인덱스
.venv/bin/python -m src.rasterize               # 페이지 이미지 (150dpi)
.venv/bin/python -m src.region_caption week12   # 영역 검출 + 크롭 + VLM 캡션

# 12주차 3-way 평가 (구성당 1프로세스 — 메모리)
.venv/bin/python -m src.week12_eval C0
.venv/bin/python -m src.week12_eval C1page
.venv/bin/python -m src.week12_eval C1region
.venv/bin/python -m src.week12_eval merge

pytest tests/ -q
```

## 데이터

LG전자 공개 사용설명서 PDF 6종 (한국어). 카테고리 3 × 단순/복잡 1개씩:

| 카테고리 | 단순 | 복잡 |
|---|---|---|
| 정수기 | WD325A** | WD520A** |
| 공기청정기 | AS181DAW | AS281DAW |
| 청소기 | K83 (유선) | O958 (무선) |

출처: [LG전자 제품 매뉴얼](https://www.lge.co.kr/support/product-manuals). 총 252페이지.
평가셋 `data/eval/golden_set_v4.csv` 61문항은 **원본 PDF를 직접 보고 손으로 작성**했다.

## 문서

| ADR | 결정 |
|---|---|
| [ADR-008](adr/ADR-008-retrieval-strategy.md) | 검색 전략 — Hybrid(BM25+Dense, RRF) + Reranker |
| [ADR-009](adr/ADR-009-agentic-rag.md) | Agentic RAG (LangGraph 4노드 + Self-Query) |
| [ADR-010](adr/ADR-010-evaluation-framework.md) | 평가 프레임워크 |
| [ADR-011](adr/ADR-011-crossmodal-eval.md) | 크로스모달 평가 프레임 + Run-2 보정 |
| [**ADR-012**](adr/ADR-012-crossmodal-conclusion.md) | **크로스모달 최종 결론 (영역 캡션 채택, 범위 한정)** |

ADR-001~007(도메인·언어·데이터 선정 등)은 `CLAUDE.md` §2에 요약되어 있다.
주차별 상세: `docs/WEEK*_TASKS.md`, `docs/WEEK*_RETROSPECTIVE.md`,
[12주 전체 회고](docs/WEEK12_RETROSPECTIVE.md).

## ADR 작성법
adr/ADR-000-template.md 참고. 기술 선택할 때마다 왜 이 기술을 선택했는지 기록.

## 보안 (Guardrail) 고려사항

성능·평가에 더해, 이 Agentic RAG를 현업에서 안전하게 운영하려면 예측 불가능한 입력과 악의적 context 속에서도 통제되는지 설명할 수 있어야 한다. 본 시스템은 **closed-domain**(LG 가전 매뉴얼 한정, 웹 검색·외부 도구 미사용 — ADR-009)이라 공격 표면이 좁다. 고려한 주요 리스크는 (1) 사용자 질문을 통한 직접 prompt injection/jailbreak, (2) 매뉴얼 PDF 본문·메타데이터를 통한 **간접(indirect) prompt injection**, (3) 검색된 context를 과신해 환각·탈선하는 경우, (4) Phase 2에서 도구(vision·검색 등)가 추가될 때 늘어나는 공격 표면이다. 현재는 검색 외 도구가 없고, **근거 부족 시 거절하는 로직(retry≤2 후 cannot_answer)과 출처 인용(Citation) 강제**가 기본 output guardrail 역할을 한다. 다만 검색된 매뉴얼 텍스트를 그대로 신뢰하므로, 외부에서 매뉴얼을 수집·갱신하게 되면 **context(retrieved chunk) 단계의 guardrail이 필요**해진다. 도구가 추가되는 Phase 2부터는 도구별 최소 권한 제어와 호출 검증을 재평가한다.

**LlamaFirewall 구성요소를 현재 4노드 그래프에 매핑하면 (가드레일 부착 위치):**

- **PromptGuard 2** → ① `retrieve` 직전 **입력(사용자 질문)** 스캔(jailbreak/injection 탐지), ② `grade`/`generate`에 들어가는 **retrieved context** 스캔(간접 injection 방어).
- **Agent Alignment Checks** → `grade_documents`·`rewrite_query`·라우팅 결정이 원래 목표(매뉴얼 QA)에서 이탈하지 않는지 점검(현재 라우팅이 단순해 리스크는 낮음).
- **CodeShield** → 현재 코드 생성·실행이 없어 **N/A**. Phase 2에서 도구 실행이 생기면 도입 검토.

**향후 개선 계획:** Phase 2 도구 도입 시 입력/context guardrail(PromptGuard 2) 시범 적용 + 도구별 권한 제어, 간접 injection 테스트 케이스를 평가셋에 추가. 가드레일 부착 위치 다이어그램(`docs/` 아키텍처 그림)도 그때 함께 갱신한다.
