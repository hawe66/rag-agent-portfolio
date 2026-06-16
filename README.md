# Advanced RAG & Multimodal AI Agent 포트폴리오 스터디

## 스터디 소개
현재 LLM 시장에서 요구하는 기술력을 기반으로 설계한
14주 완성 포트폴리오 스터디입니다.

## 최종 산출물
- Advanced RAG 시스템 GitHub repo
- Multimodal AI Agent GitHub repo
- benchmark 결과표
- 발표 슬라이드

## 14주 커리큘럼
| 단계 | 주차 | 주제 |
|---|---|---|
| 기초 정비 | 1~2주 | 협업 환경, LLM 앱 기본기 |
| Advanced RAG | 3~8주 | Baseline RAG, 전처리, Retrieval 고도화, Agentic RAG, Evaluation |
| Multimodal Agent | 9~12주 | Vision LLM, Multimodal RAG, Agent 구현 |
| 프로덕션 | 13~14주 | 서빙·배포, 포트폴리오 패키징 |

## 환경 세팅
1. Python 3.11 설치
2. 패키지 설치
pip install -r requirements.txt
3. .env.example을 .env로 복사 후 API 키 입력

## ADR 작성법
adr/ADR-000-template.md 참고
기술 선택할 때마다 왜 이 기술을 선택했는지 기록

## 보안 (Guardrail) 고려사항

성능·평가에 더해, 이 Agentic RAG를 현업에서 안전하게 운영하려면 예측 불가능한 입력과 악의적 context 속에서도 통제되는지 설명할 수 있어야 한다. 본 시스템은 **closed-domain**(LG 가전 매뉴얼 한정, 웹 검색·외부 도구 미사용 — ADR-009)이라 공격 표면이 좁다. 고려한 주요 리스크는 (1) 사용자 질문을 통한 직접 prompt injection/jailbreak, (2) 매뉴얼 PDF 본문·메타데이터를 통한 **간접(indirect) prompt injection**, (3) 검색된 context를 과신해 환각·탈선하는 경우, (4) Phase 2에서 도구(vision·검색 등)가 추가될 때 늘어나는 공격 표면이다. 현재는 검색 외 도구가 없고, **근거 부족 시 거절하는 로직(retry≤2 후 cannot_answer)과 출처 인용(Citation) 강제**가 기본 output guardrail 역할을 한다. 다만 검색된 매뉴얼 텍스트를 그대로 신뢰하므로, 외부에서 매뉴얼을 수집·갱신하게 되면 **context(retrieved chunk) 단계의 guardrail이 필요**해진다. 도구가 추가되는 Phase 2부터는 도구별 최소 권한 제어와 호출 검증을 재평가한다.

**LlamaFirewall 구성요소를 현재 4노드 그래프에 매핑하면 (가드레일 부착 위치):**

- **PromptGuard 2** → ① `retrieve` 직전 **입력(사용자 질문)** 스캔(jailbreak/injection 탐지), ② `grade`/`generate`에 들어가는 **retrieved context** 스캔(간접 injection 방어).
- **Agent Alignment Checks** → `grade_documents`·`rewrite_query`·라우팅 결정이 원래 목표(매뉴얼 QA)에서 이탈하지 않는지 점검(현재 라우팅이 단순해 리스크는 낮음).
- **CodeShield** → 현재 코드 생성·실행이 없어 **N/A**. Phase 2에서 도구 실행이 생기면 도입 검토.

**향후 개선 계획:** Phase 2 도구 도입 시 입력/context guardrail(PromptGuard 2) 시범 적용 + 도구별 권한 제어, 간접 injection 테스트 케이스를 평가셋에 추가. 가드레일 부착 위치 다이어그램(`docs/` 아키텍처 그림)도 그때 함께 갱신한다.
