1. Chat Completion 기본 패턴
   └─ messages 타입 (HumanMessage, SystemMessage, AIMessage)
   └─ system prompt 설계
   └─ streaming 패턴 (추가)

system prompt / user / assistant 역할 설명
system prompt를 어떻게 설계하는가

2. Structured Output
   └─ 개념 / JSON Schema / Pydantic / 비교 예시

Structured Output이 뭔지, 왜 쓰는지
JSON Schema 설계 방법
Pydantic 기반 스키마 정의 및 응답 파싱 방법
(더 추가하면 좋을 내용) 일반 텍스트 출력 vs Structured Output 비교 예시 코드

3. Tool Calling
   └─ 개념 / bind_tools 패턴 / 예시

Tool Calling이 뭔지, 왜 쓰는지
Function Calling 패턴 설명
(더 추가하면 좋을 내용) 간단한 예시 코드 (tool 1~2개 연결)

4. Failure Modes
   └─ hallucination / 스키마 불일치 / tool call 실패 / 실제 예시

hallucination 유형
스키마 불일치 케이스
tool call 실패 케이스
(더 추가하면 좋을 내용) 실제 예시 1~2개