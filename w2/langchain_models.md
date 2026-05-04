# LangChain Models
> 출처: https://docs.langchain.com/oss/python/langchain/models + https://reference.langchain.com/python/langchain/models

---

## Basic Usage

LLM은 텍스트 생성 외에도 **Tool calling**, **Structured output**, **Multimodality**, **Reasoning**을 지원한다.  
모델은 두 가지 방식으로 사용된다:
1. **With agents** – 에이전트 루프 안에서 동적으로 사용
2. **Standalone** – 에이전트 없이 직접 호출 (텍스트 생성, 분류, 추출 등)

### Initialize a model

```python
from langchain.chat_models import init_chat_model

# 권장 방식 (provider-agnostic)
model = init_chat_model("gpt-4o-mini")                          # OpenAI (prefix로 자동 추론)
model = init_chat_model("claude-sonnet-4-6")                    # Anthropic
model = init_chat_model("google_genai:gemini-2.5-flash-lite")   # Google (provider:model 형식)
model = init_chat_model("azure_openai:gpt-4o-mini", ...)        # Azure

# 이전 방식 (provider-specific, deprecated 아님. provider 전용 기능 필요시 사용)
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")
```

### 🧪 실험 포인트
| 실험 | 코드 |
|---|---|
| provider 자동 추론 확인 | `init_chat_model("gpt-4o-mini")` vs `init_chat_model("openai:gpt-4o-mini")` |
| configurable model (런타임에 모델 교체) | 아래 참고 |

```python
# configurable model: 런타임에 모델 교체 가능
configurable = init_chat_model(temperature=0)
configurable.invoke("hi", config={"configurable": {"model": "gpt-4o"}})
configurable.invoke("hi", config={"configurable": {"model": "claude-sonnet-4-6"}})
```

---

## Supported Models

`init_chat_model`의 `model` 인자에서 prefix로 provider 자동 추론:

| prefix | provider | 패키지 |
|---|---|---|
| `gpt-...`, `o1...`, `o3...` | openai | `langchain-openai` |
| `claude...` | anthropic | `langchain-anthropic` |
| `gemini...` | google_vertexai | `langchain-google-vertexai` |
| `amazon...` | bedrock | `langchain-aws` |
| `mistral...` | mistralai | `langchain-mistralai` |
| `deepseek...` | deepseek | `langchain-deepseek` |
| `grok...` | xai | `langchain-xai` |

provider를 명시하려면 `"provider:model"` 형식 또는 `model_provider=` kwarg 사용.

전체 목록: https://docs.langchain.com/oss/python/integrations/providers/overview

---

## Key Methods

### Imperative Methods (모델을 실제로 호출)

| 메서드 | 반환 타입 | 설명 |
|---|---|---|
| `invoke()` | `AIMessage` | 완성된 응답 한 번에 반환 |
| `ainvoke()` | `AIMessage` | async invoke |
| `stream()` | `Iterator[AIMessageChunk]` | 동기 스트리밍 |
| `astream()` | `AsyncIterator[AIMessageChunk]` | 비동기 스트리밍 |
| `astream_events()` | `AsyncIterator[StreamEvent]` | 이벤트 타입별 스트리밍 |
| `batch()` | `list[AIMessage]` | 병렬 처리 |
| `abatch()` | `list[AIMessage]` | async batch |
| `batch_as_completed()` | `Iterator[tuple[int, AIMessage]]` | 완료 순서대로 yield (인덱스 포함) |
| `abatch_as_completed()` | `AsyncIterator[tuple[int, AIMessage]]` | async batch_as_completed |
| `get_num_tokens(text)` | `int` | 텍스트의 토큰 수 반환 |
| `get_num_tokens_from_messages(messages)` | `int` | 메시지 리스트의 토큰 수 반환 |

### Declarative Methods (새로운 Runnable 생성)

| 메서드 | 설명 |
|---|---|
| `bind_tools(tools)` | tool을 바인딩한 모델 반환 |
| `with_structured_output(schema)` | 스키마에 맞게 출력을 구조화하는 wrapper 반환 |
| `with_retry(...)` | 실패 시 재시도하는 wrapper 반환 |
| `with_fallbacks(fallbacks)` | 실패 시 다른 모델로 fallback하는 wrapper 반환 |
| `with_config(config)` | config를 바인딩한 새 Runnable 반환 |
| `with_listeners(on_start, on_end, on_error)` | lifecycle 리스너 바인딩 |
| `configurable_fields(...)` | 런타임에 변경 가능한 init 인자 지정 |
| `configurable_alternatives(...)` | 런타임에 교체 가능한 대체 모델 지정 |
| `bind(...)` | 인자를 고정한 새 Runnable 반환 |
| `pipe(...)` | Runnable 연결 (`\|` 연산자와 동일) |
| `as_tool(...)` | 모델을 BaseTool로 변환 |
| `map()` | 입력 리스트 → 출력 리스트 매핑 |

### invoke() vs ainvoke()

둘의 차이는 **동기 vs 비동기** 하나다.

```python
import asyncio

# invoke: 동기. 응답 올 때까지 현재 스레드 블로킹
response = model.invoke("안녕하세요")
print(response.content)         # 텍스트
print(response.usage_metadata)  # {"input_tokens": N, "output_tokens": N}

# dict 형식 멀티턴
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user",      "content": "Translate: I love building applications."},
]
response = model.invoke(conversation)

# Message 객체 형식 (동일 결과)
from langchain.messages import HumanMessage, AIMessage, SystemMessage
conversation = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications."),
]
response = model.invoke(conversation)
```

> ⚠️ 반환값이 `str`이면 Chat model이 아니라 Legacy LLM을 사용 중인 것. Chat model은 반드시 "Chat" prefix (`ChatOpenAI` 등).

`ainvoke`가 실질적으로 빨라지는 경우 — 여러 요청을 진짜 동시에 보낼 때:

```python
async def main():
    results = await asyncio.gather(
        model.ainvoke("질문1"),
        model.ainvoke("질문2"),
        model.ainvoke("질문3"),
    )
    # 세 요청이 동시에 나가고 동시에 받음
    # invoke를 루프로 3번 돌리는 것보다 빠름

asyncio.run(main())
```

> `batch()`도 내부적으로 스레드 풀로 병렬 처리하지만, `asyncio.gather + ainvoke`는 I/O 기반 비동기라 스레드 오버헤드가 없음.

### stream() vs astream()

마찬가지로 **동기 vs 비동기** 차이.

```python
# stream: 동기. for 루프 도는 동안 현재 스레드 블로킹
for chunk in model.stream("긴 글 써줘"):
    print(chunk.content, end="", flush=True)
    # chunk 타입: AIMessageChunk

# astream: 비동기. 각 chunk 기다리는 동안 event loop이 다른 작업 처리 가능
async def main():
    async for chunk in model.astream("긴 글 써줘"):
        print(chunk.content, end="", flush=True)
```

FastAPI에서 클라이언트에 실시간 스트리밍 응답을 보내는 전형적인 패턴:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/chat")
async def chat(question: str):
    async def generate():
        async for chunk in model.astream(question):
            yield chunk.content  # chunk 즉시 클라이언트로 전송

    return StreamingResponse(generate(), media_type="text/plain")
```

> ⚠️ FastAPI의 async endpoint 안에서 `stream()`을 쓰면 event loop을 블로킹하므로 반드시 `astream()` 사용.

```python
# chunk 누적 → 전체 AIMessage 조립 (+ 연산자 지원, sync/async 공통)
full = None
for chunk in model.stream("설명해주세요"):
    full = chunk if full is None else full + chunk
print(full.content)
print(full.content_blocks)  # [{"type": "text", "text": "..."}]

# astream_events: 이벤트 타입별 필터링 가능 (on_chat_model_start/stream/end)
async for event in model.astream_events("Hello"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].text)
```

> 📌 LangGraph agent 안에서 `invoke()`를 써도 streaming mode 실행 시 LangChain이 내부적으로 자동 streaming으로 전환함.

**sync vs async 요약:**

| | 동기 | 비동기 |
|---|---|---|
| 단일 호출 | `invoke` | `ainvoke` |
| 스트리밍 | `stream` | `astream` |
| 배치 | `batch` | `abatch` |
| 사용 환경 | 스크립트, Jupyter | FastAPI, LangGraph, asyncio |

### batch()

```python
# 기본 batch (모든 응답이 완료된 후 리스트로 반환)
responses = model.batch([
    "파이썬이란?",
    "자바스크립트란?",
    "러스트란?",
])
for r in responses:
    print(r.content)

# batch_as_completed: 완료 순서대로 yield (순서 보장 X, 인덱스 포함)
for (idx, response) in model.batch_as_completed(["질문1", "질문2", "질문3"]):
    print(f"[{idx}]", response.content)

# max_concurrency 제한
model.batch(
    list_of_inputs,
    config={"max_concurrency": 5},
)
```

> ⚠️ `batch()`는 client-side 병렬 처리. OpenAI / Anthropic의 서버-side Batch API와는 다름.

### with_retry() / with_fallbacks()

```python
# with_retry: 지정 exception 발생 시 자동 재시도
model_with_retry = model.with_retry(
    retry_if_exception_type=(ValueError,),
    stop_after_attempt=3,
    wait_exponential_jitter=True,  # 지수 백오프 + jitter
)

# with_fallbacks: 첫 번째 모델 실패 시 대체 모델 사용
fallback_model = model.with_fallbacks(
    fallbacks=[init_chat_model("claude-sonnet-4-6")],
    exceptions_to_handle=(Exception,),
)
response = fallback_model.invoke("안녕하세요")
```

### with_listeners()

```python
# 호출 시작/종료/오류 시 콜백 실행
model_with_listeners = model.with_listeners(
    on_start=lambda run: print(f"시작: {run.id}"),
    on_end=lambda run: print(f"종료: {run.id}, 출력: {run.outputs}"),
    on_error=lambda run: print(f"오류: {run.error}"),
)
response = model_with_listeners.invoke("안녕하세요")
```

### get_num_tokens / get_num_tokens_from_messages

```python
# 토큰 수 사전 확인 (과금/컨텍스트 초과 방지)
n = model.get_num_tokens("안녕하세요, 반갑습니다.")
print(n)  # int

from langchain.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("리스트와 튜플의 차이는?"),
]
n = model.get_num_tokens_from_messages(messages)
print(n)  # int
```

### as_tool()

```python
# 모델 자체를 tool로 변환 → 다른 agent가 이 모델을 tool로 호출 가능
model_tool = model.as_tool(
    name="summarizer",
    description="텍스트를 요약합니다.",
)
```

### 🧪 실험 포인트
| 실험 | 코드 포인트 |
|---|---|
| `invoke` vs `stream` 체감 속도 차이 | 긴 응답으로 비교 |
| chunk `+` 연산으로 AIMessage 조립 | `full = chunk if full is None else full + chunk` |
| `batch_as_completed` 순서 비교 | 결과 idx와 실제 순서 불일치 확인 |
| `max_concurrency` 적용 | batch에 `config={"max_concurrency": 2}` |
| `with_retry` 동작 확인 | 실패하는 모델에 retry 설정 후 재시도 횟수 확인 |
| `with_fallbacks` 동작 확인 | 첫 번째 모델에 잘못된 key → fallback 모델로 응답 오는지 확인 |
| `get_num_tokens_from_messages` | max_tokens 설정 전 토큰 수 사전 계산 |
| `with_listeners` | on_start/on_end에서 run.id, run.outputs 확인 |

## BaseChatModel Attributes

`init_chat_model` / `ChatOpenAI` 등 초기화 후 인스턴스에서 접근 가능한 속성.

| 속성 | 타입 | 설명 |
|---|---|---|
| `disable_streaming` | `bool \| 'tool_calling'` | streaming bypass 여부. `'tool_calling'`이면 tool 호출 시에만 invoke로 fallback |
| `output_version` | `'v0' \| 'v1'` | AIMessage 출력 포맷. `'v1'`은 표준화된 content_blocks 형식 |
| `profile` | `ModelProfile \| None` | 모델 capability 메타데이터 (beta). context window, tool calling 지원 등 |
| `rate_limiter` | `BaseRateLimiter \| None` | 요청 속도 제한기 |
| `cache` | `BaseCache \| bool \| None` | 응답 캐시 여부 |
| `verbose` | `bool` | 응답 텍스트 출력 여부 |
| `callbacks` | `Callbacks` | run trace에 추가할 콜백 |
| `tags` | `list[str] \| None` | run trace에 추가할 태그 |
| `metadata` | `dict \| None` | run trace에 추가할 메타데이터 |

```python
# disable_streaming 예시
model = init_chat_model("gpt-4o-mini", disable_streaming="tool_calling")
# → tool 없이 stream() 호출 시: streaming
# → bind_tools() 후 stream() 호출 시: invoke로 자동 fallback

# profile 확인 (beta)
print(model.profile)
# ModelProfile(context_window=128000, supports_tool_calling=True, ...)
```

---

## Parameters

`init_chat_model()` 또는 `ChatOpenAI()` 등 초기화 시 `**kwargs`로 전달.

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `model` | str | (필수) | 모델 이름. `"openai:o1"` 형식으로 provider 포함 가능 |
| `api_key` | str | - | provider 인증 키. 보통 환경변수로 설정 |
| `temperature` | float | - | 출력 무작위성. 높을수록 창의적, 낮을수록 결정론적 |
| `max_tokens` | int | - | **output 토큰 수 제한** (input/reasoning 무관) |
| `timeout` | float | - | 응답 대기 최대 시간 (초) |
| `max_retries` | int | **6** | 재시도 횟수. 네트워크/429/5xx 자동 재시도. 401/404는 재시도 안 함 |
| `base_url` | str | - | 커스텀 API endpoint URL |
| `rate_limiter` | BaseRateLimiter | - | 요청 속도 제한기 |

```python
model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=10,   # 불안정 네트워크 환경에서 증가 권장
)
```

> `max_tokens`는 **output 전용**. reasoning 모델(o1, o3 등)의 reasoning 토큰을 제한하려면 provider별 `max_reasoning_tokens` 같은 별도 파라미터 필요.

### configurable_fields 옵션

| 값 | 동작 |
|---|---|
| `None` (기본) | 고정 모델, 런타임 변경 불가 |
| `"any"` | 모든 필드 런타임 변경 가능 (⚠️ `api_key`, `base_url`도 포함 → 보안 주의) |
| `["model", "temperature"]` | 지정 필드만 변경 가능 |

```python
# config_prefix로 여러 configurable model 구분
model = init_chat_model(
    "openai:gpt-4o",
    configurable_fields="any",
    config_prefix="llm",
    temperature=0,
)
model.invoke("hi", config={"configurable": {"llm_model": "claude-sonnet-4-6", "llm_temperature": 0.5}})
```

### 🧪 실험 포인트
| 실험 | 코드 포인트 |
|---|---|
| `max_tokens` 낮게 설정 후 응답 잘림 확인 | `max_tokens=20` |
| `temperature=0` vs `temperature=1.5` 출력 차이 | 같은 질문 반복 |
| `max_retries=0`으로 실패 즉시 확인 | 잘못된 API key로 테스트 |
| configurable model 런타임 교체 | `config={"configurable": {"model": "..."}}` |

---

## Advanced Topics (개요만)

docs의 나머지 섹션 목록 (필요시 개별 탐색):

- **Model profiles** – 모델 capability 메타데이터 (structured_output 지원 여부 등)
- **Multimodal** – 이미지/오디오/영상 입력
- **Reasoning** – o1, o3 등 reasoning 모델 파라미터
- **Local models** – Ollama 등
- **Prompt caching** – Anthropic 등 일부 provider 지원
- **Server-side tool use** – provider 내장 tool (web search 등)
- **Rate limiting** – `BaseRateLimiter`
- **Log probabilities** – `logprobs` 파라미터
- **Token usage** – `response.usage_metadata`
- **Invocation config** – `RunnableConfig` (tags, metadata, callbacks 등)
- **Configurable models** – 런타임 model/provider 교체

---

## 참고 링크

- Docs: https://docs.langchain.com/oss/python/langchain/models
- Reference (init_chat_model): https://reference.langchain.com/python/langchain/models
- BaseChatModel methods: https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/
- Integrations: https://docs.langchain.com/oss/python/integrations/chat
