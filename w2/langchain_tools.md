# LangChain Tools
> 출처: https://docs.langchain.com/oss/python/langchain/tools

---

## 개요

Tools는 agent가 실제 세계와 상호작용할 수 있게 확장해준다.  
내부적으로 tool은 **잘 정의된 입출력을 가진 callable 함수**이며, chat model에 전달된다.  
모델은 대화 맥락에 따라 어떤 tool을 언제 어떤 인자로 호출할지 결정한다.

---

## Create Tools

### Basic Tool Definition

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

- **타입 힌트 필수** – tool의 input schema가 타입 힌트로 자동 생성됨
- **docstring = tool description** – 모델이 언제 이 tool을 쓸지 판단하는 기준
- **이름 규칙**: `snake_case` 권장. 공백/특수문자는 일부 provider에서 오류 발생

### Customize Tool Properties

```python
# 커스텀 이름
@tool("web_search")
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # "web_search"

# 커스텀 description
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

### Advanced Schema Definition

복잡한 입력은 Pydantic 또는 JSON Schema로 정의:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius")
    include_forecast: bool = Field(default=False)

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"{location}: {temp}°{units[0].upper()}"
    if include_forecast:
        result += " | Forecast: Sunny"
    return result
```

### Reserved Argument Names

아래 이름은 예약어 — tool 인자로 사용 불가:

| 예약 이름 | 용도 |
|---|---|
| `config` | 내부 RunnableConfig 전달용 |
| `runtime` | ToolRuntime 주입용 |

### 🧪 실험 포인트
| 실험 | 코드 포인트 |
|---|---|
| tool schema 확인 | `search_database.args_schema.schema()` |
| tool name/description 확인 | `search_database.name`, `search_database.description` |
| Pydantic schema vs 일반 타입힌트 비교 | schema 자동 생성 결과 비교 |
| `Literal` 타입으로 선택지 제한 | `Literal["celsius", "fahrenheit"]` |

---

## Access Context (ToolRuntime)

Tool 안에서 런타임 정보에 접근하려면 `runtime: ToolRuntime` 파라미터 사용.  
이 파라미터는 **모델에게 숨겨짐** (tool schema에 나타나지 않음).

| 속성 | 설명 | 사용 예 |
|---|---|---|
| `runtime.state` | 현재 대화 상태 (messages, 커스텀 필드) | 대화 히스토리 접근 |
| `runtime.context` | 호출 시 전달된 불변 설정 (user ID 등) | 사용자 인증 정보 |
| `runtime.store` | 대화 간 영속 저장소 | 사용자 선호도 저장 |
| `runtime.stream_writer` | 실시간 업데이트 emit | 진행 상황 스트리밍 |
| `runtime.execution_info` | thread_id, run_id, 재시도 횟수 | 로깅, 디버깅 |
| `runtime.server_info` | LangGraph Server의 assistant/graph/user 정보 | 서버 환경 구분 |
| `runtime.tool_call_id` | 현재 tool 호출의 고유 ID | ToolMessage 생성 시 필요 |

### Access State

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message from the user."""
    messages = runtime.state["messages"]
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return "No user messages found"
```

### Update State (Command)

```python
from langgraph.types import Command
from langchain.tools import tool

@tool
def set_user_name(new_name: str) -> Command:
    """Set the user's name in the conversation state."""
    return Command(update={"user_name": new_name})
```

### Context (불변 설정값)

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id
    return f"Account: {user_id}"

# 호출 시
agent.invoke(
    {"messages": [...]},
    context=UserContext(user_id="user123")
)
```

### Long-term Memory (Store)

```python
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime

@tool
def save_user_info(user_id: str, name: str, runtime: ToolRuntime) -> str:
    """Save user info to persistent store."""
    runtime.store.put(("users",), user_id, {"name": name})
    return "Saved."

@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Get user info from persistent store."""
    result = runtime.store.get(("users",), user_id)
    return str(result.value) if result else "Not found"
```

> 프로덕션에서는 `InMemoryStore` 대신 `PostgresStore` 사용 권장.

### Stream Writer

```python
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer
    writer(f"Looking up data for: {city}")   # 실시간 업데이트 emit
    writer(f"Data acquired for: {city}")
    return f"It's sunny in {city}!"
```

> `stream_writer`는 LangGraph 실행 컨텍스트 안에서만 동작.

---

## ToolNode

`ToolNode`는 LangGraph 워크플로우에서 tool을 실행하는 prebuilt 노드.  
병렬 tool 실행, 에러 핸들링, state 주입을 자동 처리.

### Basic Usage

```python
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

tool_node = ToolNode([search, calculator])

builder = StateGraph(MessagesState)
builder.add_node("tools", tool_node)
```

### Tool Return Values

| 반환 타입 | 동작 | 언제 사용 |
|---|---|---|
| `str` | ToolMessage로 변환, 모델이 텍스트로 읽음 | 자연어 결과 |
| `dict` | 직렬화 후 tool output으로 전달 | 구조화된 데이터 |
| `Command` | graph state 직접 업데이트 | state 변경 필요 시 |

```python
# str 반환
@tool
def get_weather(city: str) -> str:
    return f"It is currently sunny in {city}."

# dict 반환
@tool
def get_weather_data(city: str) -> dict:
    return {"city": city, "temperature_c": 22, "conditions": "sunny"}

# Command 반환 (state 업데이트 + ToolMessage 포함)
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """Set the preferred response language."""
    return Command(
        update={
            "preferred_language": language,
            "messages": [
                ToolMessage(
                    content=f"Language set to {language}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

### Error Handling

```python
from langgraph.prebuilt import ToolNode

# 기본: invocation 에러는 catch, execution 에러는 재raise
tool_node = ToolNode(tools)

# 모든 에러 catch → LLM에게 에러 메시지 반환
tool_node = ToolNode(tools, handle_tool_errors=True)

# 커스텀 에러 메시지
tool_node = ToolNode(tools, handle_tool_errors="Something went wrong, please try again.")

# 커스텀 핸들러 함수
def handle_error(e: ValueError) -> str:
    return f"Invalid input: {e}"
tool_node = ToolNode(tools, handle_tool_errors=handle_error)

# 특정 exception만 catch
tool_node = ToolNode(tools, handle_tool_errors=(ValueError, TypeError))
```

### Route with tools_condition

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END

builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)  # tool call 있으면 "tools", 없으면 END
builder.add_edge("tools", "llm")
graph = builder.compile()
```

### 🧪 실험 포인트
| 실험 | 코드 포인트 |
|---|---|
| tool schema 자동 생성 확인 | `search.args_schema.schema()` |
| `handle_tool_errors` 각 옵션 동작 확인 | ValueError를 raise하는 tool로 테스트 |
| `Command`로 state 업데이트 확인 | graph state에서 변경사항 확인 |
| `tools_condition` 라우팅 | tool call 있는 응답 vs 없는 응답 분기 |

---

## API Reference: BaseTool / StructuredTool

`@tool` 데코레이터로 만든 함수는 내부적으로 `StructuredTool`(BaseTool 서브클래스) 인스턴스가 된다.

### 클래스 계층

```
Runnable
  └─ RunnableSerializable
       └─ BaseTool          ← 모든 tool의 추상 베이스 클래스
            └─ StructuredTool  ← @tool 데코레이터 결과물
```

### BaseTool Attributes

| 속성 | 타입 | 설명 |
|---|---|---|
| `name` | `str` | tool의 고유 이름. 모델이 tool을 식별할 때 사용 |
| `description` | `str` | 모델이 언제/왜/어떻게 쓸지 판단하는 기준. few-shot 예시 포함 가능 |
| `args_schema` | `ArgsSchema \| None` | Pydantic 모델 또는 JSON schema dict로 입력 검증 |
| `return_direct` | `bool` | `True`이면 tool 호출 후 agent 루프 즉시 종료, 결과를 바로 반환 |
| `handle_tool_error` | `bool \| str \| Callable` | `ToolException` 발생 시 처리 방식 |
| `handle_validation_error` | `bool \| str \| Callable` | `ValidationError` 발생 시 처리 방식 |
| `response_format` | `'content' \| 'content_and_artifact'` | 출력 포맷. `'content_and_artifact'`이면 `(content, artifact)` 튜플 반환 |
| `extras` | `dict \| None` | provider-specific 추가 설정 |
| `verbose` | `bool` | tool 진행 상황 로그 출력 여부 |
| `tags` | `list[str] \| None` | 트레이싱용 태그 |
| `metadata` | `dict \| None` | 트레이싱용 메타데이터 |
| `args` | `dict` | tool의 입력 인자 schema (읽기 전용) |
| `tool_call_schema` | `ArgsSchema` | 주입 인자(`runtime` 등) 제외한 schema (모델에 전달되는 것) |
| `is_single_input` | `bool` | 단일 인자만 받는 tool인지 여부 |

```python
@tool
def search(query: str, limit: int = 10) -> str:
    """Search the database."""
    return f"results for {query}"

print(search.name)              # "search"
print(search.description)       # "Search the database."
print(search.args)              # {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}}
print(search.tool_call_schema)  # runtime 같은 주입 인자 제외한 schema
print(search.is_single_input)   # False (인자가 2개)
```

### BaseTool Methods

| 메서드 | 설명 |
|---|---|
| `invoke(input)` | tool 동기 실행. input은 dict, str, ToolCall 가능 |
| `ainvoke(input)` | tool 비동기 실행 |
| `run(input)` | invoke의 low-level 버전 (직접 호출은 invoke 권장) |
| `arun(input)` | run의 async 버전 |
| `get_input_schema()` | tool의 입력 schema 반환 |

### StructuredTool 추가 Attributes

| 속성 | 타입 | 설명 |
|---|---|---|
| `func` | `Callable \| None` | tool 호출 시 실행되는 동기 함수 |
| `coroutine` | `Callable \| None` | tool 호출 시 실행되는 비동기 함수 |

### StructuredTool.from_function()

`@tool` 데코레이터 없이 명시적으로 StructuredTool을 만드는 방법:

```python
from langchain_core.tools import StructuredTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers asynchronously."""
    return a * b

tool = StructuredTool.from_function(
    func=multiply,
    coroutine=amultiply,   # async 버전 별도 지정 가능
    name="multiply",       # 기본값: 함수명
    description="Multiplies two numbers.",  # 기본값: docstring
)

print(tool.invoke({"a": 3, "b": 4}))  # 12
```

### handle_tool_error / handle_validation_error

BaseTool 수준에서 직접 에러 처리 (ToolNode 없이도 동작):

```python
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException

def risky_tool(x: int) -> str:
    if x < 0:
        raise ToolException("음수는 허용되지 않습니다.")
    return f"결과: {x * 2}"

tool = StructuredTool.from_function(
    func=risky_tool,
    handle_tool_error=True,               # 기본 에러 메시지로 catch
    # handle_tool_error="직접 작성한 에러 메시지",  # 고정 문자열
    # handle_tool_error=lambda e: f"에러: {e}",    # 커스텀 핸들러
)

result = tool.invoke({"x": -1})
print(result)  # "음수는 허용되지 않습니다." (exception 대신 문자열 반환)
```

### response_format: content_and_artifact

tool이 모델에게 보여줄 텍스트(content)와 내부적으로 저장할 데이터(artifact)를 분리해서 반환:

```python
from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def search_with_raw(query: str) -> tuple[str, list]:
    """Search and return results with raw data."""
    raw_results = [{"id": 1, "text": "결과1"}, {"id": 2, "text": "결과2"}]
    summary = f"'{query}'에 대한 결과 {len(raw_results)}건"
    return summary, raw_results  # (모델이 읽는 텍스트, 저장되는 원본 데이터)
```

### 🧪 실험 포인트
| 실험 | 코드 포인트 |
|---|---|
| `tool.args` vs `tool.tool_call_schema` 비교 | `runtime` 파라미터 있는 tool에서 차이 확인 |
| `return_direct=True` 동작 | agent 루프 즉시 종료 확인 |
| `handle_tool_error` 각 타입 | `True` / `str` / `Callable` 결과 비교 |
| `StructuredTool.from_function` vs `@tool` | 동일 결과인지 schema 비교 |
| `response_format="content_and_artifact"` | ToolMessage의 content vs artifact 확인 |
| `coroutine` 별도 지정 | sync/async 각각 다른 구현 연결 |

---

## Prebuilt Tools

LangChain은 web search, code interpreter, DB 접근 등 다양한 prebuilt tool 제공.  
→ https://docs.langchain.com/oss/python/integrations/tools

---

## 참고 링크

- Docs: https://docs.langchain.com/oss/python/langchain/tools
- Reference (BaseTool): https://reference.langchain.com/python/langchain-core/tools/base/BaseTool
- Reference (StructuredTool): https://reference.langchain.com/python/langchain-core/tools/structured/StructuredTool
- Reference (tool decorator): https://reference.langchain.com/python/langchain-core/tools/convert/tool
- Reference (ToolRuntime): https://reference.langchain.com/python/langchain/tools/#langchain.tools.ToolRuntime
- Reference (ToolNode): https://reference.langchain.com/python/langgraph/agents/#langgraph.prebuilt.tool_node.ToolNode
