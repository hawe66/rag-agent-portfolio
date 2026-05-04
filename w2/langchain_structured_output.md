# LangChain Structured Output
> 출처: https://docs.langchain.com/oss/python/langchain/structured-output

---

## 개요

Structured output은 agent가 **예측 가능한 형식**으로 데이터를 반환하게 한다.  
자연어 응답을 파싱하는 대신, JSON 객체 / Pydantic 모델 / dataclass를 바로 얻는다.

> ⚠️ 이 페이지는 **`create_agent`를 사용한 agent 컨텍스트**의 structured output을 다룬다.  
> 모델을 직접 (agent 없이) 사용하는 경우: https://docs.langchain.com/oss/python/langchain/models#structured-output

---

## Response Format

`create_agent`의 `response_format` 파라미터로 제어.  
결과는 agent 최종 state의 **`structured_response`** 키에 담긴다.

```python
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],    # tool calling 방식
        ProviderStrategy[StructuredResponseT], # provider native 방식
        type[StructuredResponseT],             # 자동 선택
        None,                                  # structured output 미사용
    ]
)
```

### 자동 전략 선택 (schema type 직접 전달 시)

```
schema type 직접 전달
      ↓
provider가 native structured output 지원? (OpenAI, Anthropic, Gemini, xAI)
      ├─ YES → ProviderStrategy
      └─ NO  → ToolStrategy
```

### 🧪 실험 포인트
| 실험 | 코드 포인트 |
|---|---|
| 자동 전략 선택 vs 명시적 전략 결과 비교 | `response_format=MySchema` vs `response_format=ProviderStrategy(MySchema)` |
| `result["structured_response"]` 타입 확인 | Pydantic이면 인스턴스, TypedDict이면 dict |

---

## Provider Strategy

provider가 native API로 structured output을 지원하는 경우. **가장 신뢰도 높음**.

```python
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]      # 필수
    strict: bool | None = None # OpenAI, xAI만 지원 (langchain>=1.2 필요)
```

지원 provider: OpenAI, Anthropic (Claude), Gemini, xAI (Grok)

### 지원 schema 타입

| 타입 | 반환값 |
|---|---|
| Pydantic `BaseModel` | validated Pydantic 인스턴스 |
| Python dataclass | dict |
| TypedDict | dict |
| JSON Schema (dict) | dict |

### 예시 (Pydantic)

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str  = Field(description="The name of the person")
    email: str = Field(description="The email address")
    phone: str = Field(description="The phone number")

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # 자동으로 ProviderStrategy 선택
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract: John Doe, john@example.com, (555) 123-4567"}]
})
print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

> 아래 두 코드는 **동일하게 동작** (provider가 지원하는 경우):
> ```python
> response_format=ProductReview               # 자동 선택
> response_format=ProviderStrategy(ProductReview)  # 명시적
> ```
> provider가 미지원이면 둘 다 ToolStrategy로 fallback.

---

## Tool Calling Strategy

native structured output 미지원 모델에서 tool calling을 이용해 구조화된 출력을 유도.  
tool calling을 지원하는 대부분의 현대 모델에서 작동한다.

```python
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]                  # 필수
    tool_message_content: str | None       # 기본: 구조화 응답 데이터 표시
    handle_errors: Union[
        bool,                              # True: 모든 에러 catch (기본값)
        str,                               # 고정 에러 메시지로 재시도
        type[Exception],                   # 특정 exception만 catch
        tuple[type[Exception], ...],       # 여러 exception
        Callable[[Exception], str],        # 커스텀 핸들러
    ]
```

### 지원 schema 타입

ProviderStrategy와 동일 + **Union types** 추가 지원:

| 타입 | 반환값 |
|---|---|
| Pydantic `BaseModel` | validated Pydantic 인스턴스 |
| Python dataclass | dict |
| TypedDict | dict |
| JSON Schema (dict) | dict |
| `Union[SchemaA, SchemaB]` | 모델이 가장 적절한 schema 선택 |

### 예시

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="Rating 1-5", ge=1, le=5)
    sentiment: Literal["positive", "negative"]
    key_points: list[str] = Field(description="1-3 words each, lowercase")

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze: 'Great product, 5/5. Fast shipping but expensive'"}]
})
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

### Custom tool_message_content

대화 히스토리에 남는 ToolMessage 내용을 커스터마이즈:

```python
agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!"
        # 기본값: "Returning structured response: {'task': '...', ...}"
    )
)
```

---

## Error Handling

모델이 structured output 생성 시 실수할 수 있다. LangChain은 자동 retry 메커니즘을 제공.

### 에러 유형 1: Multiple Structured Outputs

모델이 Union 스키마에서 여러 tool을 잘못 호출하는 경우 → 자동 retry:

```
[모델 응답] ContactInfo + EventDetails 동시 호출
     ↓
[ToolMessage] Error: Model incorrectly returned multiple structured responses...
     ↓
[모델 재시도] ContactInfo만 호출
```

### 에러 유형 2: Schema Validation Error

Pydantic 유효성 검사 실패 시 (예: `rating=10` when max=5):

```
[모델 응답] rating: 10
     ↓
[ToolMessage] Error: 1 validation error for ProductRating.rating
              Input should be less than or equal to 5
     ↓
[모델 재시도] rating: 5
```

### Error Handling 전략

```python
# 1. True (기본): 모든 에러 catch + 기본 에러 메시지
ToolStrategy(schema=ProductRating, handle_errors=True)

# 2. 고정 문자열: 모든 에러에 이 메시지로 재시도 유도
ToolStrategy(
    schema=ProductRating,
    handle_errors="Please provide a valid rating between 1-5 and include a comment."
)

# 3. 특정 exception만 catch (나머지는 raise)
ToolStrategy(schema=ProductRating, handle_errors=ValueError)

# 4. 여러 exception
ToolStrategy(schema=ProductRating, handle_errors=(ValueError, TypeError))

# 5. 커스텀 핸들러 함수
from langchain.agents.structured_output import StructuredOutputValidationError, MultipleStructuredOutputsError

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"

ToolStrategy(schema=Union[ContactInfo, EventDetails], handle_errors=custom_error_handler)

# 6. False: 에러 즉시 raise (retry 없음)
ToolStrategy(schema=ProductRating, handle_errors=False)
```

### 🧪 실험 포인트
| 실험 | 코드 포인트 |
|---|---|
| ProviderStrategy vs ToolStrategy 내부 동작 차이 확인 | 대화 히스토리 비교 (`result["messages"]`) |
| schema validation error 유도 | `rating: int = Field(ge=1, le=5)`로 범위 초과 입력 |
| Union 스키마에서 multiple outputs 에러 유도 | `Union[ContactInfo, EventDetails]`에 두 정보 혼재 입력 |
| `handle_errors=False`로 exception 직접 catch | try-except로 감싸서 테스트 |
| `strict=True` (ProviderStrategy, OpenAI) | 엄격한 schema 준수 강제 |
| `tool_message_content` 커스터마이즈 | `result["messages"]` 마지막 ToolMessage 내용 확인 |

---

## 참고 링크

- Docs: https://docs.langchain.com/oss/python/langchain/structured-output
- Models의 with_structured_output (agent 없이 사용): https://docs.langchain.com/oss/python/langchain/models#structured-output
- Reference (ToolStrategy): https://reference.langchain.com/python/langchain/agents/structured_output/
- Reference (create_agent): https://reference.langchain.com/python/langchain/agents/factory/create_agent
