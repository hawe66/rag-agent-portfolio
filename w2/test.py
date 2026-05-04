from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()




'''
-------------------------------------------------------------------------------
'''

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="당신은 친절한 한국어 요리 전문가입니다."),
    HumanMessage(content="파스타 레시피 알려줘"),
    AIMessage(content="네! 카르보나라 레시피를 알려드릴게요..."),  # 이전 응답 (멀티턴 시)
    HumanMessage(content="재료가 없으면 어떻게 해?"),
]

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage(content="당신은 Python 전문가입니다. 간결하게 답하세요."),
    HumanMessage(content="리스트 컴프리헨션이 뭔가요?"),
]

# response = llm.invoke(messages)
# print(response.content)         # 텍스트 응답
# print(response.usage_metadata)  # 토큰 사용량

import asyncio

# compare sync vs async
# compute time for sync
import time
question1 = "파스타 레시피 알려줘"
question2 = "재료가 없으면 어떻게 해?"
question3 = "칼로리는 어떻게 돼?"
def sync_call():
    start_time = time.time()
    result1 = model.invoke(question1)
    result2 = model.invoke(question2)
    result3 = model.invoke(question3)
    end_time = time.time()
    print("Sync response:", result1.content)
    print("Sync time:", end_time - start_time)

# compute time for async
async def async_call():
    start_time = time.time()
    results = await asyncio.gather(
        model.ainvoke(question1),
        model.ainvoke(question2),
        model.ainvoke(question3)
    )
    end_time = time.time()
    print("Async response:", results[0].content)
    print("Async time:", end_time - start_time)

sync_call()
asyncio.run(async_call())


'''
system_prompt = """당신은 S모 회사의 AI Agent 어시스턴트입니다.

[규칙]
- 기술 용어는 한국어로 먼저 설명하고 영어 원어를 괄호 안에 표기하세요.
- 확실하지 않은 정보는 "확인이 필요합니다"라고 명시하세요.
- 답변은 300자 이내로 유지하세요.

[금지]
- 경쟁사 제품을 직접 비교하지 마세요.
"""
'''