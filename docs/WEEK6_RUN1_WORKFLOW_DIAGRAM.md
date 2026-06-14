# Week 6 Agentic RAG Workflow Diagram

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	retrieve(retrieve)
	grade_documents(grade_documents)
	rewrite_query(rewrite_query)
	generate(generate)
	cannot_answer(cannot_answer)
	__end__([<p>__end__</p>]):::last
	__start__ --> retrieve;
	grade_documents -.-> cannot_answer;
	grade_documents -.-> generate;
	grade_documents -.-> rewrite_query;
	retrieve --> grade_documents;
	rewrite_query --> retrieve;
	cannot_answer --> __end__;
	generate --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## 노드 설명

- **retrieve**: Hybrid+Rerank 검색 (5주차 전략). metadata_filter 적용 가능
- **grade_documents**: LLM Judge로 관련성 판단 (relevant/not_relevant)
- **rewrite_query**: Self-Query (카테고리 추출) 또는 Query 재작성 (키워드 추가)
- **generate**: RAG 답변 생성
- **cannot_answer**: 2회 retry 후 답변 거절
