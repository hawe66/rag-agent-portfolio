"""Minimal repro of the RAGAS evaluation cell to capture the actual error.

Temporary diagnostic — delete after fix verified.
"""
import sys
import os
import traceback
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.insert(0, '..')

import csv
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.vectorstore import load_vectorstore
from src.retrieval import HybridRerankerRetriever, HybridRetrieverConfig, RerankConfig


@dataclass
class GoldenQuestion:
    question: str
    ground_truth: str
    reference_context: str
    q_type: str
    modality_label: str
    notes: str


def load_golden_set(path: Path):
    qs = []
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            qs.append(GoldenQuestion(
                question=row['question'],
                ground_truth=row['ground_truth'],
                reference_context=row['reference_context'],
                q_type=row['q_type'],
                modality_label=row['modality_label'],
                notes=row['notes'],
            ))
    return qs


golden = load_golden_set(Path('../data/eval/golden_set_v1.csv'))
print(f"Loaded {len(golden)} questions")

vs = load_vectorstore(Path('../data/chroma_db_c3'), collection_name="lg_manuals_c3")
print(f"VS count: {vs._collection.count()}")

retriever = HybridRerankerRetriever(
    vs,
    hybrid_config=HybridRetrieverConfig(bm25_weight=0.5, dense_weight=0.5),
    rerank_config=RerankConfig(first_stage_k=20, final_k=5),
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

RAG_PROMPT = """다음 컨텍스트를 바탕으로 질문에 답하세요.
반드시 출처(문서명, 페이지)를 명시하세요.
컨텍스트에서 답을 찾을 수 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""


def run_rag(question, retriever, llm):
    docs = retriever.invoke(question)
    parts = []
    for d in docs:
        s = d.metadata.get('source', 'unknown')
        p = d.metadata.get('page', '?')
        parts.append(f"[출처: {s} p.{p}]\n{d.page_content}")
    ctx = "\n\n".join(parts)
    return llm.invoke(RAG_PROMPT.format(context=ctx, question=question)).content, docs


from unittest.mock import MagicMock
_fake = MagicMock()
_fake.ChatVertexAI = MagicMock()
sys.modules["langchain_community.chat_models.vertexai"] = _fake

from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall

from src.ragas_helpers import merge_ragas_scores
ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

eval_qs = [q for q in golden if q.q_type != 'out_of_scope'][:2]
print(f"Evaluating {len(eval_qs)} questions")

results = []
for q in eval_qs:
    resp, docs = run_rag(q.question, retriever, llm)
    results.append({
        'question': q.question,
        'ground_truth': q.ground_truth,
        'response': resp,
        'contexts': [d.page_content for d in docs],
    })

ragas_data = {
    'user_input': [r['question'] for r in results],
    'response': [r['response'] for r in results],
    'retrieved_contexts': [r['contexts'] for r in results],
    'reference': [r['ground_truth'] for r in results],
}
dataset = Dataset.from_dict(ragas_data)

print("Running RAGAS evaluate...")
try:
    metrics = [Faithfulness(), ResponseRelevancy(), ContextPrecision(), ContextRecall()]
    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=ragas_embeddings,
    )
    print("=== SUCCESS ===")
    import pandas as pd
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    df = ragas_result.to_pandas()
    print('columns:', df.columns.tolist())
    merged = merge_ragas_scores(results, df)
    out = pd.DataFrame(merged)[['faithfulness', 'response_relevancy', 'context_precision', 'context_recall']]
    print(out)
    print('NaN counts:', out.isna().sum().to_dict())
except Exception as e:
    print("=== ERROR ===")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {e}")
    traceback.print_exc()
