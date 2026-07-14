"""
Week 11 agent tools (WEEK11_TASKS §2).

Four single-responsibility tools for the multimodal agent (`src/mm_agent.py`).
Contract shared by all tools:

- fixed output schema (documented per tool below),
- failure is signaled explicitly via ``ok=False`` + ``error`` — never swallowed,
- no tool searches or answers inside another tool's responsibility.

Heavy resources (retriever, reranker, LLM, OpenAI client) are injected via
``make_*`` factories, matching the existing project style
(``make_text_retriever``, ``make_filterable_hybrid_retriever``).

Domain truth (LIM-002 / WEEK11_TASKS §2): manual figures are vector line-art,
so OCR is expected to be weak on diagram pages — a *low OCR confidence is a
feature here*, it is the trigger for the agent's Vision escalation fallback.
"""

import base64
import io
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

OCR_LANG = "kor+eng"
VISION_MODEL = "gpt-4o"  # same model/dpi family as the Week 9 v2 captions

REFUSAL_TEXT = "제공된 문서에서 확인할 수 없습니다"

# Mirrors mm_eval_v1.ANSWER_PROMPT (same source-tag / citation contract as the
# ADR-011 anchor runs). Duplicated deliberately: importing mm_eval_v1 drags in
# the CLIP stack, which the agent does not need resident.
ANSWER_PROMPT = """다음 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트 앞에 `[제품_복잡도 p.페이지]` 형태의 출처 태그가 붙어 있습니다.
컨텍스트에 있는 정보만 사용하세요. 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.
답변 끝에 `(출처: 제품_복잡도 p.페이지)` 형태로 사용한 근거의 출처를 명시하세요.

컨텍스트:
{context}

질문: {question}

답변:"""

IMAGE_ANALYSIS_PROMPT = """당신은 LG 가전 매뉴얼 페이지 이미지를 읽는 분석가입니다.
질문과 관련된 시각 정보(부품의 위치, 아이콘의 모양, 화살표 방향, 콜아웃 라벨)를 이미지에서 직접 찾아 서술하세요.
이미지에서 확인되지 않는 내용은 추측하지 말고 "확인 불가"라고 하세요.

질문: {question}

JSON으로만 답하세요 (다른 텍스트 없이):
{{"summary": "질문과 관련해 이미지에서 확인한 내용", "confidence": 0.0에서 1.0 사이 숫자}}"""


# ---------------------------------------------------------------------------
# ocr_tool — text extraction ONLY (no search, no answering)
# ---------------------------------------------------------------------------


def ocr_tool(image_path: str) -> dict:
    """Extract printed text from a page image with tesseract (kor+eng).

    Returns ``{"text": str, "confidence": float 0..1, "ok": bool}``.
    ``confidence`` is the mean of tesseract's word-level confidences;
    a page with no recognizable words returns ``ok=True, confidence=0.0``
    (the OCR ran — finding nothing is a *result*, not an error).
    """
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(image_path)
        data = pytesseract.image_to_data(
            image, lang=OCR_LANG, output_type=pytesseract.Output.DICT
        )
        words = [
            (token.strip(), conf)
            for token, conf in zip(data["text"], data["conf"])
            if token.strip() and conf != -1
        ]
        text = " ".join(token for token, _ in words)
        confidence = (
            sum(conf for _, conf in words) / len(words) / 100.0 if words else 0.0
        )
        return {"text": text, "confidence": round(confidence, 3), "ok": True}
    except Exception as exc:
        return {"text": "", "confidence": 0.0, "ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# image_analysis_tool — query-side VLM read (gpt-4o vision)
# ---------------------------------------------------------------------------


def image_analysis_tool(
    image_path: str,
    question: str,
    bbox: tuple[int, int, int, int] | None = None,
    client: OpenAI | None = None,
    model: str = VISION_MODEL,
) -> dict:
    """Read a page image (optionally a bbox crop) against a question.

    Returns ``{"image_summary": str, "confidence": float 0..1, "ok": bool}``.
    ``bbox`` = (left, top, right, bottom) pixel crop — the seam that the
    Week 11 §6 region-crop work plugs into.
    """
    try:
        client = client or OpenAI(max_retries=8)
        path = Path(image_path)
        image_bytes = path.read_bytes()
        if bbox is not None:
            from PIL import Image

            cropped = Image.open(path).crop(tuple(bbox))
            buffer = io.BytesIO()
            cropped.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_ANALYSIS_PROMPT.format(question=question)},
                    {"type": "image_url",
                     # detail="high" — small icons/arrows vanish at low detail
                     # (confirmed in the Week 9 caption smoke test).
                     "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                ],
            }],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(raw)
            return {
                "image_summary": str(parsed.get("summary", "")),
                "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))),
                "ok": True,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            # The vision call succeeded but the model broke the JSON contract;
            # keep the paid-for content, flag reduced confidence.
            return {"image_summary": raw, "confidence": 0.5, "ok": True}
    except Exception as exc:
        return {"image_summary": "", "confidence": 0.0, "ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# rag_search_tool — caption→text RAG (Week 9's adopted cross-modal medium)
# ---------------------------------------------------------------------------


def make_rag_search_tool(retriever, reranker):
    """Wrap the modality-aware retriever (`src/mm_retrieval.py`) as a tool.

    The returned tool maps ``query`` to
    ``{"docs": list[Document], "scores": list[float], "caption_hit": bool, "ok": bool}``.
    ``scores`` are cross-encoder relevance scores for the final docs (the
    retriever itself discards them); ``caption_hit`` = at least one
    image-derived (caption) chunk reached the context.
    """

    def rag_search_tool(query: str) -> dict:
        try:
            import torch

            docs = retriever.invoke(query)
            if docs:
                pairs = [(query, doc.page_content) for doc in docs]
                with torch.inference_mode():
                    scores = [float(s) for s in reranker.predict(pairs)]
            else:
                scores = []
            caption_hit = any(
                doc.metadata.get("modality") == "image-derived" for doc in docs
            )
            return {"docs": docs, "scores": scores, "caption_hit": caption_hit, "ok": bool(docs)}
        except Exception as exc:
            return {"docs": [], "scores": [], "caption_hit": False, "ok": False, "error": str(exc)}

    return rag_search_tool


# ---------------------------------------------------------------------------
# answer_generation_tool — grounded generation with explicit refusal signal
# ---------------------------------------------------------------------------


def make_answer_generation_tool(llm):
    """Wrap the answer LLM. The returned tool maps ``(context, question)`` to
    ``{"answer": str, "is_grounded": bool, "ok": bool}``.

    ``is_grounded=False`` when the model invoked the refusal contract
    ("제공된 문서에서 확인할 수 없습니다") — the agent turns that into an
    explicit refusal instead of shipping an unsupported answer.
    """

    def answer_generation_tool(context: str, question: str) -> dict:
        try:
            answer = llm.invoke(
                ANSWER_PROMPT.format(context=context, question=question)
            ).content.strip()
            is_grounded = REFUSAL_TEXT not in answer
            return {"answer": answer, "is_grounded": is_grounded, "ok": True}
        except Exception as exc:
            return {"answer": "", "is_grounded": False, "ok": False, "error": str(exc)}

    return answer_generation_tool
