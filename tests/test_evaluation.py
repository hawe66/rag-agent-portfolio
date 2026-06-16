"""
Unit tests for src.evaluation domain metrics.

Pinned to the citation-regex bug that surfaced in the W7 sprint-1 run
(`_OM_WEB.pdf p.1` filenames broke the old `((?:water|air|vacuum)_\\w+)\\s*p?\\.?(\\d+)`
pattern and silently returned Citation Accuracy = 0%). These tests guard
the fix so a future regex tweak cannot quietly regress to 0% again.
"""

from pathlib import Path

import pytest

from src.evaluation import (
    EvalQuestion,
    citation_accuracy,
    extract_citations,
    is_refusal,
    load_golden_set,
    refusal_accuracy,
)


# ---------------------------------------------------------------------------
# Citation regex — the W7 sprint-1 bug it fixes
# ---------------------------------------------------------------------------


REAL_W7_ANSWER = (
    "정수기 필터 교체 주기는 다음과 같습니다:\n"
    "- 중금속 흡착 필터: 6개월 (10 L/일 사용 기준)\n"
    "- 바이러스 클리어 필터: 12개월 (10 L/일 사용 기준)\n"
    "(출처: waterpurifier_simple_WP_KOR_MFL71817002_03_240524_00_OM_WEB.pdf p.1)"
)


def test_extract_citations_handles_pdf_filename_between_model_and_page():
    """The original bug: a PDF filename between `_simple` and `p.1` killed the match."""
    assert extract_citations(REAL_W7_ANSWER) == [("waterpurifier_simple", 1)]


def test_extract_citations_pulls_all_matches():
    answer = (
        "공기청정기는 (출처: airpurifier_complex_AS281...pdf p.18) 참고.\n"
        "추가로 (출처: airpurifier_simple_AS181...pdf p.14)도 확인하세요."
    )
    assert extract_citations(answer) == [
        ("airpurifier_complex", 18),
        ("airpurifier_simple", 14),
    ]


def test_extract_citations_returns_empty_when_no_citation():
    assert extract_citations("제공된 문서에서 확인할 수 없습니다.") == []


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        "제공된 문서에서 확인할 수 없습니다.",
        "해당 정보가 없는 것으로 보입니다.",
        "문서에서 확인할 수 없는 내용입니다.",
        "정보를 찾을 수 없습니다.",
        "관련 내용이 포함되어 있지 않습니다.",
    ],
)
def test_is_refusal_matches_known_patterns(response: str):
    assert is_refusal(response)


def test_is_refusal_rejects_factual_answer():
    assert not is_refusal("정수기 필터는 6개월마다 교체합니다.")


# ---------------------------------------------------------------------------
# citation_accuracy: end-to-end with one factual + one out_of_scope
# ---------------------------------------------------------------------------


def _q(
    qid: str,
    q_type: str,
    reference_context: str = "",
    modality_label: str = "text-only",
) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        question="(test)",
        category="waterpurifier",
        question_type=q_type,
        retrieval_bias="",
        expected_section="",
        modality_label=modality_label,
        reference_context=reference_context,
    )


def test_citation_accuracy_counts_only_should_cite_questions():
    questions = [
        _q("F", "factual", "waterpurifier_simple p.15 (정수 필터 교체하기)"),
        _q("O", "out_of_scope"),
    ]
    answers = [
        "필터는 6개월마다 교체합니다. (출처: waterpurifier_simple_xxx.pdf p.16)",  # ±2 page tolerance
        "제공된 문서에서 확인할 수 없습니다.",
    ]
    result = citation_accuracy(questions, answers)
    assert result["accuracy"] == 1.0
    assert result["n"] == 1  # out_of_scope excluded
    assert "out_of_scope" not in result["by_q_type"]
    assert result["by_q_type"]["factual"]["correct"] == 1


def test_citation_accuracy_marks_mismatched_page_wrong():
    questions = [_q("F", "factual", "waterpurifier_simple p.15")]
    answers = ["출처: waterpurifier_simple_xxx.pdf p.99"]  # outside ±2
    result = citation_accuracy(questions, answers)
    assert result["accuracy"] == 0.0


def test_citation_accuracy_marks_mismatched_category_wrong():
    questions = [_q("F", "factual", "waterpurifier_simple p.15")]
    answers = ["출처: airpurifier_simple_xxx.pdf p.15"]
    result = citation_accuracy(questions, answers)
    assert result["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# refusal_accuracy: q_type aware
# ---------------------------------------------------------------------------


def test_refusal_accuracy_credits_correct_refusal_and_correct_answer():
    questions = [_q("F", "factual"), _q("O", "out_of_scope")]
    answers = [
        "정수기 필터는 6개월마다 교체합니다.",
        "제공된 문서에서 확인할 수 없습니다.",
    ]
    result = refusal_accuracy(questions, answers)
    assert result["accuracy"] == 1.0
    assert result["fp_rate"] == 0.0


def test_refusal_accuracy_penalizes_wrong_refusal_on_answerable():
    questions = [_q("F", "factual"), _q("O", "out_of_scope")]
    answers = [
        "제공된 문서에서 확인할 수 없습니다.",  # wrongly refused
        "제공된 문서에서 확인할 수 없습니다.",
    ]
    result = refusal_accuracy(questions, answers)
    assert result["accuracy"] == 0.5
    assert result["fp_rate"] == 1.0


# ---------------------------------------------------------------------------
# Loader sanity (light — fuller coverage lives in the W7 notebook smoke run)
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_golden_set_v1_parses_v1_csv():
    questions = load_golden_set(REPO_ROOT / "data" / "eval" / "golden_set_v1.csv")
    assert len(questions) >= 35  # v1 promises 35 (1 row drift tolerated)
    # modality distribution must show non-zero text-only and image-required
    labels = {q.modality_label for q in questions}
    assert "text-only" in labels
    # category derivation should hit the three product lines, plus possibly "unknown"
    categories = {q.category for q in questions}
    assert {"waterpurifier", "airpurifier", "vacuumcleaner"} <= categories
