"""Unit tests for ragas result merging.

The week7 notebook (cell 10) merged ragas output into per-question results
using the key `response_relevancy`, but RAGAS 0.4.3's pandas output column
is `answer_relevancy`. That mismatch raised KeyError after `evaluate()`.

These tests pin the merge behavior so future RAGAS column renames cannot
re-break the notebook silently.
"""

from __future__ import annotations

import math
import unittest

import pandas as pd

from src.ragas_helpers import merge_ragas_scores


class MergeRagasScoresTests(unittest.TestCase):
    """Verify ragas df → results merge handles 0.4.3 column names."""

    def _ragas_df(self, answer_relevancy_values: list[float]) -> pd.DataFrame:
        """Fixture: minimal ragas 0.4.3 output frame."""
        return pd.DataFrame({
            "user_input": ["q1", "q2"],
            "response": ["a1", "a2"],
            "faithfulness": [1.0, 0.5],
            "answer_relevancy": answer_relevancy_values,
            "context_precision": [1.0, 0.75],
            "context_recall": [1.0, 0.5],
        })

    def test_merges_answer_relevancy_under_response_relevancy_key(self):
        results = [{"question": "q1"}, {"question": "q2"}]
        ragas_df = self._ragas_df([0.9, 0.8])

        merged = merge_ragas_scores(results, ragas_df)

        self.assertEqual(merged[0]["response_relevancy"], 0.9)
        self.assertEqual(merged[1]["response_relevancy"], 0.8)

    def test_copies_all_four_ragas_metrics(self):
        results = [{"question": "q1"}, {"question": "q2"}]
        ragas_df = self._ragas_df([0.9, 0.8])

        merged = merge_ragas_scores(results, ragas_df)

        for row in merged:
            for key in ("faithfulness", "response_relevancy",
                        "context_precision", "context_recall"):
                self.assertIn(key, row, f"missing {key}")

    def test_preserves_original_result_fields(self):
        results = [{"question": "q1", "q_type": "factual"},
                   {"question": "q2", "q_type": "safety"}]
        ragas_df = self._ragas_df([0.9, 0.8])

        merged = merge_ragas_scores(results, ragas_df)

        self.assertEqual(merged[0]["q_type"], "factual")
        self.assertEqual(merged[1]["q_type"], "safety")

    def test_does_not_mutate_input_results(self):
        results = [{"question": "q1"}, {"question": "q2"}]
        ragas_df = self._ragas_df([0.9, 0.8])

        _ = merge_ragas_scores(results, ragas_df)

        self.assertNotIn("faithfulness", results[0])
        self.assertNotIn("response_relevancy", results[0])

    def test_accepts_response_relevancy_column_when_already_present(self):
        results = [{"question": "q1"}]
        ragas_df = pd.DataFrame({
            "faithfulness": [0.7],
            "response_relevancy": [0.6],
            "context_precision": [0.5],
            "context_recall": [0.4],
        })

        merged = merge_ragas_scores(results, ragas_df)

        self.assertEqual(merged[0]["response_relevancy"], 0.6)

    def test_handles_nan_relevancy_without_crashing(self):
        results = [{"question": "q1"}]
        ragas_df = pd.DataFrame({
            "faithfulness": [1.0],
            "answer_relevancy": [float("nan")],
            "context_precision": [1.0],
            "context_recall": [1.0],
        })

        merged = merge_ragas_scores(results, ragas_df)

        self.assertTrue(math.isnan(merged[0]["response_relevancy"]))


if __name__ == "__main__":
    unittest.main()
