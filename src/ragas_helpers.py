"""Helpers for the week7 RAGAS evaluation notebook.

RAGAS 0.4.3 emits the response-relevancy metric under the column name
`answer_relevancy`. Downstream notebook code reads it as `response_relevancy`.
This module centralizes the rename so a future RAGAS version change touches
one place.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd


_RAGAS_METRIC_COLUMNS: tuple[str, ...] = (
    "faithfulness",
    "response_relevancy",
    "context_precision",
    "context_recall",
)

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "response_relevancy": ("response_relevancy", "answer_relevancy"),
}


def _pick_column(df: pd.DataFrame, canonical: str) -> str:
    """Resolve a canonical metric name to the actual column in `df`."""
    candidates = _COLUMN_ALIASES.get(canonical, (canonical,))
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"ragas result frame missing column for {canonical!r}; "
        f"tried {candidates}, got {list(df.columns)}"
    )


def merge_ragas_scores(
    results: list[dict[str, Any]],
    ragas_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return a new list of results with ragas metrics merged in.

    Each output row carries the four canonical metrics
    (`faithfulness`, `response_relevancy`, `context_precision`, `context_recall`)
    regardless of whether ragas emitted the new (`answer_relevancy`) or old
    (`response_relevancy`) column name. Input `results` is not mutated.
    """
    if len(results) != len(ragas_df):
        raise ValueError(
            f"results/ragas_df length mismatch: {len(results)} vs {len(ragas_df)}"
        )

    resolved = {m: _pick_column(ragas_df, m) for m in _RAGAS_METRIC_COLUMNS}

    merged: list[dict[str, Any]] = []
    for i, row in enumerate(results):
        new_row = copy.copy(row)
        for canonical, source_col in resolved.items():
            new_row[canonical] = ragas_df.iloc[i][source_col]
        merged.append(new_row)
    return merged
