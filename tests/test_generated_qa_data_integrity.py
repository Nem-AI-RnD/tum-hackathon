"""
Additional integrity tests for the generated QA dataset.

These tests ensure that every item in `data/generated_qa_data_tum.json`
is validated individually and that higher level helpers such as
`to_deepeval_format` keep metadata in sync with the raw dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

import pytest

from ai_eval.resources.rag_prototype import QADataItem, RAGConfig, RAGPrototype

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "generated_qa_data_tum.json"
)

if not DATASET_PATH.exists():
    pytestmark = pytest.mark.skip(
        reason=f"Generated QA dataset missing at {DATASET_PATH}"
    )
    RAW_DATASET: List[dict] = []
else:
    with open(DATASET_PATH, "r", encoding="utf-8") as fp:
        RAW_DATASET = json.load(fp)

QA_ITEMS: List[QADataItem] = [QADataItem(**item) for item in RAW_DATASET]


def _make_rag_instance() -> RAGPrototype:
    """
    Create a minimal RAGPrototype instance without standing up infra.

    Matches the lightweight fixture used in other tests but is defined locally
    to avoid circular fixture dependencies.
    """

    config = RAGConfig(
        qdrant_url=os.getenv("QDRANT_URL", "https://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "test_key"),
        jina_api_key=os.getenv("JINA_API_KEY", "test_key"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        llamaparse_api_key=os.getenv("LLAMAPARSE_API_KEY"),
        collection_name="test_collection_integrity",
        force_recreate=True,
    )

    rag = RAGPrototype.__new__(RAGPrototype)
    rag.config = config
    return rag


def _dataset_ids(items: Iterable[QADataItem]) -> List[str]:
    """Provide deterministic pytest ids for parametrized cases."""

    ids: List[str] = []
    for item in items:
        question_fragment = item.question.strip().replace("\n", " ")
        ids.append(f"{item.index:03d}-{question_fragment[:40]}")
    return ids


@pytest.mark.parametrize(
    "qa_item",
    QA_ITEMS,
    ids=_dataset_ids(QA_ITEMS),
)
def test_generated_dataset_item_integrity(qa_item: QADataItem) -> None:
    """Validate each QA entry individually."""

    assert qa_item.index >= 0
    assert qa_item.question.strip(), "Question must not be empty"
    assert qa_item.answer.strip(), "Answer must not be empty"
    assert qa_item.context.strip(), "Context must not be empty"
    assert qa_item.location_dependency_evaluator_target_answer.strip(), (
        "Location dependency answer must be present"
    )

    # Questions should be phrased as questions for clarity.
    assert qa_item.question.strip().endswith("?"), (
        f"Question should end with '?': {qa_item.question}"
    )

    # Target answers in the dataset are expected to match the main answer.
    assert (
        qa_item.location_dependency_evaluator_target_answer.strip()
        == qa_item.answer.strip()
    ), "Target answer must match canonical answer"

    # Ensure evaluation metadata exists and scores remain inside the valid range.
    for score_name in (
        "groundedness_score",
        "question_relevancy_score",
        "faithfulness_score",
    ):
        score_value = getattr(qa_item, score_name)
        assert 1 <= score_value <= 5, f"{score_name} out of allowed range"

    for eval_name in (
        "groundedness_eval",
        "question_relevancy_eval",
        "faithfulness_eval",
    ):
        eval_value = getattr(qa_item, eval_name)
        assert eval_value and eval_value.strip(
        ), f"{eval_name} must be provided"


def test_dataset_indices_and_questions_are_unique() -> None:
    """Indices should be sequential and every question should be unique."""

    indices = [item.index for item in QA_ITEMS]
    assert indices == list(range(len(QA_ITEMS))), "Indices must be sequential"

    questions = [item.question.strip().lower() for item in QA_ITEMS]
    assert len(questions) == len(set(questions)), "Questions must be unique"


def test_to_deepeval_format_matches_dataset() -> None:
    """Ensure to_deepeval_format preserves all crucial metadata."""

    rag = _make_rag_instance()
    deepeval_cases = rag.to_deepeval_format(QA_ITEMS)

    assert len(deepeval_cases) == len(QA_ITEMS)

    for qa_item, case in zip(QA_ITEMS, deepeval_cases, strict=True):
        assert case["input"] == qa_item.question
        assert case["expected_output"] == qa_item.answer
        assert case["actual_output"] == qa_item.answer
        assert case["context"] == [qa_item.context]
        metadata = case["metadata"]
        assert metadata["index"] == qa_item.index
        assert metadata["groundedness_score"] == qa_item.groundedness_score
        assert metadata["question_relevancy_score"] == qa_item.question_relevancy_score
        assert metadata["faithfulness_score"] == qa_item.faithfulness_score
        assert metadata["target_answer"] == qa_item.location_dependency_evaluator_target_answer


def test_real_dataset_statistics_have_reasonable_values() -> None:
    """Compute stats across the full dataset to guard against regressions."""

    rag = _make_rag_instance()
    stats = rag.get_qa_statistics(QA_ITEMS)

    assert stats["total_items"] == len(QA_ITEMS)
    assert stats["quality_metrics"]["overall_average"] >= 3.0
    assert 0 <= stats["quality_metrics"]["high_quality_percentage"] <= 100

    for score_name, score_stats in stats["score_statistics"].items():
        assert 1 <= score_stats["min"] <= score_stats["max"] <= 5, (
            f"Invalid min/max for {score_name}"
        )
        assert (
            sum(score_stats["distribution"].values()) == len(QA_ITEMS)
        ), f"Distribution mismatch for {score_name}"
