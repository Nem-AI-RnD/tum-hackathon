"""
Pytest tests for QA dataset evaluation functionality.

These tests focus on:
- Loading and validating QA datasets
- Computing evaluation statistics
- Testing evaluation metrics (groundedness, question_relevancy, faithfulness)
- Dataset operations related to evaluation (merging, appending)
"""

import json
import os
from pathlib import Path
from typing import List

import pytest  # type: ignore[import-untyped]
from dotenv import load_dotenv  # type: ignore[import-untyped]

from ai_eval.resources.rag_prototype import (  # type: ignore[import-untyped]
    QADataItem,
    RAGConfig,
    RAGPrototype,
)

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def sample_qa_dataset() -> List[QADataItem]:
    """Create a sample QA dataset for testing."""
    return [
        QADataItem(
            index=0,
            question="How can I control the height of a pattern element?",
            answer="You must enter a factor in the pattern parameters.",
            location_dependency_evaluator_target_answer="You must enter a factor in the pattern parameters.",
            context="Pattern elements can be controlled using height factors.",
            groundedness_score=5,
            groundedness_eval="The context directly answers the question.",
            question_relevancy_score=5,
            question_relevancy_eval="Highly relevant question.",
            faithfulness_score=5,
            faithfulness_eval="The answer accurately reflects the context.",
        ),
        QADataItem(
            index=1,
            question="What are the two primary workflows for selecting elements?",
            answer="You can select the tool first or the elements first.",
            location_dependency_evaluator_target_answer="You can select the tool first or the elements first.",
            context="Element selection can be done before or after choosing tools.",
            groundedness_score=4,
            groundedness_eval="The context provides relevant information.",
            question_relevancy_score=4,
            question_relevancy_eval="Relevant question.",
            faithfulness_score=4,
            faithfulness_eval="The answer is mostly accurate.",
        ),
        QADataItem(
            index=2,
            question="How do I expand task areas?",
            answer="Use Ctrl+Shift double-click.",
            location_dependency_evaluator_target_answer="Use Ctrl+Shift double-click.",
            context="Task areas can be expanded with Ctrl+Shift double-click.",
            groundedness_score=5,
            groundedness_eval="Direct answer in context.",
            question_relevancy_score=3,
            question_relevancy_eval="Somewhat relevant.",
            faithfulness_score=5,
            faithfulness_eval="Completely accurate.",
        ),
    ]


@pytest.fixture
def sample_qa_dataset_file(sample_qa_dataset: List[QADataItem], tmp_path: Path) -> Path:
    """Create a temporary JSON file with sample QA data."""
    file_path = tmp_path / "test_qa_dataset.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([item.model_dump()
                  for item in sample_qa_dataset], f, indent=2)
    return file_path


@pytest.fixture
def real_qa_dataset_path() -> Path:
    """Path to the real QA dataset file."""
    return Path("data/generated_qa_data_tum.json")


@pytest.fixture
def rag_instance() -> RAGPrototype:
    """Create a minimal RAG instance for testing (no actual deployment needed)."""
    # RAGConfig will automatically load from environment variables via BaseSettings
    # Environment variable names should be: ANTHROPIC_API_KEY, JINA_API_KEY, QDRANT_API_KEY, QDRANT_URL, LLAMAPARSE_API_KEY
    # Since case_sensitive=False, these will map to the config fields
    # For tests, we provide defaults if not set in environment
    config = RAGConfig(
        qdrant_url=os.getenv("QDRANT_URL", "https://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "test_key"),
        jina_api_key=os.getenv("JINA_API_KEY", "test_key"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        llamaparse_api_key=os.getenv("LLAMAPARSE_API_KEY"),
        collection_name="test_collection",
        force_recreate=True,
    )

    # Create RAG instance but skip vector store setup for tests
    # We only need QA dataset functionality, not vector store operations
    rag = RAGPrototype.__new__(RAGPrototype)
    rag.config = config
    # Initialize only what's needed for QA dataset operations
    # Skip _setup_llm(), _setup_embeddings(), _setup_vector_store()
    return rag


class TestQADatasetLoading:
    """Tests for loading and validating QA datasets."""

    def test_load_qa_dataset_from_file(
        self, sample_qa_dataset_file: Path, rag_instance: RAGPrototype
    ):
        """Test loading a QA dataset from a JSON file."""
        qa_dataset = rag_instance.load_qa_dataset(
            sample_qa_dataset_file, validate=True)

        assert len(qa_dataset) == 3
        assert all(isinstance(item, QADataItem) for item in qa_dataset)
        assert qa_dataset[0].question == "How can I control the height of a pattern element?"
        assert qa_dataset[0].groundedness_score == 5

    def test_load_qa_dataset_validation_fails_on_invalid_data(
        self, tmp_path: Path, rag_instance: RAGPrototype
    ):
        """Test that validation fails when loading invalid data."""
        invalid_file = tmp_path / "invalid_qa.json"
        invalid_data = [
            {
                "index": 0,
                "question": "Test question?",
                "answer": "Test answer",
                # Missing required fields
            }
        ]
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump(invalid_data, f)

        with pytest.raises(ValueError, match="Invalid QA item"):
            rag_instance.load_qa_dataset(invalid_file, validate=True)

    def test_load_qa_dataset_validation_skipped(
        self, tmp_path: Path, rag_instance: RAGPrototype
    ):
        """Test that validation can be skipped for invalid data."""
        invalid_file = tmp_path / "invalid_qa.json"
        invalid_data = [{"index": 0, "question": "Test?"}]  # Missing fields
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump(invalid_data, f)

        # Should not raise when validate=False
        qa_dataset = rag_instance.load_qa_dataset(invalid_file, validate=False)
        assert len(qa_dataset) == 0  # Invalid items are skipped

    def test_load_qa_dataset_file_not_found(self, rag_instance: RAGPrototype):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            rag_instance.load_qa_dataset(
                "nonexistent_file.json", validate=True)

    def test_load_qa_dataset_invalid_json_format(
        self, tmp_path: Path, rag_instance: RAGPrototype
    ):
        """Test that ValueError is raised for invalid JSON format."""
        invalid_file = tmp_path / "not_a_list.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)

        with pytest.raises(ValueError, match="must contain a list"):
            rag_instance.load_qa_dataset(invalid_file, validate=True)

    def test_load_real_qa_dataset(
        self, real_qa_dataset_path: Path, rag_instance: RAGPrototype
    ):
        """Test loading the real QA dataset file."""
        if not real_qa_dataset_path.exists():
            pytest.skip(f"Real dataset file not found: {real_qa_dataset_path}")

        qa_dataset = rag_instance.load_qa_dataset(
            real_qa_dataset_path, validate=True)

        assert len(qa_dataset) > 0
        assert all(isinstance(item, QADataItem) for item in qa_dataset)
        # Verify all items have valid scores
        for item in qa_dataset:
            assert 1 <= item.groundedness_score <= 5
            assert 1 <= item.question_relevancy_score <= 5
            assert 1 <= item.faithfulness_score <= 5


class TestQADatasetStatistics:
    """Tests for computing QA dataset statistics."""

    def test_get_qa_statistics_basic(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test basic statistics computation."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        assert stats["total_items"] == 3
        assert "score_statistics" in stats
        assert "quality_metrics" in stats
        assert "content_statistics" in stats

    def test_get_qa_statistics_score_averages(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test that score averages are computed correctly."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        # Groundedness: (5 + 4 + 5) / 3 = 4.67
        assert stats["score_statistics"]["groundedness"]["average"] == pytest.approx(
            4.67, abs=0.01)
        # Question relevancy: (5 + 4 + 3) / 3 = 4.0
        assert stats["score_statistics"]["question_relevancy"]["average"] == pytest.approx(
            4.0, abs=0.01)
        # Faithfulness: (5 + 4 + 5) / 3 = 4.67
        assert stats["score_statistics"]["faithfulness"]["average"] == pytest.approx(
            4.67, abs=0.01)

    def test_get_qa_statistics_score_min_max(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test that min and max scores are computed correctly."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        assert stats["score_statistics"]["groundedness"]["min"] == 4
        assert stats["score_statistics"]["groundedness"]["max"] == 5
        assert stats["score_statistics"]["question_relevancy"]["min"] == 3
        assert stats["score_statistics"]["question_relevancy"]["max"] == 5

    def test_get_qa_statistics_score_distribution(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test that score distributions are computed correctly."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        # Groundedness distribution: 2 items with score 5, 1 with score 4
        dist = stats["score_statistics"]["groundedness"]["distribution"]
        assert dist["5"] == 2
        assert dist["4"] == 1
        assert dist["3"] == 0

    def test_get_qa_statistics_quality_metrics(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test that quality metrics are computed correctly."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        # Overall average: (5+4+5 + 5+4+3 + 5+4+5) / 9 = 4.44
        assert stats["quality_metrics"]["overall_average"] == pytest.approx(
            4.44, abs=0.01)

        # High quality items (all scores >= 4): items 0 and 1
        assert stats["quality_metrics"]["high_quality_items"] == 2
        assert stats["quality_metrics"]["high_quality_percentage"] == pytest.approx(
            66.7, abs=0.1)

    def test_get_qa_statistics_content_statistics(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test that content statistics are computed correctly."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        content_stats = stats["content_statistics"]
        assert "avg_question_length" in content_stats
        assert "avg_answer_length" in content_stats
        assert "avg_context_length" in content_stats

        # All should be positive numbers
        assert content_stats["avg_question_length"] > 0
        assert content_stats["avg_answer_length"] > 0
        assert content_stats["avg_context_length"] > 0

    def test_get_qa_statistics_empty_dataset(self, rag_instance: RAGPrototype):
        """Test statistics for empty dataset."""
        stats = rag_instance.get_qa_statistics([])

        assert stats["total_items"] == 0

    def test_get_qa_statistics_real_dataset(
        self, real_qa_dataset_path: Path, rag_instance: RAGPrototype
    ):
        """Test statistics computation on real dataset."""
        if not real_qa_dataset_path.exists():
            pytest.skip(f"Real dataset file not found: {real_qa_dataset_path}")

        qa_dataset = rag_instance.load_qa_dataset(
            real_qa_dataset_path, validate=True)
        stats = rag_instance.get_qa_statistics(qa_dataset)

        assert stats["total_items"] > 0
        # Verify all score statistics are present
        for score_type in ["groundedness", "question_relevancy", "faithfulness"]:
            score_stats = stats["score_statistics"][score_type]
            assert "average" in score_stats
            assert "min" in score_stats
            assert "max" in score_stats
            assert "distribution" in score_stats
            # Verify averages are within valid range
            assert 1.0 <= score_stats["average"] <= 5.0
            assert 1 <= score_stats["min"] <= 5
            assert 1 <= score_stats["max"] <= 5


class TestQADatasetPrinting:
    """Tests for printing QA dataset statistics."""

    def test_print_qa_statistics(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype, capsys
    ):
        """Test that statistics are printed correctly."""
        rag_instance.print_qa_statistics(sample_qa_dataset)

        captured = capsys.readouterr()
        assert "QA DATASET STATISTICS" in captured.out
        assert "Total Items: 3" in captured.out
        assert "Groundedness:" in captured.out
        assert "Question Relevancy:" in captured.out
        assert "Faithfulness:" in captured.out

    def test_print_qa_statistics_empty_dataset(
        self, rag_instance: RAGPrototype, capsys
    ):
        """Test printing statistics for empty dataset."""
        rag_instance.print_qa_statistics([])

        captured = capsys.readouterr()
        assert "Dataset is empty" in captured.out


class TestQADatasetMerging:
    """Tests for merging QA datasets (evaluation-focused)."""

    def test_merge_qa_datasets_basic(
        self,
        sample_qa_dataset: List[QADataItem],
        tmp_path: Path,
        rag_instance: RAGPrototype,
    ):
        """Test basic dataset merging."""
        # Create two dataset files
        file1 = tmp_path / "dataset1.json"
        file2 = tmp_path / "dataset2.json"
        output = tmp_path / "merged.json"

        dataset1 = sample_qa_dataset[:2]
        dataset2 = sample_qa_dataset[2:]

        with open(file1, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in dataset1], f)
        with open(file2, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in dataset2], f)

        merged = rag_instance.merge_qa_datasets(
            output, file1, file2, deduplicate=False)

        assert len(merged) == 3
        assert merged[0].index == 0
        assert merged[1].index == 1
        assert merged[2].index == 2

    def test_merge_qa_datasets_with_deduplication(
        self,
        sample_qa_dataset: List[QADataItem],
        tmp_path: Path,
        rag_instance: RAGPrototype,
    ):
        """Test dataset merging with deduplication."""
        file1 = tmp_path / "dataset1.json"
        file2 = tmp_path / "dataset2.json"
        output = tmp_path / "merged.json"

        # Create duplicate question in second dataset
        dataset1 = sample_qa_dataset[:2]
        dataset2 = [sample_qa_dataset[0]]  # Duplicate of first item

        with open(file1, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in dataset1], f)
        with open(file2, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in dataset2], f)

        merged = rag_instance.merge_qa_datasets(
            output, file1, file2, deduplicate=True)

        # Should have only 2 items (duplicate removed)
        assert len(merged) == 2

    def test_merge_qa_datasets_reindexing(
        self,
        sample_qa_dataset: List[QADataItem],
        tmp_path: Path,
        rag_instance: RAGPrototype,
    ):
        """Test that merged datasets are properly reindexed."""
        file1 = tmp_path / "dataset1.json"
        file2 = tmp_path / "dataset2.json"
        output = tmp_path / "merged.json"

        dataset1 = sample_qa_dataset[:1]
        dataset2 = sample_qa_dataset[1:]

        with open(file1, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in dataset1], f)
        with open(file2, "w", encoding="utf-8") as f:
            json.dump([item.model_dump() for item in dataset2], f)

        merged = rag_instance.merge_qa_datasets(
            output, file1, file2, deduplicate=False)

        # Verify indices are sequential
        for i, item in enumerate(merged):
            assert item.index == i


class TestQADatasetEvaluationMetrics:
    """Tests for individual evaluation metrics."""

    def test_groundedness_score_validation(
        self, sample_qa_dataset: List[QADataItem]
    ):
        """Test that groundedness scores are within valid range."""
        for item in sample_qa_dataset:
            assert 1 <= item.groundedness_score <= 5
            assert item.groundedness_eval is not None
            assert len(item.groundedness_eval) > 0

    def test_question_relevancy_score_validation(
        self, sample_qa_dataset: List[QADataItem]
    ):
        """Test that question relevancy scores are within valid range."""
        for item in sample_qa_dataset:
            assert 1 <= item.question_relevancy_score <= 5
            assert item.question_relevancy_eval is not None
            assert len(item.question_relevancy_eval) > 0

    def test_faithfulness_score_validation(
        self, sample_qa_dataset: List[QADataItem]
    ):
        """Test that faithfulness scores are within valid range."""
        for item in sample_qa_dataset:
            assert 1 <= item.faithfulness_score <= 5
            assert item.faithfulness_eval is not None
            assert len(item.faithfulness_eval) > 0

    def test_all_scores_present(self, sample_qa_dataset: List[QADataItem]):
        """Test that all required evaluation scores are present."""
        for item in sample_qa_dataset:
            assert hasattr(item, "groundedness_score")
            assert hasattr(item, "question_relevancy_score")
            assert hasattr(item, "faithfulness_score")
            assert hasattr(item, "groundedness_eval")
            assert hasattr(item, "question_relevancy_eval")
            assert hasattr(item, "faithfulness_eval")

    def test_real_dataset_score_validation(
        self, real_qa_dataset_path: Path, rag_instance: RAGPrototype
    ):
        """Test that all scores in real dataset are valid."""
        if not real_qa_dataset_path.exists():
            pytest.skip(f"Real dataset file not found: {real_qa_dataset_path}")

        qa_dataset = rag_instance.load_qa_dataset(
            real_qa_dataset_path, validate=True)

        for item in qa_dataset:
            # Verify score ranges
            assert 1 <= item.groundedness_score <= 5
            assert 1 <= item.question_relevancy_score <= 5
            assert 1 <= item.faithfulness_score <= 5

            # Verify evaluation text is present
            assert len(item.groundedness_eval) > 0
            assert len(item.question_relevancy_eval) > 0
            assert len(item.faithfulness_eval) > 0


class TestQADatasetQualityThresholds:
    """Tests for quality threshold analysis."""

    def test_high_quality_items_filtering(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test filtering items by quality thresholds."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        # Items with all scores >= 4
        high_quality_count = stats["quality_metrics"]["high_quality_items"]
        assert high_quality_count == 2  # Items 0 and 1

        # Manually verify
        high_quality_items = [
            item
            for item in sample_qa_dataset
            if all([
                item.groundedness_score >= 4,
                item.question_relevancy_score >= 4,
                item.faithfulness_score >= 4,
            ])
        ]
        assert len(high_quality_items) == high_quality_count

    def test_score_distribution_analysis(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test analyzing score distributions."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        # Check that distributions sum to total items
        for score_type in ["groundedness", "question_relevancy", "faithfulness"]:
            dist = stats["score_statistics"][score_type]["distribution"]
            total = sum(dist.values())
            assert total == stats["total_items"]


class TestQADatasetContentAnalysis:
    """Tests for content analysis of QA datasets."""

    def test_content_length_statistics(
        self, sample_qa_dataset: List[QADataItem], rag_instance: RAGPrototype
    ):
        """Test that content length statistics are computed."""
        stats = rag_instance.get_qa_statistics(sample_qa_dataset)

        content = stats["content_statistics"]
        assert content["avg_question_length"] > 0
        assert content["avg_answer_length"] > 0
        assert content["avg_context_length"] > 0

        # Verify averages are reasonable
        # Questions shouldn't be too long
        assert content["avg_question_length"] < 1000
        assert content["avg_answer_length"] < 5000  # Answers might be longer
        assert content["avg_context_length"] < 10000  # Contexts can be long

    def test_qa_item_required_fields(self, sample_qa_dataset: List[QADataItem]):
        """Test that all required fields are present in QA items."""
        for item in sample_qa_dataset:
            assert item.index is not None
            assert item.question is not None and len(item.question) > 0
            assert item.answer is not None and len(item.answer) > 0
            assert item.context is not None and len(item.context) > 0
            assert item.location_dependency_evaluator_target_answer is not None
